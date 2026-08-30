#!/usr/bin/env python3
"""Export and validate the hot q=1 VoXtream temp_former step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from torchtune.models import llama3_2
from voxtream_tensorrt_explicit_kv_probe import (
    ExplicitKVStep,
    ExplicitKVTRTRunner,
    attempt,
    kv_buffer_names,
)
from voxtream_tensorrt_probe import make_kv_update_exportable

TEMP_PREFIX = "temp_former."
MAX_SEQ_LEN = 2048
NUM_PHONE_STATES = 6
AUDIO_CODEBOOK_SIZE = 2050
SEMANTIC_TEMPERATURE = 0.8
SEMANTIC_TOP_P = 0.9
SEMANTIC_TOP_K = 50
CFG_GAMMA = 1.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prefill-length", type=int, default=16)
    parser.add_argument("--trajectory-steps", type=int, default=64)
    parser.add_argument("--sampling-seed", type=int, default=86420)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--skip-export", action="store_true")
    return parser.parse_args()


def build_temp_former(checkpoint: Path, device: str, batch_size: int) -> nn.Module:
    model = llama3_2.llama3_2(
        vocab_size=1,
        num_layers=12,
        num_heads=16,
        num_kv_heads=4,
        embed_dim=1024,
        max_seq_len=MAX_SEQ_LEN,
        intermediate_dim=4096,
    )
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        state = {
            key.removeprefix(TEMP_PREFIX): source.get_tensor(key)
            for key in source.keys()
            if key.startswith(TEMP_PREFIX)
        }
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(
            f"temp_former weight mismatch: missing={missing}, unexpected={unexpected}"
        )
    model = model.eval().to(device=device, dtype=torch.bfloat16)
    model.requires_grad_(False)
    with torch.device(device):
        model.setup_caches(batch_size=batch_size, dtype=torch.bfloat16)
    return model


def causal_rows(input_pos: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(MAX_SEQ_LEN, device=input_pos.device).view(1, 1, -1)
    return positions <= input_pos.unsqueeze(-1)


def prepare_hot_step(
    model: nn.Module,
    batch_size: int,
    prefill_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    if not 1 <= prefill_length < MAX_SEQ_LEN:
        raise ValueError(f"prefill length must be in [1, {MAX_SEQ_LEN - 1}]")
    torch.manual_seed(2468)
    model.reset_caches()
    prefill_hidden = torch.randn(
        batch_size,
        prefill_length,
        1024,
        device=device,
        dtype=torch.bfloat16,
    )
    prefill_pos = torch.arange(prefill_length, device=device).unsqueeze(0).repeat(
        batch_size, 1
    )
    with torch.inference_mode():
        model(
            prefill_hidden,
            input_pos=prefill_pos,
            mask=causal_rows(prefill_pos),
        )
    hidden = torch.randn(
        batch_size, 1, 1024, device=device, dtype=torch.bfloat16
    )
    input_pos = torch.full(
        (batch_size, 1), prefill_length, device=device, dtype=torch.int64
    )
    names = kv_buffer_names(model)
    state = tuple(dict(model.named_buffers())[name].clone() for name in names)
    return hidden, input_pos, causal_rows(input_pos), state


def compare_output_and_state(
    reference_output: torch.Tensor,
    candidate_output: torch.Tensor,
    state_names: tuple[str, ...],
    reference_model: nn.Module,
    candidate_state: tuple[torch.Tensor, ...],
) -> dict[str, object]:
    hidden_delta = reference_output.float() - candidate_output.float()
    reference_flat = reference_output.float().flatten()
    candidate_flat = candidate_output.float().flatten()
    cosine = torch.dot(reference_flat, candidate_flat) / (
        torch.linalg.vector_norm(reference_flat)
        * torch.linalg.vector_norm(candidate_flat)
    )
    reference_buffers = dict(reference_model.named_buffers())
    state_deltas = {
        name: float(
            (reference_buffers[name].float() - candidate.float()).abs().max().item()
        )
        for name, candidate in zip(state_names, candidate_state)
    }
    return {
        "hidden": {
            "max_abs": float(hidden_delta.abs().max().item()),
            "mean_abs": float(hidden_delta.abs().mean().item()),
            "rmse": float(torch.sqrt(torch.mean(hidden_delta.square())).item()),
            "cosine": float(cosine.item()),
        },
        "state": {
            "max_abs": max(state_deltas.values()),
            "per_tensor_max_abs": state_deltas,
        },
    }


def multinomial_one(probs: torch.Tensor) -> torch.Tensor:
    """Exact num_samples=1 path used by VoXtream's production sampler."""
    flat = probs.reshape(-1, probs.shape[-1])
    q = torch.empty_like(flat).exponential_(1)
    sampled = (flat / q).argmax(dim=-1, keepdim=True)
    return sampled.reshape(*probs.shape[:-1], 1)


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative - sorted_probs > p
    sorted_probs *= (~mask).float()
    sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))
    sampled = multinomial_one(sorted_probs)
    return torch.gather(sorted_indices, -1, sampled)


def sample_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
    top_probs, top_indices = torch.topk(probs, k, dim=-1)
    sampled = multinomial_one(top_probs)
    return top_indices.gather(-1, sampled)


def sample_semantic_token(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror the Russian demo's state/top-p then semantic/top-k sampler."""
    logits_cond, logits_uncond = torch.split(logits, 1, dim=0)
    cfg_logits = CFG_GAMMA * logits_cond + (1 - CFG_GAMMA) * logits_uncond
    conditional_joint = logits_cond.view(
        -1, NUM_PHONE_STATES, AUDIO_CODEBOOK_SIZE
    )
    cfg_joint = cfg_logits.view(-1, NUM_PHONE_STATES, AUDIO_CODEBOOK_SIZE)

    state_logits = torch.logsumexp(conditional_joint, dim=-1)
    state_probs = torch.nn.functional.softmax(
        state_logits / SEMANTIC_TEMPERATURE, dim=-1
    )
    state_token = sample_top_p(state_probs, SEMANTIC_TOP_P)

    state_index = state_token.unsqueeze(-1).expand(
        -1, -1, AUDIO_CODEBOOK_SIZE
    )
    semantic_logits = torch.gather(cfg_joint, dim=1, index=state_index).squeeze(1)
    semantic_probs = torch.nn.functional.softmax(
        semantic_logits / SEMANTIC_TEMPERATURE, dim=-1
    )
    semantic_token = sample_top_k(semantic_probs, SEMANTIC_TOP_K)
    return semantic_token, state_token.squeeze(1), state_probs, semantic_probs


def validate_engine(
    engine_path: Path,
    checkpoint: Path,
    model: nn.Module,
    state_names: tuple[str, ...],
    batch_size: int,
    prefill_length: int,
    device: str,
    trajectory_steps: int,
    sampling_seed: int,
) -> dict[str, object]:
    hidden, input_pos, mask, initial_state = prepare_hot_step(
        model, batch_size, prefill_length, device
    )
    if not 1 <= trajectory_steps <= MAX_SEQ_LEN - prefill_length:
        raise ValueError("trajectory exceeds temp_former cache length")
    runner = ExplicitKVTRTRunner(
        engine_path, state_names, initial_state, sequence_length=1
    )
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        sem_head = source.get_tensor("sem_head.weight").to(
            device=device, dtype=torch.bfloat16
        )

    torch.manual_seed(9753)
    hidden_steps = torch.randn(
        trajectory_steps,
        batch_size,
        1,
        1024,
        device=device,
        dtype=torch.bfloat16,
    )
    steps = []
    max_hidden_abs = 0.0
    max_logits_abs = 0.0
    torch.cuda.manual_seed_all(sampling_seed)
    for step_index in range(trajectory_steps):
        hidden = hidden_steps[step_index]
        input_pos = torch.full(
            (batch_size, 1),
            prefill_length + step_index,
            device=device,
            dtype=torch.int64,
        )
        mask = causal_rows(input_pos)
        with torch.inference_mode():
            reference_output = model(hidden, input_pos=input_pos, mask=mask)
            candidate_output, candidate_state = runner.step(hidden, input_pos, mask)
            reference_logits = torch.mm(
                reference_output[:, -1, :].to(torch.bfloat16), sem_head.T
            )
            candidate_logits = torch.mm(
                candidate_output[:, -1, :].to(torch.bfloat16), sem_head.T
            )
            reference_cfg = (
                1.5 * reference_logits[0].float()
                - 0.5 * reference_logits[1].float()
            )
            candidate_cfg = (
                CFG_GAMMA * candidate_logits[0].float()
                + (1 - CFG_GAMMA) * candidate_logits[1].float()
            )
            sampling_rng = torch.cuda.get_rng_state(device=hidden.device)
            (
                reference_sampled_token,
                reference_sampled_state,
                reference_state_probs,
                reference_semantic_probs,
            ) = sample_semantic_token(reference_logits)
            reference_rng_after = torch.cuda.get_rng_state(device=hidden.device)
            torch.cuda.set_rng_state(sampling_rng, device=hidden.device)
            (
                candidate_sampled_token,
                candidate_sampled_state,
                candidate_state_probs,
                candidate_semantic_probs,
            ) = sample_semantic_token(candidate_logits)
            candidate_rng_after = torch.cuda.get_rng_state(device=hidden.device)
            if not torch.equal(reference_rng_after, candidate_rng_after):
                raise RuntimeError("reference and candidate sampler consumed different RNG")
        hidden_max_abs = float(
            (reference_output.float() - candidate_output.float()).abs().max().item()
        )
        logits_max_abs = float(
            (reference_cfg - candidate_cfg).abs().max().item()
        )
        reference_top2 = torch.topk(reference_cfg, 2)
        reference_token = int(reference_top2.indices[0].item())
        candidate_token = int(torch.argmax(candidate_cfg).item())
        sampled_state_equal = bool(
            torch.equal(reference_sampled_state, candidate_sampled_state)
        )
        sampled_token_equal = bool(
            torch.equal(reference_sampled_token, candidate_sampled_token)
        )
        state_probability_delta = (
            reference_state_probs.float() - candidate_state_probs.float()
        ).abs()
        semantic_probability_delta = (
            reference_semantic_probs.float() - candidate_semantic_probs.float()
        ).abs()
        reference_top_k = torch.topk(reference_semantic_probs, SEMANTIC_TOP_K).indices
        candidate_top_k = torch.topk(candidate_semantic_probs, SEMANTIC_TOP_K).indices
        semantic_top_k_overlap = len(
            set(reference_top_k.flatten().tolist())
            & set(candidate_top_k.flatten().tolist())
        )
        max_hidden_abs = max(max_hidden_abs, hidden_max_abs)
        max_logits_abs = max(max_logits_abs, logits_max_abs)
        steps.append(
            {
                "step": step_index,
                "position": prefill_length + step_index,
                "hidden_max_abs": hidden_max_abs,
                "cfg_logits_max_abs": logits_max_abs,
                "reference_argmax": reference_token,
                "candidate_argmax": candidate_token,
                "argmax_equal": reference_token == candidate_token,
                "reference_margin": float(
                    (reference_top2.values[0] - reference_top2.values[1]).item()
                ),
                "production_sampler": {
                    "reference_state": int(reference_sampled_state.item()),
                    "candidate_state": int(candidate_sampled_state.item()),
                    "state_equal": sampled_state_equal,
                    "reference_token": int(reference_sampled_token.item()),
                    "candidate_token": int(candidate_sampled_token.item()),
                    "token_equal": sampled_token_equal,
                    "state_probs_max_abs": float(state_probability_delta.max().item()),
                    "state_probs_total_variation": float(
                        0.5 * state_probability_delta.sum().item()
                    ),
                    "semantic_probs_max_abs": float(
                        semantic_probability_delta.max().item()
                    ),
                    "semantic_probs_total_variation": float(
                        0.5 * semantic_probability_delta.sum().item()
                    ),
                    "semantic_top_k_overlap": semantic_top_k_overlap,
                },
            }
        )

    final_comparison = compare_output_and_state(
        reference_output,
        candidate_output,
        state_names,
        model,
        candidate_state,
    )
    return {
        "steps": steps,
        "production_sampler": {
            "temperature": SEMANTIC_TEMPERATURE,
            "state_top_p": SEMANTIC_TOP_P,
            "semantic_top_k": SEMANTIC_TOP_K,
            "cfg_gamma": CFG_GAMMA,
            "sampling_seed": sampling_seed,
            "state_matches": sum(
                int(step["production_sampler"]["state_equal"]) for step in steps
            ),
            "token_matches": sum(
                int(step["production_sampler"]["token_equal"]) for step in steps
            ),
            "total": len(steps),
            "max_state_probs_total_variation": max(
                float(step["production_sampler"]["state_probs_total_variation"])
                for step in steps
            ),
            "max_semantic_probs_total_variation": max(
                float(step["production_sampler"]["semantic_probs_total_variation"])
                for step in steps
            ),
            "min_semantic_top_k_overlap": min(
                int(step["production_sampler"]["semantic_top_k_overlap"])
                for step in steps
            ),
        },
        "argmax_matches": sum(int(step["argmax_equal"]) for step in steps),
        "argmax_total": len(steps),
        "all_argmax_equal": all(bool(step["argmax_equal"]) for step in steps),
        "max_hidden_abs": max_hidden_abs,
        "max_cfg_logits_abs": max_logits_abs,
        "final": final_comparison,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_kv_update_exportable()
    model = build_temp_former(args.checkpoint, args.device, args.batch_size)
    names = kv_buffer_names(model)
    if len(names) != 36:
        raise RuntimeError(f"expected 36 temp_former KV buffers, got {len(names)}")
    wrapper = ExplicitKVStep(model, names).eval()
    hidden, input_pos, mask, initial_state = prepare_hot_step(
        model, args.batch_size, args.prefill_length, args.device
    )

    with torch.inference_mode():
        reference_output = model(hidden, input_pos=input_pos, mask=mask)
        explicit_output = wrapper(hidden, input_pos, mask, *initial_state)
    torch.cuda.synchronize()
    results: dict[str, object] = {
        "torch": torch.__version__,
        "model": "temp_former",
        "batch_size": args.batch_size,
        "sequence_length": 1,
        "prefill_length": args.prefill_length,
        "max_seq_len": MAX_SEQ_LEN,
        "buffer_names": list(names),
        "input_count": 3 + len(names),
        "output_count": 1 + len(names),
        "explicit_state_bytes": sum(value.numel() * value.element_size() for value in initial_state),
        "eager_equivalence": compare_output_and_state(
            reference_output,
            explicit_output[0],
            names,
            model,
            explicit_output[1:],
        ),
    }

    if args.engine is not None:
        validation = attempt(
            "tensorrt_hot_step",
            lambda: validate_engine(
                args.engine,
                args.checkpoint,
                model,
                names,
                args.batch_size,
                args.prefill_length,
                args.device,
                args.trajectory_steps,
                args.sampling_seed,
            ),
            results,
        )
        if validation is not None:
            results["tensorrt_hot_step"].update(validation)

    metrics_path = args.output_dir / "temp-step-explicit-kv.json"
    if args.skip_export:
        metrics_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    export_hidden, export_input_pos, export_mask, export_state = prepare_hot_step(
        model, args.batch_size, args.prefill_length, args.device
    )
    inputs = (export_hidden, export_input_pos, export_mask, *export_state)
    exported = attempt(
        "torch_export",
        lambda: torch.export.export(wrapper, inputs, strict=False),
        results,
    )
    if exported is not None:
        results["torch_export"].update(
            {"graph_nodes": len(list(exported.graph.nodes))}
        )

    onnx_path = args.output_dir / "temp-step-explicit-kv.onnx"
    input_names = ["hidden", "input_pos", "mask"] + [
        name.replace(".", "_") for name in names
    ]
    output_names = ["output"] + [f"next_{name}" for name in input_names[3:]]
    onnx_result = (
        attempt(
            "onnx_export",
            lambda: torch.onnx.export(
                exported,
                (),
                onnx_path,
                input_names=input_names,
                output_names=output_names,
                dynamo=True,
                external_data=False,
            ),
            results,
        )
        if exported is not None
        else None
    )
    del onnx_result

    if results.get("onnx_export", {}).get("ok"):
        import onnx

        onnx_model = onnx.load(onnx_path, load_external_data=False)
        removed_default_reductions = 0
        for node in onnx_model.graph.node:
            if node.op_type != "ScatterND":
                continue
            kept_attributes = []
            for attribute in node.attribute:
                if attribute.name == "reduction" and attribute.s == b"none":
                    removed_default_reductions += 1
                else:
                    kept_attributes.append(attribute)
            del node.attribute[:]
            node.attribute.extend(kept_attributes)
        if removed_default_reductions:
            onnx.save(onnx_model, onnx_path)
        results["onnx_graph"] = {
            "nodes": len(onnx_model.graph.node),
            "initializers": len(onnx_model.graph.initializer),
            "inputs": len(onnx_model.graph.input),
            "outputs": len(onnx_model.graph.output),
            "bytes": onnx_path.stat().st_size,
            "removed_default_scatternd_reductions": removed_default_reductions,
        }
        attempt("onnx_check", lambda: onnx.checker.check_model(onnx_path), results)

        def parse_tensorrt() -> int:
            import tensorrt as trt

            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            if not parser.parse_from_file(str(onnx_path)):
                errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
                raise RuntimeError("\n".join(errors))
            return network.num_layers

        layers = attempt("tensorrt_parse", parse_tensorrt, results)
        if layers is not None:
            results["tensorrt_parse"]["layers"] = layers

    metrics_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
