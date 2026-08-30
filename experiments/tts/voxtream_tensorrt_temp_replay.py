#!/usr/bin/env python3
"""Replay a real temp_former trajectory through TensorRT at low memory."""

from __future__ import annotations

import argparse
import gc
import json
import resource
from pathlib import Path

import torch
from safetensors import safe_open
from voxtream_tensorrt_explicit_kv_probe import ExplicitKVTRTRunner

NUM_PHONE_STATES = 6
AUDIO_CODEBOOK_SIZE = 2050
TEMPERATURE = 0.8
TOP_P = 0.9
TOP_K = 50
CFG_GAMMA = 1.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inplace-state", action="store_true")
    return parser.parse_args()


def multinomial_one(probs: torch.Tensor) -> torch.Tensor:
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
    return torch.gather(sorted_indices, -1, multinomial_one(sorted_probs))


def sample_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
    top_probs, top_indices = torch.topk(probs, k, dim=-1)
    return top_indices.gather(-1, multinomial_one(top_probs))


def sample_semantic(logits: torch.Tensor):
    conditional, unconditional = torch.split(logits, 1, dim=0)
    cfg_logits = CFG_GAMMA * conditional + (1 - CFG_GAMMA) * unconditional
    conditional_joint = conditional.view(
        -1, NUM_PHONE_STATES, AUDIO_CODEBOOK_SIZE
    )
    cfg_joint = cfg_logits.view(-1, NUM_PHONE_STATES, AUDIO_CODEBOOK_SIZE)
    state_probs = torch.softmax(
        torch.logsumexp(conditional_joint, dim=-1) / TEMPERATURE, dim=-1
    )
    state_token = sample_top_p(state_probs, TOP_P)
    state_index = state_token.unsqueeze(-1).expand(
        -1, -1, AUDIO_CODEBOOK_SIZE
    )
    semantic_logits = torch.gather(cfg_joint, 1, state_index).squeeze(1)
    semantic_probs = torch.softmax(semantic_logits / TEMPERATURE, dim=-1)
    semantic_token = sample_top_k(semantic_probs, TOP_K)
    return semantic_token, state_token.squeeze(1), state_probs, semantic_probs


def main() -> None:
    args = parse_args()
    payload = torch.load(args.capture, map_location="cpu", weights_only=False)
    if payload.get("format") != "voxtream-temp-real-trajectory-v1":
        raise RuntimeError(f"unsupported capture format: {payload.get('format')!r}")
    state_names = tuple(payload["buffer_names"])
    initial_state = tuple(value.to(device="cuda") for value in payload["initial_state"])
    runner = ExplicitKVTRTRunner(
        args.engine,
        state_names,
        initial_state,
        sequence_length=1,
        inplace_state=args.inplace_state,
    )
    del initial_state
    gc.collect()
    torch.cuda.empty_cache()

    with safe_open(args.checkpoint, framework="pt", device="cpu") as source:
        sem_head = source.get_tensor("sem_head.weight").to(
            device="cuda", dtype=torch.bfloat16
        )

    records: list[dict[str, object]] = []
    for step_index, captured in enumerate(payload["records"]):
        hidden = captured["hidden"].to(device="cuda")
        input_pos = captured["input_pos"].to(device="cuda")
        mask = captured["mask"].to(device="cuda")
        reference_output = captured["reference_output"].to(device="cuda")
        candidate_output, _ = runner.step(hidden, input_pos, mask)
        reference_logits = torch.mm(
            reference_output[:, -1, :].to(torch.bfloat16), sem_head.T
        )
        external_candidate_logits = torch.mm(
            candidate_output[:, -1, :].to(torch.bfloat16), sem_head.T
        )
        fused_logits = runner.extra_outputs.get("semantic_logits")
        if fused_logits is None:
            candidate_logits = external_candidate_logits
            fused_head_max_abs = 0.0
        else:
            candidate_logits = fused_logits[:, -1, :]
            fused_head_max_abs = float(
                (
                    external_candidate_logits.float()
                    - candidate_logits.float()
                )
                .abs()
                .max()
                .item()
            )

        sampling_rng = captured["sampling_rng_state"]
        torch.cuda.set_rng_state(sampling_rng, device="cuda")
        (
            reference_token,
            reference_state,
            reference_state_probs,
            reference_semantic_probs,
        ) = sample_semantic(reference_logits)
        torch.cuda.set_rng_state(sampling_rng, device="cuda")
        (
            candidate_token,
            candidate_state,
            candidate_state_probs,
            candidate_semantic_probs,
        ) = sample_semantic(candidate_logits)

        hidden_delta = reference_output.float() - candidate_output.float()
        state_probability_delta = (
            reference_state_probs.float() - candidate_state_probs.float()
        ).abs()
        states_equal = bool(torch.equal(reference_state, candidate_state))
        if states_equal:
            semantic_probability_delta = (
                reference_semantic_probs.float()
                - candidate_semantic_probs.float()
            ).abs()
            reference_top_k = set(
                torch.topk(reference_semantic_probs, TOP_K).indices.flatten().tolist()
            )
            candidate_top_k = set(
                torch.topk(candidate_semantic_probs, TOP_K).indices.flatten().tolist()
            )
            semantic_total_variation = float(
                0.5 * semantic_probability_delta.sum().item()
            )
            semantic_top_k_overlap = len(reference_top_k & candidate_top_k)
        else:
            semantic_total_variation = None
            semantic_top_k_overlap = None

        actual_token = int(captured["actual_semantic_token"])
        actual_state = int(captured["actual_state_token"])
        records.append(
            {
                "step": step_index,
                "position": int(captured["position"]),
                "reference_sampler_matches_actual": (
                    int(reference_token.item()) == actual_token
                    and int(reference_state.item()) == actual_state
                ),
                "actual_token": actual_token,
                "actual_state": actual_state,
                "reference_token": int(reference_token.item()),
                "candidate_token": int(candidate_token.item()),
                "token_equal": bool(torch.equal(reference_token, candidate_token)),
                "reference_state": int(reference_state.item()),
                "candidate_state": int(candidate_state.item()),
                "state_equal": states_equal,
                "hidden_max_abs": float(hidden_delta.abs().max().item()),
                "hidden_mean_abs": float(hidden_delta.abs().mean().item()),
                "state_probs_total_variation": float(
                    0.5 * state_probability_delta.sum().item()
                ),
                "semantic_probs_total_variation": semantic_total_variation,
                "semantic_top_k_overlap": semantic_top_k_overlap,
                "fused_head_max_abs": fused_head_max_abs,
            }
        )

    final_state_deltas = {}
    for name, reference, candidate in zip(
        state_names, payload["final_state"], runner.state
    ):
        reference_cuda = reference.to(device="cuda")
        final_state_deltas[name] = float(
            (reference_cuda.float() - candidate.float()).abs().max().item()
        )
        del reference_cuda

    count = len(records)
    state_matches = sum(bool(record["state_equal"]) for record in records)
    token_matches = sum(bool(record["token_equal"]) for record in records)
    reference_sampler_matches = sum(
        bool(record["reference_sampler_matches_actual"]) for record in records
    )
    max_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "mode": (
            "teacher_forced_real_hidden_inplace_state_replay"
            if args.inplace_state
            else "teacher_forced_real_hidden_separate_replay"
        ),
        "capture": str(args.capture),
        "engine": str(args.engine),
        "engine_bytes": args.engine.stat().st_size,
        "steps": count,
        "reference_sampler_matches_actual": reference_sampler_matches,
        "state_matches": state_matches,
        "token_matches": token_matches,
        "state_match_rate": state_matches / count if count else 0.0,
        "token_match_rate": token_matches / count if count else 0.0,
        "max_hidden_abs": max(
            (float(record["hidden_max_abs"]) for record in records), default=0.0
        ),
        "fused_semantic_head": "semantic_logits" in runner.extra_outputs,
        "max_fused_head_abs": max(
            (float(record["fused_head_max_abs"]) for record in records),
            default=0.0,
        ),
        "max_state_probs_total_variation": max(
            (
                float(record["state_probs_total_variation"])
                for record in records
            ),
            default=0.0,
        ),
        "final_state_max_abs": max(final_state_deltas.values()),
        "final_state_per_tensor_max_abs": final_state_deltas,
        "max_rss_mib": max_rss_kib / 1024,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
