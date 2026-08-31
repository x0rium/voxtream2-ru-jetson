#!/usr/bin/env python3
"""Export a fixed-length temp_former cache rebuild with explicit K/V state."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch
from voxtream_tensorrt_explicit_kv_probe import (
    ExplicitKVStep,
    attempt,
    kv_buffer_names,
)
from voxtream_tensorrt_probe import make_kv_update_exportable
from voxtream_tensorrt_temp_probe import (
    MAX_SEQ_LEN,
    build_temp_former,
    causal_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=420)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def empty_state(model: torch.nn.Module, names: tuple[str, ...]) -> tuple[torch.Tensor, ...]:
    model.reset_caches()
    buffers = dict(model.named_buffers())
    return tuple(buffers[name].clone() for name in names)


def prefill_inputs(
    batch_size: int,
    sequence_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 1 <= sequence_length < MAX_SEQ_LEN:
        raise ValueError(f"sequence length must be in [1, {MAX_SEQ_LEN - 1}]")
    torch.manual_seed(420)
    hidden = torch.randn(
        batch_size,
        sequence_length,
        1024,
        device=device,
        dtype=torch.bfloat16,
    )
    input_pos = torch.arange(sequence_length, device=device, dtype=torch.int64)
    input_pos = input_pos.unsqueeze(0).repeat(batch_size, 1)
    return hidden, input_pos, causal_rows(input_pos)


def tensor_delta(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, object]:
    delta = reference.float() - candidate.float()
    return {
        "shape": list(reference.shape),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "bf16_equal": bool(torch.equal(reference, candidate)),
    }


def eager_equivalence(
    model: torch.nn.Module,
    wrapper: ExplicitKVStep,
    names: tuple[str, ...],
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, object]:
    hidden, input_pos, mask = inputs
    model.reset_caches()
    with torch.inference_mode():
        reference_output = model(hidden, input_pos=input_pos, mask=mask).clone()
    reference_buffers = dict(model.named_buffers())
    reference_state = tuple(reference_buffers[name].clone() for name in names)

    explicit_state = empty_state(model, names)
    with torch.inference_mode():
        explicit = wrapper(hidden, input_pos, mask, *explicit_state)
    output_metrics = tensor_delta(reference_output, explicit[0])
    state_metrics = {
        name: tensor_delta(reference, candidate)
        for name, reference, candidate in zip(names, reference_state, explicit[1:])
    }
    return {
        "output": output_metrics,
        "state_max_abs": max(float(item["max_abs"]) for item in state_metrics.values()),
        "all_state_bf16_equal": all(bool(item["bf16_equal"]) for item in state_metrics.values()),
        "state": state_metrics,
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
    inputs = prefill_inputs(args.batch_size, args.sequence_length, args.device)

    results: dict[str, object] = {
        "torch": torch.__version__,
        "torchtune": importlib.metadata.version("torchtune"),
        "torchao": importlib.metadata.version("torchao"),
        "model": "temp_former",
        "purpose": "sink-attention batched prefill",
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "max_seq_len": MAX_SEQ_LEN,
        "buffer_names": list(names),
    }
    equivalence = attempt(
        "eager_equivalence",
        lambda: eager_equivalence(model, wrapper, names, inputs),
        results,
    )
    if equivalence is not None:
        results["eager_equivalence"].update(equivalence)

    export_state = empty_state(model, names)
    exported = attempt(
        "torch_export",
        lambda: torch.export.export(
            wrapper,
            (*inputs, *export_state),
            strict=False,
        ),
        results,
    )
    if exported is not None:
        results["torch_export"].update({"graph_nodes": len(list(exported.graph.nodes))})

    stem = f"temp-prefill-explicit-kv-q{args.sequence_length}"
    onnx_path = args.output_dir / f"{stem}.onnx"
    input_names = ["hidden", "input_pos", "mask"] + [name.replace(".", "_") for name in names]
    output_names = ["output"] + [f"next_{name}" for name in input_names[3:]]
    if exported is not None:
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

    metrics_path = args.output_dir / f"{stem}.json"
    metrics_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
