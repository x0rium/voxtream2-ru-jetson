#!/usr/bin/env python3
"""Check that a captured production temp trajectory replays in PyTorch eager."""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from voxtream_tensorrt_temp_probe import build_temp_former, kv_buffer_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def patch_efficient_attention(transformer: torch.nn.Module) -> int:
    patched = 0
    for module in transformer.modules():
        attention_call = getattr(module, "_attention_call", None)
        if attention_call is None:
            continue

        @functools.wraps(attention_call)
        def efficient_attention(*args, _attention_call=attention_call, **kwargs):
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                return _attention_call(*args, **kwargs)

        module._attention_call = efficient_attention
        patched += 1
    return patched


def main() -> None:
    args = parse_args()
    payload = torch.load(args.capture, map_location="cpu", weights_only=False)
    model = build_temp_former(args.checkpoint, "cuda", batch_size=2)
    names = kv_buffer_names(model)
    if tuple(payload["buffer_names"]) != names:
        raise RuntimeError("capture and model temp buffer names differ")
    patched_modules = patch_efficient_attention(model)
    buffers = dict(model.named_buffers())
    for name, captured in zip(names, payload["initial_state"]):
        buffers[name].copy_(captured.to(device="cuda"))

    records = []
    with torch.inference_mode():
        for step_index, captured in enumerate(payload["records"]):
            output = model(
                captured["hidden"].to(device="cuda"),
                input_pos=captured["input_pos"].to(device="cuda"),
                mask=captured["mask"].to(device="cuda"),
            )
            reference = captured["reference_output"].to(device="cuda")
            delta = reference.float() - output.float()
            reference_flat = reference.float().flatten()
            output_flat = output.float().flatten()
            cosine = torch.dot(reference_flat, output_flat) / (
                torch.linalg.vector_norm(reference_flat)
                * torch.linalg.vector_norm(output_flat)
            )
            records.append(
                {
                    "step": step_index,
                    "position": int(captured["position"]),
                    "max_abs": float(delta.abs().max().item()),
                    "mean_abs": float(delta.abs().mean().item()),
                    "cosine": float(cosine.item()),
                }
            )

    final_state_deltas = {}
    for name, reference in zip(names, payload["final_state"]):
        final_state_deltas[name] = float(
            (
                reference.to(device="cuda").float()
                - buffers[name].float()
            )
            .abs()
            .max()
            .item()
        )
    result = {
        "mode": "pytorch_eager_efficient_attention_real_capture_replay",
        "steps": len(records),
        "patched_attention_modules": patched_modules,
        "max_hidden_abs": max(record["max_abs"] for record in records),
        "max_hidden_mean_abs": max(record["mean_abs"] for record in records),
        "min_hidden_cosine": min(record["cosine"] for record in records),
        "final_state_max_abs": max(final_state_deltas.values()),
        "final_state_per_tensor_max_abs": final_state_deltas,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
