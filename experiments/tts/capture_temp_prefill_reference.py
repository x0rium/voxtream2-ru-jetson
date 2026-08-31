#!/usr/bin/env python3
"""Capture a PyTorch q=420 prefill followed by the production q=1 step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from voxtream_tensorrt_explicit_kv_probe import kv_buffer_names
from voxtream_tensorrt_temp_prefill_probe import prefill_inputs
from voxtream_tensorrt_temp_probe import build_temp_former, causal_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=420)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def numpy_value(value: torch.Tensor) -> np.ndarray:
    value = value.detach().contiguous().cpu()
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy()
    return value.numpy()


def main() -> None:
    args = parse_args()
    batch_size = 2
    model = build_temp_former(args.checkpoint, args.device, batch_size)
    names = kv_buffer_names(model)
    hidden, input_pos, mask = prefill_inputs(batch_size, args.sequence_length, args.device)
    model.reset_caches()
    with torch.inference_mode():
        model(hidden, input_pos=input_pos, mask=mask)
        torch.manual_seed(421)
        next_hidden = torch.randn(batch_size, 1, 1024, device=args.device, dtype=torch.bfloat16)
        next_pos = torch.full(
            (batch_size, 1),
            args.sequence_length,
            device=args.device,
            dtype=torch.int64,
        )
        output = model(next_hidden, input_pos=next_pos, mask=causal_rows(next_pos))
        with safe_open(args.checkpoint, framework="pt", device="cpu") as source:
            sem_head = source.get_tensor("sem_head.weight").to(
                device=args.device, dtype=torch.bfloat16
            )
        logits = torch.mm(output[:, -1].to(torch.bfloat16), sem_head.T).unsqueeze(1)

    tensors = {
        "hidden": numpy_value(hidden),
        "next_hidden": numpy_value(next_hidden),
        "expected_output": numpy_value(output),
        "expected_logits": numpy_value(logits),
    }
    buffers = dict(model.named_buffers())
    state_keys = {}
    for index, name in enumerate(names):
        value = buffers[name]
        if name.endswith(("k_cache", "v_cache")):
            value = value[:, :, : args.sequence_length + 1]
        key = f"state_{index}"
        tensors[key] = numpy_value(value)
        state_keys[name] = key

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **tensors)
    manifest = {
        "format": "voxtream-temp-prefill-reference-v1",
        "sequence_length": args.sequence_length,
        "state_keys": state_keys,
        "numpy_archive": args.output.name,
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
