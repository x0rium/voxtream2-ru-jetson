#!/usr/bin/env python3
"""Add fixed-prompt semantic logits so sem_head never needs to enter PyTorch."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    if payload.get("format") != "voxtream-temp-fixed-prompt-prefill-v1":
        raise RuntimeError(f"unsupported cache format: {payload.get('format')!r}")
    hidden = payload["output"][:, -1, :].to(
        device=args.device, dtype=torch.bfloat16
    )
    with safe_open(args.checkpoint, framework="pt", device="cpu") as source:
        weight = source.get_tensor("sem_head.weight").to(
            device=args.device, dtype=torch.bfloat16
        )
    with torch.inference_mode():
        logits = torch.mm(hidden, weight.T).cpu()
    payload["semantic_logits"] = logits
    payload["semantic_logits_source"] = "sem_head.weight BF16 torch.mm"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        f"semantic_logits_shape={tuple(logits.shape)} "
        f"dtype={logits.dtype} output_bytes={args.output.stat().st_size}"
    )


if __name__ == "__main__":
    main()
