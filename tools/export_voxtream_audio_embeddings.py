#!/usr/bin/env python3
"""Export VoXtream audio_embeddings.weight as contiguous raw BF16 words."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with safe_open(args.checkpoint, framework="pt", device="cpu") as source:
        weight = source.get_tensor("audio_embeddings.weight")
    weight = weight.to(torch.bfloat16).contiguous()
    payload = weight.view(torch.uint16).numpy().tobytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    result = {
        "checkpoint": str(args.checkpoint),
        "output": str(args.output),
        "shape": list(weight.shape),
        "source_dtype": "float32",
        "dtype": "bfloat16",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if args.metrics is not None:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
