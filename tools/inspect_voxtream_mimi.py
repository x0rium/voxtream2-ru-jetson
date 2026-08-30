#!/usr/bin/env python3
"""Inventory Mimi decoder parameters and streaming state for TensorRT work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from moshi.models import loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-codebooks", type=int, default=16)
    return parser.parse_args()


def tensor_spec(value: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "device": str(value.device),
        "mib": round(value.numel() * value.element_size() / 1024**2, 4),
    }


def parameter_summary(module: torch.nn.Module) -> dict[str, object]:
    parameters = list(module.parameters())
    return {
        "class": type(module).__name__,
        "parameters": sum(value.numel() for value in parameters),
        "bf16_mib": round(sum(value.numel() for value in parameters) * 2 / 1024**2, 3),
    }


def main() -> None:
    args = parse_args()
    model = (
        loaders.get_mimi(
            args.checkpoint,
            device="cuda",
            num_codebooks=args.num_codebooks,
        )
        .eval()
        .to(dtype=torch.bfloat16)
    )
    components = {
        name: parameter_summary(getattr(model, name))
        for name in (
            "encoder",
            "decoder",
            "encoder_transformer",
            "decoder_transformer",
            "quantizer",
            "downsample",
            "upsample",
        )
        if hasattr(model, name) and getattr(model, name) is not None
    }
    model.streaming_forever(batch_size=1)
    states = {}
    for module_name, state in model.get_streaming_state().items():
        fields = {}
        for field_name, value in vars(state).items():
            if isinstance(value, torch.Tensor):
                fields[field_name] = tensor_spec(value)
            elif value is None or isinstance(value, (bool, int, float, str)):
                fields[field_name] = value
            else:
                fields[field_name] = type(value).__name__
        states[module_name or "<root>"] = {
            "class": type(state).__name__,
            "fields": fields,
        }
    result = {
        "checkpoint": str(args.checkpoint),
        "frame_size": model.frame_size,
        "sample_rate": model.sample_rate,
        "frame_rate": model.frame_rate,
        "num_codebooks": model.num_codebooks,
        "components": components,
        "total": parameter_summary(model),
        "streaming_states": states,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
