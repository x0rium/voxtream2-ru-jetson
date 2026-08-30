#!/usr/bin/env python3
"""Expose an existing typed ONNX intermediate as an additional graph output."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--value", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = onnx.load(args.input, load_external_data=False)
    typed_values = {
        value.name: value
        for value in (
            list(model.graph.input)
            + list(model.graph.output)
            + list(model.graph.value_info)
        )
    }
    if args.value not in typed_values:
        model = onnx.shape_inference.infer_shapes(model)
        typed_values = {
            value.name: value
            for value in (
                list(model.graph.input)
                + list(model.graph.output)
                + list(model.graph.value_info)
            )
        }
    if args.value not in typed_values:
        raise KeyError(f"ONNX value has no inferred type: {args.value}")
    if any(output.name == args.value for output in model.graph.output):
        raise ValueError(f"value is already a graph output: {args.value}")
    model.graph.output.append(typed_values[args.value])
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"added_output={args.value}")


if __name__ == "__main__":
    main()
