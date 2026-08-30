#!/usr/bin/env python3
"""Append VoXtream's BF16 semantic projection to a temp_former ONNX graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch
from onnx import TensorProto, helper
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = onnx.load(args.input, load_external_data=False)
    output_names = {value.name for value in model.graph.output}
    if "output" not in output_names:
        raise RuntimeError(f"temp graph has no 'output': {sorted(output_names)}")
    if "semantic_logits" in output_names:
        raise RuntimeError("semantic_logits is already a graph output")

    with safe_open(args.checkpoint, framework="pt", device="cpu") as source:
        weight = source.get_tensor("sem_head.weight").to(torch.bfloat16)
    if tuple(weight.shape) != (12300, 1024):
        raise RuntimeError(f"unexpected sem_head shape: {tuple(weight.shape)}")
    transposed = weight.T.contiguous()
    raw_data = transposed.view(torch.uint8).numpy().tobytes()
    initializer = onnx.TensorProto()
    initializer.name = "sem_head_weight_transposed"
    initializer.data_type = TensorProto.BFLOAT16
    initializer.dims.extend(transposed.shape)
    initializer.raw_data = raw_data
    model.graph.initializer.append(initializer)
    model.graph.node.extend(
        [
            helper.make_node(
                "Cast",
                ["output"],
                ["temp_output_bf16"],
                name="SemanticHeadInputBF16",
                to=TensorProto.BFLOAT16,
            ),
            helper.make_node(
                "MatMul",
                ["temp_output_bf16", initializer.name],
                ["semantic_logits"],
                name="SemanticHeadMatMul",
            ),
        ]
    )
    model.graph.output.append(
        helper.make_tensor_value_info(
            "semantic_logits", TensorProto.BFLOAT16, [2, 1, 12300]
        )
    )
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(
        f"semantic_head_shape={tuple(weight.shape)} "
        f"onnx_bytes={args.output.stat().st_size}"
    )


if __name__ == "__main__":
    main()
