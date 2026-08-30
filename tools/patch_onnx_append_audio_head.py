#!/usr/bin/env python3
"""Append VoXtream's BF16 acoustic projections to a dep_former ONNX graph."""

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


def scalar_initializer(name: str, value: int) -> onnx.TensorProto:
    return helper.make_tensor(name, TensorProto.INT64, (), [value])


def main() -> None:
    args = parse_args()
    model = onnx.load(args.input, load_external_data=False)
    input_names = {value.name for value in model.graph.input}
    output_names = {value.name for value in model.graph.output}
    if "input_pos" not in input_names:
        raise RuntimeError(f"dep graph has no 'input_pos': {sorted(input_names)}")
    if "output" not in output_names:
        raise RuntimeError(f"dep graph has no 'output': {sorted(output_names)}")
    if "acoustic_logits" in output_names:
        raise RuntimeError("acoustic_logits is already a graph output")

    with safe_open(args.checkpoint, framework="pt", device="cpu") as source:
        weight = source.get_tensor("audio_head").to(torch.bfloat16).contiguous()
    if tuple(weight.shape) != (15, 1024, 2050):
        raise RuntimeError(f"unexpected audio_head shape: {tuple(weight.shape)}")

    initializer = onnx.TensorProto()
    initializer.name = "audio_head_weight"
    initializer.data_type = TensorProto.BFLOAT16
    initializer.dims.extend(weight.shape)
    initializer.raw_data = weight.view(torch.uint8).numpy().tobytes()
    model.graph.initializer.extend(
        [
            initializer,
            scalar_initializer("audio_head_batch_zero", 0),
            scalar_initializer("audio_head_last_index", -1),
            scalar_initializer("audio_head_position_offset", 1),
        ]
    )

    # The first acoustic head is consumed after the q=2 init whose final
    # position is 1. Subsequent q=1 calls use positions 2..15, so
    # ``head_index = input_pos[0, -1] - 1`` selects all 15 heads without adding
    # another runtime input. Both CFG rows always share the same position.
    model.graph.node.extend(
        [
            helper.make_node(
                "Gather",
                ["input_pos", "audio_head_batch_zero"],
                ["audio_head_position_row"],
                name="AudioHeadPositionRow",
                axis=0,
            ),
            helper.make_node(
                "Gather",
                ["audio_head_position_row", "audio_head_last_index"],
                ["audio_head_last_position"],
                name="AudioHeadLastPosition",
                axis=0,
            ),
            helper.make_node(
                "Sub",
                ["audio_head_last_position", "audio_head_position_offset"],
                ["audio_head_index"],
                name="AudioHeadIndex",
            ),
            helper.make_node(
                "Gather",
                [initializer.name, "audio_head_index"],
                ["audio_head_selected_weight"],
                name="AudioHeadSelectWeight",
                axis=0,
            ),
            helper.make_node(
                "Gather",
                ["output", "audio_head_last_index"],
                ["audio_head_last_hidden_f32"],
                name="AudioHeadLastHidden",
                axis=1,
            ),
            helper.make_node(
                "Cast",
                ["audio_head_last_hidden_f32"],
                ["audio_head_last_hidden_bf16"],
                name="AudioHeadInputBF16",
                to=TensorProto.BFLOAT16,
            ),
            helper.make_node(
                "MatMul",
                ["audio_head_last_hidden_bf16", "audio_head_selected_weight"],
                ["acoustic_logits"],
                name="AudioHeadMatMul",
            ),
        ]
    )
    model.graph.output.append(
        helper.make_tensor_value_info(
            "acoustic_logits", TensorProto.BFLOAT16, [2, 2050]
        )
    )
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(
        f"audio_head_shape={tuple(weight.shape)} "
        f"audio_head_bytes={weight.numel() * weight.element_size()} "
        f"onnx_bytes={args.output.stat().st_size}"
    )


if __name__ == "__main__":
    main()
