#!/usr/bin/env python3
"""Materialize persistent K-cache graph outputs after internal attention views."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = onnx.load(args.input, load_external_data=False)
    k_outputs = {
        output.name
        for output in model.graph.output
        if output.name.startswith("next_layers_")
        and output.name.endswith("_attn_kv_cache_k_cache")
    }
    if len(k_outputs) != 12:
        raise RuntimeError(f"expected 12 K-cache graph outputs, found {len(k_outputs)}")
    internal_names = {name: f"{name}_internal" for name in k_outputs}
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name in internal_names:
                node.input[index] = internal_names[name]
        for index, name in enumerate(node.output):
            if name in internal_names:
                node.output[index] = internal_names[name]
    producer_by_internal = {}
    for node in model.graph.node:
        for name in node.output:
            if name in internal_names.values():
                producer_by_internal[name] = node
    if len(producer_by_internal) != 12:
        raise RuntimeError(
            f"expected 12 internal K producers, found {len(producer_by_internal)}"
        )

    new_nodes = []
    attention_copies = 0
    for node in model.graph.node:
        source = node.input[0] if node.input else ""
        if (
            node.op_type == "Reshape"
            and source in producer_by_internal
        ):
            external_name = next(
                name for name, internal in internal_names.items() if internal == source
            )
            attention_fp32 = f"{external_name}_attention_fp32"
            new_nodes.append(
                helper.make_node(
                    "Cast",
                    [external_name],
                    [attention_fp32],
                    name=f"Copy_{external_name}_for_attention",
                    to=TensorProto.FLOAT,
                )
            )
            node.input[0] = attention_fp32
            attention_copies += 1
        new_nodes.append(node)
        produced = [name for name in node.output if name in producer_by_internal]
        for internal_name in produced:
            external_name = next(
                name
                for name, internal in internal_names.items()
                if internal == internal_name
            )
            fp32_name = f"{external_name}_external_fp32"
            new_nodes.extend(
                [
                helper.make_node(
                    "Cast",
                    [internal_name],
                    [fp32_name],
                    name=f"Materialize_{external_name}_fp32",
                    to=TensorProto.FLOAT,
                ),
                helper.make_node(
                    "Cast",
                    [fp32_name],
                    [external_name],
                    name=f"Materialize_{external_name}_bf16",
                    to=TensorProto.BFLOAT16,
                ),
                ]
            )
    if attention_copies != 12:
        raise RuntimeError(
            f"expected 12 K-cache attention consumers, found {attention_copies}"
        )
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"materialized_k_cache_outputs={len(k_outputs)}")


if __name__ == "__main__":
    main()
