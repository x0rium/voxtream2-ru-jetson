#!/usr/bin/env python3
"""Export one dynamic temp_former graph for q=1 generation and q=420 replay."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch
from onnxscript import opset18 as op
from torchtune.modules.kv_cache import KVCache
from voxtream_tensorrt_explicit_kv_probe import ExplicitKVStep, kv_buffer_names
from voxtream_tensorrt_temp_probe import (
    MAX_SEQ_LEN,
    build_temp_former,
    causal_rows,
)


def make_dynamic_kv_update_exportable() -> None:
    """Express the cache write without symbolic index_put expansion."""

    def update(self, k_val: torch.Tensor, v_val: torch.Tensor):
        batch_size, _, sequence_length, _ = k_val.shape
        if batch_size > self.k_cache.shape[0]:
            raise ValueError(
                f"KV batch {batch_size} exceeds configured {self.k_cache.shape[0]}"
            )
        indexes = self.cache_pos[:sequence_length]
        k_out = torch.index_copy(self.k_cache, 2, indexes, k_val)
        v_out = torch.index_copy(self.v_cache, 2, indexes, v_val)
        self.k_cache.copy_(k_out)
        self.v_cache.copy_(v_out)
        self.cache_pos += sequence_length
        return self.k_cache, self.v_cache

    KVCache.update = update


def onnx_index_put_cache(self, indices, values, accumulate: bool = False):
    """Lower the known ``cache[:, :, positions] = values`` pattern."""

    if accumulate:
        raise ValueError("temporal KV update does not use accumulating index_put")
    positions = op.Reshape(
        indices[2],
        op.Constant(value_ints=[1, 1, -1, 1]),
    )
    expanded_positions = op.Expand(positions, op.Shape(values))
    return op.ScatterElements(self, expanded_positions, values, axis=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-sequence", type=int, default=420)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def remove_default_scatter_reductions(model) -> int:
    removed = 0
    for node in model.graph.node:
        if node.op_type != "ScatterND":
            continue
        kept = []
        for attribute in node.attribute:
            if attribute.name == "reduction" and attribute.s == b"none":
                removed += 1
            else:
                kept.append(attribute)
        del node.attribute[:]
        node.attribute.extend(kept)
    return removed


def main() -> None:
    args = parse_args()
    if not 2 <= args.max_sequence < MAX_SEQ_LEN:
        raise ValueError(
            f"max sequence must be in [2, {MAX_SEQ_LEN - 1}], got {args.max_sequence}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_dynamic_kv_update_exportable()
    model = build_temp_former(args.checkpoint, args.device, args.batch_size)
    names = kv_buffer_names(model)
    if len(names) != 36:
        raise RuntimeError(f"expected 36 temp_former state buffers, got {len(names)}")
    wrapper = ExplicitKVStep(model, names).eval()

    model.reset_caches()
    buffers = dict(model.named_buffers())
    state = tuple(buffers[name].clone() for name in names)
    torch.manual_seed(420)
    hidden = torch.randn(
        args.batch_size,
        args.max_sequence,
        1024,
        device=args.device,
        dtype=torch.bfloat16,
    )
    positions = torch.arange(
        args.max_sequence, device=args.device, dtype=torch.int64
    ).unsqueeze(0).repeat(args.batch_size, 1)
    mask = causal_rows(positions)
    inputs = (hidden, positions, mask, *state)
    sequence = torch.export.Dim("sequence", min=1, max=args.max_sequence)
    dynamic_shapes = (
        {1: sequence},
        {1: sequence},
        {1: sequence},
        tuple(None for _ in names),
    )

    with torch.inference_mode():
        eager = wrapper(hidden, positions, mask, *(value.clone() for value in state))
    exported = torch.export.export(
        wrapper,
        inputs,
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )
    input_names = ["hidden", "input_pos", "mask"] + [
        name.replace(".", "_") for name in names
    ]
    output_names = ["output"] + [
        f"next_{name}" for name in input_names[3:]
    ]
    onnx_path = args.output_dir / "temp-unified-dynamic-q1-q420.onnx"
    torch.onnx.export(
        exported,
        (),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamo=True,
        external_data=False,
        custom_translation_table={
            torch.ops.aten.index_put.default: onnx_index_put_cache,
        },
    )

    import onnx

    onnx_model = onnx.load(onnx_path, load_external_data=False)
    removed_reductions = remove_default_scatter_reductions(onnx_model)
    if removed_reductions:
        onnx.save(onnx_model, onnx_path)
    onnx.checker.check_model(onnx_path)
    output_shape = list(eager[0].shape)
    results = {
        "purpose": "unified q=1 generation and q=420 sink replay",
        "torch": torch.__version__,
        "packages": {
            package: importlib.metadata.version(package)
            for package in (
                "torchtune",
                "onnx",
                "onnxscript",
                "onnx-ir",
            )
        },
        "sequence_range": [1, args.max_sequence],
        "example_output_shape": output_shape,
        "state_buffers": len(names),
        "torch_export_nodes": len(list(exported.graph.nodes)),
        "onnx_nodes": len(onnx_model.graph.node),
        "onnx_inputs": len(onnx_model.graph.input),
        "onnx_outputs": len(onnx_model.graph.output),
        "onnx_bytes": onnx_path.stat().st_size,
        "removed_default_scatternd_reductions": removed_reductions,
        "onnx": str(onnx_path),
    }
    metrics_path = args.output_dir / "temp-unified-dynamic-q1-q420.json"
    metrics_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
