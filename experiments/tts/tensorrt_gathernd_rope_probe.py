#!/usr/bin/env python3
"""Check TensorRT GatherND indexing used by exported torchtune RoPE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import tensorrt as trt
import torch
from onnx import TensorProto, helper, numpy_helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def rope_cache() -> np.ndarray:
    dim = 64
    theta = 1.0 / (
        500000.0 ** (np.arange(0, dim, 2, dtype=np.float32) / dim)
    )
    positions = np.arange(2048, dtype=np.float32)
    angles = np.einsum("i,j->ij", positions, theta)
    return np.stack([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float32)


def build_onnx(path: Path, index_type: int) -> None:
    index_name = "input_pos"
    indices = index_name
    nodes = []
    if index_type == TensorProto.INT64:
        nodes.append(helper.make_node("Cast", [index_name], ["indices_i64"], to=7))
        indices = "indices_i64"
    nodes.extend(
        [
            helper.make_node("Unsqueeze", [indices, "axes"], ["gather_indices"]),
            helper.make_node(
                "GatherND", ["rope_cache", "gather_indices"], ["output"]
            ),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "rope_gathernd",
        [helper.make_tensor_value_info(index_name, index_type, [2, 1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 1, 32, 2])],
        [
            numpy_helper.from_array(rope_cache(), name="rope_cache"),
            numpy_helper.from_array(np.array([-1], dtype=np.int64), name="axes"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def build_engine(onnx_path: Path, engine_path: Path) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError(
            "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        )
    serialized = builder.build_serialized_network(
        network, builder.create_builder_config()
    )
    if serialized is None:
        raise RuntimeError("TensorRT failed to build GatherND probe")
    engine_path.write_bytes(bytes(serialized))


def run_engine(engine_path: Path, input_dtype: torch.dtype) -> list[dict[str, object]]:
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    output = torch.empty((2, 1, 32, 2), device="cuda", dtype=torch.float32)
    cache = torch.from_numpy(rope_cache()).cuda()
    records = []
    for position in (0, 1, 16, 31, 32, 63, 64, 95, 96, 107, 108, 127, 255, 1024):
        input_pos = torch.full(
            (2, 1), position, device="cuda", dtype=input_dtype
        )
        context.set_tensor_address("input_pos", input_pos.data_ptr())
        context.set_tensor_address("output", output.data_ptr())
        with torch.cuda.stream(stream):
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT GatherND enqueue failed")
        stream.synchronize()
        expected = cache[position].view(1, 1, 32, 2).expand_as(output)
        delta = expected - output
        records.append(
            {
                "position": position,
                "max_abs": float(delta.abs().max().item()),
                "output_head": output[0, 0, :3].cpu().tolist(),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, onnx_type, torch_type in (
        ("int64", TensorProto.INT64, torch.int64),
        ("int32", TensorProto.INT32, torch.int32),
    ):
        onnx_path = args.output_dir / f"rope-gathernd-{name}.onnx"
        engine_path = args.output_dir / f"rope-gathernd-{name}.engine"
        build_onnx(onnx_path, onnx_type)
        build_engine(onnx_path, engine_path)
        result[name] = run_engine(engine_path, torch_type)
    metrics = args.output_dir / "rope-gathernd.json"
    metrics.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
