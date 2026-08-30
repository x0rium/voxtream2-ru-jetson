#!/usr/bin/env python3
"""Validate the extracted first-layer K projection + RoPE TensorRT path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorrt as trt
import torch
from voxtream_tensorrt_temp_probe import build_temp_former


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--scatter-engine", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.capture, map_location="cpu", weights_only=False)
    hidden = payload["records"][0]["hidden"].to(device="cuda")
    model = build_temp_former(args.checkpoint, "cuda", batch_size=2)
    layer = model.layers[0]
    attention = layer.attn
    normalized = layer.sa_norm(hidden)
    raw_k = attention.k_proj(normalized).view(
        2, 1, attention.num_kv_heads, attention.head_dim
    )
    q_per_kv = attention.num_heads // attention.num_kv_heads

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    output = torch.empty((2, 1, 1024), device="cuda", dtype=torch.bfloat16)
    context.set_tensor_address("hidden", hidden.data_ptr())
    context.set_tensor_address("_unsafe_view", output.data_ptr())

    records = []
    with torch.inference_mode():
        for position in (0, 1, 2, 15, 16, 31, 32, 63, 64, 95, 96, 107, 108):
            input_pos = torch.full(
                (2, 1), position, device="cuda", dtype=torch.int64
            )
            context.set_tensor_address("input_pos", input_pos.data_ptr())
            with torch.cuda.stream(stream):
                if not context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError("TensorRT K-path enqueue failed")
            stream.synchronize()
            expected = attention.pos_embeddings(
                raw_k, input_pos=input_pos
            )
            expected = (
                expected.view(
                    2, 1, attention.num_kv_heads, 1, attention.head_dim
                )
                .expand(
                    2,
                    1,
                    attention.num_kv_heads,
                    q_per_kv,
                    attention.head_dim,
                )
                .reshape(2, 1, 1024)
            )
            delta = expected.float() - output.float()
            records.append(
                {
                    "position": position,
                    "max_abs": float(delta.abs().max().item()),
                    "mean_abs": float(delta.abs().mean().item()),
                }
            )
    result = {"engine": str(args.engine), "records": records}
    if args.scatter_engine is not None:
        scatter_runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        scatter_engine = scatter_runtime.deserialize_cuda_engine(
            args.scatter_engine.read_bytes()
        )
        scatter_context = scatter_engine.create_execution_context()
        k_cache = payload["initial_state"][0].to(device="cuda")
        cache_pos = payload["initial_state"][2].to(device="cuda")
        scatter_output = torch.empty_like(k_cache)
        position = int(payload["records"][0]["position"])
        input_pos = torch.full((2, 1), position, device="cuda", dtype=torch.int64)
        for name, tensor in (
            ("hidden", hidden),
            ("input_pos", input_pos),
            ("layers_0_attn_kv_cache_k_cache", k_cache),
            ("layers_0_attn_kv_cache_cache_pos", cache_pos),
            ("next_layers_0_attn_kv_cache_k_cache", scatter_output),
        ):
            scatter_context.set_tensor_address(name, tensor.data_ptr())
        with torch.cuda.stream(stream):
            if not scatter_context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT K-scatter enqueue failed")
        stream.synchronize()
        expected_written = attention.pos_embeddings(raw_k, input_pos=input_pos)
        expected_written = (
            expected_written.view(
                2, 1, attention.num_kv_heads, 1, attention.head_dim
            )
            .expand(
                2,
                1,
                attention.num_kv_heads,
                q_per_kv,
                attention.head_dim,
            )
            .reshape(2, 1, attention.num_heads, attention.head_dim)
            .transpose(1, 2)
        )
        expected_cache = k_cache.clone()
        expected_cache[:, :, cache_pos[:1]] = expected_written
        scatter_delta = expected_cache.float() - scatter_output.float()
        result["scatter"] = {
            "engine": str(args.scatter_engine),
            "max_abs": float(scatter_delta.abs().max().item()),
            "mean_abs": float(scatter_delta.abs().mean().item()),
            "written_max_abs": float(
                scatter_delta[:, :, int(cache_pos[0].item()), :]
                .abs()
                .max()
                .item()
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
