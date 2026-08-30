#!/usr/bin/env python3
"""Bitwise and latency probe for the PyTorch-free CUDA embedding kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
from cuda import cudart

from voxtream2_ru_jetson.cuda_audio_embedding import (
    CudaAudioEmbeddingCore,
    cuda_check,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--cubin", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--num-embeddings", type=int, default=32800)
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=500)
    return parser.parse_args()


def allocate(nbytes: int) -> int:
    return int(cuda_check(cudart.cudaMalloc(nbytes), "cudaMalloc(probe)"))


def upload(array: np.ndarray) -> int:
    array = np.ascontiguousarray(array)
    pointer = allocate(array.nbytes)
    cuda_check(
        cudart.cudaMemcpy(
            pointer,
            array.ctypes.data,
            array.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        ),
        "cudaMemcpy(probe H2D)",
    )
    return pointer


def download(pointer: int, shape: tuple[int, ...]) -> np.ndarray:
    output = np.empty(shape, dtype=np.uint16)
    cuda_check(
        cudart.cudaMemcpy(
            output.ctypes.data,
            pointer,
            output.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        ),
        "cudaMemcpy(probe D2H)",
    )
    return output


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    runtime = CudaAudioEmbeddingCore(
        args.weight,
        args.cubin,
        args.num_embeddings,
        args.embedding_dim,
    )
    weights = np.memmap(
        args.weight,
        mode="r",
        dtype=np.uint16,
        shape=(args.num_embeddings, args.embedding_dim),
    )
    rng = np.random.default_rng(20260830)
    shapes = ((2, 16, 108), (2, 16, 1), (1, 1))
    validations = []
    allocations: list[int] = []
    stream = int(cuda_check(cudart.cudaStreamCreate(), "cudaStreamCreate"))
    try:
        benchmark_pair: tuple[int, int, int] | None = None
        for shape in shapes:
            indices = rng.integers(
                0,
                args.num_embeddings,
                size=shape,
                dtype=np.int64,
            )
            expected = np.asarray(weights[indices]).copy()
            indices_pointer = upload(indices)
            output_pointer = allocate(expected.nbytes)
            allocations.extend((indices_pointer, output_pointer))
            runtime.launch(
                indices_pointer,
                output_pointer,
                indices.size,
                stream,
            )
            cuda_check(cudart.cudaStreamSynchronize(stream), "probe synchronize")
            actual = download(output_pointer, expected.shape)
            validations.append(
                {
                    "indices_shape": list(shape),
                    "output_shape": list(expected.shape),
                    "bitwise_equal": bool(np.array_equal(actual, expected)),
                    "actual_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                    "expected_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
                }
            )
            if shape == (2, 16, 1):
                benchmark_pair = (
                    indices_pointer,
                    output_pointer,
                    indices.size,
                )

        if benchmark_pair is None:
            raise AssertionError("benchmark fixture missing")
        for _ in range(20):
            runtime.launch(*benchmark_pair, stream)
        cuda_check(cudart.cudaStreamSynchronize(stream), "benchmark warm-up")
        rounds = []
        for _ in range(5):
            start = cuda_check(cudart.cudaEventCreate(), "cudaEventCreate(start)")
            end = cuda_check(cudart.cudaEventCreate(), "cudaEventCreate(end)")
            cuda_check(cudart.cudaEventRecord(start, stream), "record start")
            for _ in range(args.iterations):
                runtime.launch(*benchmark_pair, stream)
            cuda_check(cudart.cudaEventRecord(end, stream), "record end")
            cuda_check(cudart.cudaEventSynchronize(end), "sync end")
            elapsed = float(
                cuda_check(cudart.cudaEventElapsedTime(start, end), "elapsed")
            )
            rounds.append(elapsed / args.iterations)
            cuda_check(cudart.cudaEventDestroy(start), "destroy start")
            cuda_check(cudart.cudaEventDestroy(end), "destroy end")

        result = {
            "torch_imported": "torch" in __import__("sys").modules,
            "load_seconds": round(time.perf_counter() - started, 4),
            "runtime": runtime.metrics(),
            "validations": validations,
            "all_bitwise_equal": all(item["bitwise_equal"] for item in validations),
            "benchmark": {
                "shape": [2, 16, 1],
                "iterations_per_round": args.iterations,
                "round_mean_ms": [round(value, 6) for value in rounds],
                "median_ms": round(statistics.median(rounds), 6),
            },
        }
        if args.metrics is not None:
            args.metrics.parent.mkdir(parents=True, exist_ok=True)
            args.metrics.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        if not result["all_bitwise_equal"]:
            raise SystemExit(1)
    finally:
        cuda_check(cudart.cudaStreamDestroy(stream), "cudaStreamDestroy")
        for pointer in reversed(allocations):
            cuda_check(cudart.cudaFree(pointer), "cudaFree(probe)")
        runtime.close()


if __name__ == "__main__":
    main()
