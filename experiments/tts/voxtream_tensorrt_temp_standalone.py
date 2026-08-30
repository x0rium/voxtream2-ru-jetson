#!/usr/bin/env python3
"""Replay real temp_former TensorRT steps without importing PyTorch."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda import cudart

from voxtream2_ru_jetson.tensorrt_standalone import (
    CudaArena,
    compare_payload,
    cuda_check,
    load_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--benchmark-iterations", type=int, default=200)
    parser.add_argument("--benchmark-rounds", type=int, default=5)
    return parser.parse_args()


def copy_to_device(pointer: int, payload: bytes) -> None:
    host = np.frombuffer(payload, dtype=np.uint8)
    cuda_check(
        cudart.cudaMemcpy(
            pointer,
            host.ctypes.data,
            host.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        ),
        "cudaMemcpy(H2D)",
    )


def bind_context(
    context,
    inputs: dict[str, int],
    extra_outputs: dict[str, int],
    states: list[dict[str, object]],
    state_pointers: dict[str, int],
    output: int,
) -> None:
    bindings = dict(inputs)
    bindings["output"] = output
    bindings.update(extra_outputs)
    for state in states:
        pointer = state_pointers[str(state["input_name"])]
        bindings[str(state["input_name"])] = pointer
        bindings[str(state["output_name"])] = pointer
    for name, pointer in bindings.items():
        if not context.set_tensor_address(name, pointer):
            raise RuntimeError(f"failed to bind TensorRT tensor {name}")


def enqueue(context, stream: int) -> None:
    if not context.execute_async_v3(stream):
        raise RuntimeError("TensorRT execute_async_v3 failed")


def timed_round(operation, iterations: int, stream_handle) -> float:
    start = cuda_check(cudart.cudaEventCreate(), "cudaEventCreate(start)")
    end = cuda_check(cudart.cudaEventCreate(), "cudaEventCreate(end)")
    try:
        cuda_check(cudart.cudaEventRecord(start, stream_handle), "record start")
        for _ in range(iterations):
            operation()
        cuda_check(cudart.cudaEventRecord(end, stream_handle), "record end")
        cuda_check(cudart.cudaEventSynchronize(end), "synchronize end")
        total_ms = float(
            cuda_check(cudart.cudaEventElapsedTime(start, end), "elapsed time")
        )
    finally:
        cuda_check(cudart.cudaEventDestroy(start), "destroy start")
        cuda_check(cudart.cudaEventDestroy(end), "destroy end")
    return total_ms / iterations


def reset_state(
    fixture: Path,
    states: list[dict[str, object]],
    state_pointers: dict[str, int],
) -> None:
    for state in states:
        copy_to_device(
            state_pointers[str(state["input_name"])],
            load_payload(fixture, state["initial"]),
        )


def summarize(samples: list[float]) -> dict[str, object]:
    return {
        "rounds_ms": [round(value, 4) for value in samples],
        "median_ms": round(statistics.median(samples), 4),
        "min_ms": round(min(samples), 4),
        "max_ms": round(max(samples), 4),
    }


def bfloat16_to_float32(payload: bytes, shape: list[int]) -> np.ndarray:
    words = np.frombuffer(payload, dtype=np.uint16)
    return (words.astype(np.uint32) << 16).view(np.float32).reshape(shape)


def softmax(value: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = value - np.max(value, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def sample_categorical(probs: np.ndarray, rng: np.random.Generator) -> int:
    exponentials = rng.exponential(size=probs.shape)
    return int(np.argmax(probs / exponentials))


def sample_semantic_logits(
    payload: bytes,
    spec: dict[str, object],
    rng: np.random.Generator,
) -> tuple[int, int]:
    logits = bfloat16_to_float32(payload, spec["shape"])[:, -1, :]
    conditional = logits[0].reshape(6, 2050)
    cfg = (1.5 * logits[0] - 0.5 * logits[1]).reshape(6, 2050)
    maximum = np.max(conditional, axis=-1)
    state_logits = maximum + np.log(
        np.sum(np.exp(conditional - maximum[:, None]), axis=-1)
    )
    state_probs = softmax(state_logits / 0.8)
    order = np.argsort(state_probs)[::-1]
    sorted_probs = state_probs[order]
    keep = np.cumsum(sorted_probs) - sorted_probs <= 0.9
    filtered = sorted_probs * keep
    filtered /= filtered.sum()
    state = int(order[sample_categorical(filtered, rng)])

    semantic_probs = softmax(cfg[state] / 0.8)
    top_indices = np.argpartition(semantic_probs, -50)[-50:]
    top_probs = semantic_probs[top_indices]
    semantic = int(top_indices[sample_categorical(top_probs, rng)])
    return semantic, state


def main() -> None:
    args = parse_args()
    if "torch" in sys.modules:
        raise RuntimeError("PyTorch was imported before standalone execution")
    if args.benchmark_iterations < 1 or args.benchmark_iterations > 512:
        raise ValueError("--benchmark-iterations must be in [1, 512]")
    if args.benchmark_rounds < 1:
        raise ValueError("--benchmark-rounds must be at least 1")

    started = time.perf_counter()
    manifest = json.loads((args.fixture / "manifest.json").read_text())
    if manifest.get("format") != "voxtream-temp-tensorrt-trajectory-v1":
        raise ValueError("unsupported temp TensorRT fixture format")
    engine_payload = args.engine.read_bytes()
    engine_sha256 = hashlib.sha256(engine_payload).hexdigest()
    if engine_sha256 != manifest["engine"]["sha256"]:
        raise RuntimeError("engine does not match fixture SHA-256")

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_payload)
    del engine_payload
    if engine is None:
        raise RuntimeError(f"failed to deserialize {args.engine}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("failed to create TensorRT execution context")

    records = manifest["records"]
    states = manifest["states"]
    first_record = records[0]
    stream_handle = cuda_check(cudart.cudaStreamCreate(), "cudaStreamCreate")
    stream = int(stream_handle)
    arena = CudaArena()
    graph = None
    graph_exec = None
    comparisons: dict[str, object] = {}
    sampled_tokens: list[dict[str, int]] = []
    sampler_rng = np.random.default_rng(20260830)
    try:
        input_pointers = {
            name: arena.allocate(int(spec["nbytes"]))
            for name, spec in first_record["inputs"].items()
        }
        output_pointer = arena.allocate(int(first_record["output"]["nbytes"]))
        extra_output_pointers = {
            name: arena.allocate(int(spec["nbytes"]))
            for name, spec in first_record.get("extra_outputs", {}).items()
        }
        state_pointers = {
            str(state["input_name"]): arena.upload(
                load_payload(args.fixture, state["initial"])
            )
            for state in states
        }
        bind_context(
            context,
            input_pointers,
            extra_output_pointers,
            states,
            state_pointers,
            output_pointer,
        )

        for record in records:
            for name, spec in record["inputs"].items():
                copy_to_device(
                    input_pointers[name], load_payload(args.fixture, spec)
                )
            enqueue(context, stream)
            cuda_check(
                cudart.cudaStreamSynchronize(stream_handle),
                "trajectory synchronize",
            )
            spec = record["output"]
            actual = arena.download(output_pointer, int(spec["nbytes"]))
            expected = load_payload(args.fixture, spec)
            comparisons[f"output.{int(record['index']):03d}"] = compare_payload(
                actual, expected, spec
            )
            for name, extra_spec in record.get("extra_outputs", {}).items():
                extra_actual = arena.download(
                    extra_output_pointers[name], int(extra_spec["nbytes"])
                )
                extra_expected = load_payload(args.fixture, extra_spec)
                comparisons[
                    f"{name}.{int(record['index']):03d}"
                ] = compare_payload(extra_actual, extra_expected, extra_spec)
                if name == "semantic_logits":
                    semantic, state = sample_semantic_logits(
                        extra_actual, extra_spec, sampler_rng
                    )
                    sampled_tokens.append(
                        {
                            "step": int(record["index"]),
                            "position": int(record["position"]),
                            "semantic_token": semantic,
                            "state_token": state,
                        }
                    )

        for state in states:
            spec = state["final"]
            actual = arena.download(
                state_pointers[str(state["input_name"])], int(spec["nbytes"])
            )
            expected = load_payload(args.fixture, spec)
            comparisons[f"state.{state['input_name']}"] = compare_payload(
                actual, expected, spec
            )

        benchmark_record = records[-1]
        for name, spec in benchmark_record["inputs"].items():
            copy_to_device(input_pointers[name], load_payload(args.fixture, spec))

        direct_samples = []
        graph_samples = []
        for round_index in range(args.benchmark_rounds):
            reset_state(args.fixture, states, state_pointers)
            for _ in range(10):
                enqueue(context, stream)
            cuda_check(cudart.cudaStreamSynchronize(stream_handle), "direct warm-up")
            direct_samples.append(
                timed_round(
                    lambda: enqueue(context, stream),
                    args.benchmark_iterations,
                    stream_handle,
                )
            )

        reset_state(args.fixture, states, state_pointers)
        enqueue(context, stream)
        cuda_check(cudart.cudaStreamSynchronize(stream_handle), "graph warm-up")
        cuda_check(
            cudart.cudaStreamBeginCapture(
                stream_handle,
                cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal,
            ),
            "cudaStreamBeginCapture",
        )
        enqueue(context, stream)
        graph = cuda_check(
            cudart.cudaStreamEndCapture(stream_handle), "cudaStreamEndCapture"
        )
        graph_exec = cuda_check(
            cudart.cudaGraphInstantiate(graph, 0), "cudaGraphInstantiate"
        )
        for round_index in range(args.benchmark_rounds):
            reset_state(args.fixture, states, state_pointers)
            for _ in range(10):
                cuda_check(
                    cudart.cudaGraphLaunch(graph_exec, stream_handle),
                    "graph warm-up launch",
                )
            cuda_check(cudart.cudaStreamSynchronize(stream_handle), "graph warm-up")
            graph_samples.append(
                timed_round(
                    lambda: cuda_check(
                        cudart.cudaGraphLaunch(graph_exec, stream_handle),
                        "cudaGraphLaunch",
                    ),
                    args.benchmark_iterations,
                    stream_handle,
                )
            )
    finally:
        if graph_exec is not None:
            cuda_check(cudart.cudaGraphExecDestroy(graph_exec), "destroy graph exec")
        if graph is not None:
            cuda_check(cudart.cudaGraphDestroy(graph), "destroy graph")
        arena.close()
        cuda_check(cudart.cudaStreamDestroy(stream_handle), "destroy stream")

    all_equal = all(
        bool(comparison["bitwise_equal"])
        for comparison in comparisons.values()
    )
    direct = summarize(direct_samples)
    cuda_graph = summarize(graph_samples)
    result = {
        "runtime": "tensorrt+cuda-python+numpy",
        "torch_imported": "torch" in sys.modules,
        "engine": str(args.engine),
        "engine_sha256": engine_sha256,
        "records": len(records),
        "states": len(states),
        "inplace_state": bool(manifest["inplace_state"]),
        "comparisons": comparisons,
        "all_bitwise_equal": all_equal,
        "numpy_sampler": {
            "algorithm": "top_p_0.9_state_then_top_k_50_semantic_temperature_0.8",
            "seed": 20260830,
            "samples": sampled_tokens,
        },
        "benchmark": {
            "iterations": args.benchmark_iterations,
            "direct_enqueue": direct,
            "cuda_graph": cuda_graph,
            "speedup": round(
                float(direct["median_ms"]) / float(cuda_graph["median_ms"]), 3
            ),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "max_rss_mib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
        ),
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.metrics is not None:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(serialized)
    print(serialized)
    if not all_equal or result["torch_imported"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
