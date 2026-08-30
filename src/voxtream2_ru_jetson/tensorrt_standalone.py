#!/usr/bin/env python3
"""Validate the unified dep_former TensorRT plan without importing PyTorch."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--benchmark-iterations", type=int, default=200)
    parser.add_argument("--benchmark-rounds", type=int, default=5)
    return parser.parse_args()


def cuda_check(result, operation: str):
    error, *values = result
    if error != cudart.cudaError_t.cudaSuccess:
        _, name = cudart.cudaGetErrorName(error)
        _, message = cudart.cudaGetErrorString(error)
        raise RuntimeError(
            f"{operation} failed: {name.decode()} ({message.decode()})"
        )
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


class CudaArena:
    def __init__(self) -> None:
        self.pointers: list[int] = []

    def allocate(self, nbytes: int) -> int:
        pointer = int(cuda_check(cudart.cudaMalloc(nbytes), "cudaMalloc"))
        self.pointers.append(pointer)
        return pointer

    def upload(self, payload: bytes) -> int:
        pointer = self.allocate(len(payload))
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
        return pointer

    @staticmethod
    def download(pointer: int, nbytes: int) -> bytes:
        host = np.empty(nbytes, dtype=np.uint8)
        cuda_check(
            cudart.cudaMemcpy(
                host.ctypes.data,
                pointer,
                nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            ),
            "cudaMemcpy(D2H)",
        )
        return host.tobytes()

    def close(self) -> None:
        for pointer in reversed(self.pointers):
            cuda_check(cudart.cudaFree(pointer), "cudaFree")
        self.pointers.clear()


def load_payload(root: Path, spec: dict[str, object]) -> bytes:
    payload = (root / str(spec["path"])).read_bytes()
    if len(payload) != int(spec["nbytes"]):
        raise RuntimeError(
            f"fixture size mismatch for {spec['path']}: "
            f"{len(payload)} != {spec['nbytes']}"
        )
    digest = hashlib.sha256(payload).hexdigest()
    if digest != spec["sha256"]:
        raise RuntimeError(f"fixture SHA-256 mismatch for {spec['path']}")
    return payload


def matching_profiles(engine, sequence_length: int) -> tuple[int, ...]:
    matches = []
    for profile_index in range(engine.num_optimization_profiles):
        minimum, optimum, maximum = engine.get_tensor_profile_shape(
            "hidden", profile_index
        )
        if (
            minimum[1] <= sequence_length <= maximum[1]
            and optimum[1] == sequence_length
        ):
            matches.append(profile_index)
    return tuple(matches)


def configure_context(
    engine,
    context,
    profile_index: int,
    sequence_length: int,
    batch_size: int,
    stream: int,
) -> None:
    if context.active_optimization_profile != profile_index:
        if not context.set_optimization_profile_async(profile_index, stream):
            raise RuntimeError(f"failed to select TensorRT profile {profile_index}")
    shapes = {
        "hidden": (batch_size, sequence_length, 1024),
        "input_pos": (batch_size, sequence_length),
        "mask": (batch_size, sequence_length, 16),
    }
    for name, shape in shapes.items():
        if not context.set_input_shape(name, shape):
            raise RuntimeError(f"failed to set {name} shape to {shape}")


def bind_context(
    context,
    input_pointers: dict[str, int],
    state_inputs: dict[str, int],
    state_outputs: dict[str, int],
    output_pointer: int,
    extra_output_pointers: dict[str, int] | None = None,
) -> None:
    bindings = dict(input_pointers)
    bindings.update(state_inputs)
    bindings.update(state_outputs)
    bindings["output"] = output_pointer
    if extra_output_pointers:
        bindings.update(extra_output_pointers)
    for name, pointer in bindings.items():
        if not context.set_tensor_address(name, pointer):
            raise RuntimeError(f"failed to bind TensorRT tensor {name}")


def enqueue_context(context, stream: int) -> None:
    if not context.execute_async_v3(stream):
        raise RuntimeError("TensorRT execute_async_v3 failed")


def capture_context_graph(context, stream_handle, stream: int):
    # TensorRT requires one enqueue after shapes/addresses are set and before
    # CUDA stream capture. The explicit input and output state buffers do not
    # alias, so this warm-up is safe and deterministic.
    enqueue_context(context, stream)
    cuda_check(cudart.cudaStreamSynchronize(stream_handle), "graph warm-up")
    cuda_check(
        cudart.cudaStreamBeginCapture(
            stream_handle, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal
        ),
        "cudaStreamBeginCapture",
    )
    enqueue_context(context, stream)
    graph = cuda_check(
        cudart.cudaStreamEndCapture(stream_handle), "cudaStreamEndCapture"
    )
    graph_exec = cuda_check(
        cudart.cudaGraphInstantiate(graph, 0), "cudaGraphInstantiate"
    )
    return graph, graph_exec


def benchmark_alternating(
    operations,
    iterations: int,
    stream_handle,
) -> dict[str, float | int]:
    if iterations < 2:
        raise ValueError("--benchmark-iterations must be at least 2")
    for index in range(20):
        operations[index & 1]()
    cuda_check(cudart.cudaStreamSynchronize(stream_handle), "benchmark warm-up")
    start = cuda_check(cudart.cudaEventCreate(), "cudaEventCreate(start)")
    end = cuda_check(cudart.cudaEventCreate(), "cudaEventCreate(end)")
    try:
        cuda_check(cudart.cudaEventRecord(start, stream_handle), "cudaEventRecord(start)")
        for index in range(iterations):
            operations[index & 1]()
        cuda_check(cudart.cudaEventRecord(end, stream_handle), "cudaEventRecord(end)")
        cuda_check(cudart.cudaEventSynchronize(end), "cudaEventSynchronize")
        total_ms = float(
            cuda_check(cudart.cudaEventElapsedTime(start, end), "cudaEventElapsedTime")
        )
    finally:
        cuda_check(cudart.cudaEventDestroy(start), "cudaEventDestroy(start)")
        cuda_check(cudart.cudaEventDestroy(end), "cudaEventDestroy(end)")
    return {
        "iterations": iterations,
        "total_ms": round(total_ms, 3),
        "mean_ms": round(total_ms / iterations, 4),
        "fourteen_steps_ms": round(total_ms / iterations * 14, 3),
    }


def summarize_benchmark_rounds(rounds: list[dict[str, float | int]]) -> dict[str, object]:
    samples = [float(item["mean_ms"]) for item in rounds]
    median_ms = statistics.median(samples)
    return {
        "rounds": rounds,
        "median_ms": round(median_ms, 4),
        "min_ms": round(min(samples), 4),
        "max_ms": round(max(samples), 4),
        "fourteen_steps_median_ms": round(median_ms * 14, 3),
    }


def bfloat16_to_float32(payload: bytes) -> np.ndarray:
    words = np.frombuffer(payload, dtype=np.uint16)
    return (words.astype(np.uint32) << 16).view(np.float32)


def compare_payload(
    actual: bytes,
    expected: bytes,
    spec: dict[str, object],
) -> dict[str, object]:
    equal = actual == expected
    result: dict[str, object] = {
        "path": spec["path"],
        "dtype": spec["dtype"],
        "shape": spec["shape"],
        "bytes": len(actual),
        "bitwise_equal": equal,
        "actual_sha256": hashlib.sha256(actual).hexdigest(),
        "expected_sha256": spec["sha256"],
    }
    if not equal and spec["dtype"] == "bfloat16":
        delta = bfloat16_to_float32(actual) - bfloat16_to_float32(expected)
        result["max_abs"] = float(np.max(np.abs(delta)))
        result["mean_abs"] = float(np.mean(np.abs(delta)))
    elif not equal and spec["dtype"] == "float32":
        delta = np.frombuffer(actual, dtype=np.float32) - np.frombuffer(
            expected, dtype=np.float32
        )
        result["max_abs"] = float(np.max(np.abs(delta)))
        result["mean_abs"] = float(np.mean(np.abs(delta)))
    elif not equal:
        actual_bytes = np.frombuffer(actual, dtype=np.uint8)
        expected_bytes = np.frombuffer(expected, dtype=np.uint8)
        result["different_bytes"] = int(np.count_nonzero(actual_bytes != expected_bytes))
    return result


def main() -> None:
    args = parse_args()
    if "torch" in sys.modules:
        raise RuntimeError("PyTorch was imported before standalone execution")

    manifest_path = args.fixture / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "voxtream-explicit-kv-trajectory-v1":
        raise ValueError(f"unsupported fixture format in {manifest_path}")

    engine_payload = args.engine.read_bytes()
    engine_sha256 = hashlib.sha256(engine_payload).hexdigest()
    if engine_sha256 != manifest["engine"]["sha256"]:
        raise RuntimeError("engine does not match fixture SHA-256")

    started = time.perf_counter()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_payload)
    del engine_payload
    if engine is None:
        raise RuntimeError(f"failed to deserialize {args.engine}")

    q1_profiles = matching_profiles(engine, 1)
    q2_profiles = matching_profiles(engine, 2)
    if not q1_profiles or not q2_profiles:
        raise RuntimeError(
            f"required profiles missing: q1={q1_profiles}, q2={q2_profiles}"
        )

    stream_handle = cuda_check(cudart.cudaStreamCreate(), "cudaStreamCreate")
    stream = int(stream_handle)
    arena = CudaArena()
    comparisons: dict[str, object] = {}
    graph_metrics: dict[str, object] = {}
    graphs = []
    graph_execs = []
    try:
        q1_context = engine.create_execution_context()
        q1_graph_context = engine.create_execution_context()
        q2_context = engine.create_execution_context()
        if q1_context is None or q1_graph_context is None or q2_context is None:
            raise RuntimeError("failed to create TensorRT execution contexts")
        if len(q1_profiles) < 2:
            raise RuntimeError(
                "standalone CUDA Graph validation requires two q=1 profiles"
            )
        batch_size = int(manifest["batch_size"])
        configure_context(
            engine, q1_context, q1_profiles[0], 1, batch_size, stream
        )
        configure_context(
            engine, q1_graph_context, q1_profiles[1], 1, batch_size, stream
        )
        configure_context(
            engine, q2_context, q2_profiles[0], 2, batch_size, stream
        )
        cuda_check(cudart.cudaStreamSynchronize(stream_handle), "cudaStreamSynchronize")

        input_buffers: dict[str, dict[str, int]] = {}
        for stage in ("q2", "q1"):
            input_buffers[stage] = {
                name: arena.upload(load_payload(args.fixture, spec))
                for name, spec in manifest["inputs"][stage].items()
            }

        state_a: dict[str, int] = {}
        state_b: dict[str, int] = {}
        for state in manifest["states"]:
            input_name = state["input_name"]
            initial = load_payload(args.fixture, state["initial"])
            state_a[input_name] = arena.upload(initial)
            state_b[input_name] = arena.allocate(len(initial))

        q2_output_spec = manifest["expected_outputs"]["q2"]
        q1_output_spec = manifest["expected_outputs"]["q1"]
        q2_output = arena.allocate(int(q2_output_spec["nbytes"]))
        q1_output = arena.allocate(int(q1_output_spec["nbytes"]))
        extra_specs = manifest.get("expected_extra_outputs", {})
        extra_outputs = {
            stage: {
                name: arena.allocate(int(spec["nbytes"]))
                for name, spec in extra_specs.get(stage, {}).items()
            }
            for stage in ("q2", "q1")
        }

        q2_state_outputs = {
            state["output_name"]: state_b[state["input_name"]]
            for state in manifest["states"]
        }
        q1_state_outputs = {
            state["output_name"]: state_a[state["input_name"]]
            for state in manifest["states"]
        }
        bind_context(
            q2_context,
            input_buffers["q2"],
            state_a,
            q2_state_outputs,
            q2_output,
            extra_outputs["q2"],
        )
        bind_context(
            q1_context,
            input_buffers["q1"],
            state_b,
            q1_state_outputs,
            q1_output,
            extra_outputs["q1"],
        )
        enqueue_context(q2_context, stream)
        enqueue_context(q1_context, stream)
        cuda_check(cudart.cudaStreamSynchronize(stream_handle), "cudaStreamSynchronize")

        for stage, pointer, spec in (
            ("q2_output", q2_output, q2_output_spec),
            ("q1_output", q1_output, q1_output_spec),
        ):
            actual = arena.download(pointer, int(spec["nbytes"]))
            expected = load_payload(args.fixture, spec)
            comparisons[stage] = compare_payload(actual, expected, spec)

        for stage in ("q2", "q1"):
            for name, pointer in extra_outputs[stage].items():
                spec = extra_specs[stage][name]
                actual = arena.download(pointer, int(spec["nbytes"]))
                expected = load_payload(args.fixture, spec)
                comparisons[f"{stage}_{name}"] = compare_payload(
                    actual, expected, spec
                )

        for state in manifest["states"]:
            input_name = state["input_name"]
            for stage, pointer, spec_name in (
                ("after_q2", state_b[input_name], "after_q2"),
                ("after_q1", state_a[input_name], "after_q1"),
            ):
                spec = state[spec_name]
                actual = arena.download(pointer, int(spec["nbytes"]))
                expected = load_payload(args.fixture, spec)
                comparisons[f"{stage}.{input_name}"] = compare_payload(
                    actual, expected, spec
                )

        # Graph A owns the q1 state layout B->A and profile 0. Its warm-up and
        # replay must reproduce the already validated direct q1 step exactly.
        graph_a, graph_exec_a = capture_context_graph(
            q1_context, stream_handle, stream
        )
        graphs.append(graph_a)
        graph_execs.append(graph_exec_a)
        cuda_check(cudart.cudaGraphLaunch(graph_exec_a, stream_handle), "graph A launch")
        cuda_check(cudart.cudaStreamSynchronize(stream_handle), "graph A synchronize")
        actual = arena.download(q1_output, int(q1_output_spec["nbytes"]))
        expected = load_payload(args.fixture, q1_output_spec)
        comparisons["graph_q1_output"] = compare_payload(
            actual, expected, q1_output_spec
        )
        for name, pointer in extra_outputs["q1"].items():
            spec = extra_specs["q1"][name]
            actual = arena.download(pointer, int(spec["nbytes"]))
            expected = load_payload(args.fixture, spec)
            comparisons[f"graph_q1_{name}"] = compare_payload(
                actual, expected, spec
            )
        for state in manifest["states"]:
            spec = state["after_q1"]
            actual = arena.download(
                state_a[state["input_name"]], int(spec["nbytes"])
            )
            expected = load_payload(args.fixture, spec)
            comparisons[f"graph_after_q1.{state['input_name']}"] = compare_payload(
                actual, expected, spec
            )

        # Graph B owns the reverse A->B state layout and the second q1 profile.
        bind_context(
            q1_graph_context,
            input_buffers["q1"],
            state_a,
            {
                state["output_name"]: state_b[state["input_name"]]
                for state in manifest["states"]
            },
            q1_output,
            extra_outputs["q1"],
        )
        graph_b, graph_exec_b = capture_context_graph(
            q1_graph_context, stream_handle, stream
        )
        graphs.append(graph_b)
        graph_execs.append(graph_exec_b)

        direct_operations = (
            lambda: enqueue_context(q1_context, stream),
            lambda: enqueue_context(q1_graph_context, stream),
        )
        graph_operations = (
            lambda: cuda_check(
                cudart.cudaGraphLaunch(graph_exec_a, stream_handle),
                "cudaGraphLaunch(A)",
            ),
            lambda: cuda_check(
                cudart.cudaGraphLaunch(graph_exec_b, stream_handle),
                "cudaGraphLaunch(B)",
            ),
        )
        if args.benchmark_rounds < 1:
            raise ValueError("--benchmark-rounds must be at least 1")
        # Heat the GPU before either candidate is timed, then alternate which
        # candidate runs first to avoid assigning Jetson clock ramp to direct.
        for index in range(100):
            graph_operations[index & 1]()
        cuda_check(cudart.cudaStreamSynchronize(stream_handle), "clock warm-up")
        direct_rounds = []
        graph_rounds = []
        for round_index in range(args.benchmark_rounds):
            candidates = (
                ((direct_operations, direct_rounds), (graph_operations, graph_rounds))
                if round_index % 2 == 0
                else ((graph_operations, graph_rounds), (direct_operations, direct_rounds))
            )
            for operations, destination in candidates:
                destination.append(
                    benchmark_alternating(
                        operations,
                        args.benchmark_iterations,
                        stream_handle,
                    )
                )
        direct_metrics = summarize_benchmark_rounds(direct_rounds)
        graph_benchmark = summarize_benchmark_rounds(graph_rounds)
        graph_metrics = {
            "captures": 2,
            "profiles": [q1_profiles[0], q1_profiles[1]],
            "direct_enqueue": direct_metrics,
            "graph_replay": graph_benchmark,
            "speedup": round(
                direct_metrics["median_ms"] / graph_benchmark["median_ms"], 3
            ),
        }
    finally:
        for graph_exec in reversed(graph_execs):
            cuda_check(cudart.cudaGraphExecDestroy(graph_exec), "cudaGraphExecDestroy")
        for graph in reversed(graphs):
            cuda_check(cudart.cudaGraphDestroy(graph), "cudaGraphDestroy")
        arena.close()
        cuda_check(cudart.cudaStreamDestroy(stream_handle), "cudaStreamDestroy")

    all_equal = all(
        bool(comparison["bitwise_equal"])
        for comparison in comparisons.values()
    )
    result = {
        "runtime": "tensorrt+cuda-python",
        "torch_imported": "torch" in sys.modules,
        "tensorrt": trt.__version__,
        "cuda_runtime": cuda_check(
            cudart.cudaRuntimeGetVersion(), "cudaRuntimeGetVersion"
        ),
        "engine": str(args.engine),
        "engine_sha256": engine_sha256,
        "profiles": {"q1": list(q1_profiles), "q2": list(q2_profiles)},
        "trajectory": [2, 1],
        "cuda_graph": graph_metrics,
        "comparisons": comparisons,
        "all_bitwise_equal": all_equal,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "max_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
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
