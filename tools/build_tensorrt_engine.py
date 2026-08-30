#!/usr/bin/env python3
"""Build a TensorRT engine in a clean process with reproducible metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import tensorrt as trt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--timing-cache", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--workspace-mib", type=int, default=512)
    parser.add_argument("--optimization-level", type=int, choices=range(0, 6), default=3)
    parser.add_argument("--max-aux-streams", type=int, default=0)
    parser.add_argument(
        "--sequence-profiles",
        type=int,
        nargs="+",
        choices=(1, 2),
        help=(
            "Add fixed-shape optimization profiles for a dynamic sequence axis. "
            "Values may repeat when separate live execution contexts need the "
            "same q shape, for example: 1 1 2."
        ),
    )
    parser.add_argument(
        "--sequence-range",
        type=int,
        nargs=3,
        metavar=("MIN", "OPT", "MAX"),
        help=(
            "Add one optimization profile and replace every dynamic sequence "
            "axis with MIN/OPT/MAX. Intended for full-sequence encoders whose "
            "mask has two tied dynamic axes."
        ),
    )
    args = parser.parse_args()
    if args.sequence_profiles and args.sequence_range:
        parser.error("--sequence-profiles and --sequence-range are mutually exclusive")
    if args.sequence_range:
        minimum, optimum, maximum = args.sequence_range
        if not 1 <= minimum <= optimum <= maximum:
            parser.error("--sequence-range requires 1 <= MIN <= OPT <= MAX")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    parse_started = time.perf_counter()
    if not parser.parse_from_file(str(args.onnx)):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))
    parse_seconds = time.perf_counter() - parse_started

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.BF16)
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, args.workspace_mib * 1024 * 1024
    )
    config.builder_optimization_level = args.optimization_level
    config.max_aux_streams = args.max_aux_streams

    profile_metrics: list[dict[str, list[int]]] = []
    if args.sequence_profiles:
        dynamic_inputs = []
        for input_index in range(network.num_inputs):
            tensor = network.get_input(input_index)
            shape = tuple(tensor.shape)
            if any(dimension == -1 for dimension in shape):
                dynamic_inputs.append((tensor, shape))
        if not dynamic_inputs:
            raise ValueError(
                "--sequence-profiles was set, but the ONNX network has no "
                "dynamic inputs"
            )

        for sequence_length in args.sequence_profiles:
            profile = builder.create_optimization_profile()
            profile_shapes: dict[str, list[int]] = {}
            for tensor, network_shape in dynamic_inputs:
                if len(network_shape) < 2 or network_shape[1] != -1:
                    raise ValueError(
                        f"unsupported dynamic shape for {tensor.name}: "
                        f"{network_shape}; expected only axis 1 to be dynamic"
                    )
                if any(
                    dimension == -1
                    for axis, dimension in enumerate(network_shape)
                    if axis != 1
                ):
                    raise ValueError(
                        f"unsupported additional dynamic axis for {tensor.name}: "
                        f"{network_shape}"
                    )
                concrete_shape = tuple(
                    sequence_length if axis == 1 else dimension
                    for axis, dimension in enumerate(network_shape)
                )
                profile.set_shape(
                    tensor.name,
                    concrete_shape,
                    concrete_shape,
                    concrete_shape,
                )
                profile_shapes[tensor.name] = list(concrete_shape)
            profile_index = config.add_optimization_profile(profile)
            if profile_index < 0:
                raise RuntimeError("failed to add TensorRT optimization profile")
            profile_metrics.append(profile_shapes)

    if args.sequence_range:
        dynamic_inputs = []
        for input_index in range(network.num_inputs):
            tensor = network.get_input(input_index)
            shape = tuple(tensor.shape)
            if any(dimension == -1 for dimension in shape):
                dynamic_inputs.append((tensor, shape))
        if not dynamic_inputs:
            raise ValueError(
                "--sequence-range was set, but the ONNX network has no dynamic inputs"
            )
        minimum_length, optimum_length, maximum_length = args.sequence_range
        profile = builder.create_optimization_profile()
        profile_shapes: dict[str, list[list[int]]] = {}
        for tensor, network_shape in dynamic_inputs:
            shapes = tuple(
                tuple(
                    length if dimension == -1 else dimension
                    for dimension in network_shape
                )
                for length in (
                    minimum_length,
                    optimum_length,
                    maximum_length,
                )
            )
            profile.set_shape(tensor.name, *shapes)
            profile_shapes[tensor.name] = [list(shape) for shape in shapes]
        profile_index = config.add_optimization_profile(profile)
        if profile_index < 0:
            raise RuntimeError("failed to add TensorRT sequence-range profile")
        profile_metrics.append(profile_shapes)

    if args.timing_cache is not None:
        cache_bytes = args.timing_cache.read_bytes() if args.timing_cache.exists() else b""
        cache = config.create_timing_cache(cache_bytes)
        config.set_timing_cache(cache, ignore_mismatch=False)

    build_started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - build_started
    if serialized is None:
        raise RuntimeError("TensorRT returned no serialized engine")

    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(bytes(serialized))
    if args.timing_cache is not None:
        args.timing_cache.write_bytes(bytes(config.get_timing_cache().serialize()))

    metrics = {
        "tensorrt": trt.__version__,
        "onnx": str(args.onnx),
        "onnx_sha256": sha256(args.onnx),
        "onnx_bytes": args.onnx.stat().st_size,
        "engine": str(args.engine),
        "engine_sha256": sha256(args.engine),
        "engine_bytes": args.engine.stat().st_size,
        "network_layers": network.num_layers,
        "workspace_mib": args.workspace_mib,
        "optimization_level": args.optimization_level,
        "max_aux_streams": args.max_aux_streams,
        "optimization_profiles": profile_metrics,
        "parse_seconds": round(parse_seconds, 3),
        "build_seconds": round(build_seconds, 3),
    }
    if args.metrics is not None:
        args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
