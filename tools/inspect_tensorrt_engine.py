#!/usr/bin/env python3
"""Print TensorRT plan I/O without allocating execution buffers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorrt as trt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    args = parser.parse_args()
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError(f"failed to load {args.engine}")
    result = {
        "engine": str(args.engine),
        "bytes": args.engine.stat().st_size,
        "io": [
            {
                "index": index,
                "name": engine.get_tensor_name(index),
                "mode": str(
                    engine.get_tensor_mode(engine.get_tensor_name(index))
                ).split(".")[-1],
                "dtype": str(
                    engine.get_tensor_dtype(engine.get_tensor_name(index))
                ),
                "shape": list(
                    engine.get_tensor_shape(engine.get_tensor_name(index))
                ),
            }
            for index in range(engine.num_io_tensors)
        ],
    }
    result["inputs"] = sum(item["mode"] == "INPUT" for item in result["io"])
    result["outputs"] = sum(item["mode"] == "OUTPUT" for item in result["io"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
