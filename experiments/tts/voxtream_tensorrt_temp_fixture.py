#!/usr/bin/env python3
"""Export a real temp_former TensorRT trajectory as raw no-Torch fixture files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from voxtream_tensorrt_explicit_kv_probe import ExplicitKVTRTRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dtype_name(dtype: torch.dtype) -> str:
    return {
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.int64: "int64",
        torch.int32: "int32",
        torch.bool: "bool",
    }[dtype]


def tensor_payload(value: torch.Tensor) -> bytes:
    return (
        value.detach()
        .to(device="cpu")
        .contiguous()
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )


def write_tensor(root: Path, relative: Path, value: torch.Tensor) -> dict[str, object]:
    payload = tensor_payload(value)
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(relative),
        "dtype": dtype_name(value.dtype),
        "shape": list(value.shape),
        "nbytes": len(payload),
        "sha256": sha256(payload),
    }


def exact_shape(value: torch.Tensor, expected: tuple[int, ...]) -> torch.Tensor:
    actual = tuple(value.shape)
    if actual == expected:
        return value.contiguous()
    if len(actual) != len(expected) or any(
        source not in (1, target) for source, target in zip(actual, expected)
    ):
        raise ValueError(f"cannot broadcast fixture tensor {actual} to {expected}")
    return value.expand(expected).contiguous()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.capture, map_location="cpu", weights_only=False)
    if payload.get("format") != "voxtream-temp-real-trajectory-v1":
        raise RuntimeError(f"unsupported capture format: {payload.get('format')!r}")
    state_names = tuple(payload["buffer_names"])
    initial_state_cpu = tuple(payload["initial_state"])
    initial_state_cuda = tuple(value.to(device="cuda") for value in initial_state_cpu)
    runner = ExplicitKVTRTRunner(
        args.engine,
        state_names,
        initial_state_cuda,
        sequence_length=1,
        inplace_state=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    states = []
    for index, (name, value) in enumerate(zip(state_names, initial_state_cpu)):
        tensor_name = name.replace(".", "_")
        states.append(
            {
                "input_name": tensor_name,
                "output_name": f"next_{tensor_name}",
                "initial": write_tensor(
                    args.output_dir,
                    Path("state") / f"{index:02d}.initial.bin",
                    value,
                ),
            }
        )

    records = []
    for index, captured in enumerate(payload["records"]):
        inputs_cpu = {
            name: exact_shape(captured[name], runner.input_shapes[name])
            for name in ("hidden", "input_pos", "mask")
        }
        inputs_cuda = {
            name: value.to(device="cuda") for name, value in inputs_cpu.items()
        }
        candidate, _ = runner.step(
            inputs_cuda["hidden"],
            inputs_cuda["input_pos"],
            inputs_cuda["mask"],
        )
        record_root = Path("records") / f"{index:03d}"
        records.append(
            {
                "index": index,
                "position": int(captured["position"]),
                "inputs": {
                    name: write_tensor(
                        args.output_dir,
                        record_root / f"{name}.bin",
                        value,
                    )
                    for name, value in inputs_cpu.items()
                },
                "output": write_tensor(
                    args.output_dir,
                    record_root / "output.bin",
                    candidate,
                ),
                "extra_outputs": {
                    name: write_tensor(
                        args.output_dir,
                        record_root / f"{name}.bin",
                        value,
                    )
                    for name, value in runner.extra_outputs.items()
                },
            }
        )

    for state, value in zip(states, runner.state):
        index = states.index(state)
        state["final"] = write_tensor(
            args.output_dir,
            Path("state") / f"{index:02d}.final.bin",
            value,
        )

    manifest = {
        "format": "voxtream-temp-tensorrt-trajectory-v1",
        "engine": {
            "path": str(args.engine),
            "bytes": args.engine.stat().st_size,
            "sha256": file_sha256(args.engine),
        },
        "capture": str(args.capture),
        "inplace_state": True,
        "records": records,
        "states": states,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "records": len(records),
                "states": len(states),
                "fixture_bytes": sum(
                    path.stat().st_size
                    for path in args.output_dir.rglob("*.bin")
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
