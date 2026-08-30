#!/usr/bin/env python3
"""Add exact temporal prefill inputs to an existing torchless asset bundle.

This is a one-time offline migration for v1 bundles exported before
``prefill.hidden`` became part of the additive ABI. The inference runtime
itself remains PyTorch-free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

ALIGNMENT = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--prefill-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.output.resolve() == args.assets.resolve():
        raise ValueError("output must differ from the source asset manifest")
    output_binary = args.output.with_suffix(".bin")
    if args.output.exists() or output_binary.exists():
        raise FileExistsError("output bundle already exists")

    manifest = json.loads(args.assets.read_text())
    if manifest.get("format") != "voxtream-torchless-fixed-utterance-v1":
        raise ValueError("unsupported torchless asset format")
    if any(item["name"] == "prefill.hidden" for item in manifest["tensors"]):
        raise ValueError("asset bundle already contains prefill.hidden")
    source_binary = args.assets.parent / manifest["binary"]
    if source_binary.stat().st_size != int(manifest["binary_bytes"]):
        raise ValueError("source asset binary size mismatch")
    if file_sha256(source_binary) != manifest["binary_sha256"]:
        raise ValueError("source asset binary checksum mismatch")

    prefill = torch.load(args.prefill_cache, map_location="cpu", weights_only=False)
    if prefill.get("format") != "voxtream-temp-fixed-prompt-prefill-v1":
        raise ValueError("unsupported temporal prefill cache")
    if tuple(prefill["buffer_names"]) != tuple(manifest["prefill_buffer_names"]):
        raise ValueError("prefill cache buffer ABI does not match the asset bundle")
    hidden = prefill["hidden"].detach().contiguous().cpu()
    expected_shape = (2, int(manifest["prompt_frames"]), 1024)
    if tuple(hidden.shape) != expected_shape:
        raise ValueError(f"prefill hidden shape changed: {tuple(hidden.shape)} != {expected_shape}")
    if hidden.dtype != torch.bfloat16:
        raise TypeError(f"prefill hidden dtype changed: {hidden.dtype}")
    payload = hidden.view(torch.uint16).numpy().tobytes()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_binary, output_binary)
    offset = (output_binary.stat().st_size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
    with output_binary.open("ab") as sink:
        sink.write(b"\0" * (offset - sink.tell()))
        sink.write(payload)

    manifest["binary"] = output_binary.name
    manifest["binary_bytes"] = output_binary.stat().st_size
    manifest["binary_sha256"] = file_sha256(output_binary)
    manifest["tensors"].append(
        {
            "name": "prefill.hidden",
            "dtype": "bfloat16",
            "shape": list(hidden.shape),
            "offset": offset,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    manifest["config"].setdefault(
        "audio_window_size",
        int(manifest["config"]["max_temp_position"]) + 1,
    )
    manifest.setdefault("sources", {})["prefill_hidden_migration"] = str(
        args.prefill_cache
    )
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "manifest": str(args.output),
                "binary": str(output_binary),
                "binary_bytes": manifest["binary_bytes"],
                "prefill_hidden_shape": list(hidden.shape),
                "prefill_hidden_sha256": hashlib.sha256(payload).hexdigest(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
