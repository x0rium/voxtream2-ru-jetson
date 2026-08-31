#!/usr/bin/env python3
"""Verify file sizes and SHA-256 values from the Hugging Face manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Downloaded Hugging Face directory")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Manifest path (default: ROOT/manifest.json)",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest or root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    failed = False
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.is_file():
            print(f"MISSING  {item['path']}")
            failed = True
            continue
        size_ok = path.stat().st_size == int(item["bytes"])
        hash_ok = sha256(path) == item["sha256"]
        state = "OK" if size_ok and hash_ok else "FAILED"
        print(f"{state:7} {item['path']}")
        failed |= not (size_ok and hash_ok)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
