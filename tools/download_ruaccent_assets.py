#!/usr/bin/env python3
"""Download the RUAccent files used by the torch-free runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPOSITORY = "ruaccent/accentuator"
REVISION = "b78ae5ea1e62beaf138bed1865cd8c3b0b5ca855"
ALLOW_PATTERNS = (
    "dictionary/accents.json.gz",
    "dictionary/omographs.json.gz",
    "dictionary/yo_homographs.json.gz",
    "dictionary/yo_words.json.gz",
    "nn/nn_accent/*",
    "nn/nn_omograph/turbo3.1/*",
    "nn/nn_stress_usage_predictor/*",
    "nn/nn_yo_homograph_resolver/*",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output",
        type=Path,
        help="Directory that will contain dictionary/ and nn/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    snapshot_download(
        repo_id=REPOSITORY,
        repo_type="model",
        revision=REVISION,
        local_dir=output,
        allow_patterns=list(ALLOW_PATTERNS),
    )
    print(output)


if __name__ == "__main__":
    main()
