#!/usr/bin/env python3
"""Normalize only digit-containing sentences with RUNorm in a short-lived process."""

from __future__ import annotations

import argparse
import functools
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--workdir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import runorm.runorm as runorm_module

    original_pipeline = runorm_module.pipeline

    @functools.wraps(original_pipeline)
    def pipeline_on_cpu(*pipeline_args, **pipeline_kwargs):
        pipeline_kwargs.setdefault("device", -1)
        return original_pipeline(*pipeline_args, **pipeline_kwargs)

    runorm_module.pipeline = pipeline_on_cpu

    from runorm import RUNorm

    normalizer = RUNorm()
    normalizer.load(model_size="medium", device="cpu", workdir=str(args.workdir))
    parts = re.split(r"(?<=[.!?…])\s+", args.text)
    normalized = " ".join(
        normalizer.norm(part) if re.search(r"\d", part) else part for part in parts
    )
    print(json.dumps({"input": args.text, "normalized": normalized}, ensure_ascii=False))


if __name__ == "__main__":
    main()
