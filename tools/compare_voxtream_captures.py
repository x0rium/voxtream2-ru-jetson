#!/usr/bin/env python3
"""Compare VoXtream debug captures and locate their first divergence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    return parser.parse_args()


def first_difference(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object] | None:
    shared_shape = tuple(min(a, b) for a, b in zip(reference.shape, candidate.shape))
    shared = tuple(slice(0, size) for size in shared_shape)
    unequal = np.asarray(reference[shared] != candidate[shared])
    if unequal.any():
        index = tuple(int(value) for value in np.argwhere(unequal)[0])
        return {
            "index": index,
            "reference": np.asarray(reference[index]).tolist(),
            "candidate": np.asarray(candidate[index]).tolist(),
        }
    if reference.shape != candidate.shape:
        return {"index": "shape", "reference": reference.shape, "candidate": candidate.shape}
    return None


def main() -> None:
    args = parse_args()
    with np.load(args.reference, allow_pickle=False) as reference, np.load(
        args.candidate, allow_pickle=False
    ) as candidate:
        results: dict[str, object] = {}
        for key in sorted(set(reference.files) | set(candidate.files)):
            if key not in reference.files or key not in candidate.files:
                results[key] = {"missing_from": "reference" if key not in reference.files else "candidate"}
                continue
            reference_value = reference[key]
            candidate_value = candidate[key]
            difference = first_difference(reference_value, candidate_value)
            results[key] = {
                "reference_shape": reference_value.shape,
                "candidate_shape": candidate_value.shape,
                "exact": difference is None,
                "first_difference": difference,
            }
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
