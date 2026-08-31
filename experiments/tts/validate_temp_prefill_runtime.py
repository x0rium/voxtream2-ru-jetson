#!/usr/bin/env python3
"""Compare batched sink-attention prefill against exact q=1 replay."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from cuda import cudart

from voxtream2_ru_jetson.runtime import (
    RawBundle,
    TempDecoder,
    bfloat16_to_float32,
    download_array,
    float32_to_bfloat16,
    sample_semantic,
)
from voxtream2_ru_jetson.tensorrt_standalone import CudaArena, cuda_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--temp-engine", type=Path, required=True)
    parser.add_argument("--temp-prefill-engine", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=625)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def capture_state(decoder: TempDecoder) -> dict[str, np.ndarray]:
    captured = {}
    for name, pointer in decoder.state.items():
        template = decoder.initial[name]
        captured[name] = download_array(pointer, template.dtype, template.shape)
    return captured


def run_variant(
    assets: Path,
    temp_engine: Path,
    prefill_engine: Path | None,
    hidden_steps: list[np.ndarray],
) -> dict[str, object]:
    bundle = RawBundle(assets)
    stream_handle = cuda_check(cudart.cudaStreamCreate(), "cudaStreamCreate")
    stream = int(stream_handle)
    arena = CudaArena()
    decoder = TempDecoder(
        temp_engine,
        bundle,
        stream,
        arena,
        prefill_engine,
    )
    start_position = int(bundle.manifest["prompt_frames"])
    final_position = int(bundle.manifest["config"]["audio_window_size"])
    positions = range(start_position, final_position + 1)
    if len(hidden_steps) != len(positions):
        raise ValueError("hidden trajectory length mismatch")
    try:
        output = None
        logits = None
        for position, hidden in zip(positions, hidden_steps):
            output, logits = decoder.step(hidden, position)
        if output is None or logits is None:
            raise RuntimeError("empty validation trajectory")
        state = capture_state(decoder)
        sampled = sample_semantic(
            logits,
            bundle.manifest["config"],
            np.random.default_rng(20260831),
        )
        return {
            "output": output,
            "logits": logits,
            "state": state,
            "sampled_semantic": int(sampled[0]),
            "sampled_shift": int(sampled[1]),
            "metrics": {
                "q1_calls": decoder.calls,
                "rebuilds": decoder.sink_rebuilds,
                "logical_replay_steps": decoder.sink_replay_steps,
                "prefill_enqueues": decoder.sink_prefill_calls,
                "rebuild_seconds": decoder.sink_rebuild_seconds,
                "state_sha256": {name: array_sha256(value) for name, value in state.items()},
            },
        }
    finally:
        decoder.close()
        del decoder
        gc.collect()
        arena.close()
        cuda_check(cudart.cudaStreamDestroy(stream_handle), "cudaStreamDestroy")


def numeric_delta(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    delta = reference - candidate
    reference_flat = reference.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    cosine = np.dot(reference_flat, candidate_flat) / (
        np.linalg.norm(reference_flat) * np.linalg.norm(candidate_flat)
    )
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "cosine": float(cosine),
    }


def compare_states(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> dict[str, object]:
    per_tensor = {}
    for name in reference:
        if reference[name].dtype == np.uint16:
            item = numeric_delta(
                bfloat16_to_float32(reference[name]),
                bfloat16_to_float32(candidate[name]),
            )
        else:
            item = {
                "exact": bool(np.array_equal(reference[name], candidate[name])),
                "max_abs": int(np.max(np.abs(reference[name].astype(np.int64) - candidate[name]))),
            }
        item["byte_equal"] = bool(np.array_equal(reference[name], candidate[name]))
        per_tensor[name] = item
    return {
        "max_abs": max(float(item["max_abs"]) for item in per_tensor.values()),
        "all_byte_equal": all(bool(item["byte_equal"]) for item in per_tensor.values()),
        "per_tensor": per_tensor,
    }


def main() -> None:
    args = parse_args()
    bundle = RawBundle(args.assets)
    start_position = int(bundle.manifest["prompt_frames"])
    final_position = int(bundle.manifest["config"]["audio_window_size"])
    rng = np.random.default_rng(args.seed)
    hidden_steps = [
        float32_to_bfloat16(rng.standard_normal((2, 1, 1024), dtype=np.float32))
        for _ in range(start_position, final_position + 1)
    ]

    reference = run_variant(
        args.assets,
        args.temp_engine,
        None,
        hidden_steps,
    )
    candidate = run_variant(
        args.assets,
        args.temp_engine,
        args.temp_prefill_engine,
        hidden_steps,
    )
    reference_logits = bfloat16_to_float32(reference["logits"])
    candidate_logits = bfloat16_to_float32(candidate["logits"])
    reference_cfg = 1.5 * reference_logits[0] - 0.5 * reference_logits[1]
    candidate_cfg = 1.5 * candidate_logits[0] - 0.5 * candidate_logits[1]
    result = {
        "positions": [start_position, final_position],
        "trajectory_steps": len(hidden_steps),
        "reference": reference["metrics"],
        "candidate": candidate["metrics"],
        "output": numeric_delta(reference["output"], candidate["output"]),
        "logits": numeric_delta(reference_logits, candidate_logits),
        "cfg_logits": {
            **numeric_delta(reference_cfg, candidate_cfg),
            "reference_argmax": int(np.argmax(reference_cfg)),
            "candidate_argmax": int(np.argmax(candidate_cfg)),
        },
        "sample": {
            "reference_semantic": reference["sampled_semantic"],
            "candidate_semantic": candidate["sampled_semantic"],
            "reference_shift": reference["sampled_shift"],
            "candidate_shift": candidate["sampled_shift"],
        },
        "state": compare_states(reference["state"], candidate["state"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
