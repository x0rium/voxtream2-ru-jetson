#!/usr/bin/env python3
"""Compare the Jetson TensorRT prefill with a captured PyTorch reference."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from cuda import cudart

from voxtream2_ru_jetson.runtime import (
    RawBundle,
    TempDecoder,
    bfloat16_to_float32,
    download_array,
)
from voxtream2_ru_jetson.tensorrt_standalone import CudaArena, cuda_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--temp-engine", type=Path, required=True)
    parser.add_argument("--temp-prefill-engine", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def delta(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    difference = reference - candidate
    reference_flat = reference.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    cosine = np.dot(reference_flat, candidate_flat) / (
        np.linalg.norm(reference_flat) * np.linalg.norm(candidate_flat)
    )
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "cosine": float(cosine),
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.reference.with_suffix(".json").read_text())
    sequence_length = int(manifest["sequence_length"])
    fixture = np.load(args.reference)
    bundle = RawBundle(args.assets)
    stream_handle = cuda_check(cudart.cudaStreamCreate(), "cudaStreamCreate")
    stream = int(stream_handle)
    arena = CudaArena()
    decoder = TempDecoder(
        args.temp_engine,
        bundle,
        stream,
        arena,
        args.temp_prefill_engine,
    )
    try:
        decoder._reset_empty_cache()
        decoder.prefill.run(fixture["hidden"])
        output, logits = decoder._execute(fixture["next_hidden"], sequence_length, readback=True)
        state = {}
        for original_name, key in manifest["state_keys"].items():
            name = original_name.replace(".", "_")
            expected = fixture[key]
            template = decoder.initial[name]
            actual = download_array(decoder.state[name], template.dtype, template.shape)
            if original_name.endswith(("k_cache", "v_cache")):
                actual = actual[:, :, : sequence_length + 1]
                expected_float = bfloat16_to_float32(expected)
                actual_float = bfloat16_to_float32(actual)
                item = delta(expected_float, actual_float)
            else:
                item = {
                    "exact": bool(np.array_equal(expected, actual)),
                    "max_abs": int(
                        np.max(np.abs(expected.astype(np.int64) - actual.astype(np.int64)))
                    ),
                }
            item["byte_equal"] = bool(np.array_equal(expected, actual))
            state[original_name] = item

        expected_output = bfloat16_to_float32(fixture["expected_output"])
        expected_logits = bfloat16_to_float32(fixture["expected_logits"])
        actual_logits = bfloat16_to_float32(logits)
        expected_cfg = 1.5 * expected_logits[0] - 0.5 * expected_logits[1]
        actual_cfg = 1.5 * actual_logits[0] - 0.5 * actual_logits[1]
        result = {
            "sequence_length": sequence_length,
            "output": delta(expected_output, output),
            "logits": delta(expected_logits, actual_logits),
            "cfg_logits": {
                **delta(expected_cfg, actual_cfg),
                "reference_argmax": int(np.argmax(expected_cfg)),
                "candidate_argmax": int(np.argmax(actual_cfg)),
            },
            "state": {
                "max_abs": max(float(item["max_abs"]) for item in state.values()),
                "all_byte_equal": all(bool(item["byte_equal"]) for item in state.values()),
                "per_tensor": state,
            },
            "prefill_enqueues": decoder.prefill.calls,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        decoder.close()
        del decoder
        gc.collect()
        arena.close()
        cuda_check(cudart.cudaStreamDestroy(stream_handle), "cudaStreamDestroy")


if __name__ == "__main__":
    main()
