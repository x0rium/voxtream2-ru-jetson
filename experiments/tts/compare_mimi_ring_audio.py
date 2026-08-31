#!/usr/bin/env python3
"""Decode one code trajectory before and after the Mimi ring-cache export patch."""

from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path

import numpy as np
import torch
from moshi.modules.transformer import RingKVCache
from voxtream_tensorrt_mimi_probe import (
    build_decoder,
    make_ring_kv_update_exportable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-codebooks", type=int, default=16)
    parser.add_argument("--sample-rate", type=int, default=24_000)
    return parser.parse_args()


def decode(model, codes: torch.Tensor) -> tuple[np.ndarray, float]:
    model.reset_streaming()
    frames = []
    started = time.perf_counter()
    with torch.inference_mode():
        for frame_codes in codes:
            frames.append(model.decode(frame_codes).float().cpu().numpy().reshape(-1))
    torch.cuda.synchronize()
    return np.concatenate(frames), time.perf_counter() - started


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as sink:
        sink.setnchannels(1)
        sink.setsampwidth(2)
        sink.setframerate(sample_rate)
        sink.writeframes(pcm.tobytes())


def frame_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    samples_per_frame: int,
) -> dict[str, object]:
    if reference.shape != candidate.shape:
        raise ValueError(f"audio shape mismatch: {reference.shape} != {candidate.shape}")
    difference = reference - candidate
    reference_frames = reference.reshape(-1, samples_per_frame)
    candidate_frames = candidate.reshape(-1, samples_per_frame)
    per_frame_max = np.max(np.abs(reference_frames - candidate_frames), axis=1)
    unequal = np.flatnonzero(per_frame_max != 0)
    denominator = np.linalg.norm(reference) * np.linalg.norm(candidate)
    return {
        "samples": int(reference.size),
        "frames": int(reference_frames.shape[0]),
        "first_unequal_frame": int(unequal[0]) if unequal.size else None,
        "unequal_frames": int(unequal.size),
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "cosine": float(np.dot(reference, candidate) / denominator),
        "selected_frame_max_abs": {
            str(index): float(per_frame_max[index])
            for index in (0, 15, 124, 125, 126, 249, 250, 251, 374, 375, 376)
            if index < per_frame_max.size
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    codes = np.load(args.codes)
    if codes.ndim != 4 or codes.shape[1:] != (1, args.num_codebooks, 1):
        raise ValueError(
            "codes must have shape [frames, 1, num_codebooks, 1], "
            f"got {codes.shape}"
        )
    if codes.min() < 0 or codes.max() >= 2048:
        raise ValueError(f"Mimi code range is invalid: [{codes.min()}, {codes.max()}]")
    device_codes = torch.from_numpy(np.ascontiguousarray(codes, dtype=np.int64)).cuda()

    upstream_complete = RingKVCache.complete
    model = build_decoder(args.checkpoint, args.num_codebooks)
    upstream_audio, upstream_seconds = decode(model, device_codes)

    make_ring_kv_update_exportable()
    patched_audio, patched_seconds = decode(model, device_codes)
    RingKVCache.complete = upstream_complete

    samples_per_frame = upstream_audio.size // codes.shape[0]
    write_wav(args.output_dir / "mimi-upstream.wav", upstream_audio, args.sample_rate)
    write_wav(args.output_dir / "mimi-export-patch.wav", patched_audio, args.sample_rate)
    result = {
        "frames": int(codes.shape[0]),
        "samples_per_frame": int(samples_per_frame),
        "ring_capacity_audio_frames": 125,
        "upstream_seconds": round(upstream_seconds, 3),
        "patched_seconds": round(patched_seconds, 3),
        "patch_vs_upstream": frame_metrics(
            upstream_audio,
            patched_audio,
            samples_per_frame,
        ),
    }
    metrics_path = args.output_dir / "mimi-ring-comparison.json"
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
