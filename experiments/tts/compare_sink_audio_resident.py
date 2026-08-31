#!/usr/bin/env python3
"""Generate q=1 and batched-prefill sink variants in one resident runtime."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

from voxtream2_ru_jetson.runtime import RuntimeFiles, SynthesisRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--ruaccent-assets", type=Path, required=True)
    parser.add_argument("--phone-map", type=Path, required=True)
    parser.add_argument("--temp-engine", type=Path, required=True)
    parser.add_argument("--temp-prefill-engine", type=Path, required=True)
    parser.add_argument("--dep-engine", type=Path, required=True)
    parser.add_argument("--phone-engine", type=Path, required=True)
    parser.add_argument("--mimi-engine", type=Path, required=True)
    parser.add_argument("--mimi-state", type=Path, required=True)
    parser.add_argument("--audio-embedding-weight", type=Path, required=True)
    parser.add_argument("--audio-embedding-cubin", type=Path, required=True)
    parser.add_argument("--cuda-acoustic-control-cubin", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--max-frames", type=int, default=900)
    parser.add_argument(
        "--include-trajectory",
        action="store_true",
        help="Store generated Mimi codes and frame metadata for decoder isolation.",
    )
    return parser.parse_args()


def read_pcm(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("unexpected WAV format")
        return source.readframes(source.getnframes()), source.getframerate()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text = args.text_file.read_text().strip()
    files = RuntimeFiles(
        assets=args.assets,
        temp_engine=args.temp_engine,
        temp_prefill_engine=args.temp_prefill_engine,
        dep_engine=args.dep_engine,
        phone_engine=args.phone_engine,
        mimi_engine=args.mimi_engine,
        mimi_state=args.mimi_state,
        audio_embedding_weight=args.audio_embedding_weight,
        audio_embedding_cubin=args.audio_embedding_cubin,
        cuda_acoustic_control_cubin=args.cuda_acoustic_control_cubin,
    )
    baseline_path = args.output_dir / "sink-q1-resident.wav"
    candidate_path = args.output_dir / "sink-q420-resident.wav"
    with SynthesisRuntime(
        files,
        ruaccent_assets=args.ruaccent_assets,
        phone_map=args.phone_map,
        cuda_temp_graph=True,
        cuda_dep_graph=True,
    ) as runtime:
        prefill = runtime.temp.prefill
        if prefill is None:
            raise RuntimeError("runtime did not load the prefill engine")
        runtime.temp.prefill = None
        baseline = runtime.synthesize_to_wav(
            text,
            baseline_path,
            seed=args.seed,
            max_frames=args.max_frames,
            include_trajectory=args.include_trajectory,
        )
        runtime.temp.prefill = prefill
        candidate = runtime.synthesize_to_wav(
            text,
            candidate_path,
            seed=args.seed,
            max_frames=args.max_frames,
            include_trajectory=args.include_trajectory,
        )

    baseline_pcm, sample_rate = read_pcm(baseline_path)
    candidate_pcm, candidate_rate = read_pcm(candidate_path)
    if candidate_rate != sample_rate:
        raise ValueError("sample rate changed between variants")
    common_bytes = 0
    for left, right in zip(baseline_pcm, candidate_pcm):
        if left != right:
            break
        common_bytes += 1
    result = {
        "sample_rate": sample_rate,
        "baseline": baseline,
        "candidate": candidate,
        "pcm": {
            "baseline_bytes": len(baseline_pcm),
            "candidate_bytes": len(candidate_pcm),
            "common_prefix_bytes": common_bytes,
            "first_difference_sample": common_bytes // 2,
            "first_difference_seconds": common_bytes / 2 / sample_rate,
            "bitwise_equal": baseline_pcm == candidate_pcm,
        },
    }
    output = args.output_dir / "sink-resident-comparison.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
