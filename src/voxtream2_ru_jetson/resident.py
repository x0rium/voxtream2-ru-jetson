#!/usr/bin/env python3
"""Keep the TTS runtime loaded and stream framed PCM records over stdio."""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO, TextIO

HEADER = struct.Struct("<cI")
MAX_RECORD_BYTES = 16 * 1024 * 1024
READY = b"R"
START = b"S"
PCM = b"P"
DONE = b"D"
ERROR = b"E"
JSON_RECORDS = frozenset((READY, START, DONE, ERROR))


def detach_protocol_stdout() -> BinaryIO:
    """Reserve the original stdout for framed records and send logs to stderr."""
    sys.stdout.flush()
    protocol_fd = os.dup(sys.stdout.fileno())
    os.set_inheritable(protocol_fd, False)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    return os.fdopen(protocol_fd, "wb", buffering=0)


def write_record(sink: BinaryIO, kind: bytes, payload: bytes) -> None:
    if len(kind) != 1:
        raise ValueError("record kind must be exactly one byte")
    sink.write(HEADER.pack(kind, len(payload)))
    sink.write(payload)
    sink.flush()


def write_json_record(sink: BinaryIO, kind: bytes, payload: dict[str, object]) -> None:
    if kind not in JSON_RECORDS:
        raise ValueError(f"record {kind!r} does not carry JSON")
    write_record(
        sink,
        kind,
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(),
    )


def _read_exact(source: BinaryIO, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = source.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def iter_records(source: BinaryIO) -> Iterator[tuple[bytes, bytes]]:
    while True:
        header = _read_exact(source, HEADER.size)
        if not header:
            return
        if len(header) != HEADER.size:
            raise EOFError("truncated resident record header")
        kind, size = HEADER.unpack(header)
        if size > MAX_RECORD_BYTES:
            raise ValueError(
                f"resident record payload is too large: {size} > {MAX_RECORD_BYTES}; "
                "stdout likely contains non-protocol text"
            )
        payload = _read_exact(source, size)
        if len(payload) != size:
            raise EOFError("truncated resident record payload")
        yield kind, payload


def _request_id(request: object, fallback: int) -> object:
    if isinstance(request, dict) and "id" in request:
        return request["id"]
    return fallback


def serve_jsonl(runtime, source: TextIO, sink: BinaryIO) -> None:
    """Read one JSON request per line and emit framed metadata plus raw PCM."""
    write_json_record(
        sink,
        READY,
        {
            "event": "ready",
            "runtime_startup_seconds": round(runtime.startup_seconds, 3),
            "pcm": {
                "format": "s16le",
                "sample_rate": runtime.sample_rate,
                "channels": 1,
                "samples_per_chunk": runtime.samples_per_chunk,
                "bytes_per_chunk": runtime.samples_per_chunk * 2,
            },
        },
    )
    for request_index, line in enumerate(source):
        if not line.strip():
            continue
        request = None
        stream = None
        request_id: object = request_index
        try:
            request = json.loads(line)
            request_id = _request_id(request, request_index)
            if not isinstance(request, dict):
                raise TypeError("request must be a JSON object")
            text = request.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("request.text must be a non-empty string")
            seed = int(request.get("seed", 20260830))
            max_frames = int(request.get("max_frames", 1024))
            if max_frames < 2:
                raise ValueError("request.max_frames must be at least 2")
            write_json_record(
                sink,
                START,
                {
                    "event": "start",
                    "id": request_id,
                    "request_index": runtime.request_count,
                },
            )
            stream = runtime.synthesize_stream(
                text,
                seed=seed,
                max_frames=max_frames,
                allow_unknown_phones=bool(request.get("allow_unknown_phones", False)),
                include_trajectory=bool(request.get("include_trajectory", False)),
            )
            for chunk in stream:
                write_record(sink, PCM, chunk)
            if stream.result is None:
                raise RuntimeError("PCM stream ended without synthesis metrics")
            result = dict(stream.result)
            result["event"] = "done"
            result["id"] = request_id
            write_json_record(sink, DONE, result)
        except Exception as error:
            write_json_record(
                sink,
                ERROR,
                {
                    "event": "error",
                    "id": request_id,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
        finally:
            if stream is not None and not stream.closed:
                stream.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load VoXtream2-RU once, read JSONL requests from stdin and write "
            "length-prefixed PCM records to stdout."
        )
    )
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--ruaccent-assets", type=Path, required=True)
    parser.add_argument("--phone-map", type=Path, required=True)
    parser.add_argument("--espeak-executable", default="espeak-ng")
    parser.add_argument(
        "--text-normalizer",
        choices=("ru-normalizr", "none"),
        default="ru-normalizr",
    )
    parser.add_argument("--temp-engine", type=Path, required=True)
    parser.add_argument("--dep-engine", type=Path, required=True)
    parser.add_argument("--phone-engine", type=Path, required=True)
    parser.add_argument("--mimi-engine", type=Path, required=True)
    parser.add_argument("--mimi-state", type=Path, required=True)
    parser.add_argument("--audio-embedding-weight", type=Path, required=True)
    parser.add_argument("--audio-embedding-cubin", type=Path, required=True)
    parser.add_argument("--cuda-acoustic-control-cubin", type=Path)
    parser.add_argument("--cuda-dep-graph", action="store_true")
    parser.add_argument("--cuda-temp-graph", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TensorRT and CUDA write native diagnostics to file descriptor 1.  Keep
    # those bytes out of the framed protocol even during runtime startup.
    protocol_sink = detach_protocol_stdout()
    try:
        from .runtime import RuntimeFiles, SynthesisRuntime

        files = RuntimeFiles(
            assets=args.assets,
            temp_engine=args.temp_engine,
            dep_engine=args.dep_engine,
            phone_engine=args.phone_engine,
            mimi_engine=args.mimi_engine,
            mimi_state=args.mimi_state,
            audio_embedding_weight=args.audio_embedding_weight,
            audio_embedding_cubin=args.audio_embedding_cubin,
            cuda_acoustic_control_cubin=args.cuda_acoustic_control_cubin,
        )
        with SynthesisRuntime(
            files,
            ruaccent_assets=args.ruaccent_assets,
            phone_map=args.phone_map,
            espeak_executable=args.espeak_executable,
            text_normalizer=args.text_normalizer,
            cuda_temp_graph=args.cuda_temp_graph,
            cuda_dep_graph=args.cuda_dep_graph,
        ) as runtime:
            serve_jsonl(runtime, sys.stdin, protocol_sink)
    finally:
        protocol_sink.close()


if __name__ == "__main__":
    main()
