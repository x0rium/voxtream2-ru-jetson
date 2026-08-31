#!/usr/bin/env python3
"""Forward resident TTS PCM records to a low-latency audio player."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from collections.abc import Iterable
from typing import BinaryIO

from .resident import DONE, ERROR, PCM, READY, START, iter_records


class RawPcmSink:
    def __init__(self, command: list[str]) -> None:
        if not command:
            raise ValueError("PCM player command must not be empty")
        self.command = command
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            bufsize=0,
        )
        if self.process.stdin is None:
            raise RuntimeError("PCM player has no stdin pipe")
        self.stdin: BinaryIO = self.process.stdin
        self.closed = False
        self.bytes_written = 0

    def write(self, payload: bytes) -> None:
        if self.closed:
            raise RuntimeError("PCM sink is closed")
        try:
            written = self.stdin.write(payload)
        except BrokenPipeError as error:
            code = self.process.poll()
            raise RuntimeError(f"PCM player exited early with code {code}") from error
        if written != len(payload):
            raise RuntimeError(f"short PCM write: {written} != {len(payload)}")
        self.bytes_written += written

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.stdin.close()
        code = self.process.wait()
        if code:
            raise RuntimeError(f"PCM player exited with code {code}")


def aplay_command(
    *,
    sample_rate: int,
    channels: int,
    device: str | None,
    buffer_time_us: int,
    period_time_us: int,
) -> list[str]:
    command = [
        "aplay",
        "--quiet",
        "--file-type=raw",
        "--format=S16_LE",
        f"--rate={sample_rate}",
        f"--channels={channels}",
        f"--buffer-time={buffer_time_us}",
        f"--period-time={period_time_us}",
    ]
    if device:
        command.append(f"--device={device}")
    return command


def forward_records(
    records: Iterable[tuple[bytes, bytes]],
    sink,
    *,
    request_id: object,
) -> dict[str, object]:
    started = False
    pcm_records = 0
    for kind, payload in records:
        if kind == START:
            event = json.loads(payload)
            if event.get("id") != request_id:
                raise RuntimeError(f"unexpected resident request id: {event.get('id')!r}")
            started = True
        elif kind == PCM:
            if not started:
                raise RuntimeError("resident emitted PCM before start")
            sink.write(payload)
            pcm_records += 1
        elif kind == DONE:
            result = json.loads(payload)
            if result.get("id") != request_id:
                raise RuntimeError(f"unexpected resident request id: {result.get('id')!r}")
            result["playback_pcm_records"] = pcm_records
            return result
        elif kind == ERROR:
            event = json.loads(payload)
            raise RuntimeError(
                f"resident {event.get('error_type', 'error')}: {event.get('message', '')}"
            )
        elif kind == READY:
            raise RuntimeError("resident emitted a duplicate ready record")
        else:
            raise RuntimeError(f"unknown resident record kind: {kind!r}")
    raise EOFError("resident stopped before done")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play one resident TTS request while PCM is still being generated."
    )
    parser.add_argument("--text", required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--max-frames", type=int, default=1024)
    parser.add_argument("--device", help="ALSA PCM name, for example hw:2,0")
    parser.add_argument("--buffer-time-us", type=int, default=320_000)
    parser.add_argument("--period-time-us", type=int, default=80_000)
    parser.add_argument(
        "--player-command",
        help=(
            "Override aplay with a shell-like argv string. The command must accept "
            "raw mono s16le PCM on stdin."
        ),
    )
    parser.add_argument(
        "resident_command",
        nargs=argparse.REMAINDER,
        help="Command after -- that starts voxtream2_ru_jetson.resident.",
    )
    args = parser.parse_args()
    if args.max_frames < 2:
        parser.error("--max-frames must be at least 2")
    if args.buffer_time_us < args.period_time_us:
        parser.error("--buffer-time-us must be at least --period-time-us")
    if not args.resident_command:
        parser.error("a resident command is required after --")
    if args.resident_command[0] == "--":
        args.resident_command = args.resident_command[1:]
    return args


def main() -> None:
    args = parse_args()
    resident = subprocess.Popen(
        args.resident_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        bufsize=0,
    )
    if resident.stdin is None or resident.stdout is None:
        raise RuntimeError("resident process pipes were not created")
    records = iter_records(resident.stdout)
    try:
        kind, payload = next(records)
        if kind != READY:
            raise RuntimeError(f"resident first record is {kind!r}, expected ready")
        ready = json.loads(payload)
        pcm = ready["pcm"]
        request_id = "playback-0"
        request = {
            "id": request_id,
            "text": args.text,
            "seed": args.seed,
            "max_frames": args.max_frames,
            "include_trajectory": False,
        }
        resident.stdin.write(
            json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
            + b"\n"
        )
        resident.stdin.flush()
        player_command = (
            shlex.split(args.player_command)
            if args.player_command
            else aplay_command(
                sample_rate=int(pcm["sample_rate"]),
                channels=int(pcm["channels"]),
                device=args.device,
                buffer_time_us=args.buffer_time_us,
                period_time_us=args.period_time_us,
            )
        )
        sink = RawPcmSink(player_command)
        try:
            result = forward_records(records, sink, request_id=request_id)
        finally:
            sink.close()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        resident.stdin.close()
        try:
            code = resident.wait(timeout=10)
        except subprocess.TimeoutExpired:
            resident.terminate()
            code = resident.wait(timeout=5)
        if code:
            raise RuntimeError(f"resident process exited with code {code}")


if __name__ == "__main__":
    main()
