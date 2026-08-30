#!/usr/bin/env python3
"""Benchmark the vendored VoXtream2-RU runtime without starting Gradio."""

from __future__ import annotations

import argparse
import contextlib
import functools
import gc
import json
import os
import resource
import sys
import threading
import time
import types
import weakref
from collections import defaultdict
from itertools import repeat
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default_tensorrt_engine = Path(
        os.environ.get(
            "VOXTREAM_TENSORRT_DEP_ENGINE",
            "/data/outputs/tensorrt-explicit-kv/dep-step-explicit-kv-opt1.engine",
        )
    )
    parser.add_argument("text")
    parser.add_argument("--demo-dir", type=Path, default=Path("/data/demo"))
    parser.add_argument("--model-dir", type=Path, default=Path("/data"))
    parser.add_argument("--prompt-audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/data/outputs/voxtream2-ru.wav"))
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--speaking-rate", type=float)
    parser.add_argument("--look-ahead", default="30")
    parser.add_argument("--cache-prompt", action="store_true")
    parser.add_argument("--fixed-prompt-runtime", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--profile-sync",
        action="store_true",
        help="Profile with CUDA synchronizations instead of allocating CUDA events.",
    )
    parser.add_argument(
        "--trim-unused-model-pool-embeddings",
        action="store_true",
        help=(
            "Replace the three temporary 128256-token MODEL_POOL embeddings "
            "with one-token placeholders before Mimi is loaded. VoXtream "
            "discards these embeddings before checkpoint loading."
        ),
    )
    parser.add_argument(
        "--preload-mimi-before-voxtream",
        action="store_true",
        help=(
            "Load the fixed-runtime Mimi decoder before importing VoXtream's "
            "large CPU MODEL_POOL, avoiding fragmented Jetson NvMap allocation."
        ),
    )
    parser.add_argument(
        "--skip-ruaccent-rule-engine",
        action="store_true",
        help=(
            "Do not load RUAccent 1.5.8.3's RuleEngine. Its process_all() path "
            "does not reference the engine; this experimental flag fails loudly "
            "if a future code path tries to use it."
        ),
    )
    parser.add_argument(
        "--restore-ruaccent-rule-engine",
        action="store_true",
        help="Restore the RuleEngine stage removed from RUAccent during its v1.5.8 refactor.",
    )
    parser.add_argument("--cuda-graph-compatible-sdpa", action="store_true")
    parser.add_argument(
        "--async-dep-cache-reset",
        action="store_true",
        help="Reset dep_former KV caches without a CUDA-to-CPU .item() synchronization.",
    )
    parser.add_argument(
        "--cuda-graph-components",
        default="temp,dep,mimi",
        help="Comma-separated graph components: temp, dep, mimi",
    )
    parser.add_argument(
        "--tensorrt-dep-engine",
        type=Path,
        default=default_tensorrt_engine,
        help="Use an explicit-KV TensorRT engine for the 14 one-token dep_former steps.",
    )
    parser.add_argument(
        "--tensorrt-dep-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture the two alternating TensorRT KV-cache layouts in CUDA Graphs.",
    )
    parser.add_argument(
        "--tensorrt-dep-init-engine",
        type=Path,
        help=(
            "q=2 explicit-KV TensorRT engine for the original two-token "
            "dep_former frame init. May be the same dynamic plan as "
            "--tensorrt-dep-engine."
        ),
    )
    parser.add_argument(
        "--tensorrt-dep-init-shadow-engine",
        type=Path,
        help=(
            "Keep the accepted PyTorch q=2 init as output, but run this q=2 "
            "TensorRT engine in parallel on the real hidden states and record "
            "teacher-forced token/logit/state differences."
        ),
    )
    parser.add_argument(
        "--tensorrt-temp-shadow-engine",
        type=Path,
        help=(
            "Keep temp_former in PyTorch, but execute this q=1 TensorRT engine "
            "on the same real hidden states and compare its independent KV trajectory."
        ),
    )
    parser.add_argument(
        "--tensorrt-temp-engine",
        type=Path,
        help=(
            "Replace one-token temp_former decoding with this explicit-KV "
            "TensorRT engine. Multi-token prompt prefill remains in PyTorch."
        ),
    )
    parser.add_argument(
        "--tensorrt-phone-engine",
        type=Path,
        help=(
            "Replace phone_embeddings + phone_former with one dynamic "
            "TensorRT encoder and skip their PyTorch weights."
        ),
    )
    parser.add_argument(
        "--tensorrt-mimi-engine",
        type=Path,
        help=(
            "Replace the streaming Mimi decoder with a state-explicit "
            "TensorRT engine and unload the upstream Mimi weights."
        ),
    )
    parser.add_argument(
        "--tensorrt-mimi-state",
        type=Path,
        help=(
            "Framework-neutral JSON manifest for the initial Mimi decoder "
            "streaming state; skips loading the PyTorch Mimi checkpoint."
        ),
    )
    parser.add_argument(
        "--cuda-audio-embedding-weight",
        type=Path,
        help=(
            "Raw contiguous BF16 audio_embeddings.weight used by the "
            "PyTorch-free CUDA lookup kernel."
        ),
    )
    parser.add_argument(
        "--cuda-audio-embedding-cubin",
        type=Path,
        help="Compiled gather_bf16_words CUDA kernel for audio embeddings.",
    )
    parser.add_argument(
        "--tensorrt-temp-prefill-cache",
        type=Path,
        help=(
            "Replay a captured fixed-prompt q>1 output/KV state, allowing the "
            "PyTorch temp_former weights to be unloaded."
        ),
    )
    parser.add_argument(
        "--temp-shadow-capture",
        type=Path,
        help=(
            "Capture one contiguous real temp_former q=1 trajectory for a "
            "separate low-memory TensorRT replay process."
        ),
    )
    parser.add_argument(
        "--temp-prefill-capture",
        type=Path,
        help=(
            "Capture the real multi-token fixed-prompt temp_former call, "
            "including its exact output and post-prefill KV state."
        ),
    )
    parser.add_argument(
        "--pytorch-dep-former",
        action="store_true",
        help="Use the original PyTorch dep_former as a fallback/control baseline.",
    )
    parser.add_argument(
        "--tensorrt-dep-full-runtime",
        action="store_true",
        help=(
            "Run both dep_former init and autoregressive steps in TensorRT, "
            "then unload the original PyTorch dep_former weights."
        ),
    )
    parser.add_argument(
        "--runorm-cpu",
        action="store_true",
        help=(
            "Keep RUNorm's lazily created NER pipeline on CPU. RUNorm's public "
            "load(device='cpu') default does not forward the device to that pipeline."
        ),
    )
    args = parser.parse_args()
    if args.pytorch_dep_former:
        args.tensorrt_dep_engine = None
        args.tensorrt_dep_cuda_graph = False
    if args.tensorrt_dep_full_runtime and args.tensorrt_dep_engine is None:
        parser.error("--tensorrt-dep-full-runtime requires a TensorRT dep engine")
    if args.tensorrt_dep_full_runtime and args.tensorrt_dep_init_engine is None:
        parser.error(
            "--tensorrt-dep-full-runtime requires --tensorrt-dep-init-engine; "
            "the sequential q=1 init failed the listening A/B"
        )
    if args.tensorrt_dep_init_engine is not None and not args.tensorrt_dep_full_runtime:
        parser.error(
            "--tensorrt-dep-init-engine requires --tensorrt-dep-full-runtime"
        )
    if (
        args.tensorrt_dep_init_shadow_engine is not None
        and args.tensorrt_dep_engine is None
    ):
        parser.error(
            "--tensorrt-dep-init-shadow-engine requires a TensorRT q=1 engine"
        )
    if (
        args.tensorrt_dep_init_shadow_engine is not None
        and args.tensorrt_dep_full_runtime
    ):
        parser.error(
            "--tensorrt-dep-init-shadow-engine and "
            "--tensorrt-dep-full-runtime are mutually exclusive"
        )
    temp_modes = sum(
        value is not None
        for value in (
            args.tensorrt_temp_engine,
            args.tensorrt_temp_shadow_engine,
            args.temp_shadow_capture,
        )
    )
    if temp_modes > 1:
        parser.error(
            "--tensorrt-temp-engine, --tensorrt-temp-shadow-engine, and "
            "--temp-shadow-capture are mutually exclusive"
        )
    if (
        args.tensorrt_temp_prefill_cache is not None
        and args.tensorrt_temp_engine is None
    ):
        parser.error("--tensorrt-temp-prefill-cache requires --tensorrt-temp-engine")
    if (
        args.tensorrt_temp_prefill_cache is not None
        and not args.fixed_prompt_runtime
    ):
        parser.error(
            "--tensorrt-temp-prefill-cache requires --fixed-prompt-runtime"
        )
    if (args.cuda_audio_embedding_weight is None) != (
        args.cuda_audio_embedding_cubin is None
    ):
        parser.error(
            "--cuda-audio-embedding-weight and "
            "--cuda-audio-embedding-cubin must be used together"
        )
    if args.tensorrt_mimi_state is not None and args.tensorrt_mimi_engine is None:
        parser.error("--tensorrt-mimi-state requires --tensorrt-mimi-engine")
    if args.tensorrt_mimi_state is not None and not args.fixed_prompt_runtime:
        parser.error("--tensorrt-mimi-state requires --fixed-prompt-runtime")
    return args


ARGS = parse_args()
PROGRAM_STARTED = time.perf_counter()
PRELOADED_MIMI = None
PRELOADED_TRT_DEP = None
PRELOADED_TRT_DEP_INIT = None
PRELOADED_TRT_TEMP = None
PRELOADED_TRT_PHONE = None
PRELOADED_TRT_MIMI = None


def _read_kib_fields(path: Path) -> dict[str, int]:
    fields: dict[str, int] = {}
    try:
        for line in path.read_text().splitlines():
            try:
                key, value = line.split(":", 1)
                parts = value.split()
                if parts:
                    fields[key] = int(parts[0])
            except ValueError:
                continue
    except (FileNotFoundError, PermissionError):
        pass
    return fields


def memory_snapshot() -> dict[str, float]:
    process = _read_kib_fields(Path("/proc/self/status"))
    system = _read_kib_fields(Path("/proc/meminfo"))
    return {
        "process_rss_mib": round(process.get("VmRSS", 0) / 1024, 1),
        "process_swap_mib": round(process.get("VmSwap", 0) / 1024, 1),
        "system_available_mib": round(system.get("MemAvailable", 0) / 1024, 1),
        "system_swap_used_mib": round(
            (system.get("SwapTotal", 0) - system.get("SwapFree", 0)) / 1024, 1
        ),
    }


def discard_module_tensors(module: torch.nn.Module) -> tuple[float, float]:
    """Invalidate an obsolete backend and release all of its tensor storage."""
    parameter_bytes = 0
    buffer_bytes = 0
    for child in module.modules():
        for name, parameter in tuple(child._parameters.items()):
            if parameter is not None:
                parameter_bytes += parameter.numel() * parameter.element_size()
                child._parameters[name] = None
        for name, buffer in tuple(child._buffers.items()):
            if buffer is not None:
                buffer_bytes += buffer.numel() * buffer.element_size()
                child._buffers[name] = None
    return parameter_bytes / 1024**2, buffer_bytes / 1024**2


def discard_module_parameters(module: torch.nn.Module) -> float:
    """Release weights while preserving structural buffers needed by setup_caches()."""
    parameter_bytes = 0
    for child in module.modules():
        for name, parameter in tuple(child._parameters.items()):
            if parameter is not None:
                parameter_bytes += parameter.numel() * parameter.element_size()
                child._parameters[name] = None
    return parameter_bytes / 1024**2


class ResourceSampler:
    """Sample Jetson-wide memory, GPU load/frequency and power without dependencies."""

    def __init__(self, interval_seconds: float = 0.1) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._samples: list[dict[str, float]] = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._gpu_load = Path("/sys/devices/platform/bus@0/17000000.gpu/load")
        self._gpu_freq = Path(
            "/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq"
        )
        self._ina = self._find_ina3221()

    @staticmethod
    def _find_ina3221() -> Path | None:
        for candidate in Path("/sys/class/hwmon").glob("hwmon*"):
            try:
                if (candidate / "name").read_text().strip() == "ina3221":
                    return candidate
            except (FileNotFoundError, PermissionError):
                continue
        return None

    @staticmethod
    def _read_number(path: Path, scale: float = 1.0) -> float | None:
        try:
            return float(path.read_text().strip()) / scale
        except (FileNotFoundError, PermissionError, ValueError):
            return None

    def _power_watts(self, channel: int) -> float | None:
        if self._ina is None:
            return None
        millivolts = self._read_number(self._ina / f"in{channel}_input")
        milliamps = self._read_number(self._ina / f"curr{channel}_input")
        if millivolts is None or milliamps is None:
            return None
        return millivolts * milliamps / 1_000_000

    def _sample(self) -> dict[str, float]:
        sample = {
            "timestamp": time.perf_counter(),
            "process_cpu_seconds": time.process_time(),
            **memory_snapshot(),
        }
        gpu_load = self._read_number(self._gpu_load, scale=10.0)
        gpu_freq = self._read_number(self._gpu_freq, scale=1_000_000.0)
        if gpu_load is not None:
            sample["gpu_load_pct"] = gpu_load
        if gpu_freq is not None:
            sample["gpu_freq_mhz"] = gpu_freq
        for name, channel in (("vdd_in_w", 1), ("cpu_gpu_cv_w", 2), ("soc_w", 3)):
            value = self._power_watts(channel)
            if value is not None:
                sample[name] = value
        return sample

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self._sample()
            with self._lock:
                self._samples.append(sample)
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._thread.start()

    def summary(self, reset: bool = False) -> dict[str, float | int]:
        with self._lock:
            samples = list(self._samples)
            if reset:
                self._samples.clear()
        if not samples:
            return {"samples": 0}

        def values(name: str) -> list[float]:
            return [sample[name] for sample in samples if name in sample]

        result: dict[str, float | int] = {"samples": len(samples)}
        duration = samples[-1]["timestamp"] - samples[0]["timestamp"]
        cpu_seconds = (
            samples[-1]["process_cpu_seconds"] - samples[0]["process_cpu_seconds"]
        )
        result["sampled_seconds"] = round(duration, 3)
        result["average_cpu_cores"] = round(cpu_seconds / duration, 3) if duration else 0.0
        reducers: tuple[tuple[str, str, Callable[[list[float]], float]], ...] = (
            ("peak_process_rss_mib", "process_rss_mib", max),
            ("peak_process_swap_mib", "process_swap_mib", max),
            ("minimum_system_available_mib", "system_available_mib", min),
            ("peak_system_swap_used_mib", "system_swap_used_mib", max),
            ("average_gpu_load_pct", "gpu_load_pct", lambda xs: sum(xs) / len(xs)),
            ("peak_gpu_load_pct", "gpu_load_pct", max),
            ("average_gpu_freq_mhz", "gpu_freq_mhz", lambda xs: sum(xs) / len(xs)),
            ("peak_gpu_freq_mhz", "gpu_freq_mhz", max),
            ("average_vdd_in_w", "vdd_in_w", lambda xs: sum(xs) / len(xs)),
            ("peak_vdd_in_w", "vdd_in_w", max),
            ("average_cpu_gpu_cv_w", "cpu_gpu_cv_w", lambda xs: sum(xs) / len(xs)),
            ("peak_cpu_gpu_cv_w", "cpu_gpu_cv_w", max),
            ("average_soc_w", "soc_w", lambda xs: sum(xs) / len(xs)),
            ("peak_soc_w", "soc_w", max),
        )
        for output_name, input_name, reducer in reducers:
            data = values(input_name)
            if data:
                result[output_name] = round(reducer(data), 3)
        return result

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


class CallProfiler:
    """Collect inclusive wall time and sampled CUDA time for selected call sites."""

    def __init__(
        self,
        max_cuda_samples: int = 64,
        synchronize_cuda: bool = False,
    ) -> None:
        self.max_cuda_samples = max_cuda_samples
        self.synchronize_cuda = synchronize_cuda
        self.reset()

    def reset(self) -> None:
        self._stats: dict[str, dict[str, object]] = defaultdict(
            lambda: {"calls": 0, "wall_seconds": 0.0, "events": []}
        )

    def wrap(self, name: str, function: Callable) -> Callable:
        @functools.wraps(function)
        def measured(*args, **kwargs):
            stats = self._stats[name]
            stats["calls"] = int(stats["calls"]) + 1
            events = stats["events"]
            record_cuda = (
                torch.cuda.is_available()
                and not self.synchronize_cuda
                and len(events) < self.max_cuda_samples
                and not torch.cuda.is_current_stream_capturing()
            )
            start_event = end_event = None
            if record_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            synchronize_call = (
                self.synchronize_cuda
                and torch.cuda.is_available()
                and not torch.cuda.is_current_stream_capturing()
            )
            if synchronize_call:
                torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                if synchronize_call:
                    torch.cuda.synchronize()
                stats["wall_seconds"] = float(stats["wall_seconds"]) + (
                    time.perf_counter() - started
                )
                if start_event is not None and end_event is not None:
                    end_event.record()
                    events.append((start_event, end_event))

        return measured

    def summary(self) -> dict[str, dict[str, float | int]]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result: dict[str, dict[str, float | int]] = {}
        for name, raw in sorted(self._stats.items()):
            calls = int(raw["calls"])
            wall_seconds = float(raw["wall_seconds"])
            events = raw["events"]
            cuda_seconds = sum(start.elapsed_time(end) for start, end in events) / 1000
            cuda_samples = len(events)
            result[name] = {
                "calls": calls,
                "wall_seconds": round(wall_seconds, 4),
                "wall_ms_per_call": round(1000 * wall_seconds / calls, 3) if calls else 0.0,
                "cuda_sampled_calls": cuda_samples,
                "cuda_sampled_seconds": round(cuda_seconds, 4),
                "cuda_estimated_total_seconds": round(
                    cuda_seconds * calls / cuda_samples, 4
                ) if cuda_samples else 0.0,
            }
        return result


class NoOpProfiler:
    def reset(self) -> None:
        pass

    @staticmethod
    def wrap(name: str, function: Callable) -> Callable:
        return function

    @staticmethod
    def summary() -> dict:
        return {}


class DepFormerShadowProbe:
    """Compare rejected TensorRT q=2 against the accepted path on real inputs."""

    def __init__(
        self,
        model: torch.nn.Module,
        reference_runtime,
        shadow_runtime,
        temperature: float,
        cfg_gamma: float,
    ) -> None:
        self.model = model
        self.reference_runtime = reference_runtime
        self.shadow_runtime = shadow_runtime
        self.temperature = temperature
        self.cfg_gamma = cfg_gamma
        self.records: list[dict[str, object]] = []
        self.frame_index = -1
        self.head_index = 0

    def begin_frame(self) -> None:
        self.frame_index += 1
        self.head_index = 0

    def reference_state(self, init: bool) -> tuple[torch.Tensor, ...]:
        if init:
            buffers = dict(self.model.dep_former.named_buffers())
            return tuple(
                buffers[name] for name in self.shadow_runtime.buffer_names
            )
        return self.reference_runtime.state

    def record(
        self,
        reference_output: torch.Tensor,
        candidate_output: torch.Tensor,
        *,
        init: bool,
    ) -> None:
        head_index = self.head_index
        head = self.model.audio_head[head_index]
        reference_logits = torch.mm(
            reference_output[:, -1, :].to(torch.bfloat16), head
        )
        candidate_logits = torch.mm(
            candidate_output[:, -1, :].to(torch.bfloat16), head
        )
        reference_cfg = (
            self.cfg_gamma * reference_logits[0].float()
            + (1.0 - self.cfg_gamma) * reference_logits[1].float()
        )
        candidate_cfg = (
            self.cfg_gamma * candidate_logits[0].float()
            + (1.0 - self.cfg_gamma) * candidate_logits[1].float()
        )
        reference_top2 = torch.topk(reference_cfg, 2)
        candidate_top2 = torch.topk(candidate_cfg, 2)
        reference_probs = torch.softmax(reference_cfg / self.temperature, dim=-1)
        candidate_probs = torch.softmax(candidate_cfg / self.temperature, dim=-1)
        midpoint = 0.5 * (reference_probs + candidate_probs)
        epsilon = torch.finfo(reference_probs.dtype).tiny
        reference_probs_safe = reference_probs.clamp_min(epsilon)
        candidate_probs_safe = candidate_probs.clamp_min(epsilon)
        midpoint_safe = midpoint.clamp_min(epsilon)
        js_divergence = 0.5 * (
            torch.sum(
                reference_probs_safe
                * torch.log(reference_probs_safe / midpoint_safe)
            )
            + torch.sum(
                candidate_probs_safe
                * torch.log(candidate_probs_safe / midpoint_safe)
            )
        )
        state_delta = torch.stack(
            [
                (reference.float() - candidate.float()).abs().max()
                for reference, candidate in zip(
                    self.reference_state(init), self.shadow_runtime.state
                )
            ]
        ).max()
        reference_token = int(reference_top2.indices[0].item())
        candidate_token = int(candidate_top2.indices[0].item())
        self.records.append(
            {
                "frame": self.frame_index,
                "head": head_index,
                "token_equal": reference_token == candidate_token,
                "reference_token": reference_token,
                "candidate_token": candidate_token,
                "reference_margin": float(
                    (reference_top2.values[0] - reference_top2.values[1]).item()
                ),
                "candidate_margin": float(
                    (candidate_top2.values[0] - candidate_top2.values[1]).item()
                ),
                "hidden_max_abs": float(
                    (reference_output.float() - candidate_output.float())
                    .abs()
                    .max()
                    .item()
                ),
                "cfg_logits_max_abs": float(
                    (reference_cfg - candidate_cfg).abs().max().item()
                ),
                "js_divergence": float(js_divergence.item()),
                "state_max_abs": float(state_delta.item()),
            }
        )
        self.head_index += 1

    def summary(self) -> dict[str, object]:
        by_head: dict[int, list[dict[str, object]]] = defaultdict(list)
        for record in self.records:
            by_head[int(record["head"])].append(record)

        def summarize(records: list[dict[str, object]]) -> dict[str, object]:
            count = len(records)
            mismatches = sum(not bool(record["token_equal"]) for record in records)
            return {
                "count": count,
                "token_mismatches": mismatches,
                "token_mismatch_rate": round(mismatches / count, 6) if count else 0.0,
                "max_hidden_abs": max(
                    (float(record["hidden_max_abs"]) for record in records),
                    default=0.0,
                ),
                "max_cfg_logits_abs": max(
                    (float(record["cfg_logits_max_abs"]) for record in records),
                    default=0.0,
                ),
                "max_js_divergence": max(
                    (float(record["js_divergence"]) for record in records),
                    default=0.0,
                ),
                "max_state_abs": max(
                    (float(record["state_max_abs"]) for record in records),
                    default=0.0,
                ),
            }

        mismatches = [
            record for record in self.records if not bool(record["token_equal"])
        ]
        mismatch_frames = {int(record["frame"]) for record in mismatches}
        return {
            "mode": "teacher_forced_real_hidden",
            "frames": self.frame_index + 1,
            "overall": summarize(self.records),
            "frames_with_any_token_mismatch": len(mismatch_frames),
            "frame_mismatch_rate": (
                round(len(mismatch_frames) / (self.frame_index + 1), 6)
                if self.frame_index >= 0
                else 0.0
            ),
            "by_head": {
                str(head): summarize(records)
                for head, records in sorted(by_head.items())
            },
            "first_mismatches": mismatches[:100],
            "runtime": self.shadow_runtime.metrics(),
        }


class TempFormerShadowProbe:
    """Shadow the accepted temp_former on real production inputs."""

    def __init__(self, model, runner, buffer_names, config, sampler) -> None:
        self.model = model
        self.runner = runner
        self.buffer_names = buffer_names
        self.config = config
        self.sampler = sampler
        self.records: list[dict[str, object]] = []
        self.last_position: int | None = None
        self.segment = -1

    def _reference_state(self) -> tuple[torch.Tensor, ...]:
        buffers = dict(self.model.temp_former.named_buffers())
        return tuple(buffers[name] for name in self.buffer_names)

    def _sync_segment(self) -> None:
        self.runner.copy_state_from(self._reference_state())
        self.segment += 1

    def _sample(self, logits: torch.Tensor):
        return self.sampler(
            config=self.config,
            logits=logits,
            num_states=self.model.config.num_phone_states,
            codebook_size=(
                self.model.config.audio_vocab_size + self.model.config.audio_pad_size
            ),
            cfg_gamma=self.config.cfg_gamma,
        )

    def run(self, accepted, hidden, input_pos, mask):
        position = int(input_pos[0, -1].item())
        if self.last_position is None or position != self.last_position + 1:
            self._sync_segment()

        reference_output = accepted(hidden, input_pos, mask)
        candidate_output, candidate_state = self.runner.step(hidden, input_pos, mask)

        reference_logits = self.model.sem_head(
            reference_output[:, -1, :].to(torch.bfloat16)
        )
        candidate_logits = self.model.sem_head(
            candidate_output[:, -1, :].to(torch.bfloat16)
        )

        sampling_rng = torch.cuda.get_rng_state(device=hidden.device)
        reference_token, reference_state_token, _ = self._sample(reference_logits)
        torch.cuda.set_rng_state(sampling_rng, device=hidden.device)
        candidate_token, candidate_state_token, _ = self._sample(candidate_logits)
        # The shadow must not perturb the production sampler that runs immediately
        # after this wrapper returns to generate_frame().
        torch.cuda.set_rng_state(sampling_rng, device=hidden.device)

        num_states = self.model.config.num_phone_states
        codebook_size = (
            self.model.config.audio_vocab_size + self.model.config.audio_pad_size
        )
        reference_conditional = reference_logits[:1].view(
            -1, num_states, codebook_size
        )
        candidate_conditional = candidate_logits[:1].view(
            -1, num_states, codebook_size
        )
        reference_state_probs = torch.softmax(
            torch.logsumexp(reference_conditional, dim=-1)
            / self.config.temperature,
            dim=-1,
        )
        candidate_state_probs = torch.softmax(
            torch.logsumexp(candidate_conditional, dim=-1)
            / self.config.temperature,
            dim=-1,
        )
        state_probability_delta = (
            reference_state_probs.float() - candidate_state_probs.float()
        ).abs()

        reference_buffers = self._reference_state()
        state_max_abs = max(
            float((reference.float() - candidate.float()).abs().max().item())
            for reference, candidate in zip(reference_buffers, candidate_state)
        )
        hidden_delta = reference_output.float() - candidate_output.float()
        self.records.append(
            {
                "step": len(self.records),
                "segment": self.segment,
                "position": position,
                "reference_state": int(reference_state_token.item()),
                "candidate_state": int(candidate_state_token.item()),
                "state_equal": bool(
                    torch.equal(reference_state_token, candidate_state_token)
                ),
                "reference_token": int(reference_token.item()),
                "candidate_token": int(candidate_token.item()),
                "token_equal": bool(torch.equal(reference_token, candidate_token)),
                "hidden_max_abs": float(hidden_delta.abs().max().item()),
                "hidden_mean_abs": float(hidden_delta.abs().mean().item()),
                "state_max_abs": state_max_abs,
                "state_probs_max_abs": float(state_probability_delta.max().item()),
                "state_probs_total_variation": float(
                    0.5 * state_probability_delta.sum().item()
                ),
            }
        )
        self.last_position = position
        return reference_output

    def summary(self) -> dict[str, object]:
        count = len(self.records)
        state_matches = sum(bool(record["state_equal"]) for record in self.records)
        token_matches = sum(bool(record["token_equal"]) for record in self.records)
        return {
            "mode": "teacher_forced_real_hidden_independent_kv",
            "steps": count,
            "segments": self.segment + 1,
            "state_matches": state_matches,
            "token_matches": token_matches,
            "state_match_rate": round(state_matches / count, 6) if count else 0.0,
            "token_match_rate": round(token_matches / count, 6) if count else 0.0,
            "max_hidden_abs": max(
                (float(record["hidden_max_abs"]) for record in self.records),
                default=0.0,
            ),
            "max_state_abs": max(
                (float(record["state_max_abs"]) for record in self.records),
                default=0.0,
            ),
            "max_state_probs_total_variation": max(
                (
                    float(record["state_probs_total_variation"])
                    for record in self.records
                ),
                default=0.0,
            ),
            "first_mismatches": [
                record
                for record in self.records
                if not bool(record["state_equal"]) or not bool(record["token_equal"])
            ][:100],
            "records": self.records,
        }


class TempFormerTensorRTRuntime:
    """Use TensorRT for q=1 while PyTorch retains multi-token prompt prefill."""

    def __init__(self, model, runner, buffer_names) -> None:
        self.model = model
        self.runner = runner
        self.buffer_names = buffer_names
        self.last_position: int | None = None
        self.segments = 0
        self.calls = 0
        self.prefill_mode = "pytorch"
        self.last_semantic_logits: torch.Tensor | None = None

    def _pytorch_prefill_state(self) -> tuple[torch.Tensor, ...]:
        buffers = dict(self.model.temp_former.named_buffers())
        if all(name in buffers for name in self.buffer_names):
            return tuple(buffers[name] for name in self.buffer_names)
        # A fixed-prompt facade has already restored the TensorRT-owned state.
        return self.runner.state

    def run(self, hidden, input_pos, mask):
        position = int(input_pos[0, -1].item())
        if self.last_position is None or position != self.last_position + 1:
            # Prompt ingestion (q>1) still runs through the accepted PyTorch
            # path and owns the canonical prefix cache. Import it once at each
            # transition to one-token streaming.
            self.runner.copy_state_from(self._pytorch_prefill_state())
            self.segments += 1
        output, _ = self.runner.step(hidden, input_pos, mask)
        fused_logits = self.runner.extra_outputs.get("semantic_logits")
        self.last_semantic_logits = (
            fused_logits[:, -1, :] if fused_logits is not None else None
        )
        self.last_position = position
        self.calls += 1
        return output

    def metrics(self) -> dict[str, object]:
        return {
            "mode": "tensorrt_q1_pytorch_prefill",
            "engine": str(self.runner.engine_path),
            "engine_bytes": self.runner.engine_path.stat().st_size,
            "calls": self.calls,
            "segments": self.segments,
            "prefill_mode": self.prefill_mode,
            "pytorch_prefill_retained": self.prefill_mode == "pytorch",
            "fused_semantic_head": "semantic_logits" in self.runner.extra_outputs,
        }


class FixedPromptTempFormerFacade(torch.nn.Module):
    """Replay one verified prompt prefill while q=1 continues in TensorRT."""

    def __init__(
        self,
        runtime: TempFormerTensorRTRuntime,
        cache_payload: dict[str, object],
        device: torch.device,
        cached_semantic_logits: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if cache_payload.get("format") != "voxtream-temp-fixed-prompt-prefill-v1":
            raise RuntimeError(
                f"unsupported temp prefill cache: {cache_payload.get('format')!r}"
            )
        if tuple(cache_payload["buffer_names"]) != tuple(runtime.buffer_names):
            raise RuntimeError("temp prefill cache buffer ABI does not match engine")
        self.runtime = runtime
        self.cached_state = tuple(cache_payload["final_state"])
        self.cached_output = cache_payload["output"].to(device=device)
        self.cached_semantic_logits = cached_semantic_logits
        self.expected_hidden_shape = tuple(cache_payload["hidden"].shape)
        self.expected_input_pos = cache_payload["input_pos"]
        self.expected_mask_shape = tuple(cache_payload["mask"].shape)
        self.prefill_calls = 0

    def reset_caches(self) -> None:
        # forward(q>1) restores the complete post-prompt cache before any q=1
        # call. Avoid a redundant 192 MiB clear immediately before that copy.
        self.runtime.last_position = None
        self.runtime.last_semantic_logits = None

    def caches_are_enabled(self) -> bool:
        return True

    def forward(
        self,
        hidden: torch.Tensor,
        input_pos: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.shape[1] == 1:
            raise RuntimeError("q=1 temp_former must use the TensorRT runtime")
        if tuple(hidden.shape) != self.expected_hidden_shape:
            raise RuntimeError(
                "fixed-prompt temp hidden shape changed: "
                f"expected {self.expected_hidden_shape}, got {tuple(hidden.shape)}"
            )
        if tuple(mask.shape) != self.expected_mask_shape:
            raise RuntimeError(
                "fixed-prompt temp mask shape changed: "
                f"expected {self.expected_mask_shape}, got {tuple(mask.shape)}"
            )
        if not torch.equal(input_pos.cpu(), self.expected_input_pos):
            raise RuntimeError("fixed-prompt temp input positions changed")
        for source, target in zip(self.cached_state, self.runtime.runner.state):
            target.copy_(source)
        self.runtime.last_semantic_logits = self.cached_semantic_logits
        self.prefill_calls += 1
        return self.cached_output


class TensorRTSemanticHeadFacade(torch.nn.Module):
    """Return semantic logits fused into the immediately preceding temp enqueue."""

    def __init__(
        self,
        runtime: TempFormerTensorRTRuntime,
        fallback: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.runtime = runtime
        self.fallback = fallback

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.runtime.last_semantic_logits
        if logits is not None:
            if logits.shape[0] != hidden.shape[0]:
                raise RuntimeError(
                    "fused semantic logits batch does not match temp output"
                )
            return logits
        if self.fallback is not None:
            return self.fallback(hidden)
        raise RuntimeError("semantic head called before temp TensorRT/prefill replay")


class TempFormerPrefillCapture:
    """Capture the fixed-prompt q>1 temp call for offline cache generation."""

    def __init__(self, temp_former, output_path: Path) -> None:
        from voxtream_tensorrt_explicit_kv_probe import kv_buffer_names

        self.temp_former = temp_former
        self.output_path = Path(output_path)
        self.buffer_names = kv_buffer_names(temp_former)
        self.result: dict[str, object] | None = None

    @staticmethod
    def _cpu_clone(value: torch.Tensor) -> torch.Tensor:
        return value.detach().to(device="cpu", copy=True)

    def _state(self) -> tuple[torch.Tensor, ...]:
        buffers = dict(self.temp_former.named_buffers())
        return tuple(buffers[name] for name in self.buffer_names)

    def run(self, accepted, hidden, input_pos, mask):
        if hidden.shape[1] == 1 or self.result is not None:
            return accepted(hidden, input_pos=input_pos, mask=mask)
        output = accepted(hidden, input_pos=input_pos, mask=mask)
        payload = {
            "format": "voxtream-temp-fixed-prompt-prefill-v1",
            "buffer_names": self.buffer_names,
            "hidden": self._cpu_clone(hidden),
            "input_pos": self._cpu_clone(input_pos),
            "mask": self._cpu_clone(mask),
            "output": self._cpu_clone(output),
            "final_state": tuple(
                self._cpu_clone(value) for value in self._state()
            ),
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, self.output_path)
        self.result = {
            "path": str(self.output_path),
            "bytes": self.output_path.stat().st_size,
            "sequence_length": int(hidden.shape[1]),
            "hidden_shape": list(hidden.shape),
            "input_pos_shape": list(input_pos.shape),
            "mask_shape": list(mask.shape),
        }
        return output


class TempFormerCapture:
    """Capture a real, contiguous q=1 trajectory without a second GPU backend."""

    def __init__(self, model, output_path: Path) -> None:
        from voxtream_tensorrt_explicit_kv_probe import kv_buffer_names

        self.model = model
        self.output_path = Path(output_path)
        self.buffer_names = kv_buffer_names(model.temp_former)
        if len(self.buffer_names) != 36:
            raise RuntimeError(
                f"expected 36 temp_former KV buffers, got {len(self.buffer_names)}"
            )
        self.initial_state: tuple[torch.Tensor, ...] | None = None
        self.records: list[dict[str, object]] = []
        self.last_position: int | None = None

    @staticmethod
    def _cpu_clone(value: torch.Tensor) -> torch.Tensor:
        return value.detach().to(device="cpu", copy=True)

    def _state(self) -> tuple[torch.Tensor, ...]:
        buffers = dict(self.model.temp_former.named_buffers())
        return tuple(buffers[name] for name in self.buffer_names)

    def run(self, accepted, hidden, input_pos, mask):
        position = int(input_pos[0, -1].item())
        if self.last_position is not None and position != self.last_position + 1:
            raise RuntimeError(
                "temp shadow capture currently supports one contiguous segment; "
                f"position jumped from {self.last_position} to {position}"
            )
        if self.initial_state is None:
            self.initial_state = tuple(
                self._cpu_clone(value) for value in self._state()
            )

        reference_output = accepted(hidden, input_pos, mask)
        self.records.append(
            {
                "position": position,
                "hidden": self._cpu_clone(hidden),
                "input_pos": self._cpu_clone(input_pos),
                "mask": self._cpu_clone(mask),
                "reference_output": self._cpu_clone(reference_output),
                # generate_frame invokes sem_head and the sampler immediately
                # after this wrapper returns. Neither operation before sampling
                # consumes CUDA RNG.
                "sampling_rng_state": torch.cuda.get_rng_state(
                    device=hidden.device
                ).cpu(),
            }
        )
        self.last_position = position
        return reference_output

    def attach_actual_sample(
        self,
        previous_record_count: int,
        frame: torch.Tensor,
        predicted_state: torch.Tensor,
    ) -> None:
        if len(self.records) == previous_record_count:
            return
        if len(self.records) != previous_record_count + 1:
            raise RuntimeError("one generate_frame call captured multiple temp steps")
        record = self.records[-1]
        record["actual_semantic_token"] = int(frame[0, 0].item())
        record["actual_state_token"] = int(predicted_state.flatten()[0].item())

    def save(self) -> dict[str, object]:
        if self.initial_state is None or not self.records:
            raise RuntimeError("temp shadow capture did not observe any q=1 steps")
        final_state = tuple(self._cpu_clone(value) for value in self._state())
        payload = {
            "format": "voxtream-temp-real-trajectory-v1",
            "buffer_names": self.buffer_names,
            "initial_state": self.initial_state,
            "final_state": final_state,
            "records": self.records,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, self.output_path)
        return {
            "path": str(self.output_path),
            "bytes": self.output_path.stat().st_size,
            "steps": len(self.records),
            "first_position": int(self.records[0]["position"]),
            "last_position": int(self.records[-1]["position"]),
        }


def report_stage(stage: str, **details: object) -> None:
    """Emit progress immediately; full VoXtream startup can take minutes on Jetson."""
    payload: dict[str, object] = {
        "stage": stage,
        "elapsed_process_seconds": round(time.perf_counter() - PROGRAM_STARTED, 3),
        **memory_snapshot(),
        **details,
    }
    if torch.cuda.is_available():
        payload.update(
            cuda_allocated_mib=round(torch.cuda.memory_allocated() / 1024**2, 1),
            cuda_reserved_mib=round(torch.cuda.memory_reserved() / 1024**2, 1),
        )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def benchmark_main() -> None:
    global PRELOADED_MIMI, PRELOADED_TRT_DEP, PRELOADED_TRT_DEP_INIT, PRELOADED_TRT_TEMP, PRELOADED_TRT_PHONE, PRELOADED_TRT_MIMI
    from dataclasses import fields

    from safetensors import safe_open
    from voxtream import generator as generator_module
    from voxtream.config import SpeechGeneratorConfig
    from voxtream.generator import SpeechGenerator
    from voxtream.model import Model, ModelConfig
    from voxtream.utils.generator import set_seed
    from voxtream.utils.generator import setup as generator_setup
    from voxtream.utils.model import MODEL_POOL

    def engine_io_names(preloaded) -> set[str]:
        if preloaded is None:
            return set()
        return {
            preloaded.engine.get_tensor_name(index)
            for index in range(preloaded.engine.num_io_tensors)
        }

    temp_engine_io = engine_io_names(PRELOADED_TRT_TEMP)
    dep_engine_io = engine_io_names(PRELOADED_TRT_DEP)
    skip_temp_weights = bool(
        ARGS.fixed_prompt_runtime
        and ARGS.tensorrt_temp_engine is not None
        and ARGS.tensorrt_temp_prefill_cache is not None
    )
    skip_dep_weights = bool(ARGS.tensorrt_dep_full_runtime)
    skip_semantic_head = bool(
        skip_temp_weights and "semantic_logits" in temp_engine_io
    )
    skip_acoustic_head = bool(
        skip_dep_weights and "acoustic_logits" in dep_engine_io
    )
    skip_phone_weights = ARGS.tensorrt_phone_engine is not None
    skip_audio_embedding_weights = (
        ARGS.cuda_audio_embedding_weight is not None
    )

    if ARGS.trim_unused_model_pool_embeddings:
        trimmed_embedding_mib = 0.0
        for pooled_model in MODEL_POOL.values():
            embedding = pooled_model.tok_embeddings
            if not isinstance(embedding, torch.nn.Embedding):
                continue
            trimmed_embedding_mib += (
                embedding.weight.numel() * embedding.weight.element_size() / 1024**2
            )
            pooled_model.tok_embeddings = torch.nn.Embedding(1, embedding.embedding_dim)
            pooled_model.output = torch.nn.Identity()
        gc.collect()
    else:
        trimmed_embedding_mib = 0.0

    def load_generator_model_low_memory(config, device, dtype, batch_size):
        """Avoid holding model + state dict + CUDA copy at the same time.

        Jetson has unified memory.  Upstream load_state_dict() copies every FP32
        tensor into an already allocated FP32 model, then allocates the BF16 CUDA
        model while both CPU copies are still live.  assign=True lets the model
        adopt the safetensors storage, so it can be released before the CUDA move.
        """
        model_config_path = generator_setup.hf_hub_download(
            config.model_repo, config.model_config_name, token=config.hf_token
        )
        phoneme_dict_path = generator_setup.hf_hub_download(
            config.model_repo, config.phoneme_dict_name, token=config.hf_token
        )
        phone_to_token = json.loads(Path(phoneme_dict_path).read_text())
        model_config_params = json.loads(Path(model_config_path).read_text())
        field_names = {field.name for field in fields(ModelConfig)}
        model_config = ModelConfig(
            **{key: value for key, value in model_config_params.items() if key in field_names}
        )
        model = Model(model_config)
        skipped_components: dict[str, float] = {}
        skipped_prefixes: tuple[str, ...] = ()
        skipped_exact: set[str] = set()
        if skip_temp_weights:
            skipped_components["temp_former"] = discard_module_parameters(
                model.temp_former
            )
            skipped_prefixes += ("temp_former.",)
        if skip_dep_weights:
            skipped_components["dep_former"] = discard_module_parameters(
                model.dep_former
            )
            skipped_prefixes += ("dep_former.",)
        if skip_semantic_head:
            skipped_components["sem_head"] = discard_module_parameters(
                model.sem_head
            )
            skipped_prefixes += ("sem_head.",)
        if skip_acoustic_head:
            audio_head = model.audio_head
            skipped_components["audio_head"] = (
                audio_head.numel() * audio_head.element_size() / 1024**2
            )
            model.audio_head = None
            skipped_exact.add("audio_head")
        if skip_audio_embedding_weights:
            audio_embedding_weight = model.audio_embeddings.weight
            skipped_components["audio_embeddings"] = (
                audio_embedding_weight.numel()
                * audio_embedding_weight.element_size()
                / 1024**2
            )
            model.audio_embeddings.weight = None
            skipped_exact.add("audio_embeddings.weight")
        skipped_buffer_components: dict[str, float] = {}
        if skip_phone_weights:
            skipped_components["phone_former"] = discard_module_parameters(
                model.phone_former
            )
            skipped_prefixes += ("phone_former.",)
            phone_embedding_weight = model.phone_embeddings.weight
            skipped_components["phone_embeddings"] = (
                phone_embedding_weight.numel()
                * phone_embedding_weight.element_size()
                / 1024**2
            )
            model.phone_embeddings.weight = None
            skipped_exact.add("phone_embeddings.weight")
            phone_mask = model.phone_former_mask
            skipped_buffer_components["phone_former_mask"] = (
                phone_mask.numel() * phone_mask.element_size() / 1024**2
            )
            model.phone_former_mask = None
        model_weight_path = generator_setup.hf_hub_download(
            config.model_repo, config.model_name, token=config.hf_token
        )
        with safe_open(model_weight_path, framework="pt", device="cpu") as source:
            state_dict = {
                key: source.get_tensor(key)
                for key in source.keys()
                if key not in skipped_exact
                and not key.startswith(skipped_prefixes)
            }
        missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
        assert not missing and not unexpected, (
            f"Model weights mismatch: missing={missing}, unexpected={unexpected}"
        )
        del state_dict
        gc.collect()
        if skipped_components:
            report_stage(
                "pytorch_weight_load_skipped",
                components={
                    name: round(size, 1)
                    for name, size in skipped_components.items()
                },
                fp32_parameter_mib=round(sum(skipped_components.values()), 1),
                buffers={
                    name: round(size, 1)
                    for name, size in skipped_buffer_components.items()
                },
            )
        model = model.eval().to(device, dtype=dtype)
        model.setup_caches(max_batch_size=batch_size, dtype=dtype)
        return model, phone_to_token

    generator_setup.load_generator_model = load_generator_model_low_memory
    generator_module.load_generator_model = load_generator_model_low_memory
    resources = ResourceSampler()
    resources.start()

    config_path = ARGS.demo_dir / "generator_ru.json"
    speaking_rate_path = ARGS.demo_dir / "configs" / "speaking_rate_ru.json"
    if not speaking_rate_path.exists():
        speaking_rate_path = ARGS.demo_dir / "configs" / "speaking_rate.json"

    config = SpeechGeneratorConfig(**json.loads(config_path.read_text()))
    if ARGS.cache_prompt:
        config.cache_prompt = True
    speaking_rate_config = json.loads(speaking_rate_path.read_text())

    if ARGS.skip_ruaccent_rule_engine and ARGS.restore_ruaccent_rule_engine:
        raise ValueError(
            "--skip-ruaccent-rule-engine and --restore-ruaccent-rule-engine "
            "are mutually exclusive"
        )

    preconstructed_mimi_tensorrt = None
    if ARGS.fixed_prompt_runtime:
        prompt_cache_path = ARGS.prompt_audio.with_suffix(".prompt.npy")
        if not config.cache_prompt or not prompt_cache_path.exists():
            raise RuntimeError(
                "--fixed-prompt-runtime requires --cache-prompt and an existing "
                f"cache file: {prompt_cache_path}"
            )
        load_mimi_model = generator_module.load_mimi_model
        mimi_load_calls = 0
        if ARGS.tensorrt_mimi_state is not None:
            from voxtream_tensorrt_mimi_runtime import TensorRTMimiDecoder

            preconstructed_mimi_tensorrt = TensorRTMimiDecoder(
                ARGS.tensorrt_mimi_engine,
                initial_state_path=ARGS.tensorrt_mimi_state,
                preloaded=PRELOADED_TRT_MIMI,
            )
            PRELOADED_TRT_MIMI = None

        def load_decoder_mimi_only(*args, **kwargs):
            nonlocal mimi_load_calls
            global PRELOADED_MIMI
            mimi_load_calls += 1
            if mimi_load_calls == 1:
                if preconstructed_mimi_tensorrt is not None:
                    return preconstructed_mimi_tensorrt
                if PRELOADED_MIMI is not None:
                    mimi = PRELOADED_MIMI
                    PRELOADED_MIMI = None
                    return mimi
                return load_mimi_model(*args, **kwargs)
            return None

        # Cached prompt data already contains Mimi codes and the speaker vector.
        # The prompt-only Mimi, ReDimNet speaker encoder and VAD are dead weight
        # for a fixed-voice device and consume scarce unified Jetson memory.
        generator_module.load_mimi_model = load_decoder_mimi_only
        generator_module.load_speaker_encoder = lambda *args, **kwargs: None
        generator_module.load_silero_vad = lambda *args, **kwargs: None

    if ARGS.skip_ruaccent_rule_engine:
        class UnloadedRuleEngine:
            """Stand-in for an upstream object unused by RUAccent.process_all()."""

            def load(self, path: str) -> None:
                self.path = path

            def __getattr__(self, name: str):
                raise RuntimeError(
                    "RUAccent unexpectedly accessed its disabled RuleEngine "
                    f"attribute {name!r}"
                )

        # RUAccent imports this module only after downloading its large Koziev
        # resources. Installing a synthetic module avoids importing those
        # resources early while leaving RUAccent's normal download/load order
        # intact for every component that process_all() actually uses.
        rule_engine_module = types.ModuleType("ruaccent.rule_accent_engine")
        rule_engine_module.RuleEngine = UnloadedRuleEngine
        sys.modules[rule_engine_module.__name__] = rule_engine_module

    if ARGS.runorm_cpu:
        import runorm.runorm as runorm_module

        runorm_pipeline = runorm_module.pipeline

        @functools.wraps(runorm_pipeline)
        def runorm_pipeline_on_cpu(*args, **kwargs):
            kwargs.setdefault("device", -1)
            return runorm_pipeline(*args, **kwargs)

        runorm_module.pipeline = runorm_pipeline_on_cpu

    set_seed()
    report_stage("model_load_started")
    load_started = time.perf_counter()
    generator = SpeechGenerator(config, speaking_rate_config)
    ruaccent_rule_engine_loaded = (
        not ARGS.skip_ruaccent_rule_engine
        and hasattr(generator.ctx.phonemizer.acc, "rule_accent")
    )
    if ARGS.restore_ruaccent_rule_engine:
        from ruaccent_rule_engine import install_rule_engine_pipeline

        if not ruaccent_rule_engine_loaded:
            raise RuntimeError(
                "--restore-ruaccent-rule-engine requires a RUAccent build loaded "
                "with load_rule_engine=True"
            )
        install_rule_engine_pipeline(generator.ctx.phonemizer.acc)
    cuda_graph_components = {
        component.strip()
        for component in ARGS.cuda_graph_components.split(",")
        if component.strip()
    }
    unknown_graph_components = cuda_graph_components - {"temp", "dep", "mimi"}
    if unknown_graph_components:
        raise ValueError(
            "Unknown --cuda-graph-components: "
            + ", ".join(sorted(unknown_graph_components))
        )

    mimi_tensorrt = None
    if ARGS.tensorrt_mimi_engine is not None:
        from voxtream_tensorrt_mimi_runtime import TensorRTMimiDecoder

        if preconstructed_mimi_tensorrt is not None:
            if generator.mimi is not preconstructed_mimi_tensorrt:
                raise RuntimeError("SpeechGenerator replaced the preconstructed Mimi runtime")
            mimi_tensorrt = preconstructed_mimi_tensorrt
            report_stage(
                "pytorch_mimi_decoder_load_skipped",
                runtime=mimi_tensorrt.metrics(),
            )
        else:
            original_mimi = generator.mimi
            mimi_tensorrt = TensorRTMimiDecoder(
                ARGS.tensorrt_mimi_engine,
                original_mimi,
                preloaded=PRELOADED_TRT_MIMI,
            )
            PRELOADED_TRT_MIMI = None
            generator.mimi = mimi_tensorrt
            discarded_mimi_parameter_mib, discarded_mimi_buffer_mib = (
                discard_module_tensors(original_mimi)
            )
            del original_mimi
            report_stage(
                "pytorch_mimi_decoder_unloaded",
                discarded_parameter_mib=round(discarded_mimi_parameter_mib, 1),
                discarded_buffer_mib=round(discarded_mimi_buffer_mib, 1),
                runtime=mimi_tensorrt.metrics(),
            )
        cuda_graph_components.discard("mimi")
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    audio_embedding_cuda = None
    if ARGS.cuda_audio_embedding_weight is not None:
        from voxtream_cuda_audio_embedding import CudaAudioEmbeddingFacade

        original_audio_embeddings = generator.model.audio_embeddings
        audio_embedding_cuda = CudaAudioEmbeddingFacade(
            ARGS.cuda_audio_embedding_weight,
            ARGS.cuda_audio_embedding_cubin,
            generator.model.config.audio_vocab_size
            * generator.model.config.num_codebooks,
            generator.model.config.embedding_dim,
        )
        generator.model.audio_embeddings = audio_embedding_cuda
        discarded_audio_embedding_parameter_mib, discarded_audio_embedding_buffer_mib = (
            discard_module_tensors(original_audio_embeddings)
        )
        del original_audio_embeddings
        gc.collect()
        report_stage(
            "pytorch_audio_embeddings_unloaded",
            discarded_parameter_mib=round(
                discarded_audio_embedding_parameter_mib, 1
            ),
            discarded_buffer_mib=round(
                discarded_audio_embedding_buffer_mib, 1
            ),
            runtime=audio_embedding_cuda.metrics(),
        )

    phone_tensorrt = None
    if ARGS.tensorrt_phone_engine is not None:
        from voxtream_tensorrt_phone_runtime import (
            PhoneEmbeddingFacade,
            TensorRTPhoneEncoder,
            TensorRTPhoneFormerFacade,
        )

        original_phone_former = generator.model.phone_former
        phone_tensorrt = TensorRTPhoneEncoder(
            ARGS.tensorrt_phone_engine,
            preloaded=PRELOADED_TRT_PHONE,
        )
        PRELOADED_TRT_PHONE = None
        phone_embeddings_facade = PhoneEmbeddingFacade(
            generator.model, phone_tensorrt
        )
        generator.ctx.extract_phone_embeddings = phone_embeddings_facade
        generator.model.extract_phoneme_embeddings = phone_embeddings_facade
        generator.model.phone_former = TensorRTPhoneFormerFacade(phone_tensorrt)
        phone_pool_aliases = [
            name
            for name, pooled_model in MODEL_POOL.items()
            if pooled_model is original_phone_former
        ]
        for name in phone_pool_aliases:
            del MODEL_POOL[name]
        discarded_phone_parameter_mib, discarded_phone_buffer_mib = (
            discard_module_tensors(original_phone_former)
        )
        del original_phone_former
        gc.collect()
        report_stage(
            "pytorch_phone_encoder_unloaded",
            discarded_parameter_mib=round(discarded_phone_parameter_mib, 1),
            discarded_buffer_mib=round(discarded_phone_buffer_mib, 1),
            removed_model_pool_aliases=phone_pool_aliases,
        )

    # Each model function is independently wrapped by upstream CUDAGraphed.
    # This lets us keep unsupported or numerically sensitive pieces eager while
    # proving the remaining graph in isolation.
    generator.model._temp_former.disable = "temp" not in cuda_graph_components
    generator.model._dep_former_init.disable = "dep" not in cuda_graph_components
    generator.model._dep_former.disable = "dep" not in cuda_graph_components

    async_dep_caches = 0
    if ARGS.async_dep_cache_reset:
        from torchtune.modules.kv_cache import KVCache

        original_kv_reset = KVCache.reset
        for module in generator.model.dep_former.modules():
            if isinstance(module, KVCache):
                module._voxtream_reset_offset = torch.empty_like(module.cache_pos[:1])
                async_dep_caches += 1

        def reset_without_host_sync(self) -> None:
            offset = getattr(self, "_voxtream_reset_offset", None)
            if offset is None:
                return original_kv_reset(self)
            self.k_cache.zero_()
            self.v_cache.zero_()
            offset.copy_(self.cache_pos[:1])
            self.cache_pos.sub_(offset)

        KVCache.reset = reset_without_host_sync

    sdpa_patched_modules = 0
    if ARGS.cuda_graph_compatible_sdpa:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        def patch_attention_module(transformer: torch.nn.Module) -> int:
            patched = 0
            for module in transformer.modules():
                attention_call = getattr(module, "_attention_call", None)
                if attention_call is None:
                    continue

                @functools.wraps(attention_call)
                def efficient_attention(*args, _attention_call=attention_call, **kwargs):
                    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                        return _attention_call(*args, **kwargs)

                module._attention_call = efficient_attention
                patched += 1
            return patched

        # phone_former stays on the exact upstream backend. Only transformers
        # selected for graph capture get the capture-safe SDPA kernel.
        if "temp" in cuda_graph_components:
            sdpa_patched_modules += patch_attention_module(generator.model.temp_former)
        if "dep" in cuda_graph_components and not ARGS.tensorrt_dep_full_runtime:
            sdpa_patched_modules += patch_attention_module(generator.model.dep_former)

        if "mimi" in cuda_graph_components and mimi_tensorrt is None:
            # Mimi owns another lazy CUDAGraphed transformer and calls PyTorch
            # SDPA directly, without torchtune's _attention_call hook.
            mimi_decode = generator.mimi.decode

            @functools.wraps(mimi_decode)
            def efficient_mimi_decode(*args, **kwargs):
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                    return mimi_decode(*args, **kwargs)

            generator.mimi.decode = efficient_mimi_decode

    if "mimi" not in cuda_graph_components and mimi_tensorrt is None:
        from moshi.utils.compile import no_cuda_graph

        mimi_decode = generator.mimi.decode

        @functools.wraps(mimi_decode)
        def eager_mimi_decode(*args, **kwargs):
            with no_cuda_graph():
                return mimi_decode(*args, **kwargs)

        generator.mimi.decode = eager_mimi_decode

    if ARGS.fixed_prompt_runtime:
        # Model construction leaves hundreds of MiB in PyTorch's allocator
        # cache. They contain no live tensors but fragment Jetson unified memory
        # before phone embedding and any graph-pool allocation.
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    temp_prefill_capture = None
    if ARGS.temp_prefill_capture is not None:
        temp_prefill_capture = TempFormerPrefillCapture(
            generator.model.temp_former, ARGS.temp_prefill_capture
        )
        accepted_temp_prefill = generator.model.temp_former.forward

        @functools.wraps(accepted_temp_prefill)
        def captured_temp_prefill(hidden, input_pos, mask):
            return temp_prefill_capture.run(
                accepted_temp_prefill, hidden, input_pos, mask
            )

        generator.model.temp_former.forward = captured_temp_prefill

    temp_capture = None
    if ARGS.temp_shadow_capture is not None:
        temp_capture = TempFormerCapture(
            generator.model, ARGS.temp_shadow_capture
        )
        accepted_temp_former = generator.model._temp_former

        @functools.wraps(accepted_temp_former)
        def captured_temp_former(hidden, input_pos, mask):
            return temp_capture.run(
                accepted_temp_former, hidden, input_pos, mask
            )

        generator.model._temp_former = captured_temp_former
        accepted_generate_frame = generator.model.generate_frame

        @functools.wraps(accepted_generate_frame)
        def captured_generate_frame(*args, **kwargs):
            previous_record_count = len(temp_capture.records)
            result = accepted_generate_frame(*args, **kwargs)
            frame, predicted_state, _ = result
            temp_capture.attach_actual_sample(
                previous_record_count, frame, predicted_state
            )
            return result

        generator.model.generate_frame = captured_generate_frame

    temp_tensorrt = None
    temp_prefill_facade = None
    temp_semantic_head_facade = None
    if ARGS.tensorrt_temp_engine is not None:
        from voxtream_tensorrt_explicit_kv_probe import (
            ExplicitKVTRTRunner,
            kv_buffer_names,
        )

        temp_buffer_names = kv_buffer_names(generator.model.temp_former)
        if len(temp_buffer_names) != 36:
            raise RuntimeError(
                f"expected 36 temp_former KV buffers, got {len(temp_buffer_names)}"
            )
        temp_buffers = dict(generator.model.temp_former.named_buffers())
        temp_initial_state = tuple(
            temp_buffers[name] for name in temp_buffer_names
        )
        temp_runner = ExplicitKVTRTRunner(
            ARGS.tensorrt_temp_engine,
            temp_buffer_names,
            temp_initial_state,
            sequence_length=1,
            preloaded=PRELOADED_TRT_TEMP,
            borrow_initial_state=True,
            inplace_state=True,
            use_current_stream=True,
        )
        PRELOADED_TRT_TEMP = None
        temp_tensorrt = TempFormerTensorRTRuntime(
            generator.model, temp_runner, temp_buffer_names
        )

        def tensorrt_temp_former(hidden, input_pos, mask):
            return temp_tensorrt.run(hidden, input_pos, mask)

        generator.model._temp_former = tensorrt_temp_former
        if ARGS.tensorrt_temp_prefill_cache is not None:
            cache_payload = torch.load(
                ARGS.tensorrt_temp_prefill_cache,
                map_location="cpu",
                weights_only=False,
            )
            original_temp_former = generator.model.temp_former
            temp_parameter_mib = sum(
                parameter.numel() * parameter.element_size()
                for parameter in original_temp_former.parameters()
            ) / 1024**2
            before_temp_unload = memory_snapshot()
            before_temp_cuda_mib = torch.cuda.memory_allocated() / 1024**2
            cached_semantic_logits = cache_payload.get("semantic_logits")
            if cached_semantic_logits is not None:
                cached_semantic_logits = cached_semantic_logits.to(
                    device=generator.ctx.device
                )
            else:
                semantic_weight = getattr(
                    generator.model.sem_head, "weight", None
                )
                if semantic_weight is None:
                    raise RuntimeError(
                        "fixed-prompt cache has no semantic_logits, but PyTorch "
                        "sem_head was intentionally skipped; augment the cache "
                        "before using the fused torchless load path"
                    )
                cached_prefill_output = cache_payload["output"].to(
                    device=generator.ctx.device
                )
                cached_semantic_logits = generator.model.sem_head(
                    cached_prefill_output[:, -1, :].to(torch.bfloat16)
                )
                del cached_prefill_output
            temp_prefill_facade = FixedPromptTempFormerFacade(
                temp_tensorrt,
                cache_payload,
                generator.ctx.device,
                cached_semantic_logits=cached_semantic_logits,
            )
            temp_tensorrt.prefill_mode = "fixed_prompt_cache"
            generator.model.temp_former = temp_prefill_facade
            temp_pool_aliases = [
                name
                for name, pooled_model in MODEL_POOL.items()
                if pooled_model is original_temp_former
            ]
            for name in temp_pool_aliases:
                del MODEL_POOL[name]
            discarded_temp_parameter_mib, discarded_temp_buffer_mib = (
                discard_module_tensors(original_temp_former)
            )
            original_temp_ref = weakref.ref(original_temp_former)
            del original_temp_former
            del cache_payload
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            after_temp_unload = memory_snapshot()
            report_stage(
                "pytorch_temp_former_unloaded",
                temp_former_parameter_mib=round(temp_parameter_mib, 1),
                discarded_parameter_mib=round(
                    discarded_temp_parameter_mib, 1
                ),
                deregistered_buffer_mib=round(discarded_temp_buffer_mib, 1),
                removed_model_pool_aliases=temp_pool_aliases,
                temp_former_object_released=original_temp_ref() is None,
                process_rss_delta_mib=round(
                    after_temp_unload["process_rss_mib"]
                    - before_temp_unload["process_rss_mib"],
                    1,
                ),
                cuda_allocated_delta_mib=round(
                    torch.cuda.memory_allocated() / 1024**2
                    - before_temp_cuda_mib,
                    1,
                ),
            )
        if "semantic_logits" in temp_runner.extra_outputs:
            original_sem_head = generator.model.sem_head
            temp_semantic_head_facade = TensorRTSemanticHeadFacade(
                temp_tensorrt,
                fallback=(
                    None if temp_prefill_facade is not None else original_sem_head
                ),
            )
            generator.model.sem_head = temp_semantic_head_facade
            if temp_prefill_facade is not None:
                before_head_cuda_mib = torch.cuda.memory_allocated() / 1024**2
                discarded_head_parameter_mib, discarded_head_buffer_mib = (
                    discard_module_tensors(original_sem_head)
                )
                original_head_ref = weakref.ref(original_sem_head)
                del original_sem_head
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                report_stage(
                    "pytorch_semantic_head_unloaded",
                    discarded_parameter_mib=round(
                        discarded_head_parameter_mib, 1
                    ),
                    discarded_buffer_mib=round(discarded_head_buffer_mib, 1),
                    semantic_head_object_released=original_head_ref() is None,
                    cuda_allocated_delta_mib=round(
                        torch.cuda.memory_allocated() / 1024**2
                        - before_head_cuda_mib,
                        1,
                    ),
                )

    temp_shadow_probe = None
    if ARGS.tensorrt_temp_shadow_engine is not None:
        from voxtream.utils.model import sample_semantic_token
        from voxtream_tensorrt_explicit_kv_probe import (
            ExplicitKVTRTRunner,
            kv_buffer_names,
        )

        temp_buffer_names = kv_buffer_names(generator.model.temp_former)
        if len(temp_buffer_names) != 36:
            raise RuntimeError(
                f"expected 36 temp_former KV buffers, got {len(temp_buffer_names)}"
            )
        temp_buffers = dict(generator.model.temp_former.named_buffers())
        temp_initial_state = tuple(
            temp_buffers[name] for name in temp_buffer_names
        )
        temp_shadow_runner = ExplicitKVTRTRunner(
            ARGS.tensorrt_temp_shadow_engine,
            temp_buffer_names,
            temp_initial_state,
            sequence_length=1,
            preloaded=PRELOADED_TRT_TEMP,
        )
        PRELOADED_TRT_TEMP = None
        temp_shadow_probe = TempFormerShadowProbe(
            generator.model,
            temp_shadow_runner,
            temp_buffer_names,
            config,
            sample_semantic_token,
        )
        accepted_temp_former = generator.model._temp_former

        @functools.wraps(accepted_temp_former)
        def shadowed_temp_former(hidden, input_pos, mask):
            return temp_shadow_probe.run(
                accepted_temp_former, hidden, input_pos, mask
            )

        generator.model._temp_former = shadowed_temp_former

    tensorrt_dep = None
    tensorrt_dep_shadow = None
    shadow_probe = None
    if ARGS.tensorrt_dep_engine is not None:
        from voxtream_tensorrt_runtime import (
            TensorRTDepFormerFacade,
            TensorRTDepFormerStep,
        )

        # Do not nest the upstream one-token CUDA graph around TensorRT.
        generator.model._dep_former.disable = True
        if ARGS.tensorrt_dep_full_runtime:
            generator.model._dep_former_init.disable = True
        tensorrt_dep = TensorRTDepFormerStep(
            ARGS.tensorrt_dep_engine,
            generator.model.dep_former,
            preloaded=PRELOADED_TRT_DEP,
            init_engine_path=ARGS.tensorrt_dep_init_engine,
            preloaded_init=(
                PRELOADED_TRT_DEP_INIT
                if ARGS.tensorrt_dep_full_runtime
                else None
            ),
            capture_cuda_graph=ARGS.tensorrt_dep_cuda_graph,
            run_init=ARGS.tensorrt_dep_full_runtime,
        )
        PRELOADED_TRT_DEP = None
        if ARGS.tensorrt_dep_full_runtime:
            PRELOADED_TRT_DEP_INIT = None
        generator.model._dep_former = tensorrt_dep
        if ARGS.tensorrt_dep_init_shadow_engine is not None:
            shared_step_engine = types.SimpleNamespace(
                engine_path=tensorrt_dep.engine_path,
                runtime=tensorrt_dep.runtime,
                engine=tensorrt_dep.engine,
            )
            tensorrt_dep_shadow = TensorRTDepFormerStep(
                ARGS.tensorrt_dep_engine,
                generator.model.dep_former,
                preloaded=shared_step_engine,
                init_engine_path=ARGS.tensorrt_dep_init_shadow_engine,
                preloaded_init=PRELOADED_TRT_DEP_INIT,
                capture_cuda_graph=False,
                run_init=True,
            )
            PRELOADED_TRT_DEP_INIT = None
            shadow_probe = DepFormerShadowProbe(
                generator.model,
                tensorrt_dep,
                tensorrt_dep_shadow,
                temperature=config.temperature,
                cfg_gamma=config.cfg_ac_gamma,
            )
            accepted_dep_init = generator.model._dep_former_init
            accepted_dep_step = generator.model._dep_former

            @functools.wraps(accepted_dep_init)
            def shadowed_dep_init(hidden, input_pos, mask):
                reference = accepted_dep_init(hidden, input_pos, mask)
                tensorrt_dep_shadow.reset_caches()
                shadow_probe.begin_frame()
                candidate = tensorrt_dep_shadow(hidden, input_pos, mask)
                shadow_probe.record(reference, candidate, init=True)
                return reference

            @functools.wraps(accepted_dep_step)
            def shadowed_dep_step(hidden, input_pos, mask):
                reference = accepted_dep_step(hidden, input_pos, mask)
                candidate = tensorrt_dep_shadow(hidden, input_pos, mask)
                shadow_probe.record(reference, candidate, init=False)
                return reference

            generator.model._dep_former_init = shadowed_dep_init
            generator.model._dep_former = shadowed_dep_step
        if ARGS.tensorrt_dep_full_runtime:
            generator.model._dep_former_init = tensorrt_dep
            original_dep_former = generator.model.dep_former
            dep_former_parameter_mib = sum(
                parameter.numel() * parameter.element_size()
                for parameter in original_dep_former.parameters()
            ) / 1024**2
            before_unload = memory_snapshot()
            before_unload_cuda_allocated_mib = (
                torch.cuda.memory_allocated() / 1024**2
            )
            original_dep_former_ref = weakref.ref(original_dep_former)
            generator.model.dep_former = TensorRTDepFormerFacade(tensorrt_dep)
            dep_pool_aliases = [
                name
                for name, pooled_model in MODEL_POOL.items()
                if pooled_model is original_dep_former
            ]
            for name in dep_pool_aliases:
                del MODEL_POOL[name]
            discarded_parameter_mib, discarded_buffer_mib = discard_module_tensors(
                original_dep_former
            )
            del original_dep_former
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            after_unload = memory_snapshot()
            report_stage(
                "pytorch_dep_former_unloaded",
                dep_former_parameter_mib=round(dep_former_parameter_mib, 1),
                discarded_parameter_mib=round(discarded_parameter_mib, 1),
                discarded_buffer_mib=round(discarded_buffer_mib, 1),
                removed_model_pool_aliases=dep_pool_aliases,
                dep_former_object_released=original_dep_former_ref() is None,
                process_rss_delta_mib=round(
                    after_unload["process_rss_mib"]
                    - before_unload["process_rss_mib"],
                    1,
                ),
                cuda_allocated_delta_mib=round(
                    torch.cuda.memory_allocated() / 1024**2
                    - before_unload_cuda_allocated_mib,
                    1,
                ),
            )
            if (
                tensorrt_dep.acoustic_logits is not None
                and generator.model.audio_head is not None
            ):
                original_audio_head = generator.model.audio_head
                audio_head_parameter_mib = (
                    original_audio_head.numel()
                    * original_audio_head.element_size()
                    / 1024**2
                )
                before_head_unload = memory_snapshot()
                before_head_unload_cuda_mib = (
                    torch.cuda.memory_allocated() / 1024**2
                )
                original_audio_head_ref = weakref.ref(original_audio_head)
                generator.model.audio_head = None
                del original_audio_head
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                after_head_unload = memory_snapshot()
                report_stage(
                    "pytorch_acoustic_head_unloaded",
                    audio_head_parameter_mib=round(audio_head_parameter_mib, 1),
                    audio_head_object_released=(
                        original_audio_head_ref() is None
                    ),
                    process_rss_delta_mib=round(
                        after_head_unload["process_rss_mib"]
                        - before_head_unload["process_rss_mib"],
                        1,
                    ),
                    cuda_allocated_delta_mib=round(
                        torch.cuda.memory_allocated() / 1024**2
                        - before_head_unload_cuda_mib,
                        1,
                    ),
                )
        gc.collect()
    model_load_seconds = time.perf_counter() - load_started
    model_load_resources = resources.summary(reset=True)
    report_stage(
        "model_load_finished",
        model_load_seconds=round(model_load_seconds, 3),
        resources=model_load_resources,
    )

    profiler = (
        CallProfiler(synchronize_cuda=ARGS.profile_sync)
        if ARGS.profile or ARGS.profile_sync
        else NoOpProfiler()
    )
    inference_stream = torch.cuda.Stream() if tensorrt_dep is not None else None

    # These names resolve from voxtream.generator globals during every stream.
    generator_module.prepare_prompt = profiler.wrap(
        "prompt.prepare", generator_module.prepare_prompt
    )
    generator_module.prepare_non_streaming_text = profiler.wrap(
        "text.prepare_total", generator_module.prepare_non_streaming_text
    )
    generator_module.decode_audio_frame = profiler.wrap(
        "mimi.decode_audio_frame", generator_module.decode_audio_frame
    )
    generator_module.update_indices_and_tokens = profiler.wrap(
        "runtime.update_indices", generator_module.update_indices_and_tokens
    )
    generator_module.update_speaking_rate_params = profiler.wrap(
        "runtime.speaking_rate", generator_module.update_speaking_rate_params
    )

    generator.model.generate_frame = profiler.wrap(
        "generator.generate_frame", generator.model.generate_frame
    )
    generator.model._temp_former = profiler.wrap(
        "runtime.temp_former", generator.model._temp_former
    )
    generator.model._dep_former_init = profiler.wrap(
        "runtime.dep_former_init", generator.model._dep_former_init
    )
    generator.model.phone_former.forward = profiler.wrap(
        "transformer.phone_former", generator.model.phone_former.forward
    )
    generator.model.temp_former.forward = profiler.wrap(
        "transformer.temp_former", generator.model.temp_former.forward
    )
    generator.model.dep_former.forward = profiler.wrap(
        "transformer.dep_former", generator.model.dep_former.forward
    )
    if tensorrt_dep is not None:
        generator.model._dep_former = profiler.wrap(
            "tensorrt.dep_former", generator.model._dep_former
        )
    generator.mimi.decode = profiler.wrap("mimi.decode", generator.mimi.decode)
    generator._ensure_mimi_streaming = profiler.wrap(
        "mimi.ensure_streaming", generator._ensure_mimi_streaming
    )

    phonemizer = generator.ctx.phonemizer
    phonemizer._normalize_digits = profiler.wrap(
        "text.runorm_digits", phonemizer._normalize_digits
    )
    phonemizer._accent = profiler.wrap("text.ruaccent", phonemizer._accent)
    phonemizer.esp.phonemize = profiler.wrap(
        "text.espeak", phonemizer.esp.phonemize
    )

    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    final_audio = None
    torch.cuda.reset_peak_memory_stats()

    for run_index in range(ARGS.repeat):
        profiler.reset()
        set_seed()
        frames = []
        frame_compute_seconds = 0.0
        first_packet_seconds = None
        started = time.perf_counter()
        report_stage("generation_started", run=run_index + 1, text=ARGS.text)
        rate = repeat(ARGS.speaking_rate) if ARGS.speaking_rate is not None else None
        stream_context = (
            torch.cuda.stream(inference_stream)
            if inference_stream is not None
            else contextlib.nullcontext()
        )
        with stream_context:
            stream = generator.generate_stream(
                prompt_audio_path=ARGS.prompt_audio,
                text=ARGS.text,
                speaking_rate=rate,
                enhance_prompt=False,
                apply_vad=False,
                return_progress=False,
                min_streaming_rtf=None,
            )
            for frame, compute_seconds in stream:
                if first_packet_seconds is None:
                    first_packet_seconds = time.perf_counter() - started
                    report_stage(
                        "first_packet",
                        run=run_index + 1,
                        first_packet_seconds=round(first_packet_seconds, 3),
                    )
                frames.append(frame)
                frame_compute_seconds += float(compute_seconds)

        wall_seconds = time.perf_counter() - started
        final_audio = np.concatenate(frames) if frames else np.zeros(1, dtype=np.float32)
        audio_seconds = len(final_audio) / int(config.mimi_sr)
        call_profile = profiler.summary()
        run_resources = resources.summary(reset=True)
        report_stage(
            "generation_finished",
            run=run_index + 1,
            generation_wall_seconds=round(wall_seconds, 3),
            audio_seconds=round(audio_seconds, 3),
            frames=len(frames),
            resources=run_resources,
        )
        runs.append(
            {
                "run": run_index + 1,
                "first_packet_seconds": round(first_packet_seconds or wall_seconds, 3),
                "generation_wall_seconds": round(wall_seconds, 3),
                "frame_compute_seconds": round(frame_compute_seconds, 3),
                "audio_seconds": round(audio_seconds, 3),
                "wall_rtf": round(wall_seconds / audio_seconds, 3),
                "compute_rtf": round(frame_compute_seconds / audio_seconds, 3),
                "frames": len(frames),
                "resources": run_resources,
                "call_profile": call_profile,
            }
        )

    assert final_audio is not None
    sf.write(ARGS.output, final_audio, int(config.mimi_sr), subtype="PCM_16")
    temp_capture_result = (
        temp_capture.save() if temp_capture is not None else None
    )
    max_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    resources.stop()
    result = {
        "output": str(ARGS.output),
        "text": ARGS.text,
        "prompt_audio": str(ARGS.prompt_audio),
        "device": str(generator.ctx.device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "model_load_seconds": round(model_load_seconds, 3),
        "model_load_resources": model_load_resources,
        "look_ahead": os.environ.get("VOXTREAM_LA"),
        "speaking_rate": ARGS.speaking_rate,
        "cache_prompt": config.cache_prompt,
        "fixed_prompt_runtime": ARGS.fixed_prompt_runtime,
        "runorm_forced_cpu": ARGS.runorm_cpu,
        "dep_former_backend": "tensorrt" if tensorrt_dep is not None else "pytorch",
        "temp_former_backend": "tensorrt_q1" if temp_tensorrt is not None else "pytorch",
        "mimi_backend": "tensorrt" if mimi_tensorrt is not None else "pytorch",
        "tensorrt_dep_full_runtime": ARGS.tensorrt_dep_full_runtime,
        "ruaccent_rule_engine_loaded": ruaccent_rule_engine_loaded,
        "ruaccent_rule_engine_pipeline_restored": ARGS.restore_ruaccent_rule_engine,
        "profile_enabled": ARGS.profile or ARGS.profile_sync,
        "profile_synchronized": ARGS.profile_sync,
        "trimmed_unused_model_pool_embeddings_mib": round(trimmed_embedding_mib, 1),
        "mimi_preloaded_before_voxtream": ARGS.preload_mimi_before_voxtream,
        "cuda_graph": {
            "enabled": not bool(os.environ.get("NO_CUDA_GRAPH")),
            "components": sorted(cuda_graph_components),
        },
        "async_dep_cache_reset": ARGS.async_dep_cache_reset,
        "async_dep_caches": async_dep_caches,
        "tensorrt_dep": tensorrt_dep.metrics() if tensorrt_dep is not None else None,
        "tensorrt_dep_init_shadow": (
            shadow_probe.summary() if shadow_probe is not None else None
        ),
        "tensorrt_temp_shadow": (
            temp_shadow_probe.summary() if temp_shadow_probe is not None else None
        ),
        "tensorrt_temp": (
            temp_tensorrt.metrics() if temp_tensorrt is not None else None
        ),
        "tensorrt_phone": (
            phone_tensorrt.metrics() if phone_tensorrt is not None else None
        ),
        "tensorrt_mimi": (
            mimi_tensorrt.metrics() if mimi_tensorrt is not None else None
        ),
        "cuda_audio_embeddings": (
            audio_embedding_cuda.metrics()
            if audio_embedding_cuda is not None
            else None
        ),
        "tensorrt_temp_prefill_cache": (
            {
                "path": str(ARGS.tensorrt_temp_prefill_cache),
                "bytes": ARGS.tensorrt_temp_prefill_cache.stat().st_size,
                "calls": temp_prefill_facade.prefill_calls,
            }
            if temp_prefill_facade is not None
            else None
        ),
        "tensorrt_semantic_head": (
            {
                "backend": "fused_temp_engine",
                "pytorch_weights_retained": (
                    temp_semantic_head_facade.fallback is not None
                ),
            }
            if temp_semantic_head_facade is not None
            else None
        ),
        "temp_shadow_capture": temp_capture_result,
        "temp_prefill_capture": (
            temp_prefill_capture.result
            if temp_prefill_capture is not None
            else None
        ),
        "sdpa": {
            "manual_efficient_attention_override": ARGS.cuda_graph_compatible_sdpa,
            "patched_attention_modules": sdpa_patched_modules,
            "flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_enabled": torch.backends.cuda.math_sdp_enabled(),
            "cudnn_enabled": (
                torch.backends.cuda.cudnn_sdp_enabled()
                if hasattr(torch.backends.cuda, "cudnn_sdp_enabled")
                else None
            ),
        },
        "max_rss_mib": round(max_rss_kib / 1024, 1),
        "cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "runs": runs,
    }
    metrics_output = ARGS.output.with_suffix(".json")
    result["metrics_output"] = str(metrics_output)
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    metrics_output.write_text(serialized)
    print(serialized)


def main() -> None:
    global PRELOADED_MIMI, PRELOADED_TRT_DEP, PRELOADED_TRT_DEP_INIT, PRELOADED_TRT_TEMP, PRELOADED_TRT_PHONE, PRELOADED_TRT_MIMI
    if ARGS.repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    if ARGS.tensorrt_dep_cuda_graph and ARGS.tensorrt_dep_engine is None:
        raise SystemExit("--tensorrt-dep-cuda-graph requires --tensorrt-dep-engine")
    if ARGS.tensorrt_dep_engine is not None and not ARGS.tensorrt_dep_engine.is_file():
        raise SystemExit(
            f"TensorRT engine not found: {ARGS.tensorrt_dep_engine}. "
            "Build it, set VOXTREAM_TENSORRT_DEP_ENGINE, or use "
            "--pytorch-dep-former."
        )
    if (
        ARGS.tensorrt_dep_init_engine is not None
        and not ARGS.tensorrt_dep_init_engine.is_file()
    ):
        raise SystemExit(
            f"TensorRT init engine not found: {ARGS.tensorrt_dep_init_engine}"
        )
    if (
        ARGS.tensorrt_dep_init_shadow_engine is not None
        and not ARGS.tensorrt_dep_init_shadow_engine.is_file()
    ):
        raise SystemExit(
            "TensorRT shadow init engine not found: "
            f"{ARGS.tensorrt_dep_init_shadow_engine}"
        )
    if (
        ARGS.tensorrt_temp_shadow_engine is not None
        and not ARGS.tensorrt_temp_shadow_engine.is_file()
    ):
        raise SystemExit(
            "TensorRT temp shadow engine not found: "
            f"{ARGS.tensorrt_temp_shadow_engine}"
        )
    if (
        ARGS.tensorrt_temp_engine is not None
        and not ARGS.tensorrt_temp_engine.is_file()
    ):
        raise SystemExit(
            "TensorRT temp engine not found: "
            f"{ARGS.tensorrt_temp_engine}"
        )
    if (
        ARGS.tensorrt_temp_prefill_cache is not None
        and not ARGS.tensorrt_temp_prefill_cache.is_file()
    ):
        raise SystemExit(
            "TensorRT temp prefill cache not found: "
            f"{ARGS.tensorrt_temp_prefill_cache}"
        )
    if (
        ARGS.tensorrt_phone_engine is not None
        and not ARGS.tensorrt_phone_engine.is_file()
    ):
        raise SystemExit(
            f"TensorRT phone engine not found: {ARGS.tensorrt_phone_engine}"
        )
    if (
        ARGS.tensorrt_mimi_engine is not None
        and not ARGS.tensorrt_mimi_engine.is_file()
    ):
        raise SystemExit(
            f"TensorRT Mimi engine not found: {ARGS.tensorrt_mimi_engine}"
        )
    if (
        ARGS.tensorrt_mimi_state is not None
        and not ARGS.tensorrt_mimi_state.is_file()
    ):
        raise SystemExit(
            f"TensorRT Mimi state manifest not found: {ARGS.tensorrt_mimi_state}"
        )
    if (
        ARGS.cuda_audio_embedding_weight is not None
        and not ARGS.cuda_audio_embedding_weight.is_file()
    ):
        raise SystemExit(
            "CUDA audio embedding weight not found: "
            f"{ARGS.cuda_audio_embedding_weight}"
        )
    if (
        ARGS.cuda_audio_embedding_cubin is not None
        and not ARGS.cuda_audio_embedding_cubin.is_file()
    ):
        raise SystemExit(
            "CUDA audio embedding cubin not found: "
            f"{ARGS.cuda_audio_embedding_cubin}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    os.environ["VOXTREAM_LA"] = ARGS.look_ahead
    os.environ["VOXTREAM_BEST_OF"] = "1"

    if ARGS.tensorrt_mimi_engine is not None:
        from voxtream_tensorrt_runtime import PreloadedTensorRTEngine

        PRELOADED_TRT_MIMI = PreloadedTensorRTEngine(
            ARGS.tensorrt_mimi_engine
        )
        report_stage(
            "tensorrt_mimi_preloaded",
            engine=str(ARGS.tensorrt_mimi_engine),
            engine_bytes=ARGS.tensorrt_mimi_engine.stat().st_size,
        )

    requested_temp_engine = (
        ARGS.tensorrt_temp_engine or ARGS.tensorrt_temp_shadow_engine
    )
    if requested_temp_engine is not None:
        from voxtream_tensorrt_runtime import PreloadedTensorRTEngine

        PRELOADED_TRT_TEMP = PreloadedTensorRTEngine(
            requested_temp_engine
        )
        report_stage(
            (
                "tensorrt_temp_preloaded"
                if ARGS.tensorrt_temp_engine is not None
                else "tensorrt_temp_shadow_preloaded"
            ),
            engine=str(requested_temp_engine),
            engine_bytes=requested_temp_engine.stat().st_size,
        )

    if ARGS.tensorrt_dep_engine is not None:
        from voxtream_tensorrt_runtime import PreloadedTensorRTEngine

        PRELOADED_TRT_DEP = PreloadedTensorRTEngine(ARGS.tensorrt_dep_engine)
        report_stage(
            "tensorrt_dep_preloaded",
            engine=str(ARGS.tensorrt_dep_engine),
            engine_bytes=ARGS.tensorrt_dep_engine.stat().st_size,
        )

    if ARGS.tensorrt_phone_engine is not None:
        from voxtream_tensorrt_runtime import PreloadedTensorRTEngine

        PRELOADED_TRT_PHONE = PreloadedTensorRTEngine(
            ARGS.tensorrt_phone_engine
        )
        report_stage(
            "tensorrt_phone_preloaded",
            engine=str(ARGS.tensorrt_phone_engine),
            engine_bytes=ARGS.tensorrt_phone_engine.stat().st_size,
        )

    requested_init_engine = (
        ARGS.tensorrt_dep_init_engine
        or ARGS.tensorrt_dep_init_shadow_engine
    )
    if requested_init_engine is not None:
        shared_engine = (
            ARGS.tensorrt_dep_engine is not None
            and requested_init_engine.resolve()
            == ARGS.tensorrt_dep_engine.resolve()
        )
        if shared_engine:
            PRELOADED_TRT_DEP_INIT = PRELOADED_TRT_DEP
        else:
            from voxtream_tensorrt_runtime import PreloadedTensorRTEngine

            PRELOADED_TRT_DEP_INIT = PreloadedTensorRTEngine(
                requested_init_engine
            )
        report_stage(
            "tensorrt_dep_init_preloaded",
            engine=str(requested_init_engine),
            engine_bytes=requested_init_engine.stat().st_size,
            shared_engine=shared_engine,
            mode=(
                "shadow"
                if ARGS.tensorrt_dep_init_shadow_engine is not None
                else "runtime"
            ),
        )

    if ARGS.preload_mimi_before_voxtream and ARGS.tensorrt_mimi_state is None:
        if not ARGS.fixed_prompt_runtime:
            raise ValueError(
                "--preload-mimi-before-voxtream currently requires "
                "--fixed-prompt-runtime"
            )
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders as mimi_loaders

        config = json.loads((ARGS.demo_dir / "generator_ru.json").read_text())
        repo_cache = "models--" + config["mimi_repo"].replace("/", "--")
        cached = sorted(
            (ARGS.model_dir / "hf-cache" / "hub" / repo_cache / "snapshots").glob(
                f"*/{config['mimi_name']}"
            )
        )
        mimi_path = (
            cached[-1]
            if cached
            else Path(
                hf_hub_download(config["mimi_repo"], config["mimi_name"])
            )
        )
        PRELOADED_MIMI = (
            mimi_loaders.get_mimi(
                mimi_path,
                device="cuda",
                num_codebooks=int(config["num_codebooks"]),
            )
            .eval()
            .to(dtype=torch.bfloat16)
        )
        report_stage("mimi_preloaded_before_voxtream")

    # demo/app.py applies the RU-specific model, phonemizer, RUNorm and streaming
    # patches before importing voxtream.app.main.  Replace only that Gradio entry
    # point with this benchmark; the official inference path remains unchanged.
    stub = types.ModuleType("voxtream.app")
    stub.main = benchmark_main
    sys.modules["voxtream.app"] = stub

    sys.path.insert(0, str(ARGS.demo_dir))
    import app as russian_app

    sys.argv = ["app.py", "--model-dir", str(ARGS.model_dir)]
    russian_app.main()


if __name__ == "__main__":
    main()
