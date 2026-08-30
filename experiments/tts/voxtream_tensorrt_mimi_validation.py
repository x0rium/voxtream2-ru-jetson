#!/usr/bin/env python3
"""Validate a state-explicit Mimi TensorRT engine across ring-cache wrap."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import tensorrt as trt
import torch
from voxtream_tensorrt_mimi_probe import (
    ExplicitMimiDecoderStep,
    RingKVCache,
    build_decoder,
    decoder_state_bindings,
    make_ring_kv_update_exportable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--frames", type=int, default=130)
    parser.add_argument("--patch-equivalence-frames", type=int, default=16)
    parser.add_argument("--num-codebooks", type=int, default=16)
    return parser.parse_args()


def run_pytorch_trajectory(model, wrapper, bindings, codes):
    state = tuple(binding.get().clone() for binding in bindings)
    audio = []
    started = time.perf_counter()
    with torch.no_grad():
        for frame_codes in codes:
            output = wrapper(frame_codes, *state)
            audio.append(output[0].clone())
            state = tuple(value.clone() for value in output[1:])
    torch.cuda.synchronize()
    return audio, state, time.perf_counter() - started


class TensorRTMimiRunner:
    def __init__(self, engine_path: Path, bindings) -> None:
        self.engine_path = Path(engine_path)
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(
            self.engine_path.read_bytes()
        )
        if self.engine is None:
            raise RuntimeError(f"failed to load {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("failed to create Mimi TensorRT context")
        self.input_names = {
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index))
            == trt.TensorIOMode.INPUT
        }
        self.output_names = {
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index))
            == trt.TensorIOMode.OUTPUT
        }
        self.bindings = bindings
        self.input_name = {}
        self.output_name = {}
        for binding in bindings:
            candidates = (binding.name, f"__next_{binding.name}")
            matches = [name for name in candidates if name in self.input_names]
            if len(matches) != 1:
                raise RuntimeError(
                    f"cannot resolve input for {binding.name}: {matches}"
                )
            self.input_name[binding.name] = matches[0]
            output_name = f"next_{binding.name}"
            if output_name not in self.output_names:
                raise RuntimeError(f"missing output {output_name}")
            self.output_name[binding.name] = output_name
        self.banks = [
            {
                binding.name: binding.get().clone()
                for binding in bindings
            }
            for _ in range(2)
        ]
        self.zero_sentinels = [
            {
                dtype: torch.empty(1, device="cuda", dtype=dtype)
                for dtype in (torch.bfloat16, torch.bool, torch.int64)
            }
            for _ in range(2)
        ]
        self.audio = torch.empty(
            tuple(self.engine.get_tensor_shape("audio")),
            device="cuda",
            dtype=torch.bfloat16,
        )
        self.current_bank = 0

    def pointer(self, tensor: torch.Tensor, bank_index: int) -> int:
        if tensor.numel():
            return tensor.data_ptr()
        return self.zero_sentinels[bank_index][tensor.dtype].data_ptr()

    def reset(self) -> None:
        for bank in self.banks:
            for binding in self.bindings:
                bank[binding.name].copy_(binding.get())
        self.current_bank = 0

    def step(self, codes: torch.Tensor) -> torch.Tensor:
        source_index = self.current_bank
        target_index = source_index ^ 1
        source = self.banks[source_index]
        target = self.banks[target_index]
        if not self.context.set_tensor_address("codes", codes.data_ptr()):
            raise RuntimeError("failed to bind Mimi codes")
        if not self.context.set_tensor_address("audio", self.audio.data_ptr()):
            raise RuntimeError("failed to bind Mimi audio")
        for binding in self.bindings:
            if not self.context.set_tensor_address(
                self.input_name[binding.name],
                self.pointer(source[binding.name], source_index),
            ):
                raise RuntimeError(f"failed to bind input {binding.name}")
            if not self.context.set_tensor_address(
                self.output_name[binding.name],
                self.pointer(target[binding.name], target_index),
            ):
                raise RuntimeError(f"failed to bind output {binding.name}")
        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("Mimi TensorRT enqueue failed")
        self.current_bank = target_index
        return self.audio

    def state(self) -> tuple[torch.Tensor, ...]:
        bank = self.banks[self.current_bank]
        return tuple(bank[binding.name] for binding in self.bindings)


def compare_audio(reference, candidate):
    reference_tensor = torch.cat(reference, dim=-1).float()
    candidate_tensor = torch.cat(candidate, dim=-1).float()
    delta = reference_tensor - candidate_tensor
    return {
        "samples": reference_tensor.numel(),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(delta.square())).item()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                reference_tensor.flatten(),
                candidate_tensor.flatten(),
                dim=0,
            ).item()
        ),
        "bf16_equal_fraction": float(
            (reference_tensor == candidate_tensor).float().mean().item()
        ),
    }


def compare_state(bindings, reference, candidate):
    unequal_tensors = {}
    all_equal = True
    for binding, expected, actual in zip(bindings, reference, candidate):
        equal = bool(torch.equal(expected, actual))
        all_equal &= equal
        item = {
            "equal": equal,
            "dtype": str(expected.dtype).removeprefix("torch."),
            "shape": list(expected.shape),
        }
        if not equal and expected.dtype == torch.bfloat16 and expected.numel():
            delta = expected.float() - actual.float()
            item.update(
                max_abs=float(delta.abs().max().item()),
                mean_abs=float(delta.abs().mean().item()),
            )
        elif not equal:
            item["different"] = int(torch.count_nonzero(expected != actual).item())
        if not equal:
            unequal_tensors[binding.name] = item
    return {
        "all_equal": all_equal,
        "equal_tensors": len(bindings) - len(unequal_tensors),
        "unequal_tensors": unequal_tensors,
    }


def main() -> None:
    args = parse_args()
    if args.frames < 1 or args.patch_equivalence_frames < 1:
        raise ValueError("frame counts must be positive")
    upstream_complete = RingKVCache.complete
    model = build_decoder(args.checkpoint, args.num_codebooks)
    bindings = decoder_state_bindings(model)
    wrapper = ExplicitMimiDecoderStep(model, bindings).eval()
    generator = torch.Generator(device="cuda").manual_seed(20260830)
    codes = torch.randint(
        0,
        2048,
        (args.frames, 1, args.num_codebooks, 1),
        device="cuda",
        dtype=torch.int64,
        generator=generator,
    )

    original_audio, original_state, original_seconds = run_pytorch_trajectory(
        model,
        wrapper,
        bindings,
        codes[: args.patch_equivalence_frames],
    )
    model.reset_streaming()
    make_ring_kv_update_exportable()
    patched_audio, patched_short_state, patched_short_seconds = (
        run_pytorch_trajectory(
            model,
            wrapper,
            bindings,
            codes[: args.patch_equivalence_frames],
        )
    )
    patch_audio_comparison = compare_audio(original_audio, patched_audio)
    patch_state_comparison = compare_state(
        bindings, original_state, patched_short_state
    )

    model.reset_streaming()
    patched_audio, patched_state, patched_seconds = run_pytorch_trajectory(
        model,
        wrapper,
        bindings,
        codes,
    )
    model.reset_streaming()
    runner = TensorRTMimiRunner(args.engine, bindings)
    runner.reset()
    candidate_audio = []
    started = time.perf_counter()
    with torch.no_grad():
        for frame_codes in codes:
            candidate_audio.append(runner.step(frame_codes).clone())
    torch.cuda.synchronize()
    tensorrt_seconds = time.perf_counter() - started
    candidate_state = tuple(value.clone() for value in runner.state())

    result = {
        "engine": str(args.engine),
        "engine_bytes": args.engine.stat().st_size,
        "frames": args.frames,
        "ring_capacity_steps": 125,
        "crossed_ring_wrap": args.frames > 125,
        "state_tensors": len(bindings),
        "patch_equivalence": {
            "frames": args.patch_equivalence_frames,
            "upstream_seconds": round(original_seconds, 3),
            "patched_seconds": round(patched_short_seconds, 3),
            "audio": patch_audio_comparison,
            "state": patch_state_comparison,
        },
        "tensorrt": {
            "pytorch_seconds": round(patched_seconds, 3),
            "tensorrt_seconds": round(tensorrt_seconds, 3),
            "pytorch_ms_per_frame": round(patched_seconds / args.frames * 1000, 3),
            "tensorrt_ms_per_frame": round(
                tensorrt_seconds / args.frames * 1000, 3
            ),
            "audio": compare_audio(patched_audio, candidate_audio),
            "state": compare_state(bindings, patched_state, candidate_state),
        },
    }
    if args.metrics is not None:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    RingKVCache.complete = upstream_complete


if __name__ == "__main__":
    main()
