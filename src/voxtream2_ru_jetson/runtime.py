#!/usr/bin/env python3
"""Run VoXtream2-RU utterances without importing PyTorch.

The acoustic prompt and its prefill cache are exported ahead of time. The text
is supplied at runtime and passes through the framework-free Russian frontend.
The resident API keeps the frontend and TensorRT engines loaded between turns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import resource
import sys
import time
import wave
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda import cudart

from .cuda_acoustic_control import CudaAcousticControl
from .cuda_audio_embedding import CudaAudioEmbeddingCore
from .frontend import TorchlessRussianFrontend
from .sink_attention import SinkAttentionConfig, SinkAttentionHistory
from .tensorrt_standalone import CudaArena, cuda_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument(
        "--text",
        help="Generate Russian text with the PyTorch-free normalization frontend.",
    )
    parser.add_argument("--ruaccent-assets", type=Path)
    parser.add_argument("--phone-map", type=Path)
    parser.add_argument("--espeak-executable", default="espeak-ng")
    parser.add_argument(
        "--text-normalizer",
        choices=("ru-normalizr", "none"),
        default="ru-normalizr",
        help="Written-to-spoken normalization before RUAccent (default: ru-normalizr).",
    )
    parser.add_argument(
        "--allow-unknown-phones",
        action="store_true",
        help="Map frontend OOV phones to UNK instead of failing the quality gate.",
    )
    parser.add_argument("--temp-engine", type=Path, required=True)
    parser.add_argument("--dep-engine", type=Path, required=True)
    parser.add_argument("--phone-engine", type=Path, required=True)
    parser.add_argument("--mimi-engine", type=Path, required=True)
    parser.add_argument("--mimi-state", type=Path, required=True)
    parser.add_argument("--audio-embedding-weight", type=Path, required=True)
    parser.add_argument("--audio-embedding-cubin", type=Path, required=True)
    parser.add_argument("--cuda-acoustic-control-cubin", type=Path)
    parser.add_argument(
        "--cuda-dep-graph",
        action="store_true",
        help="Capture the complete fixed acoustic dep_former chain in one CUDA Graph.",
    )
    parser.add_argument(
        "--cuda-temp-graph",
        action="store_true",
        help="Replay the one-token temporal decoder enqueue as a CUDA Graph.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument(
        "--teacher-force-reference",
        action="store_true",
        help="Use captured tokens only as next-step inputs and report model parity.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def bfloat16_to_float32(words: np.ndarray) -> np.ndarray:
    words = np.asarray(words, dtype=np.uint16)
    return (words.astype(np.uint32) << 16).view(np.float32)


def float32_to_bfloat16(value: np.ndarray) -> np.ndarray:
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    # IEEE round-to-nearest-even, matching CUDA/PyTorch BF16 conversion.
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded >> 16).astype(np.uint16)


def sanitize_tensor_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_")


def copy_to_device(pointer: int, value: np.ndarray, operation: str = "H2D") -> None:
    value = np.ascontiguousarray(value)
    if value.nbytes == 0:
        return
    cuda_check(
        cudart.cudaMemcpy(
            pointer,
            value.ctypes.data,
            value.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        ),
        f"cudaMemcpy({operation})",
    )


def download_array(pointer: int, dtype, shape: tuple[int, ...]) -> np.ndarray:
    output = np.empty(shape, dtype=dtype)
    if output.nbytes:
        cuda_check(
            cudart.cudaMemcpy(
                output.ctypes.data,
                pointer,
                output.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            ),
            "cudaMemcpy(D2H)",
        )
    return output


class RawBundle:
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("format") != "voxtream-torchless-fixed-utterance-v1":
            raise ValueError("unsupported torchless asset format")
        self.binary_path = self.manifest_path.parent / self.manifest["binary"]
        if self.binary_path.stat().st_size != int(self.manifest["binary_bytes"]):
            raise ValueError("torchless asset binary size mismatch")
        if file_sha256(self.binary_path) != self.manifest["binary_sha256"]:
            raise ValueError("torchless asset binary checksum mismatch")
        self.binary = np.memmap(self.binary_path, mode="r", dtype=np.uint8)
        self.specs = {item["name"]: item for item in self.manifest["tensors"]}

    def array(self, name: str) -> np.ndarray:
        item = self.specs[name]
        dtype = {
            "bfloat16": np.uint16,
            "float32": np.float32,
            "float16": np.float16,
            "int64": np.int64,
            "bool": np.bool_,
        }[item["dtype"]]
        start = int(item["offset"])
        stop = start + int(item["bytes"])
        payload = self.binary[start:stop]
        if hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise ValueError(f"asset checksum mismatch for {name}")
        return payload.view(dtype).reshape(tuple(item["shape"]))


class EngineOwner:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        payload = self.path.read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(payload)
        del payload
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {self.path}")


class PhoneEncoder:
    def __init__(self, path: Path, stream: int, arena: CudaArena) -> None:
        self.owner = EngineOwner(path)
        self.context = self.owner.engine.create_execution_context()
        self.stream = stream
        self.arena = arena
        profile_maxima = [
            self.owner.engine.get_tensor_profile_shape("phone_tokens", index)[2]
            for index in range(self.owner.engine.num_optimization_profiles)
        ]
        self.max_batch = max(int(shape[0]) for shape in profile_maxima)
        self.max_sequence = max(int(shape[1]) for shape in profile_maxima)
        self.pointers = {
            "phone_tokens": arena.allocate(self.max_batch * self.max_sequence * 8),
            "input_pos": arena.allocate(self.max_batch * self.max_sequence * 8),
            "mask": arena.allocate(self.max_batch * self.max_sequence * self.max_sequence),
            "phone_embeddings": arena.allocate(self.max_batch * self.max_sequence * 1024 * 2),
        }

    def run(self, tokens: np.ndarray, look_ahead: int = 30) -> np.ndarray:
        batch, sequence = tokens.shape
        if batch > self.max_batch or sequence > self.max_sequence:
            raise ValueError(
                "phone sequence exceeds TensorRT profile: "
                f"{tokens.shape} > ({self.max_batch}, {self.max_sequence})"
            )
        positions = np.repeat(np.arange(sequence, dtype=np.int64)[None], batch, axis=0)
        rows = positions[:, :, None]
        columns = np.arange(sequence, dtype=np.int64)[None, None]
        mask = (columns >= rows - 624) & (columns <= rows + look_ahead)
        inputs = {
            "phone_tokens": np.ascontiguousarray(tokens, dtype=np.int64),
            "input_pos": positions,
            "mask": np.ascontiguousarray(mask, dtype=np.bool_),
        }
        for name, value in inputs.items():
            if not self.context.set_input_shape(name, tuple(value.shape)):
                raise RuntimeError(f"failed to set phone shape {name}={value.shape}")
            copy_to_device(self.pointers[name], value)
        output_shape = tuple(self.context.get_tensor_shape("phone_embeddings"))
        for name, pointer in self.pointers.items():
            if not self.context.set_tensor_address(name, pointer):
                raise RuntimeError(f"failed to bind phone tensor {name}")
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("phone TensorRT enqueue failed")
        cuda_check(cudart.cudaStreamSynchronize(self.stream), "phone synchronize")
        return download_array(self.pointers["phone_embeddings"], np.uint16, output_shape)


class TempDecoder:
    def __init__(
        self, path: Path, bundle: RawBundle, stream: int, arena: CudaArena
    ) -> None:
        self.owner = EngineOwner(path)
        self.context = self.owner.engine.create_execution_context()
        self.stream = stream
        self.arena = arena
        self.state_names = tuple(bundle.manifest["prefill_buffer_names"])
        self.state = {}
        self.initial = {}
        self.empty_cache_pos = {}
        for name in self.state_names:
            value = np.ascontiguousarray(bundle.array(f"temp_state.{name}"))
            pointer = arena.allocate(max(value.nbytes, 1))
            copy_to_device(pointer, value)
            tensor_name = sanitize_tensor_name(name)
            self.state[tensor_name] = pointer
            self.initial[tensor_name] = value
            if value.dtype == np.int64:
                if not name.endswith("kv_cache.cache_pos"):
                    raise TypeError(f"unexpected temporal int64 state: {name}")
                empty = np.arange(value.size, dtype=np.int64).reshape(value.shape)
                template = arena.allocate(max(empty.nbytes, 1))
                copy_to_device(template, empty)
                self.empty_cache_pos[tensor_name] = (template, empty.nbytes)
        self.hidden = arena.allocate(2 * 1 * 1024 * 2)
        self.position = arena.allocate(2 * 1 * 8)
        self.mask = arena.allocate(2 * 1 * 2048)
        self.output = arena.allocate(2 * 1 * 1024 * 4)
        self.logits = arena.allocate(2 * 1 * 12300 * 2)
        bindings = {
            "hidden": self.hidden,
            "input_pos": self.position,
            "mask": self.mask,
            "output": self.output,
            "semantic_logits": self.logits,
        }
        for name, pointer in self.state.items():
            bindings[name] = pointer
            bindings[f"next_{name}"] = pointer
        for name, pointer in bindings.items():
            if not self.context.set_tensor_address(name, pointer):
                raise RuntimeError(f"failed to bind temp tensor {name}")
        self.calls = 0
        self.graph = None
        self.graph_exec = None
        self.graph_launches = 0
        self.sink_rebuilds = 0
        self.sink_replay_steps = 0
        self.sink_rebuild_seconds = 0.0
        config = bundle.manifest["config"]
        self.audio_window_size = int(
            config.get("audio_window_size", int(config["max_temp_position"]) + 1)
        )
        if "prefill.hidden" in bundle.specs:
            self.sink_attention = SinkAttentionHistory(
                SinkAttentionConfig(audio_window_size=self.audio_window_size),
                bundle.array("prefill.hidden"),
            )
        else:
            self.sink_attention = None

    def step(self, hidden_bf16: np.ndarray, position: int) -> tuple[np.ndarray, np.ndarray]:
        if self.sink_attention is None:
            if position >= self.audio_window_size:
                raise RuntimeError(
                    "torchless assets do not contain prefill.hidden required for "
                    "sink-attention cache rebuild"
                )
            local_position = position
        else:
            if self.sink_attention.needs_rebuild(position):
                self._rebuild_cache(position)
            local_position = self.sink_attention.local_position(position)
        if local_position >= self.audio_window_size:
            raise RuntimeError(
                "sink-attention cache rebuild did not bring the position into its window"
            )
        output, logits = self._execute(hidden_bf16, local_position, readback=True)
        if self.sink_attention is not None:
            self.sink_attention.append(hidden_bf16)
        return output, logits

    def _execute(
        self,
        hidden_bf16: np.ndarray,
        position: int,
        *,
        readback: bool,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        positions = np.full((2, 1), position, dtype=np.int64)
        mask = np.zeros((2, 1, 2048), dtype=np.bool_)
        mask[:, :, : position + 1] = True
        copy_to_device(self.hidden, hidden_bf16)
        copy_to_device(self.position, positions)
        copy_to_device(self.mask, mask)
        if self.graph_exec is not None:
            cuda_check(
                cudart.cudaGraphLaunch(self.graph_exec, self.stream),
                "temp cudaGraphLaunch",
            )
            self.graph_launches += 1
        elif not self.context.execute_async_v3(self.stream):
            raise RuntimeError("temp TensorRT enqueue failed")
        cuda_check(cudart.cudaStreamSynchronize(self.stream), "temp synchronize")
        self.calls += 1
        if not readback:
            return None
        return download_array(self.output, np.float32, (2, 1, 1024)), download_array(
            self.logits, np.uint16, (2, 1, 12300)
        )

    def _reset_empty_cache(self) -> None:
        # Every visible K/V slot is overwritten by the replay. Keeping words in
        # masked slots avoids copying 192 MiB only to clear them. cache_pos is
        # the state that controls where each replayed token is written.
        for name, (template, size) in self.empty_cache_pos.items():
            cuda_check(
                cudart.cudaMemcpy(
                    self.state[name],
                    template,
                    size,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                ),
                f"cudaMemcpy(D2D reset {name})",
            )

    def _rebuild_cache(self, global_position: int) -> None:
        if self.sink_attention is None:
            raise RuntimeError("sink-attention history is unavailable")
        started = time.perf_counter()
        replay_hidden = self.sink_attention.rebuild_sequence(global_position)
        self._reset_empty_cache()
        for position in range(replay_hidden.shape[1]):
            self._execute(
                replay_hidden[:, position : position + 1],
                position,
                readback=False,
            )
        self.sink_rebuilds += 1
        self.sink_replay_steps += int(replay_hidden.shape[1])
        self.sink_rebuild_seconds += time.perf_counter() - started

    def reset(self) -> None:
        for name, value in self.initial.items():
            copy_to_device(self.state[name], value)
        if self.sink_attention is not None:
            self.sink_attention.reset()

    def capture_graph(self) -> None:
        if self.graph_exec is not None:
            return
        cuda_check(
            cudart.cudaMemsetAsync(self.hidden, 0, 2 * 1 * 1024 * 2, self.stream),
            "cudaMemsetAsync(temp graph warm-up hidden)",
        )
        cuda_check(
            cudart.cudaMemsetAsync(self.position, 0, 2 * 1 * 8, self.stream),
            "cudaMemsetAsync(temp graph warm-up position)",
        )
        cuda_check(
            cudart.cudaMemsetAsync(self.mask, 0, 2 * 1 * 2048, self.stream),
            "cudaMemsetAsync(temp graph warm-up mask)",
        )
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("temp TensorRT graph warm-up enqueue failed")
        cuda_check(
            cudart.cudaStreamSynchronize(self.stream),
            "temp CUDA Graph warm-up synchronize",
        )
        cuda_check(
            cudart.cudaStreamBeginCapture(
                self.stream,
                cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal,
            ),
            "temp cudaStreamBeginCapture",
        )
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("temp TensorRT graph capture enqueue failed")
        self.graph = cuda_check(
            cudart.cudaStreamEndCapture(self.stream),
            "temp cudaStreamEndCapture",
        )
        self.graph_exec = cuda_check(
            cudart.cudaGraphInstantiate(self.graph, 0),
            "temp cudaGraphInstantiate",
        )
        self.reset()

    def close(self) -> None:
        if self.graph_exec is not None:
            cuda_check(
                cudart.cudaGraphExecDestroy(self.graph_exec),
                "temp cudaGraphExecDestroy",
            )
            self.graph_exec = None
        if self.graph is not None:
            cuda_check(
                cudart.cudaGraphDestroy(self.graph),
                "temp cudaGraphDestroy",
            )
            self.graph = None


class DepDecoder:
    def __init__(self, path: Path, stream: int, arena: CudaArena) -> None:
        self.owner = EngineOwner(path)
        self.engine = self.owner.engine
        self.stream = stream
        self.arena = arena
        self.q1_profile = self._profile_for(1)
        self.q2_profile = self._profile_for(2)
        self.contexts = {
            1: self._context(self.q1_profile, 1),
            2: self._context(self.q2_profile, 2),
        }
        input_names = {
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index))
            == trt.TensorIOMode.INPUT
        }
        self.state_names = tuple(sorted(input_names - {"hidden", "input_pos", "mask"}))
        self.banks: list[dict[str, int]] = [{}, {}]
        self.initial: dict[str, np.ndarray] = {}
        self.reset_templates: dict[str, int] = {}
        for name in self.state_names:
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            if dtype == trt.int64:
                value = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
            elif dtype == trt.bfloat16:
                value = np.zeros(shape, dtype=np.uint16)
            else:
                raise TypeError(f"unsupported dep state dtype {dtype} for {name}")
            self.initial[name] = value
            if dtype == trt.int64:
                template = arena.allocate(value.nbytes)
                copy_to_device(template, value)
                self.reset_templates[name] = template
            for bank in self.banks:
                bank[name] = arena.allocate(max(value.nbytes, 1))
        self.inputs = {
            q: {
                "hidden": arena.allocate(2 * q * 1024 * 2),
                "input_pos": arena.allocate(2 * q * 8),
                "mask": arena.allocate(2 * q * 16),
            }
            for q in (1, 2)
        }
        self.output = {q: arena.allocate(2 * 1 * 1024 * 4) for q in (1, 2)}
        self.logits = {q: arena.allocate(2 * 2050 * 2) for q in (1, 2)}
        self.static_steps: dict[int, tuple[int, int]] = {}
        for position in range(2, 16):
            positions = np.full((2, 1), position, dtype=np.int64)
            mask = np.zeros((2, 1, 16), dtype=np.bool_)
            mask[:, :, : position + 1] = True
            position_pointer = arena.allocate(positions.nbytes)
            mask_pointer = arena.allocate(mask.nbytes)
            copy_to_device(position_pointer, positions)
            copy_to_device(mask_pointer, mask)
            self.static_steps[position] = (position_pointer, mask_pointer)
        q2_positions = np.repeat(np.asarray([[0, 1]], dtype=np.int64), 2, axis=0)
        q2_mask = np.zeros((2, 2, 16), dtype=np.bool_)
        q2_mask[:, 0, :1] = True
        q2_mask[:, 1, :2] = True
        copy_to_device(self.inputs[2]["input_pos"], q2_positions)
        copy_to_device(self.inputs[2]["mask"], q2_mask)
        self.frame_codes = arena.allocate(16 * 8)
        self.bank = 0
        self.calls = {1: 0, 2: 0}
        self.graph = None
        self.graph_exec = None
        self.graph_launches = 0
        self.reset()

    def _profile_for(self, sequence: int) -> int:
        matches = []
        for index in range(self.engine.num_optimization_profiles):
            minimum, optimum, maximum = self.engine.get_tensor_profile_shape("hidden", index)
            if minimum[1] <= sequence <= maximum[1] and optimum[1] == sequence:
                matches.append(index)
        if not matches:
            raise RuntimeError(f"dep engine has no profile for q={sequence}")
        return matches[0]

    def _context(self, profile: int, sequence: int):
        context = self.engine.create_execution_context()
        if not context.set_optimization_profile_async(profile, self.stream):
            raise RuntimeError(f"failed to select dep profile {profile}")
        for name, shape in {
            "hidden": (2, sequence, 1024),
            "input_pos": (2, sequence),
            "mask": (2, sequence, 16),
        }.items():
            if not context.set_input_shape(name, shape):
                raise RuntimeError(f"failed to set dep shape {name}={shape}")
        return context

    def reset(self) -> None:
        for bank in self.banks:
            for name, value in self.initial.items():
                copy_to_device(bank[name], value)
        self.bank = 0

    def reset_async(self) -> None:
        for bank in self.banks:
            for name, value in self.initial.items():
                if value.dtype == np.uint16:
                    # The attention mask makes unwritten K/V slots invisible,
                    # while every visible slot is replaced in sequence by q2
                    # and q1. Only cache_pos carries validity across frames.
                    # Keeping the old words avoids 16 graph memset nodes and
                    # about 1 MiB of redundant device writes per audio frame.
                    continue
                else:
                    cuda_check(
                        cudart.cudaMemcpyAsync(
                            bank[name],
                            self.reset_templates[name],
                            value.nbytes,
                            cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                            self.stream,
                        ),
                        f"cudaMemcpyAsync(D2D reset {name})",
                    )
        self.bank = 0

    def step(self, hidden_bf16: np.ndarray, positions: np.ndarray) -> np.ndarray:
        sequence = int(hidden_bf16.shape[1])
        context = self.contexts[sequence]
        positions = np.repeat(np.asarray(positions, dtype=np.int64)[None], 2, axis=0)
        mask = np.zeros((2, sequence, 16), dtype=np.bool_)
        for column, position in enumerate(positions[0]):
            mask[:, column, : int(position) + 1] = True
        for name, value in {
            "hidden": hidden_bf16,
            "input_pos": positions,
            "mask": mask,
        }.items():
            copy_to_device(self.inputs[sequence][name], value)
        source = self.banks[self.bank]
        target = self.banks[self.bank ^ 1]
        bindings = {
            **self.inputs[sequence],
            "output": self.output[sequence],
            "acoustic_logits": self.logits[sequence],
        }
        for name in self.state_names:
            bindings[name] = source[name]
            bindings[f"next_{name}"] = target[name]
        for name, pointer in bindings.items():
            if not context.set_tensor_address(name, pointer):
                raise RuntimeError(f"failed to bind dep tensor {name}")
        if not context.execute_async_v3(self.stream):
            raise RuntimeError("dep TensorRT enqueue failed")
        cuda_check(cudart.cudaStreamSynchronize(self.stream), "dep synchronize")
        self.bank ^= 1
        self.calls[sequence] += 1
        return download_array(self.logits[sequence], np.uint16, (2, 2050))

    def _enqueue_device(
        self,
        sequence: int,
        hidden_pointer: int,
        position_pointer: int,
        mask_pointer: int,
    ) -> int:
        context = self.contexts[sequence]
        source = self.banks[self.bank]
        target = self.banks[self.bank ^ 1]
        bindings = {
            "hidden": hidden_pointer,
            "input_pos": position_pointer,
            "mask": mask_pointer,
            "output": self.output[sequence],
            "acoustic_logits": self.logits[sequence],
        }
        for name in self.state_names:
            bindings[name] = source[name]
            bindings[f"next_{name}"] = target[name]
        for name, pointer in bindings.items():
            if not context.set_tensor_address(name, pointer):
                raise RuntimeError(f"failed to bind dep tensor {name}")
        if not context.execute_async_v3(self.stream):
            raise RuntimeError("dep TensorRT enqueue failed")
        self.bank ^= 1
        self.calls[sequence] += 1
        return self.logits[sequence]

    def generate_acoustic_cuda(
        self,
        dep_hidden: np.ndarray,
        semantic: int,
        cfg_gamma: float,
        controller: CudaAcousticControl,
        embedding_weight_pointer: int,
    ) -> np.ndarray:
        initial_codes = np.zeros(16, dtype=np.int64)
        initial_codes[0] = int(semantic)
        copy_to_device(self.frame_codes, initial_codes)
        copy_to_device(self.inputs[2]["hidden"], dep_hidden)
        if self.graph_exec is not None:
            cuda_check(
                cudart.cudaGraphLaunch(self.graph_exec, self.stream),
                "dep acoustic cudaGraphLaunch",
            )
            cuda_check(
                cudart.cudaStreamSynchronize(self.stream),
                "dep acoustic CUDA Graph synchronize",
            )
            self.graph_launches += 1
            self.calls[2] += 1
            self.calls[1] += 14
            controller.calls += 15
            return download_array(self.frame_codes, np.int64, (16,))

        self._enqueue_acoustic_chain(
            cfg_gamma, controller, embedding_weight_pointer
        )
        cuda_check(
            cudart.cudaStreamSynchronize(self.stream),
            "fused dep acoustic chain synchronize",
        )
        return download_array(self.frame_codes, np.int64, (16,))

    def _enqueue_acoustic_chain(
        self,
        cfg_gamma: float,
        controller: CudaAcousticControl,
        embedding_weight_pointer: int,
    ) -> None:
        self.reset_async()
        logits_pointer = self._enqueue_device(
            2,
            self.inputs[2]["hidden"],
            self.inputs[2]["input_pos"],
            self.inputs[2]["mask"],
        )
        for codebook in range(1, 16):
            controller.launch(
                logits_pointer,
                embedding_weight_pointer,
                self.frame_codes,
                codebook,
                cfg_gamma,
                self.inputs[1]["hidden"],
                self.stream,
            )
            if codebook < 15:
                position_pointer, mask_pointer = self.static_steps[codebook + 1]
                logits_pointer = self._enqueue_device(
                    1,
                    self.inputs[1]["hidden"],
                    position_pointer,
                    mask_pointer,
                )

    def capture_acoustic_graph(
        self,
        cfg_gamma: float,
        controller: CudaAcousticControl,
        embedding_weight_pointer: int,
    ) -> None:
        if self.graph_exec is not None:
            return

        # TensorRT requires every execution context to be enqueued once after
        # its shapes and addresses are configured and before stream capture.
        # The chain resets both state banks at its start, so this discarded
        # warm-up cannot affect the graph replay or the first generated frame.
        cuda_check(
            cudart.cudaMemsetAsync(
                self.inputs[2]["hidden"],
                0,
                2 * 2 * 1024 * 2,
                self.stream,
            ),
            "cudaMemsetAsync(dep graph warm-up hidden)",
        )
        cuda_check(
            cudart.cudaMemsetAsync(self.frame_codes, 0, 16 * 8, self.stream),
            "cudaMemsetAsync(dep graph warm-up codes)",
        )
        saved_calls = dict(self.calls)
        saved_control_calls = controller.calls
        self._enqueue_acoustic_chain(
            cfg_gamma, controller, embedding_weight_pointer
        )
        cuda_check(
            cudart.cudaStreamSynchronize(self.stream),
            "dep acoustic graph warm-up synchronize",
        )

        cuda_check(
            cudart.cudaStreamBeginCapture(
                self.stream,
                cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal,
            ),
            "dep acoustic cudaStreamBeginCapture",
        )
        self._enqueue_acoustic_chain(
            cfg_gamma, controller, embedding_weight_pointer
        )
        self.graph = cuda_check(
            cudart.cudaStreamEndCapture(self.stream),
            "dep acoustic cudaStreamEndCapture",
        )
        self.graph_exec = cuda_check(
            cudart.cudaGraphInstantiate(self.graph, 0),
            "dep acoustic cudaGraphInstantiate",
        )
        self.bank = 0
        self.calls = saved_calls
        controller.calls = saved_control_calls

    def close(self) -> None:
        if self.graph_exec is not None:
            cuda_check(
                cudart.cudaGraphExecDestroy(self.graph_exec),
                "dep acoustic cudaGraphExecDestroy",
            )
            self.graph_exec = None
        if self.graph is not None:
            cuda_check(
                cudart.cudaGraphDestroy(self.graph),
                "dep acoustic cudaGraphDestroy",
            )
            self.graph = None


class MimiDecoder:
    def __init__(
        self, engine_path: Path, state_path: Path, stream: int, arena: CudaArena
    ) -> None:
        self.owner = EngineOwner(engine_path)
        self.engine = self.owner.engine
        self.context = self.engine.create_execution_context()
        self.stream = stream
        self.arena = arena
        self.manifest = json.loads(Path(state_path).read_text())
        if self.manifest.get("format") != "voxtream-mimi-decoder-state-v1":
            raise ValueError("unsupported Mimi state format")
        binary_path = Path(state_path).parent / self.manifest["binary"]
        payload = binary_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != self.manifest["binary_sha256"]:
            raise ValueError("Mimi state checksum mismatch")
        self.state_names = tuple(item["name"] for item in self.manifest["tensors"])
        self.initial = {}
        for item in self.manifest["tensors"]:
            start = int(item["offset"])
            stop = start + int(item["bytes"])
            self.initial[item["name"]] = np.frombuffer(payload[start:stop], dtype=np.uint8)
        input_names = {
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index))
            == trt.TensorIOMode.INPUT
        }
        output_names = {
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index))
            == trt.TensorIOMode.OUTPUT
        }
        self.input_names = {}
        self.output_names = {}
        for name in self.state_names:
            candidates = [item for item in (name, f"__next_{name}") if item in input_names]
            if len(candidates) != 1 or f"next_{name}" not in output_names:
                raise RuntimeError(f"cannot resolve Mimi state ABI for {name}")
            self.input_names[name] = candidates[0]
            self.output_names[name] = f"next_{name}"
        self.banks = [{}, {}]
        for name, value in self.initial.items():
            for bank in self.banks:
                bank[name] = arena.allocate(max(value.nbytes, 1))
        self.codes = arena.allocate(1 * 16 * 1 * 8)
        self.audio = arena.allocate(1 * 1 * 1920 * 2)
        self.bank = 0
        self.calls = 0
        self.reset()

    def reset(self) -> None:
        for bank in self.banks:
            for name, value in self.initial.items():
                copy_to_device(bank[name], value)
        self.bank = 0

    def decode(self, codes: np.ndarray) -> np.ndarray:
        codes = np.ascontiguousarray(codes, dtype=np.int64).reshape(1, 16, 1)
        copy_to_device(self.codes, codes)
        source = self.banks[self.bank]
        target = self.banks[self.bank ^ 1]
        bindings = {"codes": self.codes, "audio": self.audio}
        for name in self.state_names:
            bindings[self.input_names[name]] = source[name]
            bindings[self.output_names[name]] = target[name]
        for name, pointer in bindings.items():
            if not self.context.set_tensor_address(name, pointer):
                raise RuntimeError(f"failed to bind Mimi tensor {name}")
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("Mimi TensorRT enqueue failed")
        cuda_check(cudart.cudaStreamSynchronize(self.stream), "Mimi synchronize")
        self.bank ^= 1
        self.calls += 1
        return bfloat16_to_float32(
            download_array(self.audio, np.uint16, (1, 1, 1920))
        ).reshape(1920)


class AudioEmbeddings:
    def __init__(
        self,
        weight: Path,
        cubin: Path,
        stream: int,
        arena: CudaArena,
    ) -> None:
        self.core = CudaAudioEmbeddingCore(weight, cubin, 16 * 2050, 1024)
        self.stream = stream
        self.indices = arena.allocate(2 * 16 * 8)
        self.output = arena.allocate(2 * 16 * 1024 * 2)

    def gather(self, tokens: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
        tokens = np.asarray(tokens, dtype=np.int64)
        codebooks = np.asarray(codebooks, dtype=np.int64)
        shifted = np.ascontiguousarray(tokens + codebooks * 2050, dtype=np.int64)
        count = int(shifted.size)
        copy_to_device(self.indices, shifted)
        self.core.launch(self.indices, self.output, count, self.stream)
        cuda_check(cudart.cudaStreamSynchronize(self.stream), "embedding synchronize")
        return download_array(self.output, np.uint16, (*shifted.shape, 1024))

    def close(self) -> None:
        self.core.close()


def stable_remove_rows(data: np.ndarray, indices: np.ndarray) -> np.ndarray:
    output = np.empty_like(data)
    for batch in range(data.shape[0]):
        removed = {int(value) for value in indices[batch] if 0 <= value < data.shape[1]}
        keep = [index for index in range(data.shape[1]) if index not in removed]
        output[batch, : len(keep)] = data[batch, keep]
        output[batch, len(keep) :] = data[batch, -1]
    return output


def softmax(value: np.ndarray) -> np.ndarray:
    value = value - np.max(value)
    exponentials = np.exp(value)
    return exponentials / exponentials.sum()


def categorical(probs: np.ndarray, rng: np.random.Generator) -> int:
    return int(np.argmax(probs / rng.exponential(size=probs.shape)))


def sample_semantic(
    logits_words: np.ndarray, config: dict[str, object], rng: np.random.Generator
) -> tuple[int, int]:
    logits = bfloat16_to_float32(logits_words).reshape(2, 6, 2050)
    conditional = logits[0]
    cfg = float(config["cfg_gamma"]) * logits[0] + (
        1.0 - float(config["cfg_gamma"])
    ) * logits[1]
    maximum = conditional.max(axis=-1)
    state_logits = maximum + np.log(
        np.exp(conditional - maximum[:, None]).sum(axis=-1)
    )
    state_probs = softmax(state_logits / float(config["temperature"]))
    order = np.argsort(state_probs)[::-1]
    ordered = state_probs[order]
    keep = np.cumsum(ordered) - ordered <= float(config["top_p"])
    filtered = ordered * keep
    filtered /= filtered.sum()
    state = int(order[categorical(filtered, rng)])
    semantic_probs = softmax(cfg[state] / float(config["temperature"]))
    top_k = int(config["top_k"])
    top_indices = np.argpartition(semantic_probs, -top_k)[-top_k:]
    semantic = int(top_indices[categorical(semantic_probs[top_indices], rng)])
    return semantic, state


def acoustic_argmax(logits_words: np.ndarray, gamma: float) -> int:
    logits = bfloat16_to_float32(logits_words)
    cfg = gamma * logits[0] + (1.0 - gamma) * logits[1]
    return int(np.argmax(cfg))


class FrameState:
    def __init__(self, prompt_phone_indices: np.ndarray, max_frames: int) -> None:
        self.indices = prompt_phone_indices
        self.maximum = int(prompt_phone_indices[0, -1, -1])
        self.eos_index = max_frames
        self.counts: dict[tuple[int, int], int] = {}
        self.dwell_start = self.maximum
        self.dwell_count = 0


def update_frame_state(
    state: FrameState,
    predicted_shift: int,
    frame_index: int,
    phone_sequence_length: int,
    config: dict[str, object],
) -> None:
    shift, count = config["phoneme_index_map"][str(predicted_shift)]
    start = state.maximum + int(shift)
    key = (start, start + int(count))
    prior = state.counts.get(key, 0)
    repeat_limit = int(config["frame_repeat_counter"])
    if prior:
        if start >= phone_sequence_length - 2 and prior == 3:
            start += 1
            key = (start, start + int(count))
        elif prior > repeat_limit:
            start += 1
            key = (start, start + int(count))
    dwell = state.dwell_count + 1 if start == state.dwell_start else 1
    if dwell > repeat_limit and start < phone_sequence_length:
        start += 1
        key = (start, start + int(count))
        dwell = 1
    if start >= phone_sequence_length:
        token = min(start, phone_sequence_length + 1)
        values = [token, token]
        state.eos_index = frame_index
    else:
        values = list(range(start, start + int(count)))
        while len(values) < 2:
            values.append(values[-1])
    state.indices = np.repeat(
        np.asarray(values, dtype=np.int64).reshape(1, 1, 2), 2, axis=0
    )
    state.maximum = values[-1]
    state.counts[key] = state.counts.get(key, 0) + 1
    state.dwell_start = start
    state.dwell_count = dwell


@dataclass(frozen=True)
class RuntimeFiles:
    assets: Path
    temp_engine: Path
    dep_engine: Path
    phone_engine: Path
    mimi_engine: Path
    mimi_state: Path
    audio_embedding_weight: Path
    audio_embedding_cubin: Path
    cuda_acoustic_control_cubin: Path | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RuntimeFiles":
        return cls(
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


class SynthesisStream(Iterator[bytes]):
    """One utterance as raw mono 24 kHz signed 16-bit PCM chunks."""

    def __init__(
        self,
        iterator: Generator[bytes, None, dict[str, object]],
        release,
    ) -> None:
        self._iterator = iterator
        self._release = release
        self.result: dict[str, object] | None = None
        self.closed = False

    def __iter__(self) -> "SynthesisStream":
        return self

    def __enter__(self) -> "SynthesisStream":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __next__(self) -> bytes:
        if self.closed:
            raise StopIteration
        try:
            return next(self._iterator)
        except StopIteration as finished:
            self.closed = True
            self.result = finished.value
            self._release()
            raise
        except BaseException:
            self.closed = True
            self._release()
            raise

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._iterator.close()
        self._release()


class SynthesisRuntime:
    """Resident PyTorch-free frontend and TensorRT execution state."""

    def __init__(
        self,
        files: RuntimeFiles,
        *,
        ruaccent_assets: Path | None = None,
        phone_map: Path | None = None,
        espeak_executable: str = "espeak-ng",
        text_normalizer: str = "ru-normalizr",
        cuda_temp_graph: bool = False,
        cuda_dep_graph: bool = False,
    ) -> None:
        if "torch" in sys.modules:
            raise RuntimeError("PyTorch was imported before torchless runtime startup")
        if (ruaccent_assets is None) != (phone_map is None):
            raise ValueError("ruaccent_assets and phone_map must be supplied together")

        started = time.perf_counter()
        self.files = files
        self.bundle = RawBundle(files.assets)
        self.config = self.bundle.manifest["config"]
        self.frontend = None
        self.frontend_load_seconds = 0.0
        if ruaccent_assets is not None and phone_map is not None:
            frontend_started = time.perf_counter()
            self.frontend = TorchlessRussianFrontend(
                ruaccent_assets,
                phone_map,
                espeak_executable,
                text_normalizer,
            )
            self.frontend_load_seconds = time.perf_counter() - frontend_started
        if "torch" in sys.modules:
            raise RuntimeError("PyTorch entered sys.modules through the text frontend")

        self.stream_handle = cuda_check(cudart.cudaStreamCreate(), "cudaStreamCreate")
        self.cuda_stream = int(self.stream_handle)
        self.arena = CudaArena()
        self.embeddings = None
        self.acoustic_control = None
        self.temp = None
        self.dep = None
        self.mimi = None
        self.closed = False
        self._active = False
        self.request_count = 0
        try:
            self.phone = PhoneEncoder(files.phone_engine, self.cuda_stream, self.arena)
            self.temp = TempDecoder(files.temp_engine, self.bundle, self.cuda_stream, self.arena)
            self.dep = DepDecoder(files.dep_engine, self.cuda_stream, self.arena)
            self.mimi = MimiDecoder(
                files.mimi_engine,
                files.mimi_state,
                self.cuda_stream,
                self.arena,
            )
            self.embeddings = AudioEmbeddings(
                files.audio_embedding_weight,
                files.audio_embedding_cubin,
                self.cuda_stream,
                self.arena,
            )
            if files.cuda_acoustic_control_cubin is not None:
                self.acoustic_control = CudaAcousticControl(files.cuda_acoustic_control_cubin)
            if cuda_temp_graph:
                self.temp.capture_graph()
            if cuda_dep_graph:
                if self.acoustic_control is None:
                    raise ValueError("cuda_dep_graph requires cuda_acoustic_control_cubin")
                self.dep.capture_acoustic_graph(
                    float(self.config["cfg_ac_gamma"]),
                    self.acoustic_control,
                    self.embeddings.core.weight_pointer,
                )
        except BaseException:
            self._close_resources()
            raise

        self.projected_speaker = bfloat16_to_float32(self.bundle.array("speaker.projected"))
        self.prompt_indices = np.asarray(self.bundle.array("prompt.phone_indices")).copy()
        self.prompt_frames = int(self.bundle.manifest["prompt_frames"])
        cached_output = self.bundle.array("prefill.output")
        if self.bundle.specs["prefill.output"]["dtype"] == "bfloat16":
            self.initial_hidden_words = np.asarray(cached_output[:, -1]).copy()
        else:
            self.initial_hidden_words = float32_to_bfloat16(cached_output[:, -1])
        self.initial_semantic_logits = (
            np.asarray(self.bundle.array("prefill.semantic_logits"))
            .reshape(2, -1)[..., :12300]
            .copy()
        )
        self.startup_seconds = time.perf_counter() - started

    def __enter__(self) -> "SynthesisRuntime":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @property
    def sample_rate(self) -> int:
        return int(self.config["sample_rate"])

    @property
    def samples_per_chunk(self) -> int:
        return int(self.config["samples_per_frame"])

    def _release_request(self) -> None:
        self._active = False

    def _close_resources(self) -> None:
        if self.temp is not None:
            self.temp.close()
            self.temp = None
        if self.dep is not None:
            self.dep.close()
            self.dep = None
        if self.acoustic_control is not None:
            self.acoustic_control.close()
            self.acoustic_control = None
        if self.embeddings is not None:
            self.embeddings.close()
            self.embeddings = None
        if hasattr(self, "arena"):
            self.arena.close()
        if hasattr(self, "stream_handle"):
            cuda_check(
                cudart.cudaStreamDestroy(self.stream_handle),
                "cudaStreamDestroy",
            )

    def close(self) -> None:
        if self.closed:
            return
        if self._active:
            raise RuntimeError("cannot close runtime while a PCM stream is active")
        self._close_resources()
        self.frontend = None
        self.closed = True

    def synthesize_stream(
        self,
        text: str | None,
        *,
        seed: int = 20260830,
        max_frames: int = 256,
        allow_unknown_phones: bool = False,
        teacher_force_reference: bool = False,
        include_trajectory: bool = True,
    ) -> SynthesisStream:
        if self.closed:
            raise RuntimeError("synthesis runtime is closed")
        if self._active:
            raise RuntimeError("synthesis runtime already has an active stream")
        if text is not None and teacher_force_reference:
            raise ValueError("teacher forcing cannot be used with dynamic text")
        if text is not None and self.frontend is None:
            raise ValueError("dynamic text requires resident RUAccent assets and a phone map")
        self._active = True
        request_index = self.request_count
        self.request_count += 1
        iterator = self._generate(
            text=text,
            seed=seed,
            max_frames=max_frames,
            allow_unknown_phones=allow_unknown_phones,
            teacher_force_reference=teacher_force_reference,
            include_trajectory=include_trajectory,
            request_index=request_index,
        )
        return SynthesisStream(iterator, self._release_request)

    def _generate(
        self,
        *,
        text: str | None,
        seed: int,
        max_frames: int,
        allow_unknown_phones: bool,
        teacher_force_reference: bool,
        include_trajectory: bool,
        request_index: int,
    ) -> Generator[bytes, None, dict[str, object]]:
        request_started = time.perf_counter()
        frontend_metrics = None
        frontend_result = None
        try:
            if text is not None:
                frontend_started = time.perf_counter()
                prompt_len = int(self.bundle.manifest["prompt_phone_len"])
                prompt_prefix = self.bundle.array("phone.tokens")[0, :prompt_len]
                frontend_result = self.frontend.prepare(text, prompt_prefix)
                frontend_finished = time.perf_counter()
                if frontend_result.unknown_phones and not allow_unknown_phones:
                    raise ValueError(
                        "frontend produced phones outside the model vocabulary: "
                        f"{frontend_result.unknown_phones}"
                    )
                frontend_metrics = {
                    "backend": (f"{self.frontend.text_normalizer_backend}+ruaccent-onnx+espeak-ng"),
                    "normalized_text": frontend_result.normalized_text,
                    "accented_text": frontend_result.accented_text,
                    "phonemes": frontend_result.phonemes,
                    "unknown_phones": list(frontend_result.unknown_phones),
                    "phone_tokens_shape": list(frontend_result.phone_tokens.shape),
                    "phone_seq_len": frontend_result.phone_seq_len,
                    "resident_load_seconds": round(self.frontend_load_seconds, 3),
                    "process_seconds": round(frontend_finished - frontend_started, 3),
                    "reused": request_index > 0,
                    "peak_rss_mib": round(
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                        1,
                    ),
                }

            self.temp.reset()
            self.dep.reset()
            self.mimi.reset()
            call_start = {
                "temp": self.temp.calls,
                "temp_cuda_graph": self.temp.graph_launches,
                "temp_sink_rebuilds": self.temp.sink_rebuilds,
                "temp_sink_replay_steps": self.temp.sink_replay_steps,
                "temp_sink_rebuild_seconds": self.temp.sink_rebuild_seconds,
                "dep_q2": self.dep.calls[2],
                "dep_q1": self.dep.calls[1],
                "mimi": self.mimi.calls,
                "audio_embedding": self.embeddings.core.calls,
                "cuda_acoustic_control": (
                    self.acoustic_control.calls if self.acoustic_control is not None else 0
                ),
                "dep_cuda_graph": self.dep.graph_launches,
            }

            if frontend_result is None:
                phone_tokens = self.bundle.array("phone.tokens")
                punctuation = self.bundle.array("phone.punctuation_indices")
                phone_sequence_length = int(self.bundle.manifest["phone_seq_len"])
                result_text = str(self.bundle.manifest["text"])
            else:
                phone_tokens = frontend_result.phone_tokens
                punctuation = frontend_result.punctuation_indices
                phone_sequence_length = frontend_result.phone_seq_len
                result_text = text
            phone_embeddings = stable_remove_rows(self.phone.run(phone_tokens), punctuation)

            state = FrameState(self.prompt_indices, max_frames)
            position = self.prompt_frames - 1
            last_hidden_words = self.initial_hidden_words.copy()
            semantic_logits = self.initial_semantic_logits.copy()
            rng = np.random.default_rng(seed)
            generation_limit = max_frames
            if teacher_force_reference:
                if not self.bundle.manifest.get("reference_matches_text", False):
                    raise ValueError("asset bundle has no reference for this text")
                reference_codes = self.bundle.array("reference.mimi_codes")[
                    :, :, self.prompt_frames :
                ]
                reference_shifts = self.bundle.array("reference.pred_shifts")
                generation_limit = min(generation_limit, reference_codes.shape[2])
            else:
                reference_codes = None
                reference_shifts = None

            trajectory: list[dict[str, object]] = []
            teacher_forced_comparisons: list[dict[str, object]] = []
            frame_times: list[float] = []
            stage_seconds = {
                "temp_input": 0.0,
                "temp_engine": 0.0,
                "semantic_and_q2_input": 0.0,
                "dep_frame": 0.0,
                "mimi": 0.0,
            }
            audio_frames = 0
            first_audio_seconds = None
            first_audio_request_seconds = None
            previous_semantic = None
            frame = None
            generation_started = time.perf_counter()

            for frame_index in range(generation_limit):
                frame_started = time.perf_counter()
                if frame_index:
                    if frame is None:
                        raise RuntimeError("previous acoustic frame is missing")
                    stage_started = time.perf_counter()
                    position += 1
                    if state.maximum >= phone_embeddings.shape[1]:
                        raise RuntimeError("phone embedding index out of range")
                    phone_chunk = np.empty((2, 1024), dtype=np.float32)
                    for batch in range(2):
                        selected = phone_embeddings[batch, state.indices[batch, 0]]
                        phone_chunk[batch] = bfloat16_to_float32(selected).sum(axis=0)
                    audio_tokens = np.repeat(
                        np.asarray(frame, dtype=np.int64).reshape(1, 16),
                        2,
                        axis=0,
                    )
                    audio_words = self.embeddings.gather(
                        audio_tokens,
                        np.arange(16, dtype=np.int64)[None],
                    )
                    hidden = phone_chunk + bfloat16_to_float32(audio_words).sum(axis=1)
                    hidden_words = float32_to_bfloat16(hidden).reshape(2, 1, 1024)
                    stage_seconds["temp_input"] += time.perf_counter() - stage_started
                    stage_started = time.perf_counter()
                    temp_output, semantic_logits = self.temp.step(hidden_words, position)
                    last_hidden_words = float32_to_bfloat16(temp_output[:, -1])
                    stage_seconds["temp_engine"] += time.perf_counter() - stage_started

                stage_started = time.perf_counter()
                semantic, predicted_shift = sample_semantic(semantic_logits, self.config, rng)
                reference_frame = None
                if teacher_force_reference:
                    reference_frame = reference_codes[0, :, frame_index]
                    semantic = int(reference_frame[0])
                    predicted_shift = int(reference_shifts[frame_index])
                speaker_hidden = bfloat16_to_float32(last_hidden_words) + (
                    self.projected_speaker * float(self.config["spk_proj_weight"])
                )
                speaker_hidden_words = float32_to_bfloat16(speaker_hidden)
                code_word = self.embeddings.gather(
                    np.asarray([[semantic]], dtype=np.int64),
                    np.asarray([[0]], dtype=np.int64),
                )[0, 0]
                code_words = np.repeat(code_word[None], 2, axis=0)
                dep_hidden = np.stack([speaker_hidden_words, code_words], axis=1)
                stage_seconds["semantic_and_q2_input"] += time.perf_counter() - stage_started

                stage_started = time.perf_counter()
                if self.acoustic_control is not None:
                    frame = self.dep.generate_acoustic_cuda(
                        dep_hidden,
                        semantic,
                        float(self.config["cfg_ac_gamma"]),
                        self.acoustic_control,
                        self.embeddings.core.weight_pointer,
                    )
                else:
                    self.dep.reset()
                    frame = [semantic]
                    acoustic_logits = self.dep.step(dep_hidden, np.asarray([0, 1]))
                    for codebook in range(1, 16):
                        token = acoustic_argmax(
                            acoustic_logits,
                            float(self.config["cfg_ac_gamma"]),
                        )
                        frame.append(token)
                        if codebook < 15:
                            word = self.embeddings.gather(
                                np.asarray([[token]], dtype=np.int64),
                                np.asarray([[codebook]], dtype=np.int64),
                            )[0, 0]
                            hidden_words = np.repeat(word.reshape(1, 1, 1024), 2, axis=0)
                            acoustic_logits = self.dep.step(
                                hidden_words,
                                np.asarray([codebook + 1]),
                            )
                stage_seconds["dep_frame"] += time.perf_counter() - stage_started
                generated_frame = np.asarray(frame, dtype=np.int64)
                if reference_frame is not None:
                    teacher_forced_comparisons.append(
                        {
                            "index": frame_index,
                            "acoustic_tokens_equal": bool(
                                np.array_equal(
                                    generated_frame[1:],
                                    reference_frame[1:],
                                )
                            ),
                            "different_acoustic_tokens": int(
                                np.count_nonzero(generated_frame[1:] != reference_frame[1:])
                            ),
                        }
                    )
                    frame = np.asarray(reference_frame, dtype=np.int64)
                else:
                    frame = generated_frame

                pcm_bytes = None
                if frame_index >= int(self.config["audio_delay_frames"]):
                    stage_started = time.perf_counter()
                    mimi_codes = np.concatenate([np.asarray([previous_semantic]), frame[1:]])
                    audio = self.mimi.decode(mimi_codes)
                    pcm = np.clip(audio, -1.0, 1.0)
                    pcm = np.rint(pcm * 32767.0).astype("<i2")
                    pcm_bytes = pcm.tobytes()
                    audio_frames += 1
                    if first_audio_seconds is None:
                        now = time.perf_counter()
                        first_audio_seconds = now - generation_started
                        first_audio_request_seconds = now - request_started
                    stage_seconds["mimi"] += time.perf_counter() - stage_started
                previous_semantic = semantic

                eos_reached = state.eos_index <= frame_index
                if not eos_reached:
                    update_frame_state(
                        state,
                        predicted_shift,
                        frame_index,
                        phone_sequence_length,
                        self.config,
                    )
                frame_times.append(time.perf_counter() - frame_started)
                if include_trajectory:
                    trajectory.append(
                        {
                            "index": frame_index,
                            "position": position,
                            "semantic": semantic,
                            "predicted_shift": predicted_shift,
                            "phone_max": state.maximum,
                            "eos_index": state.eos_index,
                            "codes": generated_frame.tolist(),
                        }
                    )
                if pcm_bytes is not None:
                    yield pcm_bytes
                if eos_reached:
                    break

            if first_audio_seconds is None or first_audio_request_seconds is None:
                raise RuntimeError("generation ended before the first PCM chunk")
            generation_compute_seconds = sum(frame_times)
            audio_seconds = audio_frames * self.samples_per_chunk / self.sample_rate
            calls = {
                "temp": self.temp.calls - call_start["temp"],
                "temp_cuda_graph": (self.temp.graph_launches - call_start["temp_cuda_graph"]),
                "temp_sink_replay": (
                    self.temp.sink_replay_steps - call_start["temp_sink_replay_steps"]
                ),
                "dep_q2": self.dep.calls[2] - call_start["dep_q2"],
                "dep_q1": self.dep.calls[1] - call_start["dep_q1"],
                "mimi": self.mimi.calls - call_start["mimi"],
                "audio_embedding": (self.embeddings.core.calls - call_start["audio_embedding"]),
                "cuda_acoustic_control": (
                    (self.acoustic_control.calls if self.acoustic_control is not None else 0)
                    - call_start["cuda_acoustic_control"]
                ),
                "dep_cuda_graph": (self.dep.graph_launches - call_start["dep_cuda_graph"]),
            }
            request_seconds = time.perf_counter() - request_started
            result = {
                "runtime": "numpy+tensorrt+cuda-python",
                "torch_imported": False,
                "resident": True,
                "request_index": request_index,
                "text": result_text,
                "frames": len(frame_times),
                "audio_frames": audio_frames,
                "audio_seconds": round(audio_seconds, 3),
                "generation_seconds": round(generation_compute_seconds, 3),
                "stream_wall_seconds": round(time.perf_counter() - generation_started, 3),
                "request_seconds": round(request_seconds, 3),
                "rtf": round(generation_compute_seconds / audio_seconds, 3),
                "ttfa_seconds": round(float(first_audio_seconds), 3),
                "request_to_first_audio_seconds": round(float(first_audio_request_seconds), 3),
                "cold_start_to_first_audio_seconds": (
                    round(
                        self.startup_seconds + float(first_audio_request_seconds),
                        3,
                    )
                    if request_index == 0
                    else None
                ),
                "runtime_startup_seconds": round(self.startup_seconds, 3),
                "frontend": frontend_metrics,
                "pcm": {
                    "format": "s16le",
                    "sample_rate": self.sample_rate,
                    "channels": 1,
                    "samples_per_chunk": self.samples_per_chunk,
                    "bytes_per_chunk": self.samples_per_chunk * 2,
                },
                "frame_ms": {
                    "mean": round(float(np.mean(frame_times)) * 1000, 3),
                    "p95": round(float(np.percentile(frame_times, 95)) * 1000, 3),
                    "max": round(float(np.max(frame_times)) * 1000, 3),
                },
                "stage_ms_per_generated_frame": {
                    name: round(seconds * 1000 / len(frame_times), 3)
                    for name, seconds in stage_seconds.items()
                },
                "calls": calls,
                "sink_attention": {
                    "enabled": self.temp.sink_attention is not None,
                    "window_size": self.temp.audio_window_size,
                    "prompt_length": (
                        self.temp.sink_attention.prompt_length
                        if self.temp.sink_attention is not None
                        else None
                    ),
                    "tail_limit": (
                        self.temp.sink_attention.tail_limit
                        if self.temp.sink_attention is not None
                        else None
                    ),
                    "position_offset": (
                        self.temp.sink_attention.position_offset
                        if self.temp.sink_attention is not None
                        else None
                    ),
                    "rebuilds": (
                        self.temp.sink_rebuilds - call_start["temp_sink_rebuilds"]
                    ),
                    "replay_steps": (
                        self.temp.sink_replay_steps
                        - call_start["temp_sink_replay_steps"]
                    ),
                    "rebuild_seconds": round(
                        self.temp.sink_rebuild_seconds
                        - call_start["temp_sink_rebuild_seconds"],
                        3,
                    ),
                },
                "teacher_forced": {
                    "enabled": teacher_force_reference,
                    "frames": len(teacher_forced_comparisons),
                    "all_acoustic_tokens_equal": bool(teacher_forced_comparisons)
                    and all(item["acoustic_tokens_equal"] for item in teacher_forced_comparisons),
                    "comparisons": teacher_forced_comparisons,
                },
                "trajectory": trajectory,
                "max_rss_mib": round(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                    1,
                ),
            }
            if "torch" in sys.modules:
                raise RuntimeError("PyTorch entered sys.modules during torchless execution")
            return result
        finally:
            self._release_request()

    def synthesize_to_wav(
        self,
        text: str | None,
        output: Path,
        **stream_options,
    ) -> dict[str, object]:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        stream = self.synthesize_stream(text, **stream_options)
        try:
            with wave.open(str(output), "wb") as sink:
                sink.setnchannels(1)
                sink.setsampwidth(2)
                sink.setframerate(self.sample_rate)
                for pcm_chunk in stream:
                    sink.writeframesraw(pcm_chunk)
        finally:
            if not stream.closed:
                stream.close()
        if stream.result is None:
            raise RuntimeError("PCM stream completed without synthesis metrics")
        result = dict(stream.result)
        result.update(
            {
                "output": str(output),
                "output_bytes": output.stat().st_size,
                "output_sha256": file_sha256(output),
            }
        )
        return result


def main() -> None:
    args = parse_args()
    metrics_path = args.metrics or args.output.with_suffix(".json")
    if metrics_path.resolve() == args.assets.resolve():
        raise ValueError(
            "metrics path would overwrite the immutable asset manifest; "
            "pass --metrics with a distinct path"
        )
    with SynthesisRuntime(
        RuntimeFiles.from_args(args),
        ruaccent_assets=args.ruaccent_assets,
        phone_map=args.phone_map,
        espeak_executable=args.espeak_executable,
        text_normalizer=args.text_normalizer,
        cuda_temp_graph=args.cuda_temp_graph,
        cuda_dep_graph=args.cuda_dep_graph,
    ) as runtime:
        result = runtime.synthesize_to_wav(
            args.text,
            args.output,
            seed=args.seed,
            max_frames=args.max_frames,
            allow_unknown_phones=args.allow_unknown_phones,
            teacher_force_reference=args.teacher_force_reference,
        )
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
