#!/usr/bin/env python3
"""Drop-in TensorRT runtime for VoXtream's streaming Mimi decoder."""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
from pathlib import Path

import tensorrt as trt
import torch


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_")


def _collect_tensor_fields(
    owner: object,
    prefix: str,
    values: dict[str, torch.Tensor],
    visited: set[int],
) -> None:
    if id(owner) in visited:
        return
    visited.add(id(owner))
    for attribute, value in vars(owner).items():
        name = f"{prefix}.{attribute}" if prefix else attribute
        if isinstance(value, torch.Tensor):
            values[_sanitize(name)] = value.detach().clone()
        elif type(value).__name__ == "RingKVCache":
            _collect_tensor_fields(value, name, values, visited)


def capture_initial_decoder_state(mimi) -> dict[str, torch.Tensor]:
    """Start upstream streaming once and snapshot only decoder-side state."""

    mimi.streaming_forever(batch_size=1)
    mimi.reset_streaming()
    values: dict[str, torch.Tensor] = {}
    visited: set[int] = set()
    prefixes = ("decoder", "decoder_transformer", "upsample")
    for module_name, state in mimi.get_streaming_state().items():
        if module_name == "decoder" or module_name.startswith(
            tuple(prefix + "." for prefix in prefixes)
        ):
            _collect_tensor_fields(state, module_name, values, visited)
    if not values:
        raise RuntimeError("Mimi decoder streaming state is empty")
    return values


def load_initial_decoder_state(manifest_path: Path) -> dict[str, torch.Tensor]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "voxtream-mimi-decoder-state-v1":
        raise ValueError(f"unsupported Mimi state format in {manifest_path}")
    binary_path = manifest_path.parent / manifest["binary"]
    payload = bytearray(binary_path.read_bytes())
    digest = hashlib.sha256(payload).hexdigest()
    if digest != manifest["binary_sha256"]:
        raise ValueError(
            f"Mimi initial-state checksum mismatch: {digest} != "
            f"{manifest['binary_sha256']}"
        )
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bool": torch.bool,
        "int64": torch.int64,
    }
    values: dict[str, torch.Tensor] = {}
    source = torch.frombuffer(payload, dtype=torch.uint8)
    for item in manifest["tensors"]:
        dtype = dtype_map[item["dtype"]]
        shape = tuple(item["shape"])
        offset = int(item["offset"])
        size = int(item["bytes"])
        if size:
            value = source[offset : offset + size].view(dtype).reshape(shape).clone()
        else:
            value = torch.empty(shape, dtype=dtype)
        values[item["name"]] = value.to(device="cuda")
    if len(values) != int(manifest["state_tensors"]):
        raise RuntimeError("duplicate tensor names in Mimi initial-state manifest")
    return values


class TensorRTMimiDecoder:
    """Implement the Mimi methods used by SpeechGenerator with one TRT plan."""

    def __init__(
        self,
        engine_path: Path,
        upstream_mimi=None,
        *,
        initial_state_path: Path | None = None,
        preloaded=None,
    ) -> None:
        self.engine_path = Path(engine_path)
        loaded = preloaded
        if loaded is None:
            self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = self.runtime.deserialize_cuda_engine(
                self.engine_path.read_bytes()
            )
        else:
            if Path(loaded.engine_path) != self.engine_path:
                raise ValueError(
                    f"preloaded {loaded.engine_path}, requested {self.engine_path}"
                )
            self.runtime = loaded.runtime
            self.engine = loaded.engine
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("failed to create Mimi TensorRT context")

        if (upstream_mimi is None) == (initial_state_path is None):
            raise ValueError(
                "provide exactly one of upstream_mimi or initial_state_path"
            )
        self.initial_state_source = (
            str(initial_state_path) if initial_state_path is not None else "pytorch"
        )
        self.initial_state = (
            load_initial_decoder_state(initial_state_path)
            if initial_state_path is not None
            else capture_initial_decoder_state(upstream_mimi)
        )
        self.state_names = tuple(self.initial_state)
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
        self.input_name: dict[str, str] = {}
        self.output_name: dict[str, str] = {}
        for name in self.state_names:
            candidates = (name, f"__next_{name}")
            matches = [candidate for candidate in candidates if candidate in self.input_names]
            if len(matches) != 1:
                raise RuntimeError(f"cannot resolve TensorRT input for {name}: {matches}")
            self.input_name[name] = matches[0]
            output_name = f"next_{name}"
            if output_name not in self.output_names:
                raise RuntimeError(f"missing TensorRT output {output_name}")
            self.output_name[name] = output_name

        expected_inputs = {"codes", *self.input_name.values()}
        expected_outputs = {"audio", *self.output_name.values()}
        if expected_inputs != self.input_names:
            raise RuntimeError(
                f"unexpected Mimi inputs: {sorted(self.input_names - expected_inputs)}"
            )
        if expected_outputs != self.output_names:
            raise RuntimeError(
                f"unexpected Mimi outputs: {sorted(self.output_names - expected_outputs)}"
            )

        self.banks = [
            {name: value.clone() for name, value in self.initial_state.items()}
            for _ in range(2)
        ]
        state_dtypes = {value.dtype for value in self.initial_state.values()}
        self.zero_sentinels = [
            {
                dtype: torch.empty(1, device="cuda", dtype=dtype)
                for dtype in state_dtypes
            }
            for _ in range(2)
        ]
        audio_shape = tuple(self.engine.get_tensor_shape("audio"))
        self.audio = torch.empty(audio_shape, device="cuda", dtype=torch.bfloat16)
        self.current_bank = 0
        self.calls = 0
        self.resets = 0
        self.default_stream_calls = 0

    def _pointer(self, tensor: torch.Tensor, bank_index: int) -> int:
        if tensor.numel():
            return tensor.data_ptr()
        return self.zero_sentinels[bank_index][tensor.dtype].data_ptr()

    def reset_streaming(self) -> None:
        for bank in self.banks:
            for name, initial in self.initial_state.items():
                bank[name].copy_(initial)
        self.current_bank = 0
        self.resets += 1

    @contextlib.contextmanager
    def streaming(self, batch_size: int = 1):
        if batch_size != 1:
            raise ValueError("TensorRT Mimi decoder currently has fixed batch=1")
        self.reset_streaming()
        yield self

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if (
            codes.device.type != "cuda"
            or codes.dtype != torch.int64
            or tuple(codes.shape) != (1, 16, 1)
            or not codes.is_contiguous()
        ):
            raise ValueError(
                "TensorRT Mimi expects contiguous CUDA int64 codes with shape [1,16,1]"
            )

        source_index = self.current_bank
        target_index = source_index ^ 1
        source = self.banks[source_index]
        target = self.banks[target_index]
        if not self.context.set_tensor_address("codes", codes.data_ptr()):
            raise RuntimeError("failed to bind Mimi codes")
        if not self.context.set_tensor_address("audio", self.audio.data_ptr()):
            raise RuntimeError("failed to bind Mimi audio")
        for name in self.state_names:
            if not self.context.set_tensor_address(
                self.input_name[name], self._pointer(source[name], source_index)
            ):
                raise RuntimeError(f"failed to bind Mimi input {name}")
            if not self.context.set_tensor_address(
                self.output_name[name], self._pointer(target[name], target_index)
            ):
                raise RuntimeError(f"failed to bind Mimi output {name}")

        stream = torch.cuda.current_stream()
        if stream == torch.cuda.default_stream():
            self.default_stream_calls += 1
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("Mimi TensorRT enqueue failed")
        self.current_bank = target_index
        self.calls += 1
        return self.audio

    def metrics(self) -> dict[str, object]:
        state_bytes = sum(
            value.numel() * value.element_size()
            for value in self.initial_state.values()
        )
        return {
            "engine": str(self.engine_path),
            "engine_bytes": self.engine_path.stat().st_size,
            "state_tensors": len(self.initial_state),
            "state_bytes_per_bank": state_bytes,
            "state_banks": len(self.banks),
            "calls": self.calls,
            "resets": self.resets,
            "default_stream_calls": self.default_stream_calls,
            "pytorch_compute": False,
            "initial_state_source": self.initial_state_source,
        }
