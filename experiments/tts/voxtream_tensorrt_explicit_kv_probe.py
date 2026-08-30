#!/usr/bin/env python3
"""Export dep_former with explicit K/V/cache-position state tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from voxtream_tensorrt_probe import (
    build_dep_former,
    causal_rows,
    make_kv_update_exportable,
    prepare_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--dynamic-sequence",
        action="store_true",
        help=(
            "Export one ONNX graph whose sequence dimension accepts q=1 and q=2. "
            "Use --sequence-length 2 so torch.export does not specialize the "
            "singleton q=1 example."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--engine", type=Path)
    parser.add_argument(
        "--init-engine",
        type=Path,
        help="Optional q=2 engine used with the q=1 --engine for full-frame validation.",
    )
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the standard closed-loop/full trajectory checks.",
    )
    parser.add_argument(
        "--validate-cuda-graph",
        action="store_true",
        help="Capture the two alternating q=1 TensorRT CUDA Graphs during full validation.",
    )
    parser.add_argument(
        "--standalone-fixture-dir",
        type=Path,
        help=(
            "Write a raw q=2 -> q=1 TensorRT trajectory for validating a "
            "runtime that does not import PyTorch. Requires one unified engine "
            "as both --engine and --init-engine."
        ),
    )
    return parser.parse_args()


def kv_buffer_names(model: nn.Module) -> tuple[str, ...]:
    suffixes = ("kv_cache.k_cache", "kv_cache.v_cache", "kv_cache.cache_pos")
    return tuple(name for name, _ in model.named_buffers() if name.endswith(suffixes))


class ExplicitKVStep(nn.Module):
    def __init__(self, model: nn.Module, buffer_names: tuple[str, ...]) -> None:
        super().__init__()
        self.model = model
        self.buffer_names = buffer_names

    def forward(
        self,
        hidden: torch.Tensor,
        input_pos: torch.Tensor,
        mask: torch.Tensor,
        *kv_state: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        replacements = dict(zip(self.buffer_names, kv_state))
        output = torch.func.functional_call(
            self.model,
            replacements,
            (hidden,),
            {"input_pos": input_pos, "mask": mask},
            strict=False,
        )
        return (output, *kv_state)


class ExplicitKVTRTRunner:
    def __init__(
        self,
        engine_path: Path,
        state_names: tuple[str, ...],
        initial_state: tuple[torch.Tensor, ...],
        sequence_length: int,
        preloaded=None,
        borrow_initial_state: bool = False,
        inplace_state: bool = False,
        use_current_stream: bool = False,
    ) -> None:
        import tensorrt as trt

        engine_path = Path(engine_path)
        self.engine_path = engine_path
        if preloaded is None:
            self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        else:
            if Path(preloaded.engine_path) != engine_path:
                raise ValueError(
                    f"preloaded {preloaded.engine_path}, requested {engine_path}"
                )
            self.runtime = preloaded.runtime
            self.engine = preloaded.engine
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.use_current_stream = use_current_stream
        self.input_staging: dict[str, torch.Tensor] = {}
        hidden_shape = tuple(self.engine.get_tensor_shape("hidden"))
        self.profile_index: int | None = None
        if -1 in hidden_shape:
            matching_profiles = []
            for profile_index in range(self.engine.num_optimization_profiles):
                minimum, optimum, maximum = self.engine.get_tensor_profile_shape(
                    "hidden", profile_index
                )
                if (
                    minimum[1] <= sequence_length <= maximum[1]
                    and optimum[1] == sequence_length
                ):
                    matching_profiles.append(profile_index)
            if not matching_profiles:
                raise RuntimeError(
                    f"engine has no optimization profile for q={sequence_length}"
                )
            self.profile_index = matching_profiles[0]
            if self.context.active_optimization_profile != self.profile_index:
                if not self.context.set_optimization_profile_async(
                    self.profile_index, self.stream.cuda_stream
                ):
                    raise RuntimeError(
                        f"failed to select TensorRT profile {self.profile_index}"
                    )
            dynamic_shapes = {}
            for name in ("hidden", "input_pos", "mask"):
                _, optimum, _ = self.engine.get_tensor_profile_shape(
                    name, self.profile_index
                )
                shape = tuple(
                    sequence_length if axis == 1 else dimension
                    for axis, dimension in enumerate(optimum)
                )
                dynamic_shapes[name] = shape
                if not self.context.set_input_shape(name, shape):
                    raise RuntimeError(
                        f"failed to set TensorRT input shape {name}={shape}"
                    )
        self.input_shapes = {
            name: tuple(self.context.get_tensor_shape(name))
            for name in ("hidden", "input_pos", "mask")
        }
        self.state_names = tuple(name.replace(".", "_") for name in state_names)
        self.state = (
            tuple(initial_state)
            if borrow_initial_state
            else tuple(value.clone() for value in initial_state)
        )
        self.next_state = (
            self.state
            if inplace_state
            else tuple(torch.empty_like(value) for value in initial_state)
        )
        batch_size = initial_state[0].shape[0]
        self.output = torch.empty(
            (batch_size, sequence_length, 1024), device="cuda"
        )
        torch_dtypes = {
            trt.DataType.BF16: torch.bfloat16,
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.INT64: torch.int64,
            trt.DataType.INT32: torch.int32,
            trt.DataType.BOOL: torch.bool,
        }
        state_output_names = {f"next_{name}" for name in self.state_names}
        self.extra_outputs: dict[str, torch.Tensor] = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if (
                self.engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT
                or name == "output"
                or name in state_output_names
            ):
                continue
            shape = tuple(self.context.get_tensor_shape(name))
            if -1 in shape:
                raise RuntimeError(f"unresolved TensorRT output shape {name}={shape}")
            self.extra_outputs[name] = torch.empty(
                shape,
                device="cuda",
                dtype=torch_dtypes[self.engine.get_tensor_dtype(name)],
            )

    def copy_state_from(self, values: tuple[torch.Tensor, ...]) -> None:
        if len(values) != len(self.state):
            raise ValueError(
                f"state tensor count changed: expected {len(self.state)}, got {len(values)}"
            )
        for source, target in zip(values, self.state):
            target.copy_(source)

    def step(
        self,
        hidden: torch.Tensor,
        input_pos: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        prepared_inputs = {
            "hidden": self._prepare_input("hidden", hidden),
            "input_pos": self._prepare_input("input_pos", input_pos),
            "mask": self._prepare_input("mask", mask),
        }
        bindings = {
            **prepared_inputs,
            "output": self.output,
        }
        bindings.update(zip(self.state_names, self.state))
        bindings.update(
            (f"next_{name}", value)
            for name, value in zip(self.state_names, self.next_state)
        )
        bindings.update(self.extra_outputs)
        for name, tensor in bindings.items():
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"failed to bind TensorRT tensor {name}")
        stream = (
            torch.cuda.current_stream()
            if self.use_current_stream
            else self.stream
        )
        if not self.use_current_stream:
            stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            if not self.context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v3 failed")
        if not self.use_current_stream:
            stream.synchronize()
        self.state, self.next_state = self.next_state, self.state
        return self.output, self.state

    def _prepare_input(self, name: str, value: torch.Tensor) -> torch.Tensor:
        expected = self.input_shapes[name]
        actual = tuple(value.shape)
        if actual != expected:
            if len(actual) != len(expected) or any(
                source not in (1, target)
                for source, target in zip(actual, expected)
            ):
                raise ValueError(
                    f"TensorRT input {name} shape mismatch: "
                    f"expected {expected}, got {actual}"
                )
            staging = self.input_staging.get(name)
            if (
                staging is None
                or staging.device != value.device
                or staging.dtype != value.dtype
            ):
                staging = torch.empty(
                    expected, device=value.device, dtype=value.dtype
                )
                self.input_staging[name] = staging
            staging.copy_(value)
            return staging
        return value if value.is_contiguous() else value.contiguous()


def validate_closed_loop(
    engine_path: Path,
    checkpoint: Path,
    model: nn.Module,
    state_names: tuple[str, ...],
    batch_size: int,
    device: str,
) -> dict[str, object]:
    hidden, input_pos, mask = prepare_inputs(model, batch_size, 1, device)
    initial_state = tuple(
        dict(model.named_buffers())[name].clone() for name in state_names
    )
    runner = ExplicitKVTRTRunner(engine_path, state_names, initial_state, 1)

    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        audio_head = source.get_tensor("audio_head").to(
            device=device, dtype=torch.bfloat16
        )
        audio_embeddings = source.get_tensor("audio_embeddings.weight").to(
            device=device, dtype=torch.bfloat16
        )

    reference_hidden = hidden
    candidate_hidden = hidden.clone()
    steps: list[dict[str, object]] = []
    for head_index in range(1, 15):
        with torch.inference_mode():
            reference_output = model(
                reference_hidden, input_pos=input_pos, mask=mask
            )
            candidate_output, candidate_state = runner.step(
                candidate_hidden, input_pos, mask
            )
            head = audio_head[head_index]
            reference_logits = torch.mm(
                reference_output[:, -1, :].to(torch.bfloat16), head
            )
            candidate_logits = torch.mm(
                candidate_output[:, -1, :].to(torch.bfloat16), head
            )
            reference_cfg = 3.0 * reference_logits[0].float() - 2.0 * reference_logits[1].float()
            candidate_cfg = 3.0 * candidate_logits[0].float() - 2.0 * candidate_logits[1].float()
            reference_token = int(torch.argmax(reference_cfg).item())
            candidate_token = int(torch.argmax(candidate_cfg).item())
            reference_top2 = torch.topk(reference_cfg, 2).values
            hidden_delta = reference_output.float() - candidate_output.float()
            reference_state = dict(model.named_buffers())
            state_max_abs = max(
                float(
                    (
                        reference_state[name].float() - candidate_value.float()
                    ).abs().max().item()
                )
                for name, candidate_value in zip(state_names, candidate_state)
            )

            steps.append(
                {
                    "head_index": head_index,
                    "position": int(input_pos[0, 0].item()),
                    "reference_token": reference_token,
                    "candidate_token": candidate_token,
                    "token_equal": reference_token == candidate_token,
                    "reference_margin": float((reference_top2[0] - reference_top2[1]).item()),
                    "cfg_logits_max_abs": float(
                        (reference_cfg - candidate_cfg).abs().max().item()
                    ),
                    "hidden_max_abs": float(hidden_delta.abs().max().item()),
                    "state_max_abs": state_max_abs,
                }
            )

            reference_embedding = audio_embeddings[
                reference_token + (head_index + 1) * 2050
            ].view(1, 1, 1024)
            candidate_embedding = audio_embeddings[
                candidate_token + (head_index + 1) * 2050
            ].view(1, 1, 1024)
            reference_hidden = reference_embedding.repeat(batch_size, 1, 1)
            candidate_hidden = candidate_embedding.repeat(batch_size, 1, 1)
            if head_index < 14:
                input_pos = input_pos + 1
                mask = causal_rows(input_pos)

    return {
        "steps": steps,
        "token_matches": sum(int(step["token_equal"]) for step in steps),
        "token_total": len(steps),
        "all_tokens_equal": all(bool(step["token_equal"]) for step in steps),
        "max_hidden_abs": max(float(step["hidden_max_abs"]) for step in steps),
        "max_state_abs": max(float(step["state_max_abs"]) for step in steps),
    }


def validate_init(
    engine_path: Path,
    checkpoint: Path,
    model: nn.Module,
    state_names: tuple[str, ...],
    batch_size: int,
    device: str,
) -> dict[str, object]:
    """Compare the real two-token frame init against one static q=2 engine."""

    hidden, input_pos, mask = prepare_inputs(model, batch_size, 2, device)
    initial_state = tuple(
        dict(model.named_buffers())[name].clone() for name in state_names
    )
    runner = ExplicitKVTRTRunner(engine_path, state_names, initial_state, 2)

    with torch.inference_mode():
        reference_output = model(hidden, input_pos=input_pos, mask=mask)
        candidate_output, candidate_state = runner.step(hidden, input_pos, mask)

    hidden_delta = reference_output.float() - candidate_output.float()
    reference_flat = reference_output.float().flatten()
    candidate_flat = candidate_output.float().flatten()
    cosine = torch.dot(reference_flat, candidate_flat) / (
        torch.linalg.vector_norm(reference_flat)
        * torch.linalg.vector_norm(candidate_flat)
    )

    reference_state = dict(model.named_buffers())
    state_deltas = {
        name: float(
            (reference_state[name].float() - candidate_value.float())
            .abs()
            .max()
            .item()
        )
        for name, candidate_value in zip(state_names, candidate_state)
    }

    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        audio_head = source.get_tensor("audio_head")[0].to(
            device=device, dtype=torch.bfloat16
        )
    reference_logits = torch.mm(
        reference_output[:, -1, :].to(torch.bfloat16), audio_head
    )
    candidate_logits = torch.mm(
        candidate_output[:, -1, :].to(torch.bfloat16), audio_head
    )
    fused_logits = runner.extra_outputs.get("acoustic_logits")
    reference_cfg = 3.0 * reference_logits[0].float() - 2.0 * reference_logits[1].float()
    candidate_cfg = 3.0 * candidate_logits[0].float() - 2.0 * candidate_logits[1].float()
    reference_top2 = torch.topk(reference_cfg, 2)
    candidate_top2 = torch.topk(candidate_cfg, 2)
    logits_delta = reference_cfg - candidate_cfg

    return {
        "hidden": {
            "max_abs": float(hidden_delta.abs().max().item()),
            "mean_abs": float(hidden_delta.abs().mean().item()),
            "rmse": float(torch.sqrt(torch.mean(hidden_delta.square())).item()),
            "cosine": float(cosine.item()),
            "bf16_equal_fraction": float(
                (reference_output.to(torch.bfloat16) == candidate_output.to(torch.bfloat16))
                .float()
                .mean()
                .item()
            ),
        },
        "state": {
            "max_abs": max(state_deltas.values()),
            "per_tensor_max_abs": state_deltas,
        },
        "cfg_logits_audio_head_0": {
            "max_abs": float(logits_delta.abs().max().item()),
            "mean_abs": float(logits_delta.abs().mean().item()),
            "reference_argmax": int(reference_top2.indices[0].item()),
            "candidate_argmax": int(candidate_top2.indices[0].item()),
            "token_equal": bool(
                reference_top2.indices[0] == candidate_top2.indices[0]
            ),
            "reference_margin": float(
                (reference_top2.values[0] - reference_top2.values[1]).item()
            ),
            "candidate_margin": float(
                (candidate_top2.values[0] - candidate_top2.values[1]).item()
            ),
            "fused_max_abs": (
                float((candidate_logits - fused_logits).float().abs().max().item())
                if fused_logits is not None
                else None
            ),
            "fused_bitwise_equal": (
                bool(torch.equal(candidate_logits, fused_logits))
                if fused_logits is not None
                else None
            ),
        },
    }


def validate_full_trajectory(
    step_engine_path: Path,
    init_engine_path: Path,
    checkpoint: Path,
    model: nn.Module,
    state_names: tuple[str, ...],
    batch_size: int,
    device: str,
    capture_cuda_graph: bool,
) -> dict[str, object]:
    """Validate q=2 init followed by all 14 autoregressive depth steps."""

    hidden, input_pos, mask = prepare_inputs(model, batch_size, 2, device)
    initial_state = tuple(
        dict(model.named_buffers())[name].clone() for name in state_names
    )
    from voxtream_tensorrt_runtime import TensorRTDepFormerStep

    runtime = TensorRTDepFormerStep(
        step_engine_path,
        model,
        init_engine_path=init_engine_path,
        capture_cuda_graph=capture_cuda_graph,
        run_init=True,
    )
    runtime.reset_caches()
    reset_state_equal = [
        bool(torch.equal(reference, candidate))
        for reference, candidate in zip(initial_state, runtime.state)
    ]

    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        audio_head = source.get_tensor("audio_head").to(
            device=device, dtype=torch.bfloat16
        )
        audio_embeddings = source.get_tensor("audio_embeddings.weight").to(
            device=device, dtype=torch.bfloat16
        )

    with torch.inference_mode():
        reference_output = model(hidden, input_pos=input_pos, mask=mask)
        candidate_output = runtime(hidden, input_pos, mask)
    steps: list[dict[str, object]] = []
    for head_index in range(15):
        with torch.inference_mode():
            head = audio_head[head_index]
            reference_logits = torch.mm(
                reference_output[:, -1, :].to(torch.bfloat16), head
            )
            candidate_logits = torch.mm(
                candidate_output[:, -1, :].to(torch.bfloat16), head
            )
            fused_logits = runtime.acoustic_logits
            reference_cfg = (
                3.0 * reference_logits[0].float()
                - 2.0 * reference_logits[1].float()
            )
            candidate_cfg = (
                3.0 * candidate_logits[0].float()
                - 2.0 * candidate_logits[1].float()
            )
            reference_token = int(torch.argmax(reference_cfg).item())
            candidate_token = int(torch.argmax(candidate_cfg).item())
            reference_top2 = torch.topk(reference_cfg, 2).values
            hidden_delta = reference_output.float() - candidate_output.float()

            reference_state = dict(model.named_buffers())
            state_max_abs = max(
                float(
                    (reference_state[name].float() - candidate_value.float())
                    .abs()
                    .max()
                    .item()
                )
                for name, candidate_value in zip(state_names, runtime.state)
            )
            steps.append(
                {
                    "head_index": head_index,
                    "reference_token": reference_token,
                    "candidate_token": candidate_token,
                    "token_equal": reference_token == candidate_token,
                    "reference_margin": float(
                        (reference_top2[0] - reference_top2[1]).item()
                    ),
                    "cfg_logits_max_abs": float(
                        (reference_cfg - candidate_cfg).abs().max().item()
                    ),
                    "hidden_max_abs": float(hidden_delta.abs().max().item()),
                    "state_max_abs": state_max_abs,
                    "fused_logits_max_abs": (
                        float(
                            (candidate_logits - fused_logits)
                            .float()
                            .abs()
                            .max()
                            .item()
                        )
                        if fused_logits is not None
                        else None
                    ),
                    "fused_logits_bitwise_equal": (
                        bool(torch.equal(candidate_logits, fused_logits))
                        if fused_logits is not None
                        else None
                    ),
                }
            )

            if head_index == 14:
                continue
            reference_embedding = audio_embeddings[
                reference_token + (head_index + 2) * 2050
            ].view(1, 1, 1024)
            candidate_embedding = audio_embeddings[
                candidate_token + (head_index + 2) * 2050
            ].view(1, 1, 1024)
            reference_hidden = reference_embedding.repeat(batch_size, 1, 1)
            candidate_hidden = candidate_embedding.repeat(batch_size, 1, 1)
            step_pos = torch.full(
                (batch_size, 1), head_index + 2, device=device, dtype=torch.int64
            )
            step_mask = causal_rows(step_pos)
            reference_output = model(
                reference_hidden, input_pos=step_pos, mask=step_mask
            )
            candidate_output = runtime(candidate_hidden, step_pos, step_mask)

    return {
        "steps": steps,
        "token_matches": sum(int(step["token_equal"]) for step in steps),
        "token_total": len(steps),
        "all_tokens_equal": all(bool(step["token_equal"]) for step in steps),
        "max_hidden_abs": max(float(step["hidden_max_abs"]) for step in steps),
        "max_state_abs": max(float(step["state_max_abs"]) for step in steps),
        "reset_state_equal": reset_state_equal,
        "all_reset_state_equal": all(reset_state_equal),
        "all_fused_logits_bitwise_equal": (
            all(bool(step["fused_logits_bitwise_equal"]) for step in steps)
            if runtime.acoustic_logits is not None
            else None
        ),
        "max_fused_logits_abs": (
            max(float(step["fused_logits_max_abs"]) for step in steps)
            if runtime.acoustic_logits is not None
            else None
        ),
        "runtime": runtime.metrics(),
    }


def _write_fixture_tensor(
    root: Path,
    relative_path: str,
    tensor: torch.Tensor,
) -> dict[str, object]:
    contiguous = tensor.detach().contiguous().cpu()
    if contiguous.dtype == torch.bfloat16:
        dtype = "bfloat16"
        payload = contiguous.view(torch.uint16).numpy().tobytes()
    elif contiguous.dtype == torch.int64:
        dtype = "int64"
        payload = contiguous.numpy().tobytes()
    elif contiguous.dtype == torch.bool:
        dtype = "bool"
        payload = contiguous.numpy().tobytes()
    elif contiguous.dtype == torch.float32:
        dtype = "float32"
        payload = contiguous.numpy().tobytes()
    else:
        raise TypeError(f"unsupported fixture dtype: {contiguous.dtype}")
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": relative_path,
        "dtype": dtype,
        "shape": list(contiguous.shape),
        "nbytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def capture_standalone_fixture(
    fixture_dir: Path,
    engine_path: Path,
    model: nn.Module,
    state_names: tuple[str, ...],
    batch_size: int,
    device: str,
) -> dict[str, object]:
    """Capture one real explicit-state trajectory for a no-Torch executor."""

    from voxtream_tensorrt_runtime import TensorRTDepFormerStep

    q2_hidden, q2_input_pos, q2_mask = prepare_inputs(
        model, batch_size, 2, device
    )
    runtime = TensorRTDepFormerStep(
        engine_path,
        model,
        init_engine_path=engine_path,
        capture_cuda_graph=False,
        run_init=True,
    )
    runtime.reset_caches()
    initial_state = tuple(value.clone() for value in runtime.state)

    with torch.inference_mode():
        q2_output = runtime(q2_hidden, q2_input_pos, q2_mask).clone()
        q2_acoustic_logits = (
            runtime.acoustic_logits.clone()
            if runtime.acoustic_logits is not None
            else None
        )
    after_q2 = tuple(value.clone() for value in runtime.state)

    torch.manual_seed(5678)
    q1_hidden = torch.randn(
        batch_size, 1, 1024, device=device, dtype=torch.bfloat16
    )
    q1_input_pos = torch.full(
        (batch_size, 1), 2, device=device, dtype=torch.int64
    )
    q1_mask = causal_rows(q1_input_pos)
    with torch.inference_mode():
        q1_output = runtime(q1_hidden, q1_input_pos, q1_mask).clone()
        q1_acoustic_logits = (
            runtime.acoustic_logits.clone()
            if runtime.acoustic_logits is not None
            else None
        )
    after_q1 = tuple(value.clone() for value in runtime.state)
    torch.cuda.synchronize()

    fixture_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        "q2": {
            "hidden": _write_fixture_tensor(
                fixture_dir, "inputs/q2-hidden.bf16", q2_hidden
            ),
            "input_pos": _write_fixture_tensor(
                fixture_dir, "inputs/q2-input-pos.i64", q2_input_pos
            ),
            "mask": _write_fixture_tensor(
                fixture_dir, "inputs/q2-mask.bool", q2_mask
            ),
        },
        "q1": {
            "hidden": _write_fixture_tensor(
                fixture_dir, "inputs/q1-hidden.bf16", q1_hidden
            ),
            "input_pos": _write_fixture_tensor(
                fixture_dir, "inputs/q1-input-pos.i64", q1_input_pos
            ),
            "mask": _write_fixture_tensor(
                fixture_dir, "inputs/q1-mask.bool", q1_mask
            ),
        },
    }
    expected_outputs = {
        "q2": _write_fixture_tensor(
            fixture_dir, "expected/q2-output.bf16", q2_output
        ),
        "q1": _write_fixture_tensor(
            fixture_dir, "expected/q1-output.bf16", q1_output
        ),
    }
    expected_extra_outputs: dict[str, dict[str, object]] = {}
    if q2_acoustic_logits is not None and q1_acoustic_logits is not None:
        expected_extra_outputs["q2"] = {
            "acoustic_logits": _write_fixture_tensor(
                fixture_dir,
                "expected/q2-acoustic-logits.bf16",
                q2_acoustic_logits,
            )
        }
        expected_extra_outputs["q1"] = {
            "acoustic_logits": _write_fixture_tensor(
                fixture_dir,
                "expected/q1-acoustic-logits.bf16",
                q1_acoustic_logits,
            )
        }
    states = []
    for name, initial, q2_state, q1_state in zip(
        state_names, initial_state, after_q2, after_q1
    ):
        tensor_name = name.replace(".", "_")
        states.append(
            {
                "input_name": tensor_name,
                "output_name": f"next_{tensor_name}",
                "initial": _write_fixture_tensor(
                    fixture_dir,
                    f"state/initial/{tensor_name}.bin",
                    initial,
                ),
                "after_q2": _write_fixture_tensor(
                    fixture_dir,
                    f"expected/after-q2/{tensor_name}.bin",
                    q2_state,
                ),
                "after_q1": _write_fixture_tensor(
                    fixture_dir,
                    f"expected/after-q1/{tensor_name}.bin",
                    q1_state,
                ),
            }
        )

    engine_payload = engine_path.read_bytes()
    manifest = {
        "format": "voxtream-explicit-kv-trajectory-v1",
        "engine": {
            "path": str(engine_path),
            "bytes": len(engine_payload),
            "sha256": hashlib.sha256(engine_payload).hexdigest(),
        },
        "batch_size": batch_size,
        "sequence": [2, 1],
        "inputs": inputs,
        "states": states,
        "expected_outputs": expected_outputs,
        "expected_extra_outputs": expected_extra_outputs,
        "capture_runtime": runtime.metrics(),
    }
    manifest_path = fixture_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return {
        "fixture_dir": str(fixture_dir),
        "manifest": str(manifest_path),
        "files": 2 + 6 + 3 * len(states) + sum(
            len(outputs) for outputs in expected_extra_outputs.values()
        ),
        "engine_sha256": manifest["engine"]["sha256"],
    }


def attempt(name: str, operation, results: dict) -> object | None:
    started = time.perf_counter()
    try:
        value = operation()
        results[name] = {"ok": True, "seconds": round(time.perf_counter() - started, 3)}
        return value
    except Exception as error:
        results[name] = {
            "ok": False,
            "seconds": round(time.perf_counter() - started, 3),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(limit=12),
        }
        return None


def main() -> None:
    args = parse_args()
    if args.dynamic_sequence and args.sequence_length != 2:
        raise ValueError("--dynamic-sequence requires --sequence-length 2")
    artifact_stem = (
        "dep-step-explicit-kv-dynamic-q1-q2"
        if args.dynamic_sequence
        else (
            "dep-step-explicit-kv"
            if args.sequence_length == 1
            else "dep-init-explicit-kv"
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_kv_update_exportable()
    model = build_dep_former(args.checkpoint, args.device, args.batch_size)
    names = kv_buffer_names(model)
    if len(names) != 12:
        raise RuntimeError(f"expected 12 explicit KV buffers, got {len(names)}: {names}")
    wrapper = ExplicitKVStep(model, names).eval()

    hidden, input_pos, mask = prepare_inputs(
        model, args.batch_size, args.sequence_length, args.device
    )
    buffers = dict(model.named_buffers())
    kv_state = tuple(buffers[name].clone() for name in names)
    with torch.inference_mode():
        reference = model(hidden, input_pos=input_pos, mask=mask)
        explicit_output = wrapper(hidden, input_pos, mask, *kv_state)
    torch.cuda.synchronize()

    results: dict[str, object] = {
        "torch": torch.__version__,
        "sequence_length": args.sequence_length,
        "dynamic_sequence": args.dynamic_sequence,
        "sequence_range": [1, 2] if args.dynamic_sequence else None,
        "buffer_names": list(names),
        "input_count": 3 + len(kv_state),
        "output_count": len(explicit_output),
        "eager_equivalence": {
            "hidden_max_abs": float((reference - explicit_output[0]).abs().max().item()),
            "hidden_equal": bool(torch.equal(reference, explicit_output[0])),
            "state_equal": [
                bool(torch.equal(dict(model.named_buffers())[name], value))
                for name, value in zip(names, explicit_output[1:])
            ],
        },
    }

    if args.engine is not None and not args.skip_validation:
        validation_name = (
            "tensorrt_closed_loop"
            if args.sequence_length == 1
            else "tensorrt_init"
        )
        validation = attempt(
            validation_name,
            lambda: (
                validate_closed_loop(
                    args.engine,
                    args.checkpoint,
                    model,
                    names,
                    args.batch_size,
                    args.device,
                )
                if args.sequence_length == 1
                else validate_init(
                    args.engine,
                    args.checkpoint,
                    model,
                    names,
                    args.batch_size,
                    args.device,
                )
            ),
            results,
        )
        if validation is not None:
            results[validation_name].update(validation)

    if args.init_engine is not None and not args.skip_validation:
        if args.engine is None or args.sequence_length != 1:
            raise ValueError(
                "--init-engine requires a q=1 --engine and --sequence-length 1"
            )
        validation = attempt(
            "tensorrt_full_trajectory",
            lambda: validate_full_trajectory(
                args.engine,
                args.init_engine,
                args.checkpoint,
                model,
                names,
                args.batch_size,
                args.device,
                args.validate_cuda_graph,
            ),
            results,
        )
        if validation is not None:
            results["tensorrt_full_trajectory"].update(validation)

    if args.standalone_fixture_dir is not None:
        if (
            args.engine is None
            or args.init_engine is None
            or args.engine.resolve() != args.init_engine.resolve()
        ):
            raise ValueError(
                "--standalone-fixture-dir requires the same unified plan in "
                "--engine and --init-engine"
            )
        fixture = attempt(
            "standalone_fixture",
            lambda: capture_standalone_fixture(
                args.standalone_fixture_dir,
                args.engine,
                model,
                names,
                args.batch_size,
                args.device,
            ),
            results,
        )
        if fixture is not None:
            results["standalone_fixture"].update(fixture)

    if args.skip_export:
        metrics = args.output_dir / f"{artifact_stem}.json"
        metrics.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    inputs = (hidden, input_pos, mask, *tuple(buffers[name].clone() for name in names))
    dynamic_shapes = None
    if args.dynamic_sequence:
        sequence = torch.export.Dim("sequence", min=1, max=2)
        dynamic_shapes = (
            {1: sequence},
            {1: sequence},
            {1: sequence},
            tuple(None for _ in names),
        )
    exported = attempt(
        "torch_export",
        lambda: torch.export.export(
            wrapper,
            inputs,
            dynamic_shapes=dynamic_shapes,
            strict=False,
        ),
        results,
    )
    if exported is not None:
        results["torch_export"].update(
            {
                "graph_nodes": len(list(exported.graph.nodes)),
                "input_mutations": [
                    str(spec)
                    for spec in exported.graph_signature.output_specs
                    if "MUTATION" in str(spec.kind)
                ],
            }
        )

    onnx_path = args.output_dir / f"{artifact_stem}.onnx"
    input_names = ["hidden", "input_pos", "mask"] + [
        name.replace(".", "_") for name in names
    ]
    output_names = ["output"] + [f"next_{name}" for name in input_names[3:]]
    onnx_result = attempt(
        "onnx_export",
        lambda: torch.onnx.export(
            exported,
            (),
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamo=True,
            external_data=False,
        ),
        results,
    ) if exported is not None else None
    del onnx_result

    if results.get("onnx_export", {}).get("ok"):
        import onnx

        onnx_model = onnx.load(onnx_path, load_external_data=False)
        graph = onnx_model.graph
        removed_default_reductions = 0
        for node in graph.node:
            if node.op_type != "ScatterND":
                continue
            kept_attributes = []
            for attribute in node.attribute:
                if attribute.name == "reduction" and attribute.s == b"none":
                    removed_default_reductions += 1
                else:
                    kept_attributes.append(attribute)
            del node.attribute[:]
            node.attribute.extend(kept_attributes)
        if removed_default_reductions:
            onnx.save(onnx_model, onnx_path)

        initializers = {item.name for item in graph.initializer}
        graph_inputs = {item.name for item in graph.input}
        results["onnx_graph"] = {
            "inputs": len(graph.input),
            "outputs": len(graph.output),
            "nodes": len(graph.node),
            "initializers": len(graph.initializer),
            "state_inputs_present": all(name in graph_inputs for name in input_names[3:]),
            "state_inputs_are_not_initializers": all(
                name not in initializers for name in input_names[3:]
            ),
            "removed_default_scatternd_reductions": removed_default_reductions,
            "input_shapes": {
                item.name: [
                    (
                        dimension.dim_param
                        if dimension.dim_param
                        else dimension.dim_value
                    )
                    for dimension in item.type.tensor_type.shape.dim
                ]
                for item in graph.input
            },
        }
        attempt("onnx_check", lambda: onnx.checker.check_model(onnx_path), results)

        def parse_tensorrt() -> int:
            import tensorrt as trt

            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            if not parser.parse_from_file(str(onnx_path)):
                errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError("\n".join(errors))
            return network.num_layers

        layers = attempt("tensorrt_parse", parse_tensorrt, results)
        if layers is not None:
            results["tensorrt_parse"]["layers"] = layers

    metrics = args.output_dir / f"{artifact_stem}.json"
    metrics.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
