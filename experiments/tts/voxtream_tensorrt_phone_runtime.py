"""TensorRT runtime for VoXtream phone embeddings + phone_former."""

from __future__ import annotations

import os
from pathlib import Path

import torch


class TensorRTPhoneEncoder:
    def __init__(self, engine_path: Path, preloaded=None) -> None:
        from voxtream_tensorrt_runtime import PreloadedTensorRTEngine

        self.engine_path = Path(engine_path)
        loaded = preloaded or PreloadedTensorRTEngine(self.engine_path)
        if loaded.engine_path != self.engine_path:
            raise ValueError(
                f"preloaded {loaded.engine_path}, requested {self.engine_path}"
            )
        self.runtime = loaded.runtime
        self.engine = loaded.engine
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("failed to create phone TensorRT context")
        self.output: torch.Tensor | None = None
        self.calls = 0
        self.sequence_lengths: list[int] = []
        minimum, optimum, maximum = self.engine.get_tensor_profile_shape(
            "phone_tokens", 0
        )
        self.profile = {
            "minimum": tuple(minimum),
            "optimum": tuple(optimum),
            "maximum": tuple(maximum),
        }

    def __call__(
        self,
        phone_tokens: torch.Tensor,
        input_pos: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        inputs = {
            "phone_tokens": phone_tokens,
            "input_pos": input_pos,
            "mask": mask,
        }
        prepared = {
            name: value if value.is_contiguous() else value.contiguous()
            for name, value in inputs.items()
        }
        sequence_length = int(phone_tokens.shape[1])
        maximum = self.profile["maximum"]
        if sequence_length > maximum[1]:
            raise ValueError(
                f"phone sequence {sequence_length} exceeds TensorRT profile "
                f"maximum {maximum[1]}"
            )
        for name, value in prepared.items():
            if not self.context.set_input_shape(name, tuple(value.shape)):
                raise RuntimeError(
                    f"failed to set phone TensorRT input {name}={tuple(value.shape)}"
                )
        output_shape = tuple(self.context.get_tensor_shape("phone_embeddings"))
        if -1 in output_shape:
            raise RuntimeError(f"unresolved phone output shape: {output_shape}")
        if self.output is None or tuple(self.output.shape) != output_shape:
            self.output = torch.empty(
                output_shape, device=phone_tokens.device, dtype=torch.bfloat16
            )
        bindings = {**prepared, "phone_embeddings": self.output}
        for name, value in bindings.items():
            if not self.context.set_tensor_address(name, value.data_ptr()):
                raise RuntimeError(f"failed to bind phone TensorRT tensor {name}")
        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("phone TensorRT enqueue failed")
        self.calls += 1
        self.sequence_lengths.append(sequence_length)
        return self.output

    def metrics(self) -> dict[str, object]:
        return {
            "engine": str(self.engine_path),
            "engine_bytes": self.engine_path.stat().st_size,
            "profile": {key: list(value) for key, value in self.profile.items()},
            "calls": self.calls,
            "sequence_lengths": self.sequence_lengths,
        }


class PhoneEmbeddingFacade:
    """Preserve extract_phoneme_embeddings semantics around a TensorRT encoder."""

    def __init__(self, model, runtime: TensorRTPhoneEncoder) -> None:
        self.model = model
        self.runtime = runtime

    def __call__(
        self,
        phone_tokens: torch.Tensor,
        input_pos: torch.Tensor | None = None,
        phoneme_embedding_indices: torch.Tensor | None = None,
        prompt_len: int | None = None,
    ) -> torch.Tensor:
        sequence_length = int(phone_tokens.shape[1])
        if input_pos is None:
            input_pos = torch.arange(
                sequence_length, device=phone_tokens.device, dtype=torch.int64
            ).unsqueeze(0).repeat(phone_tokens.shape[0], 1)

        la_env = os.environ.get("VOXTREAM_LA")
        if la_env:
            if la_env.lower() in ("full", "offline", "max"):
                look_ahead = int(self.model.la_values[-1])
            else:
                requested = max(int(la_env), 1)
                look_ahead = int(
                    min(self.model.la_values, key=lambda value: abs(value - requested))
                )
        else:
            text_length = sequence_length - (prompt_len or 0)
            index = min(
                text_length // self.model.config.max_look_ahead,
                self.model.config.max_look_ahead - 1,
            )
            look_ahead = int(self.model.la_values[index])

        rows = input_pos.unsqueeze(-1)
        columns = torch.arange(
            sequence_length, device=phone_tokens.device, dtype=input_pos.dtype
        ).view(1, 1, -1)
        mask = (
            (columns >= rows - self.model.config.phone_window_size + 1)
            & (columns <= rows + look_ahead)
        )
        phone_emb = self.runtime(phone_tokens, input_pos, mask)
        if phoneme_embedding_indices is not None:
            phone_emb = self.model.reorder_phone_emb(
                phone_emb, phoneme_embedding_indices
            )
        return phone_emb


class TensorRTPhoneFormerFacade(torch.nn.Module):
    """Weightless marker replacing the original phone transformer module."""

    def __init__(self, runtime: TensorRTPhoneEncoder) -> None:
        super().__init__()
        self.runtime = runtime
        self.max_seq_len = 2048

    def forward(self, *args, **kwargs):
        raise RuntimeError("phone_former.forward bypassed TensorRT token encoder")
