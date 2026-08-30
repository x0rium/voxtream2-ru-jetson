#!/usr/bin/env python3
"""PyTorch facade over the raw-CUDA VoXtream audio embedding lookup."""

from __future__ import annotations

from pathlib import Path

import torch

from voxtream2_ru_jetson.cuda_audio_embedding import CudaAudioEmbeddingCore


class CudaAudioEmbeddingFacade(torch.nn.Module):
    def __init__(
        self,
        weight_path: Path,
        cubin_path: Path,
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.core = CudaAudioEmbeddingCore(
            weight_path,
            cubin_path,
            self.num_embeddings,
            self.embedding_dim,
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.device.type != "cuda":
            raise ValueError("CUDA audio embedding requires CUDA indices")
        if indices.dtype != torch.int64:
            raise TypeError(f"expected int64 indices, got {indices.dtype}")
        if not indices.is_contiguous():
            indices = indices.contiguous()
        output = torch.empty(
            (*indices.shape, self.embedding_dim),
            device=indices.device,
            dtype=torch.bfloat16,
        )
        self.core.launch(
            indices.data_ptr(),
            output.data_ptr(),
            indices.numel(),
            torch.cuda.current_stream(indices.device).cuda_stream,
        )
        return output

    def metrics(self) -> dict[str, object]:
        return self.core.metrics()
