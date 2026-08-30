#!/usr/bin/env python3
"""PyTorch-free CUDA driver for VoXtream's raw BF16 audio embedding table."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from cuda import cudart


def cuda_check(result, operation: str):
    error, *values = result
    if error != cudart.cudaError_t.cudaSuccess:
        _, name = cudart.cudaGetErrorName(error)
        _, message = cudart.cudaGetErrorString(error)
        raise RuntimeError(
            f"{operation} failed: {name.decode()} ({message.decode()})"
        )
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


class CudaDriver:
    """Minimal CUDA Driver API surface for loading and launching one cubin."""

    def __init__(self) -> None:
        self.library = ctypes.CDLL("libcuda.so.1")
        self.library.cuModuleLoad.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
        ]
        self.library.cuModuleLoad.restype = ctypes.c_int
        self.library.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        self.library.cuModuleGetFunction.restype = ctypes.c_int
        self.library.cuModuleUnload.argtypes = [ctypes.c_void_p]
        self.library.cuModuleUnload.restype = ctypes.c_int
        self.library.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        self.library.cuLaunchKernel.restype = ctypes.c_int
        self.library.cuGetErrorName.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.library.cuGetErrorName.restype = ctypes.c_int
        self.library.cuGetErrorString.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self.library.cuGetErrorString.restype = ctypes.c_int

    def check(self, error: int, operation: str) -> None:
        if error == 0:
            return
        name = ctypes.c_char_p()
        message = ctypes.c_char_p()
        self.library.cuGetErrorName(error, ctypes.byref(name))
        self.library.cuGetErrorString(error, ctypes.byref(message))
        error_name = name.value.decode() if name.value else str(error)
        error_message = message.value.decode() if message.value else "unknown"
        raise RuntimeError(
            f"{operation} failed: {error_name} ({error_message})"
        )


class CudaAudioEmbeddingCore:
    """Own one raw BF16 table and launch exact bit-copy embedding gathers."""

    def __init__(
        self,
        weight_path: Path,
        cubin_path: Path,
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        self.weight_path = Path(weight_path)
        self.cubin_path = Path(cubin_path)
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.weight_nbytes = self.num_embeddings * self.embedding_dim * 2
        if self.weight_path.stat().st_size != self.weight_nbytes:
            raise ValueError(
                f"audio embedding file has {self.weight_path.stat().st_size} bytes; "
                f"expected {self.weight_nbytes}"
            )

        # Ensure the CUDA runtime has retained the primary context before the
        # Driver API loads the cubin into the current context.
        cuda_check(cudart.cudaFree(0), "cudaFree(0) context initialization")
        self.driver = CudaDriver()
        self.module = ctypes.c_void_p()
        self.function = ctypes.c_void_p()
        self.driver.check(
            self.driver.library.cuModuleLoad(
                ctypes.byref(self.module),
                str(self.cubin_path).encode(),
            ),
            "cuModuleLoad(audio embedding)",
        )
        self.driver.check(
            self.driver.library.cuModuleGetFunction(
                ctypes.byref(self.function),
                self.module,
                b"gather_bf16_words",
            ),
            "cuModuleGetFunction(gather_bf16_words)",
        )

        self.weight_pointer = int(
            cuda_check(cudart.cudaMalloc(self.weight_nbytes), "cudaMalloc(weights)")
        )
        weight_words = np.memmap(self.weight_path, mode="r", dtype=np.uint16)
        cuda_check(
            cudart.cudaMemcpy(
                self.weight_pointer,
                weight_words.ctypes.data,
                self.weight_nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            ),
            "cudaMemcpy(audio embedding weights H2D)",
        )
        del weight_words
        self.calls = 0
        self.indices = 0
        self.closed = False

    def launch(
        self,
        indices_pointer: int,
        output_pointer: int,
        index_count: int,
        stream: int,
    ) -> None:
        if self.closed:
            raise RuntimeError("audio embedding runtime is closed")
        index_count = int(index_count)
        element_count = index_count * self.embedding_dim
        threads = 256
        blocks = (element_count + threads - 1) // threads

        weight_arg = ctypes.c_uint64(self.weight_pointer)
        indices_arg = ctypes.c_uint64(int(indices_pointer))
        output_arg = ctypes.c_uint64(int(output_pointer))
        count_arg = ctypes.c_int64(index_count)
        dim_arg = ctypes.c_int32(self.embedding_dim)
        arguments = (ctypes.c_void_p * 5)(
            ctypes.cast(ctypes.byref(weight_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(indices_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(output_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(count_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(dim_arg), ctypes.c_void_p),
        )
        self.driver.check(
            self.driver.library.cuLaunchKernel(
                self.function,
                blocks,
                1,
                1,
                threads,
                1,
                1,
                0,
                ctypes.c_void_p(int(stream)),
                arguments,
                None,
            ),
            "cuLaunchKernel(gather_bf16_words)",
        )
        self.calls += 1
        self.indices += index_count

    def metrics(self) -> dict[str, object]:
        return {
            "backend": "raw_bf16_cuda",
            "weight": str(self.weight_path),
            "weight_bytes": self.weight_nbytes,
            "cubin": str(self.cubin_path),
            "cubin_bytes": self.cubin_path.stat().st_size,
            "shape": [self.num_embeddings, self.embedding_dim],
            "calls": self.calls,
            "indices": self.indices,
        }

    def close(self) -> None:
        if self.closed:
            return
        if self.weight_pointer:
            cuda_check(cudart.cudaFree(self.weight_pointer), "cudaFree(weights)")
            self.weight_pointer = 0
        if self.module:
            self.driver.check(
                self.driver.library.cuModuleUnload(self.module),
                "cuModuleUnload(audio embedding)",
            )
            self.module = ctypes.c_void_p()
        self.closed = True
