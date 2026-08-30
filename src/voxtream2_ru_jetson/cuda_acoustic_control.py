#!/usr/bin/env python3
"""PyTorch-free CUDA control kernel for the dep_former acoustic loop."""

from __future__ import annotations

import ctypes
from pathlib import Path

from .cuda_audio_embedding import CudaDriver


class CudaAcousticControl:
    def __init__(self, cubin_path: Path) -> None:
        self.cubin_path = Path(cubin_path)
        self.driver = CudaDriver()
        self.module = ctypes.c_void_p()
        self.function = ctypes.c_void_p()
        self.driver.check(
            self.driver.library.cuModuleLoad(
                ctypes.byref(self.module), str(self.cubin_path).encode()
            ),
            "cuModuleLoad(acoustic control)",
        )
        self.driver.check(
            self.driver.library.cuModuleGetFunction(
                ctypes.byref(self.function),
                self.module,
                b"acoustic_cfg_argmax_embed",
            ),
            "cuModuleGetFunction(acoustic_cfg_argmax_embed)",
        )
        self.calls = 0
        self.closed = False

    def launch(
        self,
        logits_pointer: int,
        embedding_weight_pointer: int,
        frame_codes_pointer: int,
        codebook: int,
        cfg_gamma: float,
        hidden_pointer: int,
        stream: int,
    ) -> None:
        logits_arg = ctypes.c_uint64(int(logits_pointer))
        weight_arg = ctypes.c_uint64(int(embedding_weight_pointer))
        codes_arg = ctypes.c_uint64(int(frame_codes_pointer))
        codebook_arg = ctypes.c_int32(int(codebook))
        gamma_arg = ctypes.c_float(float(cfg_gamma))
        hidden_arg = ctypes.c_uint64(int(hidden_pointer))
        arguments = (ctypes.c_void_p * 6)(
            ctypes.cast(ctypes.byref(logits_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(weight_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(codes_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(codebook_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(gamma_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(hidden_arg), ctypes.c_void_p),
        )
        self.driver.check(
            self.driver.library.cuLaunchKernel(
                self.function,
                1,
                1,
                1,
                256,
                1,
                1,
                0,
                ctypes.c_void_p(int(stream)),
                arguments,
                None,
            ),
            "cuLaunchKernel(acoustic_cfg_argmax_embed)",
        )
        self.calls += 1

    def close(self) -> None:
        if self.closed:
            return
        if self.module:
            self.driver.check(
                self.driver.library.cuModuleUnload(self.module),
                "cuModuleUnload(acoustic control)",
            )
            self.module = ctypes.c_void_p()
        self.closed = True
