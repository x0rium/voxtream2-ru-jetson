#!/usr/bin/env python3
"""Probe CUDA Graph compatibility of PyTorch SDPA backends on Jetson."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def warmup_and_capture(function):
    """Warm up and capture on one side stream, per NVIDIA guidance."""
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            function()
    side_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side_stream):
        captured = function()
    return captured, graph


def probe(name: str, backend: SDPBackend | None, query_length: int) -> dict:
    torch.manual_seed(7)
    device = torch.device("cuda")
    q = torch.randn(2, 8, query_length, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn(2, 8, 16, 128, device=device, dtype=torch.bfloat16)
    v = torch.randn(2, 8, 16, 128, device=device, dtype=torch.bfloat16)
    mask = torch.ones(2, 1, query_length, 16, device=device, dtype=torch.bool)

    def attention():
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
        )

    context = sdpa_kernel([backend]) if backend is not None else None
    try:
        default_eager = attention().clone()
        if context is None:
            eager = attention()
            captured, graph = warmup_and_capture(attention)
        else:
            with context:
                eager = attention()
                captured, graph = warmup_and_capture(attention)
        graph.replay()
        torch.cuda.synchronize()
        max_abs_diff = float((captured - eager).abs().max().float().cpu())
        default_max_abs_diff = float((captured - default_eager).abs().max().float().cpu())
        return {
            "backend": name,
            "query_length": query_length,
            "capture": "ok",
            "exact_equal": bool(torch.equal(captured, eager)),
            "max_abs_diff": max_abs_diff,
            "default_eager_exact_equal": bool(torch.equal(captured, default_eager)),
            "default_eager_max_abs_diff": default_max_abs_diff,
        }
    except Exception as error:  # noqa: BLE001 - this is a compatibility probe
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        return {
            "backend": name,
            "query_length": query_length,
            "capture": "failed",
            "error": f"{type(error).__name__}: {error}",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=("default", "flash", "efficient", "math", "cudnn"), required=True
    )
    parser.add_argument("--query-length", type=int, choices=(1, 2), required=True)
    parser.add_argument("--inner-context", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    backends = {
        "default": None,
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
        "cudnn": getattr(SDPBackend, "CUDNN_ATTENTION", None),
    }
    if args.backend == "cudnn" and backends["cudnn"] is None:
        raise RuntimeError("CUDNN_ATTENTION is unavailable in this PyTorch build")

    selected_backend = backends[args.backend]
    if args.inner_context and selected_backend is not None:
        def probe_with_inner_context(name, backend, query_length):
            # Reuse probe machinery with a backend function whose selection
            # happens inside the callable being captured.
            torch.manual_seed(7)
            device = torch.device("cuda")
            q = torch.randn(2, 8, query_length, 128, device=device, dtype=torch.bfloat16)
            k = torch.randn(2, 8, 16, 128, device=device, dtype=torch.bfloat16)
            v = torch.randn(2, 8, 16, 128, device=device, dtype=torch.bfloat16)
            mask = torch.ones(2, 1, query_length, 16, device=device, dtype=torch.bool)

            def default_attention():
                return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

            def selected_attention():
                with sdpa_kernel([backend]):
                    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

            try:
                default_eager = default_attention().clone()
                eager = selected_attention()
                captured, graph = warmup_and_capture(selected_attention)
                graph.replay()
                torch.cuda.synchronize()
                return {
                    "backend": name,
                    "query_length": query_length,
                    "inner_context": True,
                    "capture": "ok",
                    "exact_equal": bool(torch.equal(captured, eager)),
                    "default_eager_exact_equal": bool(torch.equal(captured, default_eager)),
                    "default_eager_max_abs_diff": float(
                        (captured - default_eager).abs().max().float().cpu()
                    ),
                }
            except Exception as error:  # noqa: BLE001
                return {
                    "backend": name,
                    "query_length": query_length,
                    "inner_context": True,
                    "capture": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }

        test = probe_with_inner_context(args.backend, selected_backend, args.query_length)
    else:
        test = probe(args.backend, selected_backend, args.query_length)

    result = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_enabled": torch.backends.cuda.math_sdp_enabled(),
        "test": test,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
