#!/usr/bin/env python3
"""Localize temp TensorRT error on the first real captured step."""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from voxtream_tensorrt_explicit_kv_probe import ExplicitKVTRTRunner
from voxtream_tensorrt_temp_probe import build_temp_former, kv_buffer_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def patch_efficient_attention(transformer: torch.nn.Module) -> None:
    for module in transformer.modules():
        attention_call = getattr(module, "_attention_call", None)
        if attention_call is None:
            continue

        @functools.wraps(attention_call)
        def efficient_attention(*args, _attention_call=attention_call, **kwargs):
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                return _attention_call(*args, **kwargs)

        module._attention_call = efficient_attention


def main() -> None:
    args = parse_args()
    payload = torch.load(args.capture, map_location="cpu", weights_only=False)
    model = build_temp_former(args.checkpoint, "cuda", batch_size=2)
    patch_efficient_attention(model)
    names = kv_buffer_names(model)
    buffers = dict(model.named_buffers())
    for name, captured in zip(names, payload["initial_state"]):
        buffers[name].copy_(captured.to(device="cuda"))
    runner = ExplicitKVTRTRunner(
        args.engine,
        names,
        tuple(buffers[name] for name in names),
        sequence_length=1,
    )

    captured = payload["records"][0]
    hidden = captured["hidden"].to(device="cuda")
    input_pos = captured["input_pos"].to(device="cuda")
    mask = captured["mask"].to(device="cuda")
    with torch.inference_mode():
        # The capture already contains the exact accepted PyTorch output.  Do
        # not execute the full reference transformer here: keeping PyTorch and
        # the TensorRT engine resident simultaneously is close enough to the
        # 8 GB Jetson limit that SDPA's temporary allocation can OOM.
        reference = captured["reference_output"].to(device="cuda")
        candidate, candidate_state = runner.step(hidden, input_pos, mask)
        # runner.step() reuses its TensorRT output buffers.  Keep the first real
        # step intact before running the diagnostic position sweep below.
        candidate = candidate.clone()
        candidate_state = tuple(value.clone() for value in candidate_state)
        layer = model.layers[0]
        normalized = layer.sa_norm(hidden)
        attention = layer.attn
        batch_size, sequence_length, _ = normalized.shape
        raw_k = attention.k_proj(normalized).view(
            batch_size,
            sequence_length,
            attention.num_kv_heads,
            attention.head_dim,
        )
        q_per_kv = attention.num_heads // attention.num_kv_heads
        written_position = int(payload["initial_state"][2][0].item())

        def expected_k_for_position(position: int) -> torch.Tensor:
            position_tensor = torch.full_like(input_pos, position)
            expected_k = attention.pos_embeddings(
                raw_k, input_pos=position_tensor
            )
            expected_k = (
                expected_k.view(
                    batch_size,
                    sequence_length,
                    attention.num_kv_heads,
                    1,
                    attention.head_dim,
                )
                .expand(
                    batch_size,
                    sequence_length,
                    attention.num_kv_heads,
                    q_per_kv,
                    attention.head_dim,
                )
                .reshape(
                    batch_size,
                    sequence_length,
                    attention.num_heads,
                    attention.head_dim,
                )
                .transpose(1, 2)[:, :, 0, :]
            )
            if attention.k_norm is not None:
                expected_k = attention.k_norm(expected_k.unsqueeze(2)).squeeze(2)
            return expected_k

        candidate_written_k = candidate_state[0][
            :, :, written_position, :
        ]
        rope_position_errors = []
        pair_position_errors = [[] for _ in range(attention.head_dim // 2)]
        for position in range(2048):
            expected_k = expected_k_for_position(position)
            delta = expected_k.float() - candidate_written_k.float()
            rope_position_errors.append(
                {
                    "position": position,
                    "max_abs": float(delta.abs().max().item()),
                    "mean_abs": float(delta.abs().mean().item()),
                }
            )
            active_candidate = candidate_written_k[:batch_size, ::q_per_kv].float()
            active_expected = expected_k[:, ::q_per_kv].float()
            for pair_index in range(attention.head_dim // 2):
                pair_slice = slice(2 * pair_index, 2 * pair_index + 2)
                pair_delta = (
                    active_expected[..., pair_slice]
                    - active_candidate[..., pair_slice]
                )
                pair_position_errors[pair_index].append(
                    {
                        "position": position,
                        "max_abs": float(pair_delta.abs().max().item()),
                        "mean_abs": float(pair_delta.abs().mean().item()),
                    }
                )

        dynamic_position_checks = []
        positions = (0, 1, 2, 15, 16, 31, 32, 63, 64, 95, 96, 107, 108)
        for position in positions:
            runner.copy_state_from(
                tuple(payload["initial_state"])
            )
            runtime_position = torch.full_like(input_pos, position)
            _, runtime_state = runner.step(hidden, runtime_position, mask)
            runtime_k = runtime_state[0][:, :, written_position, :]
            expected_k = expected_k_for_position(position)
            delta = expected_k.float() - runtime_k.float()
            dynamic_position_checks.append(
                {
                    "position": position,
                    "max_abs": float(delta.abs().max().item()),
                    "mean_abs": float(delta.abs().mean().item()),
                }
            )
    output_delta = reference.float() - candidate.float()
    result = {
        "position": int(captured["position"]),
        "output_max_abs": float(output_delta.abs().max().item()),
        "output_mean_abs": float(output_delta.abs().mean().item()),
        "best_matching_rope_positions": sorted(
            rope_position_errors, key=lambda item: item["mean_abs"]
        )[:10],
        "best_position_by_rope_pair": [
            {
                "pair": pair_index,
                **min(errors, key=lambda item: item["mean_abs"]),
            }
            for pair_index, errors in enumerate(pair_position_errors)
        ],
        "dynamic_position_checks": dynamic_position_checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
