#!/usr/bin/env python3
"""Probe q=2 TensorRT error across input scales and actual-like structure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from voxtream_tensorrt_explicit_kv_probe import (
    ExplicitKVTRTRunner,
    kv_buffer_names,
)
from voxtream_tensorrt_probe import (
    build_dep_former,
    causal_rows,
    make_kv_update_exportable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    value = tensor.float()
    return {
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
        "std": float(value.std().item()),
        "rms": float(value.square().mean().sqrt().item()),
    }


def main() -> None:
    args = parse_args()
    make_kv_update_exportable()
    model = build_dep_former(args.checkpoint, args.device, args.batch_size)
    names = kv_buffer_names(model)
    model.reset_caches()
    initial_state = tuple(
        dict(model.named_buffers())[name].clone() for name in names
    )
    runner = ExplicitKVTRTRunner(args.engine, names, initial_state, 2)

    with safe_open(args.checkpoint, framework="pt", device="cpu") as source:
        audio_head = source.get_tensor("audio_head")[0].to(
            device=args.device, dtype=torch.bfloat16
        )
        audio_embeddings = source.get_tensor("audio_embeddings.weight").to(
            device=args.device, dtype=torch.bfloat16
        )

    torch.manual_seed(1234)
    base = torch.randn(
        args.batch_size, 2, 1024, device=args.device, dtype=torch.bfloat16
    )
    embedding = audio_embeddings[1056].view(1, 1, 1024).repeat(
        args.batch_size, 1, 1
    )
    correlated = torch.randn(
        1, 1, 1024, device=args.device, dtype=torch.bfloat16
    ).repeat(args.batch_size, 1, 1)
    independent = torch.randn(
        args.batch_size, 1, 1024, device=args.device, dtype=torch.bfloat16
    )

    cases: list[tuple[str, torch.Tensor]] = []
    for scale in (0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0):
        cases.append((f"iid_scale_{scale:g}", base * scale))
    for scale in (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0):
        cases.append(
            (
                f"structured_independent_first_scale_{scale:g}",
                torch.cat([independent * scale, embedding], dim=1),
            )
        )
        cases.append(
            (
                f"structured_correlated_first_scale_{scale:g}",
                torch.cat([correlated * scale, embedding], dim=1),
            )
        )

    input_pos = torch.arange(2, device=args.device).unsqueeze(0).repeat(
        args.batch_size, 1
    )
    mask = causal_rows(input_pos)
    results: list[dict[str, object]] = []
    for name, hidden in cases:
        model.reset_caches()
        for initial, value in zip(initial_state, runner.state):
            value.copy_(initial)
        for initial, value in zip(initial_state, runner.next_state):
            value.copy_(initial)
        with torch.inference_mode():
            reference = model(hidden, input_pos=input_pos, mask=mask)
            candidate, candidate_state = runner.step(hidden, input_pos, mask)

        reference_logits = torch.mm(
            reference[:, -1, :].to(torch.bfloat16), audio_head
        )
        candidate_logits = torch.mm(
            candidate[:, -1, :].to(torch.bfloat16), audio_head
        )
        reference_cfg = 3.0 * reference_logits[0].float() - 2.0 * reference_logits[1].float()
        candidate_cfg = 3.0 * candidate_logits[0].float() - 2.0 * candidate_logits[1].float()
        reference_token = int(torch.argmax(reference_cfg).item())
        candidate_token = int(torch.argmax(candidate_cfg).item())
        reference_state = dict(model.named_buffers())
        state_max_abs = max(
            float(
                (reference_state[state_name].float() - candidate_value.float())
                .abs()
                .max()
                .item()
            )
            for state_name, candidate_value in zip(names, candidate_state)
        )
        results.append(
            {
                "case": name,
                "position_0": tensor_stats(hidden[:, 0, :]),
                "position_1": tensor_stats(hidden[:, 1, :]),
                "hidden_max_abs": float(
                    (reference.float() - candidate.float()).abs().max().item()
                ),
                "cfg_logits_max_abs": float(
                    (reference_cfg - candidate_cfg).abs().max().item()
                ),
                "reference_token": reference_token,
                "candidate_token": candidate_token,
                "token_equal": reference_token == candidate_token,
                "state_max_abs": state_max_abs,
            }
        )

    payload = {
        "checkpoint": str(args.checkpoint),
        "engine": str(args.engine),
        "audio_embedding": tensor_stats(embedding),
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
