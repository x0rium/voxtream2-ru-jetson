#!/usr/bin/env python3
"""Compare an extracted first temp_former block TensorRT engine to PyTorch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorrt as trt
import torch
from voxtream_tensorrt_temp_probe import build_temp_former, kv_buffer_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefill-state-outputs", action="store_true")
    return parser.parse_args()


def torch_dtype(dtype: trt.DataType) -> torch.dtype:
    return {
        trt.DataType.BF16: torch.bfloat16,
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.INT32: torch.int32,
        trt.DataType.BOOL: torch.bool,
    }[dtype]


def main() -> None:
    args = parse_args()
    payload = torch.load(args.capture, map_location="cpu", weights_only=False)
    model = build_temp_former(args.checkpoint, "cuda", batch_size=2)
    names = kv_buffer_names(model)
    buffers = dict(model.named_buffers())
    for name, captured in zip(names, payload["initial_state"]):
        buffers[name].copy_(captured.to(device="cuda"))
    captured = payload["records"][0]
    hidden = captured["hidden"].to(device="cuda")
    input_pos = captured["input_pos"].to(device="cuda")
    mask = captured["mask"].to(device="cuda")
    with torch.inference_mode():
        reference = model.layers[0](
            hidden, input_pos=input_pos, mask=mask
        )

    inputs = {"hidden": hidden, "input_pos": input_pos, "mask": mask}
    inputs.update(
        (name.replace(".", "_"), value.to(device="cuda"))
        for name, value in zip(names, payload["initial_state"])
    )
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    context = engine.create_execution_context()
    for name in ("hidden", "input_pos", "mask"):
        expected = tuple(engine.get_tensor_shape(name))
        if tuple(inputs[name].shape) != expected:
            inputs[name] = inputs[name].expand(expected).contiguous()
    outputs = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
            continue
        outputs[name] = torch.empty(
            tuple(engine.get_tensor_shape(name)),
            device="cuda",
            dtype=torch_dtype(engine.get_tensor_dtype(name)),
        )
    if args.prefill_state_outputs:
        for output_name, output in outputs.items():
            if not output_name.startswith("next_"):
                continue
            input_name = output_name.removeprefix("next_")
            if input_name in inputs:
                output.copy_(inputs[input_name])
    for name, tensor in {**inputs, **outputs}.items():
        if engine.get_tensor_mode(name) in (
            trt.TensorIOMode.INPUT,
            trt.TensorIOMode.OUTPUT,
        ):
            context.set_tensor_address(name, tensor.data_ptr())
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT layer0 block enqueue failed")
    stream.synchronize()

    output_delta = reference.float() - outputs["add_6"].float()
    state = {}
    for name in names[:3]:
        output_name = f"next_{name.replace('.', '_')}"
        delta = buffers[name].float() - outputs[output_name].float()
        state[name] = {
            "max_abs": float(delta.abs().max().item()),
            "mean_abs": float(delta.abs().mean().item()),
        }
    reference_k = buffers[names[0]].float()
    candidate_k = outputs[f"next_{names[0].replace('.', '_')}"] .float()
    attention_k_scale = model.layers[0].attn.head_dim ** -0.25
    scaled_k_delta = candidate_k - reference_k * attention_k_scale
    written_position = int(payload["initial_state"][2][0].item())
    scaled_written_k_delta = scaled_k_delta[:, :, written_position, :]
    initial_k = payload["initial_state"][0].to(device="cuda").float()
    initial_written_k = initial_k[:, :, written_position, :]
    reference_written_k = reference_k[:, :, written_position, :]
    candidate_written_k = candidate_k[:, :, written_position, :]
    candidate_shape = candidate_k.shape

    def layout_error(value: torch.Tensor) -> dict[str, float]:
        value = value.reshape(candidate_shape)
        delta = candidate_k - value
        return {
            "max_abs": float(delta.abs().max().item()),
            "mean_abs": float(delta.abs().mean().item()),
        }

    # TensorRT may legally fuse the K-cache view used by QK^T, but it must not
    # expose that physical view at the canonical graph output.  Check the two
    # layouts adjacent to the public K output explicitly.
    pre_public_transpose = (
        reference_k.permute(1, 0, 2, 3).contiguous().reshape(candidate_shape)
    )
    qk_matmul_view = (
        reference_k.reshape(-1, reference_k.shape[2], reference_k.shape[3])
        .transpose(1, 2)
        .contiguous()
        .reshape(candidate_shape)
    )
    attention = model.layers[0].attn
    normalized_hidden = model.layers[0].sa_norm(hidden)
    active_batch, sequence_length, _ = normalized_hidden.shape
    raw_unique_k = attention.k_proj(normalized_hidden).view(
        active_batch,
        sequence_length,
        attention.num_kv_heads,
        attention.head_dim,
    )

    def expand_heads(value: torch.Tensor) -> torch.Tensor:
        q_per_kv = attention.num_heads // attention.num_kv_heads
        return (
            value.view(
                active_batch,
                sequence_length,
                attention.num_kv_heads,
                1,
                attention.head_dim,
            )
            .expand(
                active_batch,
                sequence_length,
                attention.num_kv_heads,
                q_per_kv,
                attention.head_dim,
            )
            .reshape(
                active_batch,
                sequence_length,
                attention.num_heads,
                attention.head_dim,
            )
            .transpose(1, 2)
        )

    raw_expanded_k = expand_heads(raw_unique_k)
    rope_expanded_k = expand_heads(
        attention.pos_embeddings(raw_unique_k, input_pos=input_pos)
    )
    normalized_raw_k = (
        attention.k_norm(raw_expanded_k)
        if attention.k_norm is not None
        else raw_expanded_k
    )
    normalized_rope_k = (
        attention.k_norm(rope_expanded_k)
        if attention.k_norm is not None
        else rope_expanded_k
    )
    candidate_active_written_k = candidate_written_k[:active_batch]
    rope_cache_probe = None
    if "val_19" in outputs:
        expected_rope_cache = attention.pos_embeddings.cache[input_pos].float()
        candidate_rope_cache = outputs["val_19"].float()
        rope_cache_delta = expected_rope_cache - candidate_rope_cache
        rope_cache_probe = {
            "shape": list(candidate_rope_cache.shape),
            "max_abs": float(rope_cache_delta.abs().max().item()),
            "mean_abs": float(rope_cache_delta.abs().mean().item()),
        }

    def written_error(value: torch.Tensor) -> dict[str, float]:
        value = value[:, :, 0, :].float()
        delta = candidate_active_written_k - value
        return {
            "reference_abs_mean": float(value.abs().mean().item()),
            "max_abs": float(delta.abs().max().item()),
            "mean_abs": float(delta.abs().mean().item()),
        }
    candidate_head_matches = []
    for candidate_head in range(candidate_written_k.shape[1]):
        head_errors = []
        for reference_head in range(reference_written_k.shape[1]):
            delta = (
                candidate_written_k[:, candidate_head, :]
                - reference_written_k[:, reference_head, :]
            )
            head_errors.append(
                {
                    "reference_head": reference_head,
                    "mean_abs": float(delta.abs().mean().item()),
                    "max_abs": float(delta.abs().max().item()),
                }
            )
        candidate_head_matches.append(
            {
                "candidate_head": candidate_head,
                "best": min(head_errors, key=lambda item: item["mean_abs"]),
            }
        )
    candidate_unique_heads = candidate_written_k[:, ::4, :]
    reference_unique_heads = reference_written_k[:, ::4, :]
    candidate_dim_matches = []
    for candidate_dim in range(candidate_unique_heads.shape[-1]):
        matches = []
        for reference_dim in range(reference_unique_heads.shape[-1]):
            candidate_values = candidate_unique_heads[:, :, candidate_dim]
            reference_values = reference_unique_heads[:, :, reference_dim]
            for sign in (1.0, -1.0):
                matches.append(
                    {
                        "reference_dim": reference_dim,
                        "sign": int(sign),
                        "mean_abs": float(
                            (candidate_values - sign * reference_values)
                            .abs()
                            .mean()
                            .item()
                        ),
                    }
                )
        candidate_dim_matches.append(
            {
                "candidate_dim": candidate_dim,
                "best": min(matches, key=lambda item: item["mean_abs"]),
            }
        )
    result = {
        "output_max_abs": float(output_delta.abs().max().item()),
        "output_mean_abs": float(output_delta.abs().mean().item()),
        "attention_k_scale": attention_k_scale,
        "candidate_vs_scaled_reference_k_max_abs": float(
            scaled_k_delta.abs().max().item()
        ),
        "candidate_vs_scaled_reference_k_mean_abs": float(
            scaled_k_delta.abs().mean().item()
        ),
        "candidate_vs_scaled_reference_written_k_max_abs": float(
            scaled_written_k_delta.abs().max().item()
        ),
        "candidate_vs_scaled_reference_written_k_mean_abs": float(
            scaled_written_k_delta.abs().mean().item()
        ),
        "written_k": {
            "initial_abs_mean": float(initial_written_k.abs().mean().item()),
            "reference_abs_mean": float(reference_written_k.abs().mean().item()),
            "candidate_abs_mean": float(candidate_written_k.abs().mean().item()),
            "candidate_vs_initial_max_abs": float(
                (candidate_written_k - initial_written_k).abs().max().item()
            ),
            "candidate_vs_initial_mean_abs": float(
                (candidate_written_k - initial_written_k).abs().mean().item()
            ),
            "candidate_head_matches": candidate_head_matches,
            "candidate_dim_matches": candidate_dim_matches,
        },
        "candidate_k_layout_hypotheses": {
            "canonical": layout_error(reference_k),
            "pre_public_batch_head_transpose": layout_error(pre_public_transpose),
            "qk_matmul_transposed_seq_dim": layout_error(qk_matmul_view),
            "scaled_qk_matmul_transposed_seq_dim": layout_error(
                qk_matmul_view * attention_k_scale
            ),
        },
        "candidate_written_k_value_hypotheses": {
            "raw_projection_no_rope_no_norm": written_error(raw_expanded_k),
            "rope_without_norm": written_error(rope_expanded_k),
            "norm_without_rope": written_error(normalized_raw_k),
            "reference_rope_then_norm": written_error(normalized_rope_k),
        },
        "rope_cache_intermediate": rope_cache_probe,
        "state": state,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
