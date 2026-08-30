#!/usr/bin/env python3
"""Export and validate phone_embeddings + phone_former as one TensorRT graph."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from torchtune.models import llama3_2

PHONE_PREFIX = "phone_former."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-export", action="store_true")
    return parser.parse_args()


class PhoneEncoder(nn.Module):
    def __init__(self, transformer: nn.Module, embeddings: nn.Embedding) -> None:
        super().__init__()
        self.transformer = transformer
        self.embeddings = embeddings

    def forward(
        self,
        phone_tokens: torch.Tensor,
        input_pos: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.embeddings(phone_tokens)
        return self.transformer(hidden, input_pos=input_pos, mask=mask).to(
            dtype=hidden.dtype
        )


def build_phone_encoder(checkpoint: Path, device: str) -> PhoneEncoder:
    transformer = llama3_2.llama3_2(
        vocab_size=1,
        num_layers=6,
        num_heads=8,
        num_kv_heads=2,
        embed_dim=1024,
        max_seq_len=2048,
        intermediate_dim=4096,
    )
    transformer.tok_embeddings = nn.Identity()
    transformer.output = nn.Identity()
    embeddings = nn.Embedding(166, 1024)

    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        transformer_state = {
            key.removeprefix(PHONE_PREFIX): source.get_tensor(key)
            for key in source.keys()
            if key.startswith(PHONE_PREFIX)
        }
        embedding_weight = source.get_tensor("phone_embeddings.weight")
    missing, unexpected = transformer.load_state_dict(
        transformer_state, strict=False, assign=True
    )
    if missing or unexpected:
        raise RuntimeError(
            f"phone_former weight mismatch: missing={missing}, unexpected={unexpected}"
        )
    embeddings.weight = nn.Parameter(embedding_weight, requires_grad=False)
    model = PhoneEncoder(transformer, embeddings).eval().to(
        device=device, dtype=torch.bfloat16
    )
    model.requires_grad_(False)
    return model


def prepare_inputs(
    batch_size: int,
    sequence_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1234)
    tokens = torch.randint(
        0, 166, (batch_size, sequence_length), device=device, dtype=torch.int64
    )
    input_pos = torch.arange(
        sequence_length, device=device, dtype=torch.int64
    ).unsqueeze(0).repeat(batch_size, 1)
    rows = torch.arange(sequence_length, device=device).view(-1, 1)
    columns = torch.arange(sequence_length, device=device).view(1, -1)
    mask_2d = (columns <= rows + 30) & (columns >= rows - 624)
    mask = mask_2d.unsqueeze(0).repeat(batch_size, 1, 1)
    return tokens, input_pos, mask


def attempt(name: str, operation, results: dict) -> object | None:
    started = time.perf_counter()
    try:
        value = operation()
        results[name] = {
            "ok": True,
            "seconds": round(time.perf_counter() - started, 3),
        }
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


def run_tensorrt(
    engine_path: Path,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    import tensorrt as trt

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"failed to deserialize {engine_path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("failed to create TensorRT context")
    stream = torch.cuda.Stream()
    names = ("phone_tokens", "input_pos", "mask")
    for name, value in zip(names, inputs):
        if not context.set_input_shape(name, tuple(value.shape)):
            raise RuntimeError(f"failed to set {name} shape={tuple(value.shape)}")
    output_shape = tuple(context.get_tensor_shape("phone_embeddings"))
    if -1 in output_shape:
        raise RuntimeError(f"unresolved output shape: {output_shape}")
    output = torch.empty(output_shape, device="cuda", dtype=torch.bfloat16)
    for name, value in zip(names, inputs):
        if not context.set_tensor_address(name, value.data_ptr()):
            raise RuntimeError(f"failed to bind {name}")
    if not context.set_tensor_address("phone_embeddings", output.data_ptr()):
        raise RuntimeError("failed to bind phone_embeddings")
    with torch.cuda.stream(stream):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT enqueue failed")
    stream.synchronize()
    return output


def main() -> None:
    args = parse_args()
    if not 2 <= args.sequence_length <= args.max_sequence_length <= 2048:
        raise ValueError("require 2 <= sequence-length <= max-sequence-length <= 2048")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = build_phone_encoder(args.checkpoint, args.device)
    inputs = prepare_inputs(args.batch_size, args.sequence_length, args.device)
    with torch.inference_mode():
        reference = model(*inputs)
    torch.cuda.synchronize()
    results: dict[str, object] = {
        "torch": torch.__version__,
        "sequence_length": args.sequence_length,
        "max_sequence_length": args.max_sequence_length,
        "input_shapes": [list(value.shape) for value in inputs],
        "reference_shape": list(reference.shape),
        "parameter_mib": round(
            sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2,
            1,
        ),
    }

    if args.engine is not None:
        candidate = attempt(
            "tensorrt_inference", lambda: run_tensorrt(args.engine, inputs), results
        )
        if candidate is not None:
            delta = reference.float() - candidate.float()
            results["tensorrt_inference"].update(
                max_abs=float(delta.abs().max().item()),
                mean_abs=float(delta.abs().mean().item()),
                cosine=float(
                    torch.nn.functional.cosine_similarity(
                        reference.float().flatten(),
                        candidate.float().flatten(),
                        dim=0,
                    ).item()
                ),
                bf16_equal_fraction=float(
                    (reference == candidate).float().mean().item()
                ),
            )

    metrics_path = args.output_dir / "phone-encoder.json"
    if args.skip_export:
        metrics_path.write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))
        return

    sequence = torch.export.Dim(
        "sequence", min=2, max=args.max_sequence_length
    )
    dynamic_shapes = (
        {1: sequence},
        {1: sequence},
        {1: sequence, 2: sequence},
    )
    def export_with_shape_generic_attention():
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel([SDPBackend.MATH]):
            return torch.export.export(
                model, inputs, dynamic_shapes=dynamic_shapes, strict=False
            )

    exported = attempt(
        "torch_export", export_with_shape_generic_attention, results
    )
    onnx_path = args.output_dir / "phone-encoder.onnx"
    if exported is not None:
        results["torch_export"]["graph_nodes"] = len(list(exported.graph.nodes))
        attempt(
            "onnx_export",
            lambda: torch.onnx.export(
                exported,
                (),
                onnx_path,
                input_names=["phone_tokens", "input_pos", "mask"],
                output_names=["phone_embeddings"],
                dynamo=True,
                external_data=False,
            ),
            results,
        )
    if results.get("onnx_export", {}).get("ok"):
        import onnx

        model_onnx = onnx.load(onnx_path, load_external_data=False)
        results["onnx_graph"] = {
            "bytes": onnx_path.stat().st_size,
            "nodes": len(model_onnx.graph.node),
            "initializers": len(model_onnx.graph.initializer),
            "input_shapes": {
                value.name: [
                    dimension.dim_param or dimension.dim_value
                    for dimension in value.type.tensor_type.shape.dim
                ]
                for value in model_onnx.graph.input
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

    metrics_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
