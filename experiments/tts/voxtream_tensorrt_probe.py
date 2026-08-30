#!/usr/bin/env python3
"""Probe exportability of VoXtream's dep_former without loading the full TTS stack."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from torchtune.models import llama3_2
from torchtune.modules.kv_cache import KVCache

DEP_PREFIX = "dep_former."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, choices=(1, 2), default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--build-engine", action="store_true")
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--skip-export", action="store_true")
    return parser.parse_args()


class DepFormerStep(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        hidden: torch.Tensor,
        input_pos: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(hidden, input_pos=input_pos, mask=mask)


def make_kv_update_exportable() -> None:
    """Remove only torchtune's Python tensor assert from the cache update."""

    def update(self, k_val: torch.Tensor, v_val: torch.Tensor):
        batch_size, _, sequence_length, _ = k_val.shape
        if batch_size > self.k_cache.shape[0]:
            raise ValueError(
                f"KV batch {batch_size} exceeds configured {self.k_cache.shape[0]}"
            )
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, self.cache_pos[:sequence_length]] = k_val
        v_out[:, :, self.cache_pos[:sequence_length]] = v_val
        self.cache_pos += sequence_length
        return k_out, v_out

    KVCache.update = update


def build_dep_former(checkpoint: Path, device: str, batch_size: int) -> nn.Module:
    # vocab_size=1 avoids allocating the unused 128256-token embedding.  The
    # VoXtream runtime replaces both embedding/output layers with Identity too.
    model = llama3_2.llama3_2(
        vocab_size=1,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        embed_dim=1024,
        max_seq_len=2048,
        intermediate_dim=8192,
    )
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()

    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        state = {
            key.removeprefix(DEP_PREFIX): source.get_tensor(key)
            for key in source.keys()
            if key.startswith(DEP_PREFIX)
        }
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"weight mismatch: missing={missing}, unexpected={unexpected}")

    model = model.eval().to(device=device, dtype=torch.bfloat16)
    model.requires_grad_(False)
    with torch.device(device):
        model.setup_caches(
            batch_size=batch_size,
            dtype=torch.bfloat16,
            decoder_max_seq_len=16,
        )
    return model


def causal_rows(input_pos: torch.Tensor, max_length: int = 16) -> torch.Tensor:
    causal = torch.tril(
        torch.ones((max_length, max_length), dtype=torch.bool, device=input_pos.device)
    )
    return causal[input_pos, :]


def prepare_inputs(
    model: nn.Module,
    batch_size: int,
    sequence_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1234)
    model.reset_caches()
    if sequence_length == 1:
        # Reproduce the real incremental step: the first two cache positions are
        # populated by dep_former_init before one-token calls begin.
        initial_hidden = torch.randn(
            batch_size, 2, 1024, device=device, dtype=torch.bfloat16
        )
        initial_pos = torch.arange(2, device=device).unsqueeze(0).repeat(batch_size, 1)
        with torch.inference_mode():
            model(initial_hidden, input_pos=initial_pos, mask=causal_rows(initial_pos))
        input_pos = torch.full(
            (batch_size, 1), 2, device=device, dtype=torch.int64
        )
    else:
        input_pos = torch.arange(2, device=device).unsqueeze(0).repeat(batch_size, 1)
    hidden = torch.randn(
        batch_size, sequence_length, 1024, device=device, dtype=torch.bfloat16
    )
    return hidden, input_pos, causal_rows(input_pos)


def attempt(name: str, operation, results: dict) -> object | None:
    started = time.perf_counter()
    try:
        value = operation()
        results[name] = {"ok": True, "seconds": round(time.perf_counter() - started, 3)}
        return value
    except Exception as error:  # compatibility probe: preserve the exact blocker
        results[name] = {
            "ok": False,
            "seconds": round(time.perf_counter() - started, 3),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(limit=12),
        }
        return None


def write_tensor_raw(path: Path, tensor: torch.Tensor) -> None:
    """Write a TensorRT --loadInputs compatible, contiguous raw tensor."""
    cpu = tensor.detach().contiguous().cpu()
    if cpu.dtype == torch.bfloat16:
        cpu = cpu.view(torch.uint16)
    path.write_bytes(cpu.numpy().tobytes())


def summarize_ms(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p90_index = min(len(ordered) - 1, int(0.9 * len(ordered)))
    return {
        "iterations": len(samples),
        "mean_ms": round(statistics.mean(samples), 4),
        "median_ms": round(statistics.median(samples), 4),
        "p90_ms": round(ordered[p90_index], 4),
        "min_ms": round(ordered[0], 4),
        "max_ms": round(ordered[-1], 4),
    }


def benchmark_pytorch_cuda_graph(
    model: nn.Module,
    wrapper: nn.Module,
    batch_size: int,
    sequence_length: int,
    device: str,
) -> dict[str, float]:
    # Warm the kernels without consuming positions needed by the captured run.
    inputs = prepare_inputs(model, batch_size, sequence_length, device)
    with torch.inference_mode():
        wrapper(*inputs)
        wrapper(*inputs)
    torch.cuda.synchronize()

    inputs = prepare_inputs(model, batch_size, sequence_length, device)
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        graph_output = wrapper(*inputs)
    del graph_output

    samples: list[float] = []
    # Reset the cache between short rounds so the position never exceeds 15.
    # Discard the first round while Jetson clocks are still ramping up.
    for round_index in range(6):
        prepare_inputs(model, batch_size, sequence_length, device)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
        for start, end in zip(starts, ends):
            start.record()
            graph.replay()
            end.record()
        torch.cuda.synchronize()
        if round_index > 0:
            samples.extend(
                start.elapsed_time(end) for start, end in zip(starts, ends)
            )
    torch.cuda.synchronize()
    return summarize_ms(samples)


class TensorRTRunner:
    def __init__(self, engine_path: Path, inputs: tuple[torch.Tensor, ...]) -> None:
        import tensorrt as trt

        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.output = torch.empty(
            (inputs[0].shape[0], inputs[0].shape[1], 1024), device="cuda"
        )
        tensors = dict(zip(("hidden", "input_pos", "mask"), inputs))
        tensors["output"] = self.output
        for name, tensor in tensors.items():
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"failed to bind TensorRT tensor {name}")

    def enqueue(self) -> None:
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")

    def execute(self) -> torch.Tensor:
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.enqueue()
        self.stream.synchronize()
        return self.output

    def benchmark_cuda_graph(self) -> dict[str, float]:
        for _ in range(50):
            self.enqueue()
        self.stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            self.enqueue()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(200)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(200)]
        with torch.cuda.stream(self.stream):
            for start, end in zip(starts, ends):
                start.record()
                graph.replay()
                end.record()
        self.stream.synchronize()
        return summarize_ms(
            [start.elapsed_time(end) for start, end in zip(starts, ends)]
        )


def validate_tensorrt(
    engine_path: Path,
    checkpoint: Path,
    model: nn.Module,
    wrapper: nn.Module,
    batch_size: int,
    sequence_length: int,
    device: str,
) -> dict[str, object]:
    inputs = prepare_inputs(model, batch_size, sequence_length, device)
    with torch.inference_mode():
        reference = wrapper(*inputs)
        runner = TensorRTRunner(engine_path, inputs)
        candidate = runner.execute()

    delta = reference.float() - candidate.float()
    reference_flat = reference.float().flatten()
    candidate_flat = candidate.float().flatten()
    cosine = torch.dot(reference_flat, candidate_flat) / (
        torch.linalg.vector_norm(reference_flat)
        * torch.linalg.vector_norm(candidate_flat)
    )

    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        audio_head = source.get_tensor("audio_head")[1].to(
            device=device, dtype=torch.bfloat16
        )
    reference_logits = torch.mm(reference[:, -1, :].to(torch.bfloat16), audio_head)
    candidate_logits = torch.mm(candidate[:, -1, :].to(torch.bfloat16), audio_head)
    reference_cfg = 3.0 * reference_logits[0].float() - 2.0 * reference_logits[1].float()
    candidate_cfg = 3.0 * candidate_logits[0].float() - 2.0 * candidate_logits[1].float()
    reference_top2 = torch.topk(reference_cfg, 2)
    candidate_top2 = torch.topk(candidate_cfg, 2)
    logits_delta = reference_cfg - candidate_cfg

    return {
        "cuda_graph_benchmark": runner.benchmark_cuda_graph(),
        "hidden": {
            "max_abs": float(delta.abs().max().item()),
            "mean_abs": float(delta.abs().mean().item()),
            "rmse": float(torch.sqrt(torch.mean(delta.square())).item()),
            "cosine": float(cosine.item()),
            "bf16_equal_fraction": float(
                (reference.to(torch.bfloat16) == candidate.to(torch.bfloat16))
                .float()
                .mean()
                .item()
            ),
        },
        "cfg_logits_audio_head_1": {
            "max_abs": float(logits_delta.abs().max().item()),
            "mean_abs": float(logits_delta.abs().mean().item()),
            "reference_argmax": int(reference_top2.indices[0].item()),
            "candidate_argmax": int(candidate_top2.indices[0].item()),
            "reference_margin": float(
                (reference_top2.values[0] - reference_top2.values[1]).item()
            ),
            "candidate_margin": float(
                (candidate_top2.values[0] - candidate_top2.values[1]).item()
            ),
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_kv_update_exportable()
    results: dict[str, object] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": args.device,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
    }

    model = build_dep_former(args.checkpoint, args.device, args.batch_size)
    wrapper = DepFormerStep(model)
    inputs = prepare_inputs(model, args.batch_size, args.sequence_length, args.device)
    with torch.inference_mode():
        eager = wrapper(*inputs)
    torch.cuda.synchronize()
    for name, tensor in zip(("hidden", "input_pos", "mask"), inputs):
        write_tensor_raw(
            args.output_dir / f"dep-step-s{args.sequence_length}-{name}.bin", tensor
        )
    write_tensor_raw(
        args.output_dir / f"dep-step-s{args.sequence_length}-eager-output.bin", eager
    )
    results["eager"] = {
        "shape": list(eager.shape),
        "dtype": str(eager.dtype),
        "finite": bool(torch.isfinite(eager).all().item()),
    }

    cuda_graph_benchmark = attempt(
        "pytorch_cuda_graph_benchmark",
        lambda: benchmark_pytorch_cuda_graph(
            model, wrapper, args.batch_size, args.sequence_length, args.device
        ),
        results,
    )
    if cuda_graph_benchmark is not None:
        results["pytorch_cuda_graph_benchmark"].update(cuda_graph_benchmark)

    if args.engine is not None:
        validation = attempt(
            "tensorrt_validation",
            lambda: validate_tensorrt(
                args.engine,
                args.checkpoint,
                model,
                wrapper,
                args.batch_size,
                args.sequence_length,
                args.device,
            ),
            results,
        )
        if validation is not None:
            results["tensorrt_validation"].update(validation)

    if args.skip_export:
        metrics = args.output_dir / f"dep-step-s{args.sequence_length}.json"
        metrics.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    # torch.export is the front door used by current Torch-TensorRT Dynamo.
    inputs = prepare_inputs(model, args.batch_size, args.sequence_length, args.device)
    exported = attempt(
        "torch_export",
        lambda: torch.export.export(wrapper, inputs, strict=False),
        results,
    )
    if exported is not None:
        results["torch_export"]["graph_nodes"] = len(list(exported.graph.nodes))
        results["torch_export"]["mutated_buffers"] = [
            str(spec)
            for spec in exported.graph_signature.output_specs
            if "BUFFER_MUTATION" in str(spec.kind)
        ]
        attempt(
            "torch_export_save",
            lambda: torch.export.save(
                exported, args.output_dir / f"dep-step-s{args.sequence_length}.pt2"
            ),
            results,
        )

    # The legacy exporter avoids an onnxscript dependency and gives TensorRT's
    # parser a concrete compatibility target immediately.
    onnx_path = args.output_dir / f"dep-step-s{args.sequence_length}.onnx"
    inputs = prepare_inputs(model, args.batch_size, args.sequence_length, args.device)
    attempt(
        "onnx_export",
        lambda: torch.onnx.export(
            wrapper,
            inputs,
            onnx_path,
            input_names=["hidden", "input_pos", "mask"],
            output_names=["output"],
            opset_version=18,
            # PyTorch 2.7's legacy constant folder mixes CPU shape constants
            # with CUDA cache buffers for this model. TensorRT folds constants
            # during engine build anyway, so disabling this pass loses nothing.
            do_constant_folding=False,
            dynamo=False,
        ),
        results,
    )

    if results["onnx_export"]["ok"]:
        import onnx

        attempt("onnx_check", lambda: onnx.checker.check_model(onnx_path), results)

        def parse_with_tensorrt():
            import tensorrt as trt

            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            if not parser.parse_from_file(str(onnx_path)):
                errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
                raise RuntimeError("\n".join(errors))
            return builder, network

        parsed = attempt("tensorrt_parse", parse_with_tensorrt, results)
        if parsed is not None:
            builder, network = parsed
            results["tensorrt_parse"]["layers"] = network.num_layers

            if args.build_engine:
                def build_engine():
                    import tensorrt as trt

                    config = builder.create_builder_config()
                    config.set_flag(trt.BuilderFlag.BF16)
                    config.set_memory_pool_limit(
                        trt.MemoryPoolType.WORKSPACE, 512 * 1024 * 1024
                    )
                    engine = builder.build_serialized_network(network, config)
                    if engine is None:
                        raise RuntimeError("TensorRT returned no serialized engine")
                    path = args.output_dir / f"dep-step-s{args.sequence_length}.engine"
                    path.write_bytes(engine)
                    return len(engine)

                engine_size = attempt("tensorrt_build", build_engine, results)
                if engine_size is not None:
                    results["tensorrt_build"]["engine_bytes"] = engine_size

    metrics = args.output_dir / f"dep-step-s{args.sequence_length}.json"
    metrics.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
