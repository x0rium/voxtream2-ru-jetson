#!/usr/bin/env python3
"""Export one state-explicit Mimi streaming decoder step to ONNX/TensorRT."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch
from moshi.models import loaders
from moshi.modules.transformer import KVCacheResult, RingKVCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-codebooks", type=int, default=16)
    parser.add_argument("--skip-export", action="store_true")
    return parser.parse_args()


def attempt(name: str, operation, results: dict[str, object]):
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
            "traceback": traceback.format_exc(limit=16),
        }
        return None


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_")


def make_ring_kv_update_exportable() -> None:
    """Use ONNX-exportable indexed writes for the fixed batch=1 decoder."""

    def complete(self, k, v, exec_mask):
        if self.respect_exec_mask is not True:
            raise RuntimeError("Mimi decoder expects respect_exec_mask=True")
        batch_size, heads, sequence_length, dimension = k.shape
        if batch_size != 1:
            raise ValueError("Mimi TensorRT decoder currently has fixed batch=1")
        indexes = torch.arange(
            sequence_length,
            device=self.end_offset.device,
            dtype=self.end_offset.dtype,
        )
        indexes = (indexes + self.end_offset[0]) % self.capacity

        # Equivalent to RingKVCache.scatter_ for B=1. This syntax lowers to
        # ScatterND, unlike aten.scatter.src which PyTorch 2.7 cannot export.
        self.cache[0][:, :, indexes, :] = k
        self.cache[1][:, :, indexes, :] = v
        keys = self.cache[0]
        values = self.cache[1]

        cache_indexes = torch.arange(
            self.capacity,
            device=self.end_offset.device,
            dtype=torch.long,
        )
        last_offset = self.end_offset.view(-1, 1) + sequence_length - 1
        end_index = last_offset % self.capacity
        delta = cache_indexes - end_index
        positions = torch.where(
            delta <= 0,
            last_offset + delta,
            last_offset + delta - self.capacity,
        )
        self.end_offset[:] = torch.where(
            exec_mask,
            self.end_offset + sequence_length,
            self.end_offset,
        )
        invalid = cache_indexes >= self.end_offset.view(-1, 1)
        positions = torch.where(
            invalid,
            torch.full_like(positions, -1),
            positions,
        )
        return KVCacheResult(keys, values, positions)

    RingKVCache.complete = complete


@dataclass
class StateBinding:
    name: str
    owner: object
    attribute: str
    initial: torch.Tensor

    def set(self, value: torch.Tensor) -> None:
        setattr(self.owner, self.attribute, value)

    def get(self) -> torch.Tensor:
        value = getattr(self.owner, self.attribute)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state {self.name} stopped being a tensor")
        return value


def collect_tensor_fields(
    owner: object,
    prefix: str,
    bindings: list[StateBinding],
    visited: set[int],
) -> None:
    if id(owner) in visited:
        return
    visited.add(id(owner))
    for attribute, value in vars(owner).items():
        name = f"{prefix}.{attribute}" if prefix else attribute
        if isinstance(value, torch.Tensor):
            bindings.append(
                StateBinding(sanitize(name), owner, attribute, value.clone())
            )
        elif type(value).__name__ == "RingKVCache":
            collect_tensor_fields(value, name, bindings, visited)


def decoder_state_bindings(model) -> list[StateBinding]:
    bindings: list[StateBinding] = []
    visited: set[int] = set()
    prefixes = ("decoder", "decoder_transformer", "upsample")
    for module_name, state in model.get_streaming_state().items():
        if module_name == "decoder" or module_name.startswith(
            tuple(prefix + "." for prefix in prefixes)
        ):
            collect_tensor_fields(state, module_name, bindings, visited)
    duplicates = [
        name
        for name in {binding.name for binding in bindings}
        if sum(binding.name == name for binding in bindings) > 1
    ]
    if duplicates:
        raise RuntimeError(f"duplicate Mimi state names: {duplicates}")
    return bindings


class ExplicitMimiDecoderStep(torch.nn.Module):
    def __init__(self, model, bindings: list[StateBinding]) -> None:
        super().__init__()
        self.model = model
        self.bindings = bindings

    def forward(
        self, codes: torch.Tensor, *state: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        for binding, value in zip(self.bindings, state):
            binding.set(value)
        emb = self.model.quantizer.decode(codes)
        emb = self.model._to_encoder_framerate(emb)
        (emb,) = self.model.decoder_transformer(emb)
        audio = self.model.decoder(emb)
        return (audio, *(binding.get() for binding in self.bindings))


def build_decoder(checkpoint: Path, num_codebooks: int):
    model = (
        loaders.get_mimi(
            checkpoint,
            device="cuda",
            num_codebooks=num_codebooks,
        )
        .eval()
        .to(dtype=torch.bfloat16)
    )
    model.requires_grad_(False)
    # Fixed-prompt TTS never encodes audio. Remove 74 MiB of BF16 encoder-side
    # weights before export so only the resident decoder path is captured.
    model.encoder = torch.nn.Identity()
    model.encoder_transformer = None
    model.downsample = torch.nn.Identity()
    gc.collect()
    torch.cuda.empty_cache()
    model.streaming_forever(batch_size=1)
    return model


def write_initial_state(
    output_dir: Path,
    bindings: list[StateBinding],
    state: tuple[torch.Tensor, ...],
) -> dict[str, object]:
    """Write a framework-neutral, aligned snapshot of decoder streaming state."""

    binary_path = output_dir / "mimi-decoder-initial-state.bin"
    manifest_path = output_dir / "mimi-decoder-initial-state.json"
    payload = bytearray()
    tensors = []
    for binding, value in zip(bindings, state):
        padding = (-len(payload)) % 64
        payload.extend(b"\0" * padding)
        offset = len(payload)
        raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        payload.extend(raw)
        tensors.append(
            {
                "name": binding.name,
                "dtype": str(value.dtype).removeprefix("torch."),
                "shape": list(value.shape),
                "offset": offset,
                "bytes": len(raw),
            }
        )
    binary_path.write_bytes(payload)
    manifest = {
        "format": "voxtream-mimi-decoder-state-v1",
        "binary": binary_path.name,
        "binary_bytes": len(payload),
        "binary_sha256": hashlib.sha256(payload).hexdigest(),
        "alignment": 64,
        "state_tensors": len(tensors),
        "tensors": tensors,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "manifest": str(manifest_path),
        "manifest_bytes": manifest_path.stat().st_size,
        "binary": str(binary_path),
        "binary_bytes": binary_path.stat().st_size,
        "binary_sha256": manifest["binary_sha256"],
        "state_tensors": len(tensors),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_ring_kv_update_exportable()
    model = build_decoder(args.checkpoint, args.num_codebooks)
    bindings = decoder_state_bindings(model)
    wrapper = ExplicitMimiDecoderStep(model, bindings).eval()
    codes = torch.randint(
        0,
        2048,
        (1, args.num_codebooks, 1),
        device="cuda",
        dtype=torch.int64,
    )
    state = tuple(binding.initial.clone() for binding in bindings)
    initial_state_artifact = write_initial_state(args.output_dir, bindings, state)
    results: dict[str, object] = {
        "torch": torch.__version__,
        "checkpoint": str(args.checkpoint),
        "input_codes_shape": list(codes.shape),
        "state_count": len(bindings),
        "state_bytes": sum(value.numel() * value.element_size() for value in state),
        "initial_state_artifact": initial_state_artifact,
        "states": {
            binding.name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
                "bytes": value.numel() * value.element_size(),
            }
            for binding, value in zip(bindings, state)
        },
    }
    with torch.inference_mode():
        eager = attempt("eager_step", lambda: wrapper(codes, *state), results)
    if eager is not None:
        results["eager_step"].update(
            output_shape=list(eager[0].shape),
            output_dtype=str(eager[0].dtype).removeprefix("torch."),
        )
    if args.skip_export:
        print(json.dumps(results, indent=2))
        return

    # Recreate zero state because the eager check above advanced every cache.
    model.reset_streaming()
    state = tuple(binding.get().clone() for binding in bindings)

    def export_math_attention():
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel([SDPBackend.MATH]):
            return torch.export.export(
                wrapper,
                (codes, *state),
                strict=False,
            )

    exported = attempt("torch_export", export_math_attention, results)
    onnx_path = args.output_dir / "mimi-decoder-step.onnx"
    input_names = ["codes", *(binding.name for binding in bindings)]
    output_names = ["audio", *(f"next_{binding.name}" for binding in bindings)]
    if exported is not None:
        results["torch_export"].update(
            graph_nodes=len(list(exported.graph.nodes)),
            inputs=len(input_names),
            outputs=len(output_names),
        )
        attempt(
            "onnx_export",
            lambda: torch.onnx.export(
                exported,
                (),
                onnx_path,
                input_names=input_names,
                output_names=output_names,
                dynamo=True,
                external_data=False,
            ),
            results,
        )
    if results.get("onnx_export", {}).get("ok"):
        import onnx

        onnx_model = onnx.load(onnx_path, load_external_data=False)
        removed_default_scatter_reductions = 0
        for node in onnx_model.graph.node:
            if node.op_type != "ScatterND":
                continue
            kept_attributes = []
            for attribute in node.attribute:
                if attribute.name == "reduction" and attribute.s == b"none":
                    removed_default_scatter_reductions += 1
                else:
                    kept_attributes.append(attribute)
            del node.attribute[:]
            node.attribute.extend(kept_attributes)
        if removed_default_scatter_reductions:
            onnx.save(onnx_model, onnx_path)
        results["onnx_graph"] = {
            "path": str(onnx_path),
            "bytes": onnx_path.stat().st_size,
            "nodes": len(onnx_model.graph.node),
            "initializers": len(onnx_model.graph.initializer),
            "removed_default_scatter_reductions": (
                removed_default_scatter_reductions
            ),
            "inputs": [value.name for value in onnx_model.graph.input],
            "outputs": [value.name for value in onnx_model.graph.output],
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
                raise RuntimeError(
                    "\n".join(
                        str(parser.get_error(index))
                        for index in range(parser.num_errors)
                    )
                )
            return network.num_layers

        parsed_layers = attempt("tensorrt_parse", parse_tensorrt, results)
        if parsed_layers is not None:
            results["tensorrt_parse"]["layers"] = parsed_layers

    metrics_path = args.output_dir / "mimi-decoder-step.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
