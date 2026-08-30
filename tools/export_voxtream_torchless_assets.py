#!/usr/bin/env python3
"""Export the remaining fixed VoXtream assets to a PyTorch-free binary ABI.

This is an offline conversion tool and is intentionally allowed to import
PyTorch.  The resulting manifest and binary are consumed by the standalone
runtime, which asserts that PyTorch has never been imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ALIGNMENT = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-cache", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--prompt-cache", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text", default="Привет, я работаю на Джетсоне.")
    parser.add_argument(
        "--phonemes",
        help="Precomputed RUAccent/ESpeak phone string for a new input phrase.",
    )
    parser.add_argument("--phone-map", type=Path)
    return parser.parse_args()


def aligned(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def tensor_payload(value: torch.Tensor) -> tuple[str, list[int], bytes]:
    value = value.detach().contiguous().cpu()
    if value.dtype == torch.bfloat16:
        payload = value.view(torch.uint16).numpy().tobytes()
        dtype = "bfloat16"
    else:
        dtype_map = {
            torch.float32: "float32",
            torch.float16: "float16",
            torch.int64: "int64",
            torch.bool: "bool",
        }
        try:
            dtype = dtype_map[value.dtype]
        except KeyError as error:
            raise TypeError(f"unsupported torch dtype: {value.dtype}") from error
        payload = value.numpy().tobytes()
    return dtype, list(value.shape), payload


def numpy_payload(value: np.ndarray) -> tuple[str, list[int], bytes]:
    value = np.ascontiguousarray(value)
    dtype_map = {
        np.dtype(np.float32): "float32",
        np.dtype(np.float16): "float16",
        np.dtype(np.int64): "int64",
        np.dtype(np.bool_): "bool",
    }
    try:
        dtype = dtype_map[value.dtype]
    except KeyError as error:
        raise TypeError(f"unsupported NumPy dtype: {value.dtype}") from error
    return dtype, list(value.shape), value.tobytes()


class BundleWriter:
    def __init__(self) -> None:
        self.binary = bytearray()
        self.tensors: list[dict[str, object]] = []

    def add_payload(
        self, name: str, dtype: str, shape: list[int], payload: bytes
    ) -> None:
        offset = aligned(len(self.binary))
        self.binary.extend(b"\0" * (offset - len(self.binary)))
        self.binary.extend(payload)
        self.tensors.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": shape,
                "offset": offset,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )

    def add_tensor(self, name: str, value: torch.Tensor) -> None:
        self.add_payload(name, *tensor_payload(value))

    def add_numpy(self, name: str, value: np.ndarray) -> None:
        self.add_payload(name, *numpy_payload(value))


def main() -> None:
    args = parse_args()
    prefill = torch.load(args.prefill_cache, map_location="cpu", weights_only=False)
    if prefill.get("format") != "voxtream-temp-fixed-prompt-prefill-v1":
        raise ValueError("unsupported fixed-prompt cache")
    semantic_logits = prefill.get("semantic_logits")
    if semantic_logits is None:
        raise ValueError("prefill cache must contain fused semantic_logits")

    capture = np.load(args.capture)
    prompt = np.load(args.prompt_cache, allow_pickle=True).item()
    prompt_len = int(capture["prompt_phone_len"])
    prompt_frames = int(capture["prompt_frames"])
    if prompt_len != prompt_frames:
        raise ValueError("current fixed-prompt ABI requires prompt_len == prompt_frames")

    # Reconstruct the exact CFG phone batch. For the original fixture, the
    # debug capture stores only its conditional row. A new phrase may instead
    # provide the RUAccent/ESpeak result; this keeps slow linguistic frontend
    # work outside the strict inference process without replaying audio tokens.
    if args.phonemes is None:
        conditional_tokens = np.asarray(capture["phone_tokens"], dtype=np.int64)
        phone_tokens = np.repeat(conditional_tokens, 2, axis=0)
        phone_tokens[1, prompt_len:] = 163
        punctuation_indices = np.repeat(
            np.asarray(capture["punct_del_indices"], dtype=np.int64), 2, axis=0
        )
        reference_matches_text = True
    else:
        if args.phone_map is None:
            raise ValueError("--phonemes requires --phone-map")
        phone_to_token = json.loads(args.phone_map.read_text())
        punctuation_symbols = (".", ",", "?", "!")
        phones: list[str] = []
        punctuation_insertions: list[int] = []
        punctuation_tokens: list[int] = []
        for word in args.phonemes.split():
            for phone in word.split("|"):
                if phone:
                    phones.append(phone)
            if phones and phones[-1].endswith(punctuation_symbols):
                phone = phones.pop()
                symbol = phone[-1]
                if phone[:-1]:
                    phones.append(phone[:-1])
                punctuation_tokens.append(int(phone_to_token[symbol]))
                punctuation_insertions.append(len(phones))
        generated = [
            int(phone_to_token.get(phone, phone_to_token["unk"])) for phone in phones
        ]
        punctuation_delete = (
            np.asarray(punctuation_insertions, dtype=np.int64)
            + np.arange(len(punctuation_insertions), dtype=np.int64)
        )
        generated = np.insert(
            np.asarray(generated, dtype=np.int64),
            punctuation_insertions,
            punctuation_tokens,
        )
        generated = np.concatenate(
            [generated, np.asarray([120, 165], dtype=np.int64)]
        )
        prompt_prefix = np.asarray(capture["phone_tokens"], dtype=np.int64)[
            0, :prompt_len
        ]
        conditional = np.concatenate([prompt_prefix, generated])
        unconditional = np.concatenate(
            [prompt_prefix, np.full(generated.shape, 163, dtype=np.int64)]
        )
        phone_tokens = np.stack([conditional, unconditional])
        punctuation_indices = np.repeat(
            (punctuation_delete + prompt_len)[None], 2, axis=0
        )
        reference_matches_text = False

    # Recreate delay_audio_tokens() and the unconditional CFG audio prompt from
    # the immutable prompt cache rather than copying generated trajectory data.
    raw_codes = np.asarray(prompt["audio_tokens"], dtype=np.int64)
    delayed_codes = np.full(
        (1, raw_codes.shape[1], raw_codes.shape[2] + 1), 2049, dtype=np.int64
    )
    delayed_codes[:, 0, 1:] = raw_codes[:, 0]
    delayed_codes[:, 1:, 2:] = raw_codes[:, 1:, :-1]
    prompt_codes = np.repeat(delayed_codes, 2, axis=0)
    unconditional_mask = prompt_codes[1] != 2049
    prompt_codes[1, unconditional_mask] = 2048

    prompt_phone_indices = np.repeat(
        np.arange(prompt_frames, dtype=np.int64)[:, None], 2, axis=1
    )[None]
    prompt_phone_indices = np.repeat(prompt_phone_indices, 2, axis=0)

    raw_speaker = torch.from_numpy(
        np.asarray(capture["spk_embedding_raw"], dtype=np.float32)
    )
    with safe_open(args.model, framework="pt", device="cpu") as source:
        speaker_weight = source.get_tensor("spk_emb_proj.weight")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the production BF16 projection")
    projected_speaker = torch.nn.functional.linear(
        raw_speaker.to(device="cuda", dtype=torch.bfloat16),
        speaker_weight.to(device="cuda", dtype=torch.bfloat16),
    ).to(dtype=torch.bfloat16, device="cpu")

    writer = BundleWriter()
    writer.add_tensor("prefill.output", prefill["output"])
    writer.add_tensor("prefill.semantic_logits", semantic_logits)
    for name, state in zip(prefill["buffer_names"], prefill["final_state"]):
        writer.add_tensor(f"temp_state.{name}", state)
    writer.add_numpy("phone.tokens", phone_tokens)
    writer.add_numpy("phone.punctuation_indices", punctuation_indices)
    writer.add_numpy("prompt.audio_tokens", prompt_codes)
    writer.add_numpy("prompt.phone_indices", prompt_phone_indices)
    writer.add_tensor("speaker.projected", projected_speaker)
    if reference_matches_text:
        writer.add_numpy(
            "reference.pred_shifts",
            np.asarray(capture["pred_shifts"], dtype=np.int64),
        )
        writer.add_numpy(
            "reference.mimi_codes",
            np.asarray(capture["mimi_codes"], dtype=np.int64),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    binary_path = args.output.with_suffix(".bin")
    binary_path.write_bytes(writer.binary)
    manifest = {
        "format": "voxtream-torchless-fixed-utterance-v1",
        "binary": binary_path.name,
        "binary_bytes": len(writer.binary),
        "binary_sha256": hashlib.sha256(writer.binary).hexdigest(),
        "alignment": ALIGNMENT,
        "text": args.text,
        "prompt_frames": prompt_frames,
        "prompt_phone_len": prompt_len,
        "phone_seq_len": int(phone_tokens.shape[1] - 2 - punctuation_indices.shape[1]),
        "reference_matches_text": reference_matches_text,
        "config": {
            "audio_pad_token": 2049,
            "mimi_vocab_size": 2048,
            "audio_vocab_size": 2050,
            "num_codebooks": 16,
            "num_phone_states": 6,
            "num_phones_per_frame": 2,
            "frame_repeat_counter": 25,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50,
            "cfg_gamma": 1.5,
            "cfg_ac_gamma": 3.0,
            "spk_proj_weight": 1.5,
            "sample_rate": 24000,
            "samples_per_frame": 1920,
            "audio_delay_frames": 1,
            "max_temp_position": 624,
            "phoneme_index_map": {
                "0": [0, 1],
                "1": [0, 2],
                "2": [1, 1],
                "3": [1, 2],
                "4": [2, 1],
                "5": [2, 2],
            },
        },
        "prefill_buffer_names": list(prefill["buffer_names"]),
        "sources": {
            "prefill_cache": str(args.prefill_cache),
            "capture": str(args.capture),
            "prompt_cache": str(args.prompt_cache),
            "model": str(args.model),
        },
        "tensors": writer.tensors,
    }
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({
        "manifest": str(args.output),
        "binary": str(binary_path),
        "binary_bytes": len(writer.binary),
        "tensors": len(writer.tensors),
        "phone_tokens": list(phone_tokens.shape),
        "phone_seq_len": manifest["phone_seq_len"],
        "projected_speaker": list(projected_speaker.shape),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
