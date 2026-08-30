#!/usr/bin/env python3
"""PyTorch-free Russian text frontend for the VoXtream2-RU runtime.

Written text is expanded by ru-normalizr, then processed by RUAccent ONNX,
espeak-ng and model-specific phone fixes. Hugging Face Transformers imports
PyTorch by default when it is installed, even though RUAccent only asks it for
tokenizers. Disabling framework discovery before importing RUAccent keeps this
process strictly framework-free.
"""

from __future__ import annotations

import collections
import gzip
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

os.environ["USE_TORCH"] = "0"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"


VOWELS = set("aeiouyɑɛɔʌəɵɨæøœɐɒʉʊɪ")
RU_VOWELS = set("аеёиоуыэюя")
PUNCTUATION = (".", ",", "?", "!")
PROCLITICS = {"в": "v", "с": "s", "к": "k", "ж": "ʒ", "б": "b"}
STRIP_WORD = ".,?!—-«»\"'()…:;–„“”+"
NON_PHONE = ".,?!—-«»\"'()…:;–„“”"
SILENCE_AFTER = ".,!?”".replace("”", "")
PUNCTUATION_MAP = {
    ":": ",",
    ";": ",",
    "—": ",",
    "–": ",",
    "…": ".",
    "«": "",
    "»": "",
    "„": "",
    "“": "",
    "”": "",
    '"': "",
    "(": "",
    ")": "",
}
DEFAULT_ESPEAK_PUNCTUATION = ';:,.!?¡¿—…"«»“”'

INTERJECTION_FIX = {
    "хм": "x m",
    "хмм": "x m m",
    "кхм": "k x m",
    "гм": "ɡ m",
    "тс": "t s",
    "тсс": "t s s",
    "пф": "p f",
    "пфф": "p f f",
    "бр": "b r",
    "брр": "b r r",
}
INTERJECTION_REDUP = {
    "м": "m",
    "ш": "ʃ",
    "р": "r",
    "с": "s",
    "ф": "f",
    "х": "x",
    "ж": "ʒ",
}


class PunctuationPosition(Enum):
    BEGIN = 0
    END = 1
    MIDDLE = 2


PunctuationIndex = collections.namedtuple(
    "PunctuationIndex", ["punctuation", "position"]
)


def load_text_normalizer(backend: str) -> Callable[[str], str]:
    """Load a framework-free written-to-spoken Russian normalizer."""
    if backend == "none":
        return lambda text: text
    if backend != "ru-normalizr":
        raise ValueError(f"unsupported Russian text normalizer: {backend}")

    from ru_normalizr import NormalizeOptions, Normalizer

    normalizer = Normalizer(NormalizeOptions.tts())
    if "torch" in sys.modules:
        raise RuntimeError("text normalization imported PyTorch")
    return normalizer.normalize


def normalize_punctuation(text: str) -> str:
    for source, target in PUNCTUATION_MAP.items():
        text = text.replace(source, target)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"([,.!?])\1+", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def interjection_redup(word: str) -> str | None:
    letters = [character for character in word if character != "-"]
    if (
        len(letters) >= 2
        and len(set(letters)) == 1
        and letters[0] in INTERJECTION_REDUP
    ):
        return " ".join(
            [INTERJECTION_REDUP[letters[0]]] * min(len(letters), 3)
        )
    for base, pronunciation in INTERJECTION_FIX.items():
        rest = letters[len(base) :]
        if word.startswith(base) and rest and set(rest) <= {base[-1]}:
            tail = INTERJECTION_REDUP.get(base[-1]) or pronunciation.split()[-1]
            return pronunciation + " " + " ".join([tail] * min(len(rest), 2))
    parts = word.split("-")
    if len(parts) >= 2 and all(part in INTERJECTION_FIX for part in parts):
        return " ".join(INTERJECTION_FIX[part] for part in parts)
    return None


def stressed_vowel_index(accented_word: str) -> int:
    index = -1
    vowel_count = 0
    text = accented_word.lower()
    cursor = 0
    while cursor < len(text):
        if text[cursor] == "+":
            if cursor + 1 < len(text) and text[cursor + 1] in RU_VOWELS:
                index = vowel_count
            cursor += 1
            continue
        if text[cursor] in RU_VOWELS:
            vowel_count += 1
        cursor += 1
    return index


def _punctuation_regex() -> re.Pattern[str]:
    return re.compile(
        rf"(\s*[{re.escape(DEFAULT_ESPEAK_PUNCTUATION)}]+\s*)+"
    )


def strip_punctuation_to_restore(
    text: str,
) -> tuple[list[str], list[PunctuationIndex]]:
    matches = list(re.finditer(_punctuation_regex(), text))
    if not matches:
        return [text], []
    if len(matches) == 1 and matches[0].group() == text:
        return [], [
            PunctuationIndex(text, PunctuationPosition.BEGIN)
        ]
    punctuation: list[PunctuationIndex] = []
    for match in matches:
        position = PunctuationPosition.MIDDLE
        if match == matches[0] and text.startswith(match.group()):
            position = PunctuationPosition.BEGIN
        elif match == matches[-1] and text.endswith(match.group()):
            position = PunctuationPosition.END
        punctuation.append(PunctuationIndex(match.group(), position))
    chunks: list[str] = []
    remaining = text
    for index, item in enumerate(punctuation):
        split = remaining.split(item.punctuation)
        prefix, suffix = split[0], item.punctuation.join(split[1:])
        remaining = suffix
        if prefix == "":
            continue
        chunks.append(prefix)
        if index == len(punctuation) - 1 and suffix:
            chunks.append(suffix)
    return chunks, punctuation


def restore_punctuation(
    text: list[str], punctuation: list[PunctuationIndex]
) -> list[str]:
    if not punctuation:
        return text
    if not text:
        return ["".join(item.punctuation for item in punctuation)]
    current = punctuation[0]
    rest = punctuation[1:]
    if current.position == PunctuationPosition.BEGIN:
        return restore_punctuation(
            [current.punctuation + text[0]] + text[1:], rest
        )
    if current.position == PunctuationPosition.END:
        return [text[0] + current.punctuation] + restore_punctuation(
            text[1:], rest
        )
    if len(text) == 1:
        return restore_punctuation([text[0] + current.punctuation], rest)
    return restore_punctuation(
        [text[0] + current.punctuation + text[1]] + text[2:], rest
    )


def espeak_phonemize(
    text: str,
    separator: str = "|",
    executable: str = "espeak-ng",
) -> str:
    chunks, punctuation = strip_punctuation_to_restore(text.strip())
    phonemized: list[str] = []
    for chunk in chunks:
        completed = subprocess.run(
            [
                executable,
                "-q",
                "-b",
                "1",
                "-v",
                "ru",
                "--ipa=1",
                chunk,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        value = ""
        for line in completed.stdout.splitlines():
            decoded = re.sub(r"\(.+?\)", "", line.decode("utf-8").strip())
            value += decoded.strip()
        phonemized.append(value.replace("_", separator))
    return restore_punctuation(phonemized, punctuation)[0]


def _read_gzip_json(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def load_ruaccent_without_downloader(assets: Path):
    if "torch" in sys.modules:
        raise RuntimeError("PyTorch entered the process before RUAccent startup")
    from ruaccent import RUAccent

    assets = Path(assets)
    required = (
        assets / "dictionary" / "omographs.json.gz",
        assets / "dictionary" / "yo_words.json.gz",
        assets / "dictionary" / "yo_homographs.json.gz",
        assets / "dictionary" / "accents.json.gz",
        assets / "nn" / "nn_omograph" / "turbo3.1" / "model.onnx",
        assets / "nn" / "nn_accent" / "model.onnx",
        assets / "nn" / "nn_stress_usage_predictor" / "model.onnx",
        assets / "nn" / "nn_yo_homograph_resolver" / "model.onnx",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete RUAccent assets: {missing}")

    accent = RUAccent()
    accent.workdir = str(assets)
    accent.custom_dict = {}
    accent.tiny_mode = False
    accent.omographs = _read_gzip_json(required[0])
    accent.omographs.update({"коса": ["к+оса", "кос+а"]})
    accent.omograph_model.load(
        str(assets / "nn" / "nn_omograph" / "turbo3.1"), device="CPU"
    )
    accent.yo_words = _read_gzip_json(required[1])
    accent.accent_model.load(str(assets / "nn" / "nn_accent"), device="CPU")
    accent.yo_homographs = _read_gzip_json(required[2])
    accent.yo_homograph_model.load(
        str(assets / "nn" / "nn_yo_homograph_resolver"), device="CPU"
    )
    accent.accents = _read_gzip_json(required[3])
    accent.accents.update(accent.letters_accent)
    accent.stress_usage_predictor.load(
        str(assets / "nn" / "nn_stress_usage_predictor"), device="CPU"
    )
    if "torch" in sys.modules:
        raise RuntimeError("RUAccent imported PyTorch despite USE_TORCH=0")
    return accent


@dataclass
class FrontendResult:
    text: str
    normalized_text: str
    accented_text: str
    phonemes: str
    phone_tokens: np.ndarray
    punctuation_indices: np.ndarray
    phone_seq_len: int
    unknown_phones: tuple[str, ...]


class TorchlessRussianFrontend:
    def __init__(
        self,
        assets: Path,
        phone_map: Path,
        espeak_executable: str = "espeak-ng",
        text_normalizer: str = "ru-normalizr",
    ) -> None:
        self.text_normalizer_backend = text_normalizer
        self.text_normalizer = load_text_normalizer(text_normalizer)
        self.accent = load_ruaccent_without_downloader(assets)
        self.phone_to_token = json.loads(Path(phone_map).read_text())
        self.espeak_executable = espeak_executable

    def accent_text(self, text: str) -> str:
        if "+" not in text:
            return self.accent.process_all(text)
        plain = text.replace("+", "")
        automatic = self.accent.process_all(plain)
        manual_words = text.split()
        automatic_words = automatic.split()
        if len(manual_words) != len(automatic_words):
            return automatic
        return " ".join(
            manual if "+" in manual else auto
            for manual, auto in zip(manual_words, automatic_words)
        )

    def phonemize(self, text: str, separator: str = "|") -> tuple[str, str, str]:
        normalized = normalize_punctuation(self.text_normalizer(text))
        accented = self.accent_text(normalized)
        sequence = espeak_phonemize(
            accented.replace("+", ""),
            separator=separator,
            executable=self.espeak_executable,
        )
        espeak_words = sequence.split()
        accented_words = accented.split()
        if len(espeak_words) != len(accented_words):
            return normalized, accented, sequence
        output: list[str] = []
        for espeak_word, accented_word in zip(espeak_words, accented_words):
            phones = [phone for phone in espeak_word.split(separator) if phone]
            bare = accented_word.strip(STRIP_WORD).lower()
            if bare in PROCLITICS:
                keep = PROCLITICS[bare]
                tail = (
                    phones[-1][-1]
                    if phones and phones[-1][-1] in "".join(PUNCTUATION)
                    else ""
                )
                phones = [keep + tail] if tail else [keep]
            else:
                fixed = INTERJECTION_FIX.get(bare) or interjection_redup(bare)
                if fixed is not None:
                    tail = (
                        phones[-1][-1]
                        if phones and phones[-1][-1] in "".join(PUNCTUATION)
                        else ""
                    )
                    phones = fixed.split()
                    if tail:
                        phones[-1] += tail
            tail = ""
            if phones and phones[-1] and phones[-1][-1] in "".join(PUNCTUATION):
                tail = phones[-1][-1]
                phones[-1] = phones[-1][:-1]
                if not phones[-1]:
                    phones.pop()
            phones = [
                phone
                for phone in (value.strip(NON_PHONE) for value in phones)
                if phone
            ]
            stress_index = stressed_vowel_index(accented_word)
            if stress_index >= 0 and phones:
                cleaned = [phone.replace("ˈ", "").replace("ˌ", "") for phone in phones]
                vowel_positions = [
                    index
                    for index, phone in enumerate(cleaned)
                    if any(character in VOWELS for character in phone)
                ]
                if stress_index < len(vowel_positions):
                    position = vowel_positions[stress_index]
                    cleaned[position] = "ˈ" + cleaned[position]
                    phones = cleaned
            word = separator.join(phones)
            output.append(word + tail if tail else word)
            if tail and tail in SILENCE_AFTER:
                output.append("sil")
        return normalized, accented, " ".join(output)

    def prepare(
        self,
        text: str,
        prompt_prefix: np.ndarray,
        max_sequence: int = 512,
        silence_token: int = 120,
        eos_token: int = 165,
        unknown_token: int = 163,
    ) -> FrontendResult:
        normalized, accented, phonemes = self.phonemize(text)
        phones: list[str] = []
        punctuation_insertions: list[int] = []
        punctuation_tokens: list[int] = []
        unknown: set[str] = set()
        for word in phonemes.split():
            for phone in word.split("|"):
                if phone:
                    phones.append(phone)
            if phones and phones[-1].endswith(PUNCTUATION):
                value = phones.pop()
                symbol = value[-1]
                if value[:-1]:
                    phones.append(value[:-1])
                punctuation_tokens.append(int(self.phone_to_token[symbol]))
                punctuation_insertions.append(len(phones))
        generated: list[int] = []
        for phone in phones:
            if phone not in self.phone_to_token:
                unknown.add(phone)
                generated.append(int(self.phone_to_token.get("unk", unknown_token)))
            else:
                generated.append(int(self.phone_to_token[phone]))
        punctuation_delete = (
            np.asarray(punctuation_insertions, dtype=np.int64)
            + np.arange(len(punctuation_insertions), dtype=np.int64)
        )
        generated_array = np.insert(
            np.asarray(generated, dtype=np.int64),
            punctuation_insertions,
            punctuation_tokens,
        )
        generated_array = np.concatenate(
            [
                generated_array,
                np.asarray([silence_token, eos_token], dtype=np.int64),
            ]
        )
        prompt_prefix = np.ascontiguousarray(prompt_prefix, dtype=np.int64)
        if prompt_prefix.ndim != 1:
            raise ValueError("prompt phone prefix must be one-dimensional")
        if prompt_prefix.size + generated_array.size > max_sequence:
            raise ValueError(
                "text frontend produced a phone sequence longer than the "
                f"TensorRT profile: {prompt_prefix.size + generated_array.size} "
                f"> {max_sequence}"
            )
        conditional = np.concatenate([prompt_prefix, generated_array])
        unconditional = np.concatenate(
            [
                prompt_prefix,
                np.full(generated_array.shape, unknown_token, dtype=np.int64),
            ]
        )
        phone_tokens = np.stack([conditional, unconditional])
        punctuation_indices = np.repeat(
            (punctuation_delete + prompt_prefix.size)[None], 2, axis=0
        )
        return FrontendResult(
            text=text,
            normalized_text=normalized,
            accented_text=accented,
            phonemes=phonemes,
            phone_tokens=phone_tokens,
            punctuation_indices=punctuation_indices,
            phone_seq_len=int(
                phone_tokens.shape[1] - 2 - punctuation_indices.shape[1]
            ),
            unknown_phones=tuple(sorted(unknown)),
        )
