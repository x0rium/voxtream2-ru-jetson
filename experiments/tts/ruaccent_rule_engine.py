"""Restore RUAccent's v1.5.7 RuleEngine stage on the v1.5.8 tokenizer."""

from __future__ import annotations

import re
import types

from ruaccent.text_preprocessor import TextPreprocessor


def _preserve_case(source: str, accented: str) -> str:
    """Copy source casing while ignoring '+' stress markers in the target."""

    if source.casefold() != accented.replace("+", "").casefold():
        return accented
    result = []
    source_index = 0
    for character in accented:
        if character == "+":
            result.append(character)
            continue
        original = source[source_index]
        result.append(character.upper() if original.isupper() else character.lower())
        source_index += 1
    return "".join(result)


def process_all_internal_with_rule_engine(accent, text: str) -> tuple[str, list[dict]]:
    """Run the pre-v1.5.8 RuleEngine stage, then the current neural stages."""

    text = re.sub(accent.normalize, "", text)
    outputs: list[str] = []
    decisions: list[dict] = []
    for sentence in TextPreprocessor.split_by_sentences(text):
        words, remaining_text = TextPreprocessor.split_by_words(sentence)
        if not words:
            outputs.append("".join(remaining_text))
            continue

        rule_tokens = accent.rule_accent.accentuate(sentence)
        aligned = len(rule_tokens) == len(words)
        if aligned:
            for index, rule_token in enumerate(rule_tokens):
                if "+" not in rule_token:
                    continue
                original = words[index]
                words[index] = _preserve_case(original, rule_token)
                decisions.append(
                    {
                        "sentence": sentence,
                        "original": original,
                        "rule": words[index],
                        "position": index,
                    }
                )

        stress_usages = accent.extract_entities(
            accent.stress_usage_predictor.predict_stress_usage(sentence)
        )
        processed_words = accent._process_yo(words, sentence)
        processed_words = accent._process_omographs(processed_words)
        processed_words = accent._process_accent(processed_words, stress_usages)
        processed_sentence = "".join(
            [left + word for left, word in zip(remaining_text, processed_words)]
            + [remaining_text[-1]]
        )
        outputs.append(accent.delete_spaces_before_punc(processed_sentence))
        if not aligned:
            decisions.append(
                {
                    "sentence": sentence,
                    "alignment_error": True,
                    "words": words,
                    "rule_tokens": rule_tokens,
                }
            )
    return "".join(outputs), decisions


def install_rule_engine_pipeline(accent) -> None:
    """Replace process_all() while preserving its skip_regex behavior."""

    def process_all(self, text: str, skip_regex: str | None = None) -> str:
        if not skip_regex:
            return process_all_internal_with_rule_engine(self, text)[0]

        pattern = re.compile(skip_regex)
        matches = list(pattern.finditer(text))
        if not matches:
            return process_all_internal_with_rule_engine(self, text)[0]

        cursor = 0
        output = []
        for match in matches:
            if match.start() > cursor:
                output.append(
                    process_all_internal_with_rule_engine(self, text[cursor : match.start()])[0]
                )
            output.append(match.group(0))
            cursor = match.end()
        if cursor < len(text):
            output.append(process_all_internal_with_rule_engine(self, text[cursor:])[0])
        return "".join(output)

    accent.process_all = types.MethodType(process_all, accent)
