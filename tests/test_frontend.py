import unittest

from voxtream2_ru_jetson.frontend import (
    PhoneSequenceTooLongError,
    interjection_redup,
    load_text_normalizer,
    normalize_punctuation,
    restore_punctuation,
    split_text_at_natural_boundary,
    stressed_vowel_index,
    strip_punctuation_to_restore,
)


class FrontendUnitTests(unittest.TestCase):
    def test_none_normalizer_is_identity(self) -> None:
        text = "Версия 2.7.1"
        self.assertEqual(load_text_normalizer("none")(text), text)

    def test_unknown_normalizer_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported Russian text normalizer"):
            load_text_normalizer("missing")

    def test_punctuation_is_canonicalized(self) -> None:
        self.assertEqual(normalize_punctuation("Привет —  мир!!!"), "Привет, мир!")

    def test_stress_marker_maps_to_vowel_index(self) -> None:
        self.assertEqual(stressed_vowel_index("Дж+етсоне"), 0)
        self.assertEqual(stressed_vowel_index("автон+омно"), 2)

    def test_interjection_reduplication_is_bounded(self) -> None:
        self.assertEqual(interjection_redup("мммм"), "m m m")

    def test_punctuation_round_trip(self) -> None:
        text = "Привет, мир!"
        chunks, punctuation = strip_punctuation_to_restore(text)
        self.assertEqual("".join(restore_punctuation(chunks, punctuation)), text)

    def test_long_text_split_prefers_sentence_boundary(self) -> None:
        left, right = split_text_at_natural_boundary(
            "Первое короткое предложение. Второе предложение немного длиннее. "
            "Третье завершает проверку."
        )
        self.assertEqual(left, "Первое короткое предложение.")
        self.assertTrue(right.startswith("Второе предложение"))

    def test_long_text_split_falls_back_to_clause(self) -> None:
        left, right = split_text_at_natural_boundary(
            "Сначала работает первая часть, затем продолжается вторая часть без точки"
        )
        self.assertTrue(left.endswith(","))
        self.assertTrue(right.startswith("затем"))

    def test_long_text_split_requires_a_safe_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "no safe whitespace boundary"):
            split_text_at_natural_boundary("длинноесловобезпробелов")

    def test_phone_sequence_error_exposes_profile_sizes(self) -> None:
        error = PhoneSequenceTooLongError(767, 640)
        self.assertEqual(error.actual, 767)
        self.assertEqual(error.maximum, 640)
        self.assertIn("767 > 640", str(error))


if __name__ == "__main__":
    unittest.main()
