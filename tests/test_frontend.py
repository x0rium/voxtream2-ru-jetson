import unittest

from voxtream2_ru_jetson.frontend import (
    interjection_redup,
    load_text_normalizer,
    normalize_punctuation,
    restore_punctuation,
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


if __name__ == "__main__":
    unittest.main()
