import unittest

import numpy as np

from voxtream2_ru_jetson.sink_attention import (
    SinkAttentionConfig,
    SinkAttentionHistory,
)


class SinkAttentionHistoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.prompt = np.arange(2 * 3 * 2, dtype=np.uint16).reshape(2, 3, 2)
        self.history = SinkAttentionHistory(
            SinkAttentionConfig(audio_window_size=9),
            self.prompt,
        )

    def test_matches_upstream_tail_budget_and_rebuild_offset(self) -> None:
        self.assertEqual(self.history.tail_limit, 4)
        for value in range(6):
            self.history.append(np.full((2, 1, 2), value, dtype=np.uint16))

        self.assertFalse(self.history.needs_rebuild(8))
        self.assertTrue(self.history.needs_rebuild(9))
        rebuilt = self.history.rebuild_sequence(9)

        np.testing.assert_array_equal(rebuilt[:, :3], self.prompt)
        np.testing.assert_array_equal(
            rebuilt[:, 3:, 0],
            np.asarray([[2, 3, 4, 5], [2, 3, 4, 5]], dtype=np.uint16),
        )
        self.assertEqual(rebuilt.shape, (2, 7, 2))
        self.assertEqual(self.history.position_offset, 2)
        self.assertEqual(self.history.local_position(9), 7)
        self.assertEqual(self.history.compactions, 1)

    def test_reset_restores_first_segment(self) -> None:
        self.history.append(np.zeros((2, 1, 2), dtype=np.uint16))
        self.history.rebuild_sequence(9)
        self.history.reset()
        self.assertEqual(self.history.position_offset, 0)
        self.assertEqual(self.history.tail_length, 0)
        self.assertEqual(self.history.compactions, 0)

    def test_rejects_wrong_tail_shape_and_early_rebuild(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape changed"):
            self.history.append(np.zeros((2, 2, 2), dtype=np.uint16))
        with self.assertRaisesRegex(ValueError, "before the window"):
            self.history.rebuild_sequence(8)

    def test_production_window_rebuilds_to_position_420(self) -> None:
        history = SinkAttentionHistory(
            SinkAttentionConfig(audio_window_size=625),
            np.zeros((2, 108, 4), dtype=np.uint16),
        )
        for _ in range(517):
            history.append(np.zeros((2, 1, 4), dtype=np.uint16))
        rebuilt = history.rebuild_sequence(625)
        self.assertEqual(history.tail_limit, 312)
        self.assertEqual(rebuilt.shape, (2, 420, 4))
        self.assertEqual(history.position_offset, 205)
        self.assertEqual(history.local_position(625), 420)


if __name__ == "__main__":
    unittest.main()
