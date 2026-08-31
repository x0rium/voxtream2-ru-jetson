import json
import unittest

from voxtream2_ru_jetson.playback import aplay_command, forward_records
from voxtream2_ru_jetson.resident import DONE, ERROR, PCM, START


class FakeSink:
    def __init__(self):
        self.payloads = []

    def write(self, payload):
        self.payloads.append(payload)


class PlaybackTests(unittest.TestCase):
    def test_forwards_pcm_until_matching_done(self):
        records = [
            (START, json.dumps({"id": "turn"}).encode()),
            (PCM, b"one"),
            (PCM, b"two"),
            (DONE, json.dumps({"id": "turn", "rtf": 0.8}).encode()),
        ]
        sink = FakeSink()
        result = forward_records(records, sink, request_id="turn")
        self.assertEqual(sink.payloads, [b"one", b"two"])
        self.assertEqual(result["playback_pcm_records"], 2)

    def test_resident_error_is_not_silenced(self):
        records = [
            (
                ERROR,
                json.dumps(
                    {"id": "turn", "error_type": "ValueError", "message": "bad text"}
                ).encode(),
            )
        ]
        with self.assertRaisesRegex(RuntimeError, "ValueError: bad text"):
            forward_records(records, FakeSink(), request_id="turn")

    def test_aplay_uses_explicit_latency_and_device(self):
        command = aplay_command(
            sample_rate=24000,
            channels=1,
            device="hw:2,0",
            buffer_time_us=320000,
            period_time_us=80000,
        )
        self.assertIn("--buffer-time=320000", command)
        self.assertIn("--period-time=80000", command)
        self.assertIn("--device=hw:2,0", command)


if __name__ == "__main__":
    unittest.main()
