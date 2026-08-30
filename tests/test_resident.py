import io
import json
import unittest

from voxtream2_ru_jetson.resident import (
    DONE,
    ERROR,
    PCM,
    READY,
    START,
    iter_records,
    serve_jsonl,
)


class FragmentedReader:
    def __init__(self, payload):
        self._source = io.BytesIO(payload)

    def read(self, size):
        return self._source.read(min(size, 1))


class FakeStream:
    def __init__(self, chunks, result):
        self._chunks = iter(chunks)
        self._final_result = result
        self.result = None
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            self.result = self._final_result
            self.closed = True
            raise

    def close(self):
        self.closed = True


class FakeRuntime:
    startup_seconds = 12.3456
    sample_rate = 24000
    samples_per_chunk = 1920

    def __init__(self):
        self.request_count = 0
        self.requests = []

    def synthesize_stream(self, text, **options):
        request_index = self.request_count
        self.request_count += 1
        self.requests.append((text, options))
        return FakeStream(
            [b"first", b"second"],
            {
                "request_index": request_index,
                "text": text,
                "trajectory": [],
            },
        )


class ResidentProtocolTests(unittest.TestCase):
    def test_streams_raw_pcm_and_completion_metrics(self):
        runtime = FakeRuntime()
        source = io.StringIO('{"id":"turn-1","text":"Привет","seed":7,"max_frames":10}\n')
        sink = io.BytesIO()

        serve_jsonl(runtime, source, sink)

        records = list(iter_records(io.BytesIO(sink.getvalue())))
        self.assertEqual(
            [kind for kind, _ in records],
            [READY, START, PCM, PCM, DONE],
        )
        self.assertEqual(records[2][1], b"first")
        self.assertEqual(records[3][1], b"second")
        ready = json.loads(records[0][1])
        done = json.loads(records[-1][1])
        self.assertEqual(ready["pcm"]["bytes_per_chunk"], 3840)
        self.assertEqual(done["id"], "turn-1")
        self.assertEqual(done["request_index"], 0)
        self.assertEqual(runtime.requests[0][1]["seed"], 7)
        self.assertFalse(runtime.requests[0][1]["include_trajectory"])

    def test_bad_request_is_reported_without_stopping_loop(self):
        runtime = FakeRuntime()
        source = io.StringIO('{"id":"bad","text":""}\n{"id":"good","text":"Работаем"}\n')
        sink = io.BytesIO()

        serve_jsonl(runtime, source, sink)

        records = list(iter_records(io.BytesIO(sink.getvalue())))
        self.assertEqual(records[1][0], ERROR)
        self.assertEqual(json.loads(records[1][1])["id"], "bad")
        self.assertEqual(records[-1][0], DONE)
        self.assertEqual(json.loads(records[-1][1])["id"], "good")

    def test_truncated_record_is_rejected(self):
        with self.assertRaisesRegex(EOFError, "truncated resident record header"):
            list(iter_records(io.BytesIO(b"P\x05")))

    def test_fragmented_pipe_reads_are_reassembled(self):
        sink = io.BytesIO()
        serve_jsonl(FakeRuntime(), io.StringIO('{"text":"Тест"}\n'), sink)
        records = list(iter_records(FragmentedReader(sink.getvalue())))
        self.assertEqual(records[0][0], READY)
        self.assertEqual(records[-1][0], DONE)


if __name__ == "__main__":
    unittest.main()
