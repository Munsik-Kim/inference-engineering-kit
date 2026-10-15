"""Adapted existing SSE contracts; no real model/engine is involved."""
import asyncio
import unittest
from inference_kit.adapters.http import block,sse_events
from inference_kit.adapters.protocol import Adapter
from http_mock import MockServer,fixture

class BytesContent:
    def __init__(self, chunks):
        self.chunks = chunks

    async def iter_any(self):
        for chunk in self.chunks:
            yield chunk

class SSETests(unittest.IsolatedAsyncioTestCase):
    async def test_byte_fragmentation_unicode_crlf_multiline_comments(self):
        raw = ':comment\r\nevent:message\r\ndata:{"text":\r\ndata:"가"}\r\n\r\ndata:[DONE]\n\n'.encode()
        frames = [f async for f in sse_events(BytesContent([raw[i:i + 1] for i in range(len(raw))]))]
        self.assertEqual(frames, ['{"text":\n"가"}', '[DONE]'])

    async def test_unterminated_frame_is_error(self):
        with self.assertRaisesRegex(ValueError, "inside an SSE"):
            _ = [f async for f in sse_events(BytesContent([b'data:{}\n']))]

class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_engines_portable_without_token_ids(self):
        for engine in ("vllm", "sglang"):
            with self.subTest(engine=engine):
                async with MockServer(engine) as mock:
                    result = await block(mock.url, Adapter(engine), fixture()["rows"], 2, output_tokens=32)
                    self.assertEqual(result["summary"]["status"], "COMPLETE")
                    self.assertEqual(mock.posts, 4)
                    self.assertTrue(all(isinstance(b["prompt"], str) and "return_token_ids" not in b for b in mock.received))
                    self.assertEqual(mock.peak, 2)

    async def test_empty_choices_finish_and_usage_do_not_change_content_timing(self):
        async with MockServer("vllm", delay=.03) as mock:
            record = (await block(mock.url, Adapter("vllm"), fixture()["rows"][:1], 1, output_tokens=4))["records"][0]
        self.assertEqual(record["status"], "OK")
        trace = record["trace"]
        content = [t for t in trace if t["text_chars"]]
        self.assertEqual(record["ttft_seconds"], content[0]["seconds"])
        self.assertEqual(record["last_content_seconds"], content[-1]["seconds"])
        self.assertGreater(record["ttft_seconds"], trace[0]["seconds"])
        self.assertGreater(record["http_complete_seconds"], record["last_content_seconds"])
        self.assertAlmostEqual(record["tpot_stream_estimate_seconds"], (content[-1]["seconds"] - content[0]["seconds"]) / 3)

    async def test_grouped_chunks_count_usage_tokens_not_events(self):
        async with MockServer("sglang") as mock:
            result = await block(mock.url, Adapter("sglang"), fixture()["rows"][:1], 1, output_tokens=32)
        row = result["records"][0]
        self.assertEqual(row["content_event_count"], 2)
        self.assertEqual(row["usage"]["completion_tokens"], 32)
        self.assertAlmostEqual(result["summary"]["output_tokens_per_second"], 32 / result["summary"]["elapsed_seconds"])

    async def test_semaphore_caps_active_requests_and_keeps_input_order(self):
        for cap in (1, 4):
            async with MockServer("vllm", delay=.02) as mock:
                result = await block(mock.url, Adapter("vllm"), fixture()["rows"], cap)
                self.assertEqual(mock.peak, cap)
                self.assertEqual([r["request_id"] for r in result["records"]], [r["id"] for r in fixture()["rows"]])
                if cap == 1:
                    self.assertGreater(result["records"][-1]["client_queue_seconds"], .05)
                    self.assertLess(result["records"][-1]["ttft_seconds"], result["records"][-1]["client_queue_seconds"])

    async def test_strict_ids_for_source_checked_engines(self):
        for engine in ("vllm", "sglang"):
            async with MockServer(engine) as mock:
                result = await block(mock.url, Adapter(engine), fixture()["rows"][:1], 1,
                                     output_tokens=4, profile="token_ids")
            self.assertEqual(result["summary"]["status"], "COMPLETE")
            self.assertIsNotNone(result["records"][0]["output_ids_hash"])
            self.assertEqual(result["records"][0]["content_event_count"], 2)

    async def test_missing_or_wrong_ids_fails_without_text_fallback(self):
        for scenario in ("reject_ids", "missing_ids", "wrong_prompt_ids"):
            async with MockServer("sglang", scenario=scenario) as mock:
                result = await block(mock.url, Adapter("sglang"), fixture()["rows"][:1], 1,
                                     output_tokens=4, profile="token_ids")
                self.assertEqual(mock.posts, 1)
            self.assertEqual(result["summary"]["status"], "INVALID")
            self.assertIsNone(result["summary"]["output_tokens_per_second"])
            self.assertIsNone(result["records"][0]["tpot_stream_estimate_seconds"])

    async def test_faults_are_retained_and_never_retried(self):
        for scenario in ("http_error", "bad_json", "stream_error", "truncated", "missing_usage", "wrong_length", "wrong_output_length", "early_stop", "timeout"):
            with self.subTest(scenario=scenario):
                async with MockServer("vllm", scenario=scenario) as mock:
                    result = await block(mock.url, Adapter("vllm"), fixture()["rows"][:1], 1,
                                         output_tokens=4, timeout=.06 if scenario == "timeout" else 2)
                    self.assertEqual(mock.posts, 1)
                self.assertEqual(len(result["records"]), 1)
                self.assertEqual(result["summary"]["failures"], 1)
                self.assertIsNone(result["summary"]["output_tokens_per_second"])
                self.assertIsNotNone(result["records"][0]["error"])

    async def test_single_token_tpot_is_null(self):
        async with MockServer("sglang") as mock:
            result = await block(mock.url, Adapter("sglang"), fixture()["rows"][:1], 1, output_tokens=1)
        self.assertEqual(result["summary"]["status"], "COMPLETE")
        self.assertIsNone(result["records"][0]["tpot_stream_estimate_seconds"])

    async def test_http_completion_waits_for_eof_after_done(self):
        async with MockServer("sglang", scenario="delayed_eof") as mock:
            result = await block(mock.url, Adapter("sglang"), fixture()["rows"][:1], 1, output_tokens=4)
        row = result["records"][0]
        self.assertEqual(row["status"], "OK")
        self.assertGreater(row["http_complete_seconds"] - row["trace"][-1]["seconds"], .03)

    async def test_invalid_block_is_not_summarized_after_dropping_failed_rows(self):
        async with MockServer("sglang", scenario="http_error") as mock:
            result = await block(mock.url, Adapter("sglang"), fixture()["rows"], 2)
        self.assertEqual(result["summary"]["failures"], 4)
        self.assertEqual(len(result["records"]), 4)
        self.assertNotIn("ttft_seconds", result["summary"])
