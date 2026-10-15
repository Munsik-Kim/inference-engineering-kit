"""CPU protocol fixtures, deliberately NOT inference or a tokenizer emulator."""
import asyncio
import json

from aiohttp import web

from inference_kit.adapters.http import sha
SCHEMA="synthetic-http-test-only"


def fixture():
    rows = []
    for i in range(4):
        ids = [100 + i, 201, 202, 203]
        text = f"CPU mock prompt {i}"
        rows.append({"id": f"mock-{i}", "input_length": 4, "token_ids": ids,
                     "prompt_hash": sha(ids), "prompt_text": text, "text_hash": sha(text)})
    return {"schema": SCHEMA, "scope": "MOCK_ONLY", "tokenizer_files": None,
            "rows": rows, "rows_hash": sha(rows)}


class MockServer:
    def __init__(self, engine, scenario="normal", delay=.01):
        self.engine, self.scenario, self.delay = engine, scenario, delay
        self.active = self.peak = self.posts = 0
        self.received = []
        self.runner = None

    async def __aenter__(self):
        app = web.Application()
        app.router.add_post("/v1/completions", self.completion)
        async def health(request):
            return web.json_response({"status": "ok"})
        async def models(request):
            return web.json_response({"data": [{"id": "serving-lab"}]})
        app.router.add_get("/health", health)
        app.router.add_get("/v1/models", models)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = next(iter(self.runner.sites))._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        return self

    async def __aexit__(self, *args):
        await self.runner.cleanup()

    async def completion(self, request):
        self.posts += 1
        body = await request.json()
        self.received.append(body)
        if self.engine == "sglang" and "add_special_tokens" in body:
            return web.json_response({"error": "unsupported SGLang completion extension"}, status=400)
        if self.scenario == "http_error":
            return web.json_response({"error": "mock capacity failure"}, status=503)
        if self.scenario == "reject_ids" and isinstance(body["prompt"], list):
            return web.json_response({"error": "this installed version has no ID input"}, status=422)
        self.active += 1
        self.peak = max(self.peak, self.active)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        try:
            await response.prepare(request)

            async def send(payload):
                if isinstance(payload, str):
                    await response.write(payload.encode())
                    return
                # Each HTTP write is NOT one SSE event and NOT one model token.
                # Exercise comments, no space after colon, CRLF, UTF-8 fragmentation.
                raw = ("event:message\r\n: fixture\r\ndata:" + json.dumps(payload, ensure_ascii=False) + "\r\n\r\n").encode()
                for i in range(0, len(raw), 7):
                    await response.write(raw[i:i + 7])

            def choice(text="", ids=None, finish=None, prompt_ids=None):
                value = {"index": 0, "text": text, "finish_reason": finish}
                if ids is not None:
                    value["token_ids"] = ids
                if prompt_ids is not None:
                    value["prompt_token_ids"] = prompt_ids
                return {"choices": [value]}

            ids_mode = bool(body.get("return_token_ids"))
            prompt = body["prompt"]
            prompt_ids = prompt if isinstance(prompt, list) else [100, 201, 202, 203]
            if self.scenario == "wrong_prompt_ids":
                prompt_ids = [999] * len(prompt_ids)
            if self.scenario == "missing_ids":
                ids_mode = False
            if self.engine == "vllm":
                await send(choice())  # Must not start TTFT.
            if self.scenario == "timeout":
                await asyncio.sleep(.2)
            await asyncio.sleep(self.delay)
            if self.scenario == "bad_json":
                await send("data:{oops}\n\n")
                await response.write_eof()
                return response
            if self.scenario == "stream_error":
                await send({"error": {"message": "mock OOM"}})
                await response.write_eof()
                return response
            n = body["max_tokens"]
            first_ids = [900]
            # Grouped 31-token delta demonstrates why chunk count is not token count.
            await send(choice("가", first_ids if ids_mode else None,
                              prompt_ids=prompt_ids if ids_mode else None))
            if n > 1:
                await asyncio.sleep(self.delay)
                await send(choice("나다", list(range(901, 900 + n)) if ids_mode else None))
            if self.scenario == "truncated":
                await response.write_eof()
                return response
            await asyncio.sleep(self.delay)
            finish = "stop" if self.scenario == "early_stop" else "length"
            await send(choice(finish=finish))  # Must not extend last content.
            await asyncio.sleep(self.delay)
            if self.scenario != "missing_usage":
                usage = {"prompt_tokens": len(prompt_ids), "completion_tokens": n,
                         "total_tokens": len(prompt_ids) + n}
                if self.scenario == "wrong_length":
                    usage["prompt_tokens"] += 1
                    usage["total_tokens"] += 1
                if self.scenario == "wrong_output_length":
                    usage["completion_tokens"] -= 1
                    usage["total_tokens"] -= 1
                await send({"choices": [], "usage": usage})
            await send("data: [DONE]\r\n\r\n")
            if self.scenario == "delayed_eof":
                await asyncio.sleep(.04)
            await response.write_eof()
        except (ConnectionResetError, RuntimeError):
            # Client timeout is expected in fault fixtures.
            pass
        finally:
            self.active -= 1
        return response
