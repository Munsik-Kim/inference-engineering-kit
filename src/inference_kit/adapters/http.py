# Derived from lab client; original SSE parser/timing formulas retained.
"""Common asynchronous HTTP/SSE client. No engine SDK and no POST retries."""
import asyncio
import json
import math
import time


import hashlib

def sha(value):
    # Preserve lab/Case009 ASCII-escaped JSON hashing, distinct from artifact digest.
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",", ":")).encode()).hexdigest()


async def sse_events(content):
    """Parse SSE frames, not TCP chunks; preserve fragmented UTF-8/CRLF."""
    pending = b""
    data = []
    async for chunk in content.iter_any():
        pending += chunk
        if len(pending) > 1024 * 1024:
            raise ValueError("SSE line exceeds 1 MiB")
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            line = line.removesuffix(b"\r")
            if not line:
                if data:
                    yield b"\n".join(data).decode("utf-8")
                    data.clear()
            elif line.startswith(b"data:"):
                value = line[5:]
                data.append(value[1:] if value.startswith(b" ") else value)
                if sum(map(len, data)) > 1024 * 1024:
                    raise ValueError("SSE event exceeds 1 MiB")
            # Comments, id, event and retry fields do not affect timing.
    if pending or data:
        raise ValueError("Stream ended inside an SSE frame")


def _ids(value):
    if not isinstance(value, list) or any(type(i) is not int or i < 0 for i in value):
        raise ValueError("Invalid token IDs in response")
    return value


async def request(session, url, adapter, row, semaphore, model, output_tokens, profile, abort_event=None):
    queued = time.perf_counter()
    async with semaphore:
        start = time.perf_counter()
        first = last = None
        finish = usage = None
        ids, text_parts, trace = [], [], []
        prompt_ids = None
        done = False
        error = None
        http_submit = None
        try:
            if abort_event is not None and abort_event.is_set():
                raise ValueError("New request submission stopped by ownership/resource gate")
            body = adapter.body(row, model, output_tokens, profile)
            http_submit = time.perf_counter()
            async with session.post(url.rstrip("/") + "/v1/completions", json=body) as response:
                if response.status != 200:
                    raise ValueError(f"HTTP {response.status}: {(await response.text())[:300]}")
                if response.content_type != "text/event-stream":
                    raise ValueError("Expected text/event-stream")
                async for frame in sse_events(response.content):
                    now = time.perf_counter()
                    if frame.strip() == "[DONE]":
                        if done:
                            raise ValueError("Duplicate [DONE]")
                        done = True
                        continue
                    if done:
                        raise ValueError("SSE data after [DONE]")
                    event = adapter.event(json.loads(frame))
                    delta = _ids(event["ids"]) if event["ids"] is not None else []
                    if event["prompt_ids"] is not None:
                        observed = _ids(event["prompt_ids"])
                        if prompt_ids is not None and prompt_ids != observed:
                            raise ValueError("Prompt IDs changed within stream")
                        prompt_ids = observed
                    meaningful = bool(event["text"]) if profile == "text" else bool(delta)
                    if meaningful:
                        if finish is not None:
                            raise ValueError("Content after finish_reason")
                        first = now if first is None else first
                        last = now
                    if event["finish"] is not None:
                        finish = event["finish"]
                    ids.extend(delta)
                    text_parts.append(event["text"])
                    if event["usage"] is not None:
                        usage = event["usage"]
                    trace.append({"seconds": now - start, "text_chars": len(event["text"]),
                                  "token_ids_count": len(delta), "choice": event["choice"],
                                  "finish_reason": event["finish"], "usage": event["usage"]})
            if not done:
                raise ValueError("Missing [DONE]; truncated stream")
            if first is None or not isinstance(usage, dict):
                raise ValueError("Missing meaningful content/usage")
            if (type(usage.get("prompt_tokens")) is not int or
                    type(usage.get("completion_tokens")) is not int or
                    usage["prompt_tokens"] != row["input_length"] or
                    usage["completion_tokens"] != output_tokens):
                raise ValueError("Actual token length differs from frozen workload")
            if usage.get("total_tokens") != row["input_length"] + output_tokens:
                raise ValueError("Invalid total_tokens")
            if finish != "length":
                raise ValueError("Expected forced-length finish_reason=length")
            if profile == "token_ids":
                if prompt_ids != row["token_ids"]:
                    raise ValueError("Missing/mismatched prompt_token_ids; strict capability NOT verified")
                if len(ids) != output_tokens:
                    raise ValueError("Missing/mismatched token_ids; strict capability NOT verified")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        completed = time.perf_counter()
        valid = error is None
        return {"request_id": row["id"], "prompt_hash": row["prompt_hash"],
                "text_hash": row.get("text_hash"), "engine": adapter.engine, "profile": profile,
                "status": "OK" if valid else "ERROR", "error": error,
                "started_monotonic": start, "completed_monotonic": completed,
                "queued_monotonic": queued, "semaphore_acquired_monotonic": start,
                "http_submit_monotonic": http_submit,
                "first_content_monotonic": first, "last_content_monotonic": last,
                "client_queue_seconds": start - queued, "http_complete_seconds": completed - start,
                "ttft_seconds": first - start if valid else None,
                "last_content_seconds": last - start if valid else None,
                "tpot_stream_estimate_seconds": (last - first) / (output_tokens - 1) if valid and output_tokens > 1 else None,
                "usage": usage, "finish_reason": finish, "saw_done": done,
                "output_text_hash": sha("".join(text_parts)),
                "output_text": "".join(text_parts), "observed_prompt_ids": prompt_ids,
                "output_token_ids": ids if ids else None,
                "output_ids_hash": sha(ids) if profile == "token_ids" else None,
                "sse_event_count": len(trace), "content_event_count": sum(bool(t["text_chars"] if profile == "text" else t["token_ids_count"]) for t in trace),
                "trace": trace}


def quantile(values, probability):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * probability
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def summary(records, elapsed):
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Invalid block duration")
    failed = sum(r["status"] != "OK" for r in records)
    result = {"status": "INVALID" if failed else "COMPLETE", "n_requests": len(records),
              "failures": failed, "elapsed_seconds": elapsed,
              "output_tokens_per_second": None}
    # Do not create an apparently faster result by removing failed requests.
    if failed:
        return result
    for key in ("ttft_seconds", "last_content_seconds", "tpot_stream_estimate_seconds"):
        values = [r[key] for r in records if r[key] is not None]
        result[key] = {"p50": quantile(values, .5), "p95": quantile(values, .95)}
    result["output_tokens_per_second"] = sum(r["usage"]["completion_tokens"] for r in records) / elapsed
    return result


async def block(url, adapter, rows, concurrency, model="serving-lab", output_tokens=32,
                profile="text", timeout=30, abort_event=None, record_callback=None):
    if concurrency < 1 or timeout <= 0 or not rows:
        raise ValueError("Invalid block parameters")
    import aiohttp  # Optional HTTP runtime; CPU summaries/parser import without it.
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout),
                                     connector=aiohttp.TCPConnector(limit=0)) as session:
        start = time.perf_counter()
        async def one(row):
            record = await request(session, url, adapter, row, semaphore, model,
                                   output_tokens, profile, abort_event)
            if record_callback is not None:
                record_callback(record)
            return record
        records = await asyncio.gather(*(one(row) for row in rows))
        elapsed = time.perf_counter() - start
    return {"concurrency": concurrency, "records": records, "summary": summary(records, elapsed)}
