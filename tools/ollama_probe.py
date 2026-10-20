"""Standalone Ollama protocol probe: bypasses FastAPI and the browser.

Run from project root with the same Python environment as Uvicorn:
    python tools/ollama_probe.py --url http://127.0.0.1:11434 --model llama3.2:3b
    python tools/ollama_probe.py --url http://127.0.0.1:11434 --model llama3.2:3b --non-stream

Uses a harmless, fixed test prompt. Doesn't read local notes or user settings.
"""
import argparse
import asyncio
import json
import sys
import time

import httpx


async def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose raw Ollama /api/chat responses")
    parser.add_argument("--url", required=True, help="Ollama base URL (e.g. http://127.0.0.1:11434)")
    parser.add_argument("--model", required=True, help="Exact Ollama model name")
    parser.add_argument("--think", default="off", help="off, on, default, low, medium, high")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP read timeout in seconds")
    parser.add_argument("--non-stream", action="store_true", help="Ask for a single JSON response")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Ответь одним словом: готово"}],
        "stream": not args.non_stream,
        "options": {"num_predict": 48, "num_ctx": 2048, "temperature": 0},
    }
    if args.think != "default":
        payload["think"] = {"off": False, "on": True}.get(args.think, args.think)
    print(f"Ollama URL: {base}, model: {args.model}, think: {args.think}, stream: {not args.non_stream}")
    timeout = httpx.Timeout(connect=10, read=args.timeout, write=30, pool=10)
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            try:
                response = await client.get(base + "/api/version")
                print(f"GET /api/version -> HTTP {response.status_code}, body={response.text[:120]!r}")
            except httpx.HTTPError as exc:
                print(f"GET /api/version -> {type(exc).__name__}: {exc}")
            started = time.perf_counter()
            async with client.stream("POST", base + "/api/chat", json=payload) as response:
                print(f"POST /api/chat -> HTTP {response.status_code} after {time.perf_counter() - started:.2f}s")
                print(f"Content-Type: {response.headers.get('content-type', '(none)')}")
                if response.status_code != 200:
                    print(f"HTTP error body: {(await response.aread())[:600]!r}")
                    return 2
                if args.non_stream:
                    body = await response.aread()
                    try:
                        item = json.loads(body)
                    except ValueError as exc:
                        print(f"BAD JSON: {exc}; body prefix={body[:200]!r}")
                        return 3
                    print("Single JSON response: valid. keys:", list(item)[:15])
                    print("Reply:", repr((item.get("message") or {}).get("content", "")[:150]))
                    return 0
                count = 0
                saw_done = False
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith((":", "event:", "id:", "retry:")):
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    count += 1
                    if count <= 3:
                        print(f"Line {count}, {time.perf_counter() - started:.1f}s: {line[:150]!r}")
                    try:
                        item = json.loads(line)
                    except (ValueError, UnicodeError) as exc:
                        print(f"INVALID JSON line {count}: {type(exc).__name__}: {exc}")
                        print(f"RAW prefix: {line[:250]!r}")
                        print("This response is not valid Ollama NDJSON. Check reverse proxy/server.")
                        return 3
                    if not isinstance(item, dict):
                        print(f"Line {count}: unexpected JSON type {type(item).__name__}")
                        return 3
                    if item.get("error"):
                        print(f"UPSTREAM ERROR: {item['error']}")
                        return 4
                    if item.get("done") is True:
                        saw_done = True
                        print(f"Done after {time.perf_counter() - started:.1f}s; lines={count}; "
                              f"eval_count={item.get('eval_count')}; "
                              f"prompt_eval_count={item.get('prompt_eval_count')}")
                        break
                if not saw_done:
                    print(f"INCOMPLETE: connection ended after {count} lines without done=true")
                    return 5
                return 0
    except httpx.HTTPError as exc:
        print(f"HTTP TRANSPORT ERROR: {type(exc).__name__}: {exc}")
        return 6


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
