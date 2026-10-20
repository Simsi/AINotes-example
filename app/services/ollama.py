"""Ollama transport. Log protocol/timings, never request or response content by default.

Ollama returns NDJSON. Browser communication uses SSE (app/api/chat.py).
"""
from __future__ import annotations

from collections.abc import AsyncIterator
import json
import logging
import time

import httpx

from app.diagnostics import safe_preview
from app.schemas import Settings

log = logging.getLogger("notes.ollama")


class OllamaStreamFormatError(RuntimeError):
    """The upstream response is not Ollama /api/chat NDJSON."""


def ms_since(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def explain_invalid_line(line: str, content_type: str, exc: Exception) -> OllamaStreamFormatError:
    return OllamaStreamFormatError(
        f"Некорректный JSON в потоке Ollama: {type(exc).__name__}: {exc}; "
        f"Content-Type={content_type or '(отсутствует)'}, "
        f"длина строки={len(line)}, начало строки={safe_preview(line)}. "
        "Ожидается NDJSON от /api/chat. Проверьте прямой curl к Ollama и промежуточный прокси."
    )


class OllamaClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.url = settings.ollama_url

    async def models(self) -> list[str]:
        start = time.perf_counter()
        log.info("probe.start endpoint=/api/tags method=GET url=%s", self.url)
        try:
            async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
                response = await client.get(f"{self.url}/api/tags")
                log.info("probe.headers endpoint=/api/tags status=%s content_type=%s elapsed_ms=%d",
                         response.status_code, response.headers.get("content-type", "-"), ms_since(start))
                response.raise_for_status()
                payload = response.json()
            models = sorted(model["name"] for model in payload.get("models", []) if model.get("name"))
            log.info("probe.success endpoint=/api/tags models=%d elapsed_ms=%d", len(models), ms_since(start))
            log.debug("probe.models names=%s", models)
            return models
        except Exception as exc:
            log.warning("probe.failed endpoint=/api/tags exception=%s elapsed_ms=%d",
                        type(exc).__name__, ms_since(start))
            raise

    async def thinking_options(self, model: str) -> dict:
        """Metadata is optional. Its failure MUST NOT interrupt inference."""
        start = time.perf_counter()
        log.info("probe.start endpoint=/api/show method=POST model=%s", model)
        try:
            async with httpx.AsyncClient(timeout=6, trust_env=False) as client:
                response = await client.post(f"{self.url}/api/show", json={"model": model})
                log.info("probe.headers endpoint=/api/show status=%d elapsed_ms=%d",
                         response.status_code, ms_since(start))
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            log.warning("probe.optional_failed endpoint=/api/show model=%s exception=%s elapsed_ms=%d; "
                        "chat remains available", model, type(exc).__name__, ms_since(start))
            return {"values": None, "default": None,
                    "warning": f"Метаданные thinking недоступны: {type(exc).__name__}. Режим можно задать вручную."}
        thinking = data.get("thinking")
        if not isinstance(thinking, dict):
            log.info("probe.success endpoint=/api/show thinking_metadata=absent elapsed_ms=%d", ms_since(start))
            return {"values": None, "default": None}
        result = {
            "values": thinking.get("values") if isinstance(thinking.get("values"), list) else None,
            "default": thinking.get("default"),
        }
        log.info("probe.success endpoint=/api/show thinking_options=%s elapsed_ms=%d",
                 len(result["values"]) if result["values"] is not None else "unknown", ms_since(start))
        return result

    async def diagnostics(self) -> dict:
        """Lightweight probes only: don't load/infer the selected model."""
        result = {"reachable": False, "version": None, "loaded_models": [], "warnings": []}
        log.info("probe.start endpoint=/api/version,/api/ps url=%s", self.url)
        async with httpx.AsyncClient(timeout=httpx.Timeout(8, connect=5), trust_env=False) as client:
            start = time.perf_counter()
            try:
                response = await client.get(f"{self.url}/api/version")
                log.info("probe.headers endpoint=/api/version status=%d elapsed_ms=%d",
                         response.status_code, ms_since(start))
                response.raise_for_status()
                result["version"] = response.json().get("version")
                result["reachable"] = True
            except (httpx.HTTPError, ValueError) as exc:
                log.warning("probe.failed endpoint=/api/version exception=%s elapsed_ms=%d",
                            type(exc).__name__, ms_since(start))
                result["warnings"].append(f"/api/version: {type(exc).__name__}: {str(exc)[:200]}")
                return result
            start = time.perf_counter()
            try:
                response = await client.get(f"{self.url}/api/ps")
                log.info("probe.headers endpoint=/api/ps status=%d elapsed_ms=%d",
                         response.status_code, ms_since(start))
                response.raise_for_status()
                result["loaded_models"] = [
                    m.get("name") or m.get("model") for m in response.json().get("models", [])
                    if m.get("name") or m.get("model")
                ]
            except (httpx.HTTPError, ValueError) as exc:
                log.warning("probe.failed endpoint=/api/ps exception=%s elapsed_ms=%d",
                            type(exc).__name__, ms_since(start))
                result["warnings"].append(f"/api/ps: {type(exc).__name__}: {str(exc)[:200]}")
        log.info("probe.complete reachable=%s version=%s loaded_models=%d",
                 result["reachable"], result["version"], len(result["loaded_models"]))
        return result

    async def embed(self, text: str) -> list[float]:
        if not self.settings.embedding_model:
            raise ValueError("Не выбрана embedding-модель")
        start = time.perf_counter()
        log.debug("embedding.request model=%s input_chars=%d", self.settings.embedding_model, len(text))
        try:
            async with httpx.AsyncClient(timeout=90, trust_env=False) as client:
                response = await client.post(f"{self.url}/api/embed", json={
                    "model": self.settings.embedding_model, "input": text,
                })
                log.debug("embedding.headers status=%d elapsed_ms=%d", response.status_code, ms_since(start))
                response.raise_for_status()
                data = response.json()
            vectors = data.get("embeddings", [])
            if not vectors or not vectors[0]:
                raise ValueError("Ollama вернула пустой embedding")
            log.debug("embedding.done dims=%d elapsed_ms=%d", len(vectors[0]), ms_since(start))
            return vectors[0]
        except Exception as exc:
            log.warning("embedding.failed model=%s exception=%s elapsed_ms=%d",
                        self.settings.embedding_model, type(exc).__name__, ms_since(start))
            raise

    async def stream_chat(self, messages: list[dict]) -> AsyncIterator[dict]:
        """Events: connected, thinking, token, usage. Exact token usage only at done."""
        if not self.settings.chat_model:
            raise ValueError("Выбери модель чата в настройках")
        options = {
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "num_ctx": self.settings.num_ctx,
            "num_predict": self.settings.num_predict,
        }
        mode = self.settings.think_mode
        think = {"off": False, "on": True, "default": None}.get(mode, mode)
        payload = {"model": self.settings.chat_model, "messages": messages,
                   "options": options, "stream": True}
        if think is not None:
            payload["think"] = think

        started = time.perf_counter()
        stage = "initializing"
        lines = 0
        bytes_seen = 0
        content_chunks = 0
        thinking_chunks = 0
        content_chars = 0
        done = False
        roles = [m.get("role", "unknown") for m in messages]
        log.info("upstream.prepare endpoint=/api/chat url=%s model=%s think=%s "
                 "messages=%d roles=%s chars_by_message=%s options=%s timeout_connect_s=10 timeout_read_s=%s",
                 self.url, self.settings.chat_model, mode, len(messages), roles,
                 [len(str(m.get("content", ""))) for m in messages], options,
                 self.settings.read_timeout_seconds)
        try:
            timeout = httpx.Timeout(connect=10, read=self.settings.read_timeout_seconds, write=30, pool=10)
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                stage = "waiting_for_http_headers"
                log.info("upstream.send endpoint=/api/chat method=POST stream=true elapsed_ms=%d", ms_since(started))
                async with client.stream(
                    "POST", f"{self.url}/api/chat", json=payload,
                    headers={"Accept": "application/x-ndjson", "Accept-Encoding": "identity",
                             "Cache-Control": "no-cache"},
                ) as response:
                    stage = "reading_http_response"
                    content_type = response.headers.get("content-type", "")
                    log.info("upstream.headers status=%d elapsed_ms=%d content_type=%s "
                             "transfer_encoding=%s content_encoding=%s server=%s",
                             response.status_code, ms_since(started), content_type or "-",
                             response.headers.get("transfer-encoding", "-"),
                             response.headers.get("content-encoding", "-"),
                             response.headers.get("server", "-"))
                    if response.is_error:
                        error_body = (await response.aread()).decode("utf-8", errors="replace")[:750]
                        try:
                            error_body = json.loads(error_body).get("error", error_body)
                        except (ValueError, AttributeError):
                            pass
                        log.warning("upstream.http_error status=%d elapsed_ms=%d body_chars=%d preview=%s",
                                    response.status_code, ms_since(started), len(str(error_body)),
                                    safe_preview(str(error_body)))
                        raise RuntimeError(f"Ollama HTTP {response.status_code}: {error_body}")
                    if "text/html" in content_type.lower():
                        raise OllamaStreamFormatError(
                            "Сервер /api/chat вернул HTML с HTTP 200 вместо потока NDJSON "
                            f"(Content-Type={content_type}). Проверьте URL Ollama и прокси."
                        )
                    yield {"type": "connected", "status_code": response.status_code}
                    stage = "reading_ndjson"
                    log.info("upstream.stream.begin elapsed_ms=%d; waiting for first NDJSON record", ms_since(started))
                    async for raw_line in response.aiter_lines():
                        # Only count non-empty protocol lines for parse statistics.
                        line = raw_line.strip()
                        if not line:
                            continue
                        if line.startswith((":", "event:", "id:", "retry:")):
                            log.debug("upstream.sse_control chars=%d", len(line))
                            continue
                        if line.startswith("data:"):
                            line = line[5:].strip()
                            if not line:
                                continue
                        lines += 1
                        bytes_seen += len(line.encode("utf-8"))
                        log.debug("upstream.line n=%d bytes=%d first_char=%s elapsed_ms=%d",
                                  lines, len(line.encode("utf-8")), ascii(line[:1]), ms_since(started))
                        if line == "[DONE]":
                            raise OllamaStreamFormatError(
                                'Ollama /api/chat прислал SSE [DONE] вместо финальной записи '
                                '{"done":true}. Возможно, адрес указывает на OpenAI-совместимый прокси.'
                            )
                        try:
                            item = json.loads(line)
                        except (json.JSONDecodeError, UnicodeError) as exc:
                            log.error("upstream.invalid_json line=%d chars=%d error=%s position=%s "
                                      "content_type=%s preview=%s",
                                      lines, len(line), type(exc).__name__, getattr(exc, "pos", "-"),
                                      content_type or "-", safe_preview(line))
                            raise explain_invalid_line(line, content_type, exc) from exc
                        if not isinstance(item, dict):
                            raise OllamaStreamFormatError(
                                "Ollama /api/chat должна возвращать JSON-объекты по строкам; "
                                f"получено {type(item).__name__}.")
                        if item.get("error"):
                            log.warning("upstream.ollama_error line=%d error_preview=%s",
                                        lines, safe_preview(str(item["error"])))
                            raise RuntimeError(f"Ollama: {item['error']}")
                        if "choices" in item:
                            raise OllamaStreamFormatError(
                                "Получен формат OpenAI Chat Completions, а ожидался "
                                "формат Ollama /api/chat. Проверьте прокси и его маршрутизацию.")
                        message = item.get("message") or {}
                        if not isinstance(message, dict):
                            raise OllamaStreamFormatError("Поле message в потоке Ollama не является объектом.")
                        thinking = message.get("thinking") or ""
                        content = message.get("content") or ""
                        log.debug("upstream.record n=%d keys=%s thinking_chars=%d content_chars=%d done=%s",
                                  lines, list(item.keys()), len(thinking), len(content), item.get("done"))
                        if thinking:
                            thinking_chunks += 1
                            log.debug("upstream.thinking chunk=%d chars=%d elapsed_ms=%d",
                                      thinking_chunks, len(thinking), ms_since(started))
                            yield {"type": "thinking", "text": thinking}
                        if content:
                            content_chunks += 1
                            content_chars += len(content)
                            log.debug("upstream.content chunk=%d chars=%d total_chars=%d elapsed_ms=%d",
                                      content_chunks, len(content), content_chars, ms_since(started))
                            yield {"type": "token", "text": content}
                        if item.get("done") is True:
                            done = True
                            usage = {key: item.get(key) for key in (
                                "model", "done_reason", "total_duration", "load_duration",
                                "prompt_eval_count", "prompt_eval_cached_count",
                                "prompt_eval_duration", "eval_count", "eval_duration",
                            ) if item.get(key) is not None}
                            log.info("upstream.done lines=%d bytes=%d content_chunks=%d thinking_chunks=%d "
                                     "chars=%d prompt_tokens=%s output_tokens=%s reason=%s elapsed_ms=%d",
                                     lines, bytes_seen, content_chunks, thinking_chunks,
                                     content_chars, usage.get("prompt_eval_count"), usage.get("eval_count"),
                                     usage.get("done_reason", "-"), ms_since(started))
                            yield {"type": "usage", "usage": usage}
                            break
                        if not ("message" in item or "done" in item):
                            raise OllamaStreamFormatError(
                                "Строка потока не содержит полей message/done формата Ollama. "
                                f"Content-Type={content_type or '(отсутствует)'}. "
                                f"Начало: {safe_preview(line)}"
                            )
            if not done:
                raise RuntimeError("Ollama закрыла поток без финального done")
        except Exception as exc:
            log.error("upstream.failed stage=%s exception=%s elapsed_ms=%d lines=%d bytes=%d "
                      "content_chunks=%d thinking_chunks=%d", stage, type(exc).__name__,
                      ms_since(started), lines, bytes_seen, content_chunks, thinking_chunks)
            raise
        finally:
            log.debug("upstream.cleanup stage=%s elapsed_ms=%d done=%s",
                      stage, ms_since(started), done)
