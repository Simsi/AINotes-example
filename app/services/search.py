"""FTS5 + optional cosine ranking of Ollama embeddings (small local datasets)."""
import asyncio
import json
import logging
import math
import re
import time
from app.database import db
from app.schemas import Settings
from app.services.ollama import OllamaClient
from app.services.settings import load_settings

log = logging.getLogger("notes.search")
_index_lock = asyncio.Lock()
_indexing = False


def fts_query(query: str) -> str:
    # Quotes prevent user input from being interpreted as FTS5 syntax.
    tokens = re.findall(r"[^\W_]{2,}", query.lower(), re.UNICODE)[:12]
    return " OR ".join('"' + token.replace('"', '') + '"' for token in tokens)


def _lexical(query: str, exclude_note_id: int | None = None, limit: int = 30) -> list[dict]:
    expression = fts_query(query)
    if not expression:
        return []
    with db() as conn:
        rows = conn.execute("""
            SELECT c.id, c.note_id, c.text, n.title
              FROM chunks_fts f
              JOIN note_chunks c ON c.id = f.rowid
              JOIN notes n ON n.id = c.note_id
             WHERE chunks_fts MATCH ? AND (? IS NULL OR c.note_id != ?)
             ORDER BY bm25(chunks_fts)
             LIMIT ?
        """, (expression, exclude_note_id, exclude_note_id, limit)).fetchall()
    return [dict(r) for r in rows]


def search_notes(query: str) -> list[dict]:
    if not query.strip():
        return []
    hits = _lexical(query, limit=70)
    by_note: dict[int, dict] = {}
    for hit in hits:
        if hit["note_id"] not in by_note:
            by_note[hit["note_id"]] = {
                "id": hit["note_id"], "title": hit["title"],
                "snippet": hit["text"][:180],
            }
    with db() as conn:
        # Title matching works even when notes have no body.
        titles = conn.execute("SELECT id, title, content FROM notes WHERE title LIKE ? LIMIT 50",
                              (f"%{query.strip()}%",)).fetchall()
    for note in titles:
        by_note.setdefault(note["id"], {"id": note["id"], "title": note["title"],
                                        "snippet": note["content"][:180]})
    return list(by_note.values())[:50]


def _stored_vectors(model: str, excluded: int | None) -> list[dict]:
    with db() as conn:
        rows = conn.execute("""
            SELECT c.id, c.note_id, c.text, c.embedding, n.title
            FROM note_chunks c JOIN notes n ON n.id=c.note_id
            WHERE c.embedding_model=? AND c.embedding IS NOT NULL
              AND (? IS NULL OR c.note_id != ?)
        """, (model, excluded, excluded)).fetchall()
    return [dict(r) for r in rows]


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return sum(x*y for x, y in zip(a, b)) / (norm_a * norm_b)


async def related_chunks(query: str, excluded_note_id: int, settings: Settings) -> list[dict]:
    started = time.perf_counter()
    log.info("search.start mode=%s exclude_note=%d query_chars=%d max_chunks=%d embedding_model=%s",
             settings.search_mode, excluded_note_id, len(query), settings.context_chunks,
             settings.embedding_model or "-")
    if settings.search_mode == "off" or settings.context_chunks == 0:
        log.info("search.skipped reason=disabled")
        return []
    lexical = await asyncio.to_thread(_lexical, query, excluded_note_id, 30)
    log.info("search.fts.completed candidates=%d elapsed_ms=%d",
             len(lexical), round((time.perf_counter() - started) * 1000))
    rankings: list[list[dict]] = [lexical]
    if settings.search_mode == "hybrid" and settings.embedding_model:
        try:
            vectors = await asyncio.to_thread(_stored_vectors, settings.embedding_model, excluded_note_id)
            log.info("search.hybrid.stored_vectors count=%d model=%s", len(vectors), settings.embedding_model)
            if vectors:
                embedding = await OllamaClient(settings).embed(query[:3500])
                semantic = sorted(vectors, key=lambda r: cosine(embedding, json.loads(r["embedding"])), reverse=True)
                rankings.append(semantic[:30])
                log.info("search.hybrid.completed semantic_candidates=%d elapsed_ms=%d",
                         min(len(semantic), 30), round((time.perf_counter() - started) * 1000))
        except Exception as exc:
            log.warning("search.hybrid.fallback reason=%s elapsed_ms=%d",
                        type(exc).__name__, round((time.perf_counter() - started) * 1000))
    combined: dict[int, dict] = {}
    for ranked in rankings:
        for i, chunk in enumerate(ranked, start=1):
            key = chunk["id"]
            if key not in combined:
                combined[key] = {"id": key, "note_id": chunk["note_id"],
                                 "title": chunk["title"], "text": chunk["text"], "score": 0.0}
            combined[key]["score"] += 1 / (60 + i)
    result: list[dict] = []
    per_note: dict[int, int] = {}
    for chunk in sorted(combined.values(), key=lambda r: r["score"], reverse=True):
        if per_note.get(chunk["note_id"], 0) >= 2:
            continue
        per_note[chunk["note_id"]] = per_note.get(chunk["note_id"], 0) + 1
        result.append(chunk)
        if len(result) >= settings.context_chunks:
            break
    log.info("search.complete chunks=%d notes=%d elapsed_ms=%d", len(result),
             len({c["note_id"] for c in result}), round((time.perf_counter() - started) * 1000))
    return result


def _pending(note_id: int | None, model: str) -> list[dict]:
    with db() as conn:
        rows = conn.execute("""
            SELECT id, text FROM note_chunks
             WHERE (embedding IS NULL OR embedding_model IS NULL OR embedding_model != ?)
               AND (? IS NULL OR note_id = ?)
             ORDER BY id
        """, (model, note_id, note_id)).fetchall()
    return [dict(r) for r in rows]


def _set_vector(chunk_id: int, chunk_text: str, model: str, vector: list[float]):
    with db() as conn:
        # A modified/deleted chunk cannot receive a stale embedding.
        conn.execute("UPDATE note_chunks SET embedding=?, embedding_model=? WHERE id=? AND text=?",
                     (json.dumps(vector), model, chunk_id, chunk_text))


async def index_embeddings(note_id: int | None = None) -> None:
    global _indexing
    async with _index_lock:
        started = time.perf_counter()
        settings = await asyncio.to_thread(load_settings)
        if not settings.embedding_model:
            log.debug("index.skipped reason=no_embedding_model")
            return
        log.info("index.begin note=%s model=%s", note_id or "all", settings.embedding_model)
        _indexing = True
        try:
            chunks = await asyncio.to_thread(_pending, note_id, settings.embedding_model)
            client = OllamaClient(settings)
            log.info("index.pending count=%d", len(chunks))
            completed = 0
            for chunk in chunks:
                try:
                    vector = await client.embed(chunk["text"])
                    await asyncio.to_thread(_set_vector, chunk["id"], chunk["text"],
                                            settings.embedding_model, vector)
                    completed += 1
                    log.debug("index.chunk.done id=%d chars=%d dimensions=%d progress=%d/%d",
                              chunk["id"], len(chunk["text"]), len(vector), completed, len(chunks))
                except Exception as exc:
                    log.warning("index.paused chunk=%d exception=%s completed=%d/%d",
                                chunk["id"], type(exc).__name__, completed, len(chunks))
                    break
            log.info("index.end completed=%d pending_before=%d elapsed_ms=%d",
                     completed, len(chunks), round((time.perf_counter() - started) * 1000))
        finally:
            _indexing = False


def index_status() -> dict:
    settings = load_settings()
    with db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM note_chunks").fetchone()[0]
        ready = conn.execute(
            "SELECT COUNT(*) FROM note_chunks WHERE embedding IS NOT NULL AND embedding_model=?",
            (settings.embedding_model,),
        ).fetchone()[0] if settings.embedding_model else 0
    return {"total": total, "ready": ready, "pending": total - ready,
            "indexing": _indexing, "embedding_model": settings.embedding_model}
