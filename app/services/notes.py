"""Note CRUD and atomic chunk refresh."""
import sqlite3
import logging
from fastapi import HTTPException
from app.database import db, utc_now
from app.schemas import NoteIn


def chunk_text(text: str, size: int = 900, overlap: int = 120) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        stop = min(len(text), start + size)
        if stop < len(text):
            sep = max(text.rfind("\n", start + size // 2, stop), text.rfind(" ", start + size // 2, stop))
            if sep > start + size // 2:
                stop = sep + 1
        piece = text[start:stop].strip()
        if piece:
            chunks.append(piece)
        if stop == len(text):
            break
        start = max(start + 1, stop - overlap)
    return chunks


def _asdict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


def _rechunk(conn: sqlite3.Connection, note_id: int, content: str) -> None:
    conn.execute("DELETE FROM note_chunks WHERE note_id = ?", (note_id,))
    conn.executemany(
        "INSERT INTO note_chunks(note_id, chunk_index, text) VALUES (?, ?, ?)",
        ((note_id, i, part) for i, part in enumerate(chunk_text(content))),
    )


def list_notes() -> list[dict]:
    with db() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT id, title, updated_at FROM notes ORDER BY updated_at DESC, id DESC"
        ).fetchall()]


def get_note(note_id: int) -> dict:
    with db() as conn:
        row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Заметка не найдена")
    return dict(row)


def create_note(note: NoteIn) -> dict:
    now = utc_now()
    with db() as conn:
        cursor = conn.execute(
            "INSERT INTO notes(title, content, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (note.title, note.content, now, now),
        )
        note_id = cursor.lastrowid
        _rechunk(conn, note_id, note.content)
    return get_note(note_id)


def update_note(note_id: int, note: NoteIn) -> tuple[dict, bool]:
    with db() as conn:
        current = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        if current is None:
            raise HTTPException(404, "Заметка не найдена")
        changed_content = note.content != current["content"]
        if changed_content or note.title != current["title"]:
            conn.execute("UPDATE notes SET title=?, content=?, updated_at=? WHERE id=?",
                         (note.title, note.content, utc_now(), note_id))
            if changed_content:
                _rechunk(conn, note_id, note.content)
    return get_note(note_id), changed_content


def delete_note(note_id: int) -> None:
    with db() as conn:
        changed = conn.execute("DELETE FROM notes WHERE id = ?", (note_id,)).rowcount
    if not changed:
        raise HTTPException(404, "Заметка не найдена")


def list_messages(note_id: int) -> list[dict]:
    import json
    get_note(note_id)
    with db() as conn:
        rows = conn.execute(
            "SELECT id, role, content, sources_json, created_at FROM chat_messages "
            "WHERE note_id=? ORDER BY id", (note_id,)
        ).fetchall()
    return [{**dict(row), "sources": json.loads(row["sources_json"])} for row in rows]


def add_message(note_id: int, role: str, content: str, sources: list | None = None) -> None:
    import json
    with db() as conn:
        conn.execute(
            "INSERT INTO chat_messages(note_id, role, content, sources_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (note_id, role, content, json.dumps(sources or [], ensure_ascii=False), utc_now()),
        )


def add_exchange(note_id: int, question: str, answer: str, sources: list) -> None:
    """Persist both turns atomically; failures never leave a half-dialogue."""
    import json
    now = utc_now()
    with db() as conn:
        conn.executemany(
            "INSERT INTO chat_messages(note_id, role, content, sources_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [(note_id, "user", question, "[]", now),
             (note_id, "assistant", answer, json.dumps(sources, ensure_ascii=False), now)],
        )


def recent_messages(note_id: int, limit: int) -> list[dict]:
    """Return only complete user/assistant exchanges in chronological order.

    Older versions could persist a lone user message after a failed request.
    Passing it on produced adjacent user turns and contaminated the conversation.
    The configured limit is the maximum *number of messages*, so an odd
    limit rounds down to the nearest complete exchange.
    """
    pair_limit = limit // 2
    if pair_limit <= 0:
        return []
    with db() as conn:
        # Fetch beyond the requested limit in case old records have orphans.
        rows = conn.execute(
            "SELECT role, content FROM chat_messages WHERE note_id=? "
            "ORDER BY id DESC LIMIT ?", (note_id, min(200, pair_limit * 4 + 20))
        ).fetchall()

    exchanges = []
    latest_assistant = None
    for row in rows:  # most recent first
        if row["role"] == "assistant":
            latest_assistant = {"role": "assistant", "content": row["content"]}
        elif row["role"] == "user" and latest_assistant is not None:
            exchanges.append(({"role": "user", "content": row["content"]}, latest_assistant))
            latest_assistant = None
            if len(exchanges) >= pair_limit:
                break
    result = [message for pair in reversed(exchanges) for message in pair]
    if len(rows) != len(result):
        logging.getLogger("notes.notes").debug(
            "history.filtered note=%d fetched=%d valid_messages=%d skipped_or_truncated=%d",
            note_id, len(rows), len(result), len(rows) - len(result),
        )
    return result


def clear_chat(note_id: int) -> None:
    get_note(note_id)
    with db() as conn:
        conn.execute("DELETE FROM chat_messages WHERE note_id=?", (note_id,))
