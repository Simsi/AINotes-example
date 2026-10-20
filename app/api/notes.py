import asyncio
from fastapi import APIRouter, BackgroundTasks, Response
from app.schemas import NoteIn
from app.services import notes as service
from app.services.search import related_chunks, search_notes, index_embeddings
from app.services.settings import load_settings

router = APIRouter(prefix="/api", tags=["Заметки"])


@router.get("/notes")
def list_notes():
    return service.list_notes()


@router.get("/search")
def search(q: str = ""):
    return search_notes(q)


@router.post("/notes", status_code=201)
def create_note(note: NoteIn, background_tasks: BackgroundTasks):
    result = service.create_note(note)
    if note.content:
        background_tasks.add_task(index_embeddings, result["id"])
    return result


@router.get("/notes/{note_id}")
def get_note(note_id: int):
    return service.get_note(note_id)


@router.put("/notes/{note_id}")
def update_note(note_id: int, note: NoteIn, background_tasks: BackgroundTasks):
    result, content_changed = service.update_note(note_id, note)
    if content_changed:
        background_tasks.add_task(index_embeddings, note_id)
    return result


@router.delete("/notes/{note_id}", status_code=204)
def delete_note(note_id: int):
    service.delete_note(note_id)
    return Response(status_code=204)


@router.get("/notes/{note_id}/related")
async def related_notes(note_id: int):
    note = await asyncio.to_thread(service.get_note, note_id)
    settings = await asyncio.to_thread(load_settings)
    # Prefer the current note's vocabulary, not a generic instruction.
    query = note["title"] + " " + note["content"][:2800]
    result = await related_chunks(query, note_id, settings)
    seen: set[int] = set()
    response = []
    for chunk in result:
        if chunk["note_id"] not in seen:
            seen.add(chunk["note_id"])
            response.append({"id": chunk["note_id"], "title": chunk["title"],
                             "snippet": chunk["text"][:180]})
    return response


@router.get("/notes/{note_id}/messages")
def messages(note_id: int):
    return service.list_messages(note_id)


@router.delete("/notes/{note_id}/messages", status_code=204)
def clear_messages(note_id: int):
    service.clear_chat(note_id)
    return Response(status_code=204)
