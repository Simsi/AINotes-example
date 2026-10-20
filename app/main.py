from contextlib import asynccontextmanager
from pathlib import Path
import logging
import time
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.api import notes, chat, settings
from app.database import init_db
from app.diagnostics import configure_logging, new_request_id, trace_context

STATIC = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    init_db()
    yield


app = FastAPI(title="Notes + Ollama", version="0.1.0", lifespan=lifespan)
log = logging.getLogger("notes.http")


@app.middleware("http")
async def correlate_requests(request: Request, call_next):
    # Generate on the server: do not trust arbitrary user-supplied log identifiers.
    identifier = new_request_id()
    request.state.request_id = identifier
    started = time.perf_counter()
    path = request.url.path
    important = (path.startswith("/api/ollama/") or
                 path.endswith("/chat/stream") or
                 path.endswith("/rewrite/stream") or
                 path in ("/api/settings", "/api/index/rebuild"))
    with trace_context(identifier):
        if important:
            log.info("http.begin method=%s path=%s", request.method, path)
        try:
            response = await call_next(request)
        except Exception:
            log.exception("http.exception method=%s path=%s elapsed_ms=%d",
                          request.method, path, int((time.perf_counter() - started) * 1000))
            raise
        response.headers["X-Request-ID"] = identifier
        if important or response.status_code >= 400:
            log.info("http.headers method=%s path=%s status=%d elapsed_ms=%d%s",
                     request.method, path, response.status_code,
                     int((time.perf_counter() - started) * 1000),
                     " (SSE: stream continues)" if path.endswith("/stream") else "")
        else:
            log.debug("http.headers method=%s path=%s status=%d elapsed_ms=%d",
                      request.method, path, response.status_code,
                      int((time.perf_counter() - started) * 1000))
        return response


app.include_router(notes.router)
app.include_router(chat.router)
app.include_router(settings.router)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def homepage():
    return FileResponse(STATIC / "index.html")


@app.get("/health", tags=["Система"])
def health():
    return {"status": "ok"}
