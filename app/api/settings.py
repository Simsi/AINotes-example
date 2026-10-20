import httpx
import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from app.schemas import Settings
from app.services.settings import load_settings, save_settings
from app.services.ollama import OllamaClient
from app.services.search import index_embeddings, index_status

router = APIRouter(prefix="/api", tags=["Настройки и Ollama"])
log = logging.getLogger("notes.settings")


@router.get("/settings", response_model=Settings)
def get_settings():
    return load_settings()


@router.put("/settings", response_model=Settings)
def put_settings(settings: Settings, background_tasks: BackgroundTasks):
    before = load_settings()
    save_settings(settings)
    log.info("settings.saved url=%s chat_model=%s embedding_model=%s think=%s "
             "search=%s timeout_s=%d temperature=%s num_ctx=%d num_predict=%d "
             "system_prompt_chars=%d",
             settings.ollama_url, settings.chat_model or "-", settings.embedding_model or "-",
             settings.think_mode, settings.search_mode, settings.read_timeout_seconds,
             settings.temperature, settings.num_ctx, settings.num_predict, len(settings.system_prompt))
    if settings.embedding_model and (before.embedding_model != settings.embedding_model or
                                     before.ollama_url != settings.ollama_url):
        log.info("index.scheduled reason=settings_changed model=%s", settings.embedding_model)
        background_tasks.add_task(index_embeddings)
    return settings


@router.get("/ollama/models")
async def get_ollama_models():
    settings = load_settings()
    try:
        models = await OllamaClient(settings).models()
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        raise HTTPException(502, f"Не удалось получить список моделей Ollama: {exc}") from exc
    return {"models": models}


@router.get("/ollama/thinking")
async def get_model_thinking(model: str = Query(min_length=1, max_length=160)):
    settings = load_settings()
    try:
        return await OllamaClient(settings).thinking_options(model)
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        # Metadata is optional: never disable normal chat because /api/show
        # is unavailable or times out on a particular Ollama installation.
        return {"values": None, "default": None,
                "warning": f"Информация о thinking недоступна: {type(exc).__name__}"}


@router.get("/ollama/diagnostics")
async def ollama_diagnostics():
    settings = load_settings()
    result = await OllamaClient(settings).diagnostics()
    result["selected_model"] = settings.chat_model
    result["model_loaded"] = settings.chat_model in result["loaded_models"] if settings.chat_model else False
    return result


@router.get("/index/status")
def get_index_status():
    return index_status()


@router.post("/index/rebuild", status_code=202)
def rebuild_index(background_tasks: BackgroundTasks):
    settings = load_settings()
    if not settings.embedding_model:
        raise HTTPException(400, "Сначала выбери embedding-модель")
    # Existing matching vectors are reused; changed models are reindexed.
    log.info("index.scheduled reason=user_rebuild model=%s", settings.embedding_model)
    background_tasks.add_task(index_embeddings)
    return {"status": "scheduled"}
