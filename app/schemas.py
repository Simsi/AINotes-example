import re
from typing import Literal
from pydantic import BaseModel, Field, field_validator


class NoteIn(BaseModel):
    title: str = Field(default="Без названия", max_length=200)
    content: str = Field(default="", max_length=200_000)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        return value.strip() or "Без названия"


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    ignore_history: bool = False


class RewriteIn(BaseModel):
    instruction: str = Field(default="Улучши стиль, ясность и структуру текста, сохранив смысл.", min_length=3, max_length=1000)


class Settings(BaseModel):
    ollama_url: str = "http://127.0.0.1:11434"
    chat_model: str = Field(default="", max_length=160)
    embedding_model: str = Field(default="", max_length=160)
    temperature: float = Field(default=0.5, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    num_ctx: int = Field(default=4096, ge=512, le=131072)
    num_predict: int = Field(default=1024, ge=64, le=16384)
    think_mode: str = Field(default="off", min_length=2, max_length=40)
    read_timeout_seconds: int = Field(default=300, ge=15, le=1800)
    system_prompt: str = Field(
        default=("Ты полезный ассистент для работы с личными заметками. "
                 "Отвечай по-русски, если пользователь не просит иначе. "
                 "Опирайся на предоставленные заметки. Если сведений недостаточно, скажи об этом. "
                 "Не выдумывай факты и источники. Если используешь другую заметку, "
                 "укажи её ссылку в формате [note:ID]."),
        max_length=10_000,
    )
    search_mode: Literal["off", "fts", "hybrid"] = "fts"
    context_chunks: int = Field(default=4, ge=0, le=8)
    history_messages: int = Field(default=12, ge=0, le=40)

    @field_validator("think_mode")
    @classmethod
    def validate_think_mode(cls, value: str) -> str:
        if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", value):
            raise ValueError("Некорректный режим thinking")
        return value

    @field_validator("ollama_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        from urllib.parse import urlsplit
        value = value.strip().rstrip("/")
        parts = urlsplit(value)
        if (parts.scheme not in {"http", "https"} or not parts.hostname
                or parts.username or parts.password or parts.query or parts.fragment
                or parts.path not in {"", "/"}):
            raise ValueError("Укажи базовый HTTP(S) адрес Ollama: http://host:11434")
        return value
