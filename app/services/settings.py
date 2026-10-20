import json
from app.database import db
from app.schemas import Settings


def load_settings() -> Settings:
    with db() as conn:
        row = conn.execute("SELECT json FROM app_settings WHERE id=1").fetchone()
    return Settings.model_validate_json(row["json"]) if row else Settings()


def save_settings(settings: Settings) -> Settings:
    with db() as conn:
        conn.execute("INSERT INTO app_settings(id, json) VALUES (1, ?) "
                     "ON CONFLICT(id) DO UPDATE SET json=excluded.json",
                     (settings.model_dump_json(),))
    return settings
