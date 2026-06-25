from __future__ import annotations
from dataclasses import dataclass
import os

def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v.strip() if v else default

def _env_int(name: str, default: int) -> int:
    v = _env(name, "")
    try:
        return int(v) if v else default
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    v = _env(name, "")
    try:
        return float(v) if v else default
    except Exception:
        return default

def _env_bool(name: str, default: bool) -> bool:
    v = _env(name, "").lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default

@dataclass(frozen=True)
class Settings:
    chroma_path: str = _env("CHROMA_PATH","chroma_db")
    collection_medical: str = _env("COLLECTION_MEDICAL","connaissances_medicales")
    embed_model: str = _env("EMBED_MODEL","all-MiniLM-L6-v2")
    jsonl_path: str = _env("JSONL_PATH","data/base_connaissances_medicales.jsonl")

settings = Settings()