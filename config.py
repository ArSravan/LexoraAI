"""Application settings.

Resolution order (later wins): built-in defaults < environment variables / .env.
Every environment variable is the upper-cased field name with a ``LEXORA_`` prefix,
e.g. ``max_upload_mb`` -> ``LEXORA_MAX_UPLOAD_MB``. No secrets live in the code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Mapping, get_type_hints

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
ENV_PREFIX = "LEXORA_"

load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    storage_dir: Path = ROOT / "storage"
    max_upload_mb: int = 50
    scanned_min_chars: int = 30   # a page with fewer extracted characters counts as "no text"
    seed: int = 42
    ocr_enabled: bool = True
    ocr_lang: str = "eng"
    ocr_dpi: int = 200
    tesseract_cmd: str = ""       # empty = Windows default path, then PATH
    tables_enabled: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:9b"
    ollama_timeout: float = 600.0       # seconds to wait for the model (a 9B model on CPU is slow)
    ollama_temperature: float = 0.1
    ollama_num_ctx: int = 8192          # context window: prompt + answer must fit
    ollama_num_predict: int = 1024      # maximum new tokens
    ollama_think: str = "off"           # off | on | auto (auto = leave to the model's default)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_enabled: bool = True
    chunk_max_chars: int = 1000
    chunk_overlap: int = 150
    retrieval_mode: str = "hybrid"      # bm25 | dense | hybrid
    initial_top_k: int = 20             # candidates fetched per retriever / kept after fusion
    final_top_k: int = 5                # passages finally handed to the LLM
    rrf_k: int = 60                     # Reciprocal Rank Fusion constant
    dense_min_similarity: float = 0.0   # dense hits below this cosine similarity are dropped
    query_decomposition: bool = True    # split compound (multi-hop) questions into sub-queries


def _cast(raw: str, typ: type):
    if typ is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return typ(raw)


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build Settings from defaults plus ``LEXORA_*`` variables found in ``env``."""
    env = os.environ if env is None else env
    hints = get_type_hints(Settings)
    values = {}
    for f in fields(Settings):
        raw = env.get(ENV_PREFIX + f.name.upper())
        if raw is None or raw == "":
            continue
        values[f.name] = _cast(raw, hints[f.name])
    return Settings(**values)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings (read once). Tests call ``get_settings.cache_clear()``."""
    return load_settings()
