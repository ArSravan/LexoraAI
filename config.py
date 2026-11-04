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