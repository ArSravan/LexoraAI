"""API entry point.   python main.py     (or: uvicorn main:app)"""
from __future__ import annotations

import os

from api.app import app  # noqa: F401  (uvicorn imports `main:app`)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=os.getenv("API_HOST", "127.0.0.1"), port=int(os.getenv("API_PORT", "8000")),
                reload=False)
