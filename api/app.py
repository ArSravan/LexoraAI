"""The FastAPI application."""
from __future__ import annotations

import warnings

from fastapi import FastAPI

from api.routes import router

warnings.filterwarnings("ignore", category=FutureWarning)


def create_app() -> FastAPI:
    app = FastAPI(title="Lexora AI API", version="1.0.0",
                  description="Evidence-grounded legal document intelligence. Local-first: no external APIs.")
    app.include_router(router, prefix="/api")
    return app


app = create_app()
