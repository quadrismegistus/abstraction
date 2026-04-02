"""
Abstraction Explorer — FastAPI backend for interactive data exploration.
"""

import os
import sys

# Monkeypatch LLTK from local checkout before any imports that might use it
_LLTK_PATH = os.path.expanduser("~/github/lltk")
if _LLTK_PATH not in sys.path:
    sys.path.insert(0, _LLTK_PATH)

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db import init_db, get_db_path
from .routes import arc, meta, trajectory, passage


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build/check SQLite database on startup."""
    db_path = get_db_path()
    init_db(db_path)
    yield


app = FastAPI(title="Abstraction Explorer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(meta.router, prefix="/api/meta", tags=["meta"])
app.include_router(arc.router, prefix="/api/arc", tags=["arc"])
app.include_router(trajectory.router, prefix="/api/trajectory", tags=["trajectory"])
app.include_router(passage.router, prefix="/api/passage", tags=["passage"])
