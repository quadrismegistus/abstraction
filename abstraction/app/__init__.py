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

from .db import init_db
from .routes import arc, meta, trajectory, passage, decompose


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build/check scores database on startup."""
    init_db()
    yield


app = FastAPI(title="Abstraction Explorer", lifespan=lifespan)

# Default CORS origins: the SvelteKit frontend dev server (default port 1784,
# see `abstraction app --frontend-port`) plus Vite's own dev/preview defaults
# (5173/4173). Override with ABSTRACTION_CORS_ORIGINS=origin1,origin2 when
# serving the frontend from elsewhere (e.g. another host on the LAN).
_DEFAULT_CORS_ORIGINS = [
    "http://localhost:1784",
    "http://127.0.0.1:1784",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]
_cors_env = os.environ.get("ABSTRACTION_CORS_ORIGINS", "")
_cors_origins = (
    [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env.strip()
    else _DEFAULT_CORS_ORIGINS
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(meta.router, prefix="/api/meta", tags=["meta"])
app.include_router(arc.router, prefix="/api/arc", tags=["arc"])
app.include_router(trajectory.router, prefix="/api/trajectory", tags=["trajectory"])
app.include_router(passage.router, prefix="/api/passage", tags=["passage"])
app.include_router(decompose.router, prefix="/api/decompose", tags=["decompose"])
