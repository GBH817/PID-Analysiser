"""PID Analysiser - FastAPI entry point.

Serves the REST API under /api and the static frontend at /.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import router

app = FastAPI(title="PID Analysiser", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


@app.get("/", include_in_schema=False)
def index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"detail": "frontend not built yet"}


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
