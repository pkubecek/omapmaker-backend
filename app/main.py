"""
main.py — FastAPI aplikace OMapMaker backend.

Spuštění:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.jobs import router as jobs_router
from .routes.download import router as download_router

app = FastAPI(
    title="OMapMaker API",
    description="Backend pro generování orientačních map z LiDAR dat.",
    version="7.0.0",
)

# Explicitní seznam povolených origins (z env proměnné nebo výchozí seznam)
_EXTRA_ORIGINS = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "").split(",")
    if o.strip()
]

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://omapmaker.vercel.app",
    "https://omapmaker-lro9dvn3t-josef-kubecek-s-projects.vercel.app",
] + _EXTRA_ORIGINS

app.add_middleware(
    CORSMiddleware,
    # allow_origins="*" — povolí všechny origins; bezpečnější je allow_origin_regex níže
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=(
        r"https://.*\.vercel\.app"          # všechny Vercel preview deploymenty
        r"|https://.*\.up\.railway\.app"    # Railway (pro případ server→server volání)
        r"|http://localhost(:\d+)?"         # lokální vývoj na libovolném portu
        r"|http://127\.0\.0\.1(:\d+)?"
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(jobs_router, prefix="/api")
app.include_router(download_router, prefix="/api")


@app.get("/")
async def root():
    return {"app": "OMapMaker API", "version": "7.0.0", "status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}