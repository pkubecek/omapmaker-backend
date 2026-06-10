"""
main.py — FastAPI aplikace OMapMaker backend.

Spuštění:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.jobs import router as jobs_router
from .routes.download import router as download_router

app = FastAPI(
    title="OMapMaker API",
    description="Backend pro generování orientačních map z LiDAR dat.",
    version="7.0.0",
)

# CORS — umožní frontendu na localhost:3000 (nebo jiné doméně) volat API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        # Pro produkci přidej svou doménu, např. "https://omapmaker.example.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs_router, prefix="/api")
app.include_router(download_router, prefix="/api")


@app.get("/")
async def root():
    return {"app": "OMapMaker API", "version": "7.0.0", "status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}
