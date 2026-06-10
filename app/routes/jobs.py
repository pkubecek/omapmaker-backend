"""
routes/jobs.py — FastAPI endpointy pro správu jobů.
POST /api/jobs         — spustit job
GET  /api/jobs/{id}    — stav jobu
GET  /api/jobs/{id}/png  — stáhnout PNG
GET  /api/jobs/{id}/gpkg — stáhnout GPKG
"""
import os
import uuid
import json
import threading
import shutil
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse

from ..core.pipeline import run_pipeline

router = APIRouter()

# In-memory job store  { job_id: { status, progress, step, error, png_path, gpkg_path } }
JOBS: dict = {}

# Výstupní složka pro joby
JOBS_DIR = os.environ.get("OMAPMAKER_JOBS_DIR", "./jobs")
os.makedirs(JOBS_DIR, exist_ok=True)


def _save_file(upload: UploadFile, dest_dir: str) -> str:
    path = os.path.join(dest_dir, upload.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path


@router.post("/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    dtm: UploadFile = File(...),
    dsm: UploadFile = File(...),
    zabaged: list[UploadFile] = File(default=[]),
    isom: list[UploadFile] = File(default=[]),
    params: str = Form(...),
):
    """Přijme soubory + parametry, spustí analýzu na pozadí, vrátí job_id."""
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Ulož nahrané soubory
    dtm_path = _save_file(dtm, job_dir)
    dsm_path = _save_file(dsm, job_dir)
    zabaged_paths = [_save_file(f, job_dir) for f in zabaged if f.filename]
    isom_paths = [_save_file(f, job_dir) for f in isom if f.filename]

    try:
        params_dict = json.loads(params)
    except Exception:
        params_dict = {}

    JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "step": "Ve frontě...",
        "error": None,
        "png_path": None,
        "gpkg_path": None,
    }

    def _progress_cb(pct: int, msg: str):
        JOBS[job_id]["progress"] = pct
        JOBS[job_id]["step"] = msg
        JOBS[job_id]["status"] = "running"

    def _run():
        try:
            result = run_pipeline(
                job_id=job_id,
                params=params_dict,
                file_paths={
                    "dtm": dtm_path,
                    "dsm": dsm_path,
                    "zabaged": zabaged_paths,
                    "isom": isom_paths,
                },
                output_dir=job_dir,
                progress_cb=_progress_cb,
            )
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["progress"] = 100
            JOBS[job_id]["step"] = "Hotovo!"
            JOBS[job_id]["png_path"] = result.get("png_path")
            JOBS[job_id]["gpkg_path"] = result.get("gpkg_path")
        except Exception as e:
            import traceback
            traceback.print_exc()
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(e)
            JOBS[job_id]["step"] = f"Chyba: {e}"

    # Spusť v background threadu (pro produkci použij Celery nebo ProcessPoolExecutor)
    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return {"job_id": job_id}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Vrátí aktuální stav jobu."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job nenalezen.")
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "step": job["step"],
        "error": job.get("error"),
    }


@router.get("/jobs/{job_id}/png")
async def get_png(job_id: str):
    """Vrátí vygenerovanou PNG mapu."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job nenalezen.")
    if job["status"] != "done":
        raise HTTPException(status_code=425, detail="Job ještě není hotový.")
    png_path = job.get("png_path")
    if not png_path or not os.path.exists(png_path):
        raise HTTPException(status_code=404, detail="PNG soubor nenalezen.")
    return FileResponse(
        png_path,
        media_type="image/png",
        filename=f"OMap_{job_id}.png",
    )


@router.get("/jobs/{job_id}/gpkg")
async def get_gpkg(job_id: str):
    """Vrátí GPKG export pro OpenOrienteering Mapper."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job nenalezen.")
    if job["status"] != "done":
        raise HTTPException(status_code=425, detail="Job ještě není hotový.")
    gpkg_path = job.get("gpkg_path")
    if not gpkg_path or not os.path.exists(gpkg_path):
        raise HTTPException(status_code=404, detail="GPKG soubor nenalezen.")
    return FileResponse(
        gpkg_path,
        media_type="application/geopackage+sqlite3",
        filename=f"OOM_{job_id}.gpkg",
    )
