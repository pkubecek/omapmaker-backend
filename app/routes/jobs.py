"""
routes/jobs.py — FastAPI endpointy pro správu jobů.
Joby se ukládají na disk (JSON) aby přežily restart kontejneru.
"""
import os
import uuid
import json
import threading
import shutil

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from ..core.pipeline import run_pipeline

router = APIRouter()

JOBS_DIR = os.environ.get("OMAPMAKER_JOBS_DIR", "./jobs")
os.makedirs(JOBS_DIR, exist_ok=True)


def _job_path(job_id: str) -> str:
    return os.path.join(JOBS_DIR, job_id, "job.json")


def _read_job(job_id: str) -> dict | None:
    path = _job_path(job_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _write_job(job_id: str, data: dict):
    path = _job_path(job_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _save_file(upload: UploadFile, dest_dir: str) -> str:
    path = os.path.join(dest_dir, upload.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path


@router.post("/jobs")
async def create_job(
    dtm: UploadFile = File(...),
    dsm: UploadFile = File(...),
    zabaged: list[UploadFile] = File(default=[]),
    zabaged_sidecar: list[UploadFile] = File(default=[]),
    isom: list[UploadFile] = File(default=[]),
    isom_sidecar: list[UploadFile] = File(default=[]),
    params: str = Form(...),
):
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    dtm_path = _save_file(dtm, job_dir)
    dsm_path = _save_file(dsm, job_dir)

    # Sidecar soubory (.dbf, .shx, .prj) ulož do stejné složky jako .shp
    # aby je geopandas/fiona při read_file() automaticky našlo podle názvu
    for f in zabaged_sidecar:
        if f.filename:
            _save_file(f, job_dir)
    for f in isom_sidecar:
        if f.filename:
            _save_file(f, job_dir)

    # .shp soubory — ty se předají do pipeline jako cesty
    zabaged_paths = [_save_file(f, job_dir) for f in zabaged if f.filename]
    isom_paths = [_save_file(f, job_dir) for f in isom if f.filename]

    try:
        params_dict = json.loads(params)
    except Exception:
        params_dict = {}

    _write_job(job_id, {
        "status": "queued",
        "progress": 0,
        "step": "Ve frontě...",
        "error": None,
        "png_path": None,
        "gpkg_path": None,
    })

    def _progress_cb(pct: int, msg: str):
        job = _read_job(job_id) or {}
        job["status"] = "running"
        job["progress"] = pct
        job["step"] = msg
        _write_job(job_id, job)

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
            _write_job(job_id, {
                "status": "done",
                "progress": 100,
                "step": "Hotovo!",
                "error": None,
                "png_path": result.get("png_path"),
                "gpkg_path": result.get("gpkg_path"),
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            _write_job(job_id, {
                "status": "error",
                "progress": 0,
                "step": f"Chyba: {e}",
                "error": str(e),
                "png_path": None,
                "gpkg_path": None,
            })

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = _read_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job nenalezen.")
    return {"job_id": job_id, **job}


@router.get("/jobs/{job_id}/png")
async def get_png(job_id: str):
    job = _read_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job nenalezen.")
    if job["status"] != "done":
        raise HTTPException(status_code=425, detail="Job ještě není hotový.")
    png_path = job.get("png_path")
    if not png_path or not os.path.exists(png_path):
        raise HTTPException(status_code=404, detail="PNG nenalezeno.")
    return FileResponse(png_path, media_type="image/png", filename=f"OMap_{job_id}.png")


@router.get("/jobs/{job_id}/gpkg")
async def get_gpkg(job_id: str):
    job = _read_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job nenalezen.")
    if job["status"] != "done":
        raise HTTPException(status_code=425, detail="Job ještě není hotový.")
    gpkg_path = job.get("gpkg_path")
    if not gpkg_path or not os.path.exists(gpkg_path):
        raise HTTPException(status_code=404, detail="GPKG nenalezeno.")
    return FileResponse(gpkg_path, media_type="application/geopackage+sqlite3",
                        filename=f"OOM_{job_id}.gpkg")


@router.get("/crt/{filename}")
async def get_crt(filename: str):
    """Vrátí .crt soubor pro import do OpenOrienteering Mapperu."""
    for search_dir in [
        ".",
        os.path.join(os.path.dirname(__file__), "..", ".."),
        os.path.join(os.path.dirname(__file__), ".."),
    ]:
        path = os.path.join(search_dir, filename)
        if os.path.exists(path) and filename.endswith(".crt"):
            return FileResponse(path, media_type="application/octet-stream",
                                filename=filename)
    raise HTTPException(status_code=404, detail=f"Soubor {filename} nenalezen.")