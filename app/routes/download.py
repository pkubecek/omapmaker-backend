"""
routes/download.py — stahování dat z ČÚZK, GUGiK (Polsko) a BEV (Rakousko) jako background job.
"""
import os
import uuid
import json
import threading
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..core.downloader import download_cuzk as _download_cuzk
from ..core.poland_downloader import download_poland as _download_poland
from ..core.austria_downloader import download_austria as _download_austria

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_status_helpers(base_dir: str):
    os.makedirs(base_dir, exist_ok=True)

    def status_path(dl_id):
        return os.path.join(base_dir, dl_id, "status.json")

    def read_status(dl_id):
        p = status_path(dl_id)
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f)

    def write_status(dl_id, data):
        p = status_path(dl_id)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(data, f)

    return read_status, write_status


JOBS_BASE = os.environ.get("OMAPMAKER_JOBS_DIR", "./jobs")

_cuzk_read, _cuzk_write = _make_status_helpers(JOBS_BASE + "/cuzk")
_pl_read,   _pl_write   = _make_status_helpers(JOBS_BASE + "/poland")
_at_read,   _at_write   = _make_status_helpers(JOBS_BASE + "/austria")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class BboxModel(BaseModel):
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float

class CuzkRequest(BaseModel):
    bbox: BboxModel
    dsm_type: str = "DMPOK"

class PolandRequest(BaseModel):
    bbox: BboxModel
    use_lidar_point_cloud: bool = True

class AustriaRequest(BaseModel):
    bbox: BboxModel


# ---------------------------------------------------------------------------
# Generic background runner
# ---------------------------------------------------------------------------

def _run_download(dl_id, download_fn, kwargs, read_fn, write_fn, extra_status_fields=None):
    """Spustí download funkci v threadu, zapisuje stav."""
    def cb(msg):
        s = read_fn(dl_id) or {}
        s["step"] = msg
        s["progress"] = min(s.get("progress", 0) + 5, 90)
        write_fn(dl_id, s)

    def _run():
        try:
            result = download_fn(progress_cb=cb, **kwargs)
            status = {
                "status": "done",
                "progress": 100,
                "step": "Hotovo!",
                "dmr_path": result.get("dmr_path"),
                "dmp_path": result.get("dmp_path"),
                "crs": result.get("crs"),
                "error": None,
            }
            if extra_status_fields:
                status.update(extra_status_fields)
            write_fn(dl_id, status)
        except Exception as e:
            import traceback; traceback.print_exc()
            write_fn(dl_id, {
                "status": "error",
                "progress": 0,
                "step": f"Chyba: {e}",
                "dmr_path": None,
                "dmp_path": None,
                "crs": None,
                "error": str(e),
            })

    threading.Thread(target=_run, daemon=True).start()


def _file_response(read_fn, dl_id, field, label):
    s = read_fn(dl_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Download nenalezen.")
    if s["status"] != "done":
        raise HTTPException(status_code=425, detail="Stahování ještě neskončilo.")
    path = s.get(field)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{label} soubor nenalezen.")
    return FileResponse(path, media_type="application/octet-stream",
                        filename=os.path.basename(path))


# ---------------------------------------------------------------------------
# ČÚZK (Česká republika)
# ---------------------------------------------------------------------------

@router.post("/download/cuzk")
async def start_cuzk_download(req: CuzkRequest):
    dl_id = str(uuid.uuid4())[:8]
    out_dir = os.path.join(JOBS_BASE, "cuzk", dl_id)
    os.makedirs(out_dir, exist_ok=True)
    _cuzk_write(dl_id, {"status": "running", "progress": 0, "step": "Spouštím stahování...",
                         "dmr_path": None, "dmp_path": None, "crs": "EPSG:5514", "error": None})
    _run_download(dl_id, _download_cuzk,
                  {"bbox": req.bbox.model_dump(), "dsm_type": req.dsm_type, "out_dir": out_dir},
                  _cuzk_read, _cuzk_write)
    return {"download_id": dl_id}

@router.get("/download/cuzk/{dl_id}")
async def get_cuzk_status(dl_id: str):
    s = _cuzk_read(dl_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Download nenalezen.")
    return {"download_id": dl_id, **s}

@router.get("/download/cuzk/{dl_id}/dmr")
async def get_cuzk_dmr(dl_id: str):
    return _file_response(_cuzk_read, dl_id, "dmr_path", "DMR")

@router.get("/download/cuzk/{dl_id}/dmp")
async def get_cuzk_dmp(dl_id: str):
    return _file_response(_cuzk_read, dl_id, "dmp_path", "DMP")


# ---------------------------------------------------------------------------
# GUGiK (Polsko)
# ---------------------------------------------------------------------------

@router.post("/download/poland")
async def start_poland_download(req: PolandRequest):
    dl_id = str(uuid.uuid4())[:8]
    out_dir = os.path.join(JOBS_BASE, "poland", dl_id)
    os.makedirs(out_dir, exist_ok=True)
    _pl_write(dl_id, {"status": "running", "progress": 0, "step": "Spouštím stahování z GUGiK...",
                       "dmr_path": None, "dmp_path": None, "crs": "EPSG:2180", "error": None})
    _run_download(dl_id, _download_poland,
                  {"bbox": req.bbox.model_dump(), "use_lidar_point_cloud": req.use_lidar_point_cloud,
                   "out_dir": out_dir},
                  _pl_read, _pl_write)
    return {"download_id": dl_id}

@router.get("/download/poland/{dl_id}")
async def get_poland_status(dl_id: str):
    s = _pl_read(dl_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Download nenalezen.")
    return {"download_id": dl_id, **s}

@router.get("/download/poland/{dl_id}/dmr")
async def get_poland_dmr(dl_id: str):
    return _file_response(_pl_read, dl_id, "dmr_path", "DMR")

@router.get("/download/poland/{dl_id}/dmp")
async def get_poland_dmp(dl_id: str):
    return _file_response(_pl_read, dl_id, "dmp_path", "DMP")


# ---------------------------------------------------------------------------
# BEV (Rakousko)
# ---------------------------------------------------------------------------

@router.post("/download/austria")
async def start_austria_download(req: AustriaRequest):
    dl_id = str(uuid.uuid4())[:8]
    out_dir = os.path.join(JOBS_BASE, "austria", dl_id)
    os.makedirs(out_dir, exist_ok=True)
    _at_write(dl_id, {"status": "running", "progress": 0, "step": "Spouštím stahování z BEV...",
                       "dmr_path": None, "dmp_path": None, "crs": "EPSG:3035", "error": None})
    _run_download(dl_id, _download_austria,
                  {"bbox": req.bbox.model_dump(), "out_dir": out_dir},
                  _at_read, _at_write)
    return {"download_id": dl_id}

@router.get("/download/austria/{dl_id}")
async def get_austria_status(dl_id: str):
    s = _at_read(dl_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Download nenalezen.")
    return {"download_id": dl_id, **s}

@router.get("/download/austria/{dl_id}/dmr")
async def get_austria_dmr(dl_id: str):
    return _file_response(_at_read, dl_id, "dmr_path", "DTM")

@router.get("/download/austria/{dl_id}/dmp")
async def get_austria_dmp(dl_id: str):
    return _file_response(_at_read, dl_id, "dmp_path", "DSM")