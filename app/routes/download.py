"""
routes/download.py — stahování dat z ČÚZK ATOM a polského GUGiK jako background job.

Endpointy:
  POST /api/download/cuzk          — spustí stahování ČR, vrátí download_id
  GET  /api/download/cuzk/{id}     — stav stahování
  GET  /api/download/cuzk/{id}/dmr — stáhne DMR soubor
  GET  /api/download/cuzk/{id}/dmp — stáhne DMP soubor

  POST /api/download/poland          — spustí stahování PL, vrátí download_id
  GET  /api/download/poland/{id}     — stav stahování
  GET  /api/download/poland/{id}/dmr — stáhne DTM soubor (LAZ)
  GET  /api/download/poland/{id}/dmp — stáhne DSM soubor (LAZ, pokud dostupný)
"""
import os
import uuid
import json
import threading
import ssl
import urllib.request
import xml.etree.ElementTree as ET
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ..core.downloader import download_cuzk as _download_cuzk
from ..core.poland_downloader import download_poland as _download_poland

router = APIRouter()

DOWNLOADS_DIR = os.environ.get("OMAPMAKER_JOBS_DIR", "./jobs") + "/cuzk"
os.makedirs(DOWNLOADS_DIR, exist_ok=True)


def _status_path(dl_id):
    return os.path.join(DOWNLOADS_DIR, dl_id, "status.json")

def _read_status(dl_id):
    p = _status_path(dl_id)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)

def _write_status(dl_id, data):
    p = _status_path(dl_id)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f)


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
    use_lidar_point_cloud: bool = True  # True = LAZ, False = NMT rastr (rychlejší)


# ============================================================
# ČÚZK (Česká republika)
# ============================================================

@router.post("/download/cuzk")
async def start_cuzk_download(req: CuzkRequest):
    """Spustí stahování ČÚZK na pozadí, vrátí download_id."""
    dl_id = str(uuid.uuid4())[:8]
    out_dir = os.path.join(DOWNLOADS_DIR, dl_id)
    os.makedirs(out_dir, exist_ok=True)

    _write_status(dl_id, {
        "status": "running",
        "progress": 0,
        "step": "Spouštím stahování...",
        "dmr_path": None,
        "dmp_path": None,
        "crs": "EPSG:5514",
        "error": None,
    })

    def _run():
        def cb(msg):
            s = _read_status(dl_id) or {}
            s["step"] = msg
            s["progress"] = min(s.get("progress", 0) + 5, 90)
            _write_status(dl_id, s)

        try:
            result = _download_cuzk(
                bbox=req.bbox.model_dump(),
                dsm_type=req.dsm_type,
                out_dir=out_dir,
                progress_cb=cb,
            )
            _write_status(dl_id, {
                "status": "done",
                "progress": 100,
                "step": "Hotovo!",
                "dmr_path": result.get("dmr_path"),
                "dmp_path": result.get("dmp_path"),
                "crs": "EPSG:5514",
                "error": None,
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            _write_status(dl_id, {
                "status": "error",
                "progress": 0,
                "step": f"Chyba: {e}",
                "dmr_path": None,
                "dmp_path": None,
                "crs": None,
                "error": str(e),
            })

    threading.Thread(target=_run, daemon=True).start()
    return {"download_id": dl_id}


@router.get("/download/cuzk/{dl_id}")
async def get_cuzk_status(dl_id: str):
    s = _read_status(dl_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Download nenalezen.")
    return {"download_id": dl_id, **s}


@router.get("/download/cuzk/{dl_id}/dmr")
async def get_cuzk_dmr(dl_id: str):
    s = _read_status(dl_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Download nenalezen.")
    if s["status"] != "done":
        raise HTTPException(status_code=425, detail="Stahování ještě neskončilo.")
    path = s.get("dmr_path")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="DMR soubor nenalezen.")
    return FileResponse(path, media_type="application/octet-stream",
                        filename=os.path.basename(path))


@router.get("/download/cuzk/{dl_id}/dmp")
async def get_cuzk_dmp(dl_id: str):
    s = _read_status(dl_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Download nenalezen.")
    if s["status"] != "done":
        raise HTTPException(status_code=425, detail="Stahování ještě neskončilo.")
    path = s.get("dmp_path")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="DMP soubor nenalezen.")
    return FileResponse(path, media_type="application/octet-stream",
                        filename=os.path.basename(path))


# ============================================================
# GUGiK (Polsko)
# ============================================================

@router.post("/download/poland")
async def start_poland_download(req: PolandRequest):
    """Spustí stahování polských LiDAR/NMT dat na pozadí, vrátí download_id."""
    dl_id = str(uuid.uuid4())[:8]
    out_dir = os.path.join(DOWNLOADS_DIR, dl_id)
    os.makedirs(out_dir, exist_ok=True)

    _write_status(dl_id, {
        "status": "running",
        "progress": 0,
        "step": "Spouštím stahování (PL GUGiK)...",
        "dmr_path": None,
        "dmp_path": None,
        "crs": "EPSG:2180",
        "error": None,
        "country": "pl",
    })

    def _run():
        step_counter = {"n": 0}

        def cb(msg):
            step_counter["n"] += 1
            s = _read_status(dl_id) or {}
            s["step"] = msg
            # Progres: 0–90 lineárně (počet kroků neznáme předem)
            s["progress"] = min(5 + step_counter["n"] * 3, 90)
            _write_status(dl_id, s)

        try:
            result = _download_poland(
                bbox=req.bbox.model_dump(),
                out_dir=out_dir,
                use_lidar_point_cloud=req.use_lidar_point_cloud,
                progress_cb=cb,
            )
            _write_status(dl_id, {
                "status": "done",
                "progress": 100,
                "step": "Hotovo!",
                "dmr_path": result.get("dmr_path"),
                "dmp_path": result.get("dmp_path") or None,
                "crs": result.get("crs", "EPSG:2180"),
                "error": None,
                "country": "pl",
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            _write_status(dl_id, {
                "status": "error",
                "progress": 0,
                "step": f"Chyba: {e}",
                "dmr_path": None,
                "dmp_path": None,
                "crs": None,
                "error": str(e),
                "country": "pl",
            })

    threading.Thread(target=_run, daemon=True).start()
    return {"download_id": dl_id}


@router.get("/download/poland/{dl_id}")
async def get_poland_status(dl_id: str):
    s = _read_status(dl_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Download nenalezen.")
    return {"download_id": dl_id, **s}


@router.get("/download/poland/{dl_id}/dmr")
async def get_poland_dmr(dl_id: str):
    s = _read_status(dl_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Download nenalezen.")
    if s["status"] != "done":
        raise HTTPException(status_code=425, detail="Stahování ještě neskončilo.")
    path = s.get("dmr_path")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="DTM soubor nenalezen.")
    return FileResponse(path, media_type="application/octet-stream",
                        filename=os.path.basename(path))


@router.get("/download/poland/{dl_id}/dmp")
async def get_poland_dmp(dl_id: str):
    s = _read_status(dl_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Download nenalezen.")
    if s["status"] != "done":
        raise HTTPException(status_code=425, detail="Stahování ještě neskončilo.")
    path = s.get("dmp_path")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="DSM soubor nenalezen (nebo nebyl dostupný).")
    return FileResponse(path, media_type="application/octet-stream",
                        filename=os.path.basename(path))


# ============================================================
# DEBUG — dočasný endpoint pro diagnostiku WFS
# GET /api/download/debug/poland-wfs
# ============================================================

_SSL_CTX_DBG = ssl.create_default_context()
_SSL_CTX_DBG.check_hostname = False
_SSL_CTX_DBG.verify_mode = ssl.CERT_NONE

_WFS_ENDPOINTS = {
    "lidar": "https://mapy.geoportal.gov.pl/wss/service/PZGIK/DanePomiaroweLidarEVRF2007/WFS/Skorowidze",
    "nmt":   "https://mapy.geoportal.gov.pl/wss/service/PZGIK/NumerycznyModelTerenuEVRF2007/WFS/Skorowidze",
    "nmpt":  "https://mapy.geoportal.gov.pl/wss/service/PZGIK/NumerycznyModelPowierzchniEVRF2007/WFS/Skorowidze",
}

@router.get("/download/debug/poland-wfs")
async def debug_poland_wfs(
    lat: float = 50.05,
    lon: float = 19.95,
    span: float = 0.1,
):
    """
    Diagnostický endpoint.
    Příklad: /api/download/debug/poland-wfs?lat=50.05&lon=19.95&span=0.1
    """
    result = {}
    headers = {"User-Agent": "OMapMaker/7 diagnostic"}

    for name, wfs_url in _WFS_ENDPOINTS.items():
        caps_url = f"{wfs_url}?SERVICE=WFS&REQUEST=GetCapabilities"
        try:
            req = urllib.request.Request(caps_url, headers=headers)
            with urllib.request.urlopen(req, timeout=20, context=_SSL_CTX_DBG) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            root = ET.fromstring(raw)
            type_names = [
                el.text.strip()
                for el in root.iter()
                if (el.tag.endswith("}Name") or el.tag == "Name")
                and el.text and ":" in el.text
            ]
            result[name] = {"caps_ok": True, "type_names": type_names}
        except Exception as e:
            result[name] = {"caps_ok": False, "error": str(e)}
            continue

        # Zkus GetFeature s prvním TypeName
        if type_names:
            tn = type_names[0]
            bbox_str = f"{lon-span},{lat-span},{lon+span},{lat+span},urn:ogc:def:crs:EPSG::4326"
            feat_url = (
                f"{wfs_url}?SERVICE=WFS&REQUEST=GetFeature&VERSION=2.0.0"
                f"&TYPENAMES={tn}&SRSNAME=urn:ogc:def:crs:EPSG::4326"
                f"&BBOX={bbox_str}&COUNT=3"
            )
            try:
                req2 = urllib.request.Request(feat_url, headers=headers)
                with urllib.request.urlopen(req2, timeout=20, context=_SSL_CTX_DBG) as resp2:
                    raw2 = resp2.read().decode("utf-8", errors="replace")
                result[name]["feature_sample"] = raw2[:2000]
                result[name]["feature_url"] = feat_url
            except Exception as e2:
                result[name]["feature_error"] = str(e2)
                result[name]["feature_url"] = feat_url

    return JSONResponse(result)