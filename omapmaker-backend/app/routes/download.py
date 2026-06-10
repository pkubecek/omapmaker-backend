"""
routes/download.py — endpoint pro stahování dat z ČÚZK ATOM.
POST /api/download/cuzk
"""
import threading
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..core.downloader import download_cuzk

router = APIRouter()


class BboxModel(BaseModel):
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float


class CuzkRequest(BaseModel):
    bbox: BboxModel
    dsm_type: str = "DMPOK"   # "DMPOK" nebo "DMP1G"
    out_dir: str = "./cuzk_data"


@router.post("/download/cuzk")
async def trigger_cuzk_download(req: CuzkRequest):
    """
    Stáhne LiDAR dlaždice z ČÚZK ATOM pro zadanou oblast.
    Běží synchronně (může trvat minuty) — vhodné zavolat s delším timeoutem.
    Pro produkci doporučujeme přesunout do background jobu stejně jako /api/jobs.
    """
    try:
        result = download_cuzk(
            bbox=req.bbox.model_dump(),
            dsm_type=req.dsm_type,
            out_dir=req.out_dir,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chyba stahování: {e}")
