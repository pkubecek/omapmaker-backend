"""
austria_downloader.py — stahování DGM/DOM z BEV (Bundesamt für Eich- und Vermessungswesen)
Rakousko poskytuje 1m GeoTIFF dlaždice (55 kachlí) volně ke stažení pod CC-BY-4.0.
Kachle jsou v EPSG:3035 (LAEA Europe), každá 50×50 km.

Zdroj: https://data.bev.gv.at
"""
import os
import json
import zipfile
import tempfile
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
import numpy as np

from pyproj import Transformer

# ---------------------------------------------------------------------------
# Kompletní seznam 55 kachlí AT v EPSG:3035 (N, E souřadnice levého dolního rohu)
# Kachle jsou 50000 x 50000 m (50×50 km)
# URL pattern: https://data.bev.gv.at/download/ALS/DTM/CRS3035RES50000mN{N}E{E}.zip
# ---------------------------------------------------------------------------

BEV_BASE = "https://data.bev.gv.at/download/ALS"
TILE_SIZE = 50000  # 50 km

# Všechny kachle pro Rakousko (N, E) v EPSG:3035
# Odvozeno z názvů metadatových záznamů na data.bev.gv.at
AT_DTM_TILES = [
    (2650000, 4250000), (2650000, 4300000), (2650000, 4350000),
    (2650000, 4400000), (2650000, 4450000), (2650000, 4500000),
    (2650000, 4550000), (2650000, 4600000),
    (2700000, 4250000), (2700000, 4300000), (2700000, 4350000),
    (2700000, 4400000), (2700000, 4450000), (2700000, 4500000),
    (2700000, 4550000), (2700000, 4600000), (2700000, 4650000),
    (2700000, 4700000), (2700000, 4750000), (2700000, 4800000),
    (2750000, 4300000), (2750000, 4350000), (2750000, 4400000),
    (2750000, 4450000), (2750000, 4500000), (2750000, 4550000),
    (2750000, 4600000), (2750000, 4650000), (2750000, 4700000),
    (2750000, 4750000), (2750000, 4800000),
    (2800000, 4350000), (2800000, 4400000), (2800000, 4450000),
    (2800000, 4500000), (2800000, 4550000), (2800000, 4600000),
    (2800000, 4650000), (2800000, 4700000), (2800000, 4750000),
    (2800000, 4800000),
    (2850000, 4400000), (2850000, 4450000), (2850000, 4500000),
    (2850000, 4550000), (2850000, 4600000), (2850000, 4650000),
    (2850000, 4700000), (2850000, 4750000), (2850000, 4800000),
    (2900000, 4500000), (2900000, 4550000), (2900000, 4600000),
    (2900000, 4650000), (2900000, 4700000),
]


def _tile_name(n: int, e: int) -> str:
    return f"CRS3035RES50000mN{n}E{e}"


def _tile_url(n: int, e: int, model: str = "DTM") -> str:
    """
    model: 'DTM' pro DGM, 'DSM' pro DOM
    """
    name = _tile_name(n, e)
    return f"{BEV_BASE}/{model}/{name}.zip"


def _wgs84_to_epsg3035(lat: float, lon: float):
    """Transformuje WGS84 souřadnice do EPSG:3035."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    e, n = t.transform(lon, lat)
    return n, e  # vrátíme (N, E)


def _find_overlapping_tiles(bbox: dict) -> list:
    """
    Vrátí seznam (n, e) kachlí které překrývají bbox v WGS84.
    bbox: { min_lat, min_lon, max_lat, max_lon }
    """
    # Transformuj rohy bbox do EPSG:3035
    corners = [
        (bbox["min_lat"], bbox["min_lon"]),
        (bbox["min_lat"], bbox["max_lon"]),
        (bbox["max_lat"], bbox["min_lon"]),
        (bbox["max_lat"], bbox["max_lon"]),
    ]
    t = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    ns, es = [], []
    for lat, lon in corners:
        e3, n3 = t.transform(lon, lat)
        ns.append(n3)
        es.append(e3)

    n_min, n_max = min(ns), max(ns)
    e_min, e_max = min(es), max(es)

    # Najdi překrývající kachle
    overlapping = []
    for (tn, te) in AT_DTM_TILES:
        # Kachle pokrývá [tn, tn+TILE_SIZE) x [te, te+TILE_SIZE)
        if (tn + TILE_SIZE > n_min and tn < n_max and
                te + TILE_SIZE > e_min and te < e_max):
            overlapping.append((tn, te))

    return overlapping


def _download_tile(url: str, out_dir: str, progress_cb=None) -> list:
    """
    Stáhne ZIP soubor z BEV a rozbalí GeoTIFF soubory.
    Vrátí seznam cest k extrahovaným TIF souborům.
    """
    tif_files = []
    zip_path = os.path.join(out_dir, os.path.basename(url))

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "OMapMaker/1.0 (orienteering map generator)"
        })
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
        with open(zip_path, "wb") as f:
            f.write(data)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            if progress_cb:
                progress_cb(f"Kachle nenalezena (404): {os.path.basename(url)}")
            return []
        raise

    # Rozbal ZIP
    extract_dir = zip_path.replace(".zip", "_extracted")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            if member.lower().endswith(".tif") or member.lower().endswith(".tiff"):
                z.extract(member, extract_dir)
                tif_files.append(os.path.join(extract_dir, member))

    os.remove(zip_path)
    return tif_files


def merge_tif_files(
    tif_files: list,
    output_path: str,
    clip_bbox_wgs84: tuple = None,
    progress_cb=None,
) -> bool:
    """
    Merguje více GeoTIFF souborů do jednoho a volitelně ořízne na bbox.
    Používá rasterio/gdal.
    """
    if not tif_files:
        print("[austria] Žádné TIF soubory k mergování")
        return False

    try:
        import rasterio
        from rasterio.merge import merge
        from rasterio.crs import CRS
        from rasterio.warp import transform_bounds
        from rasterio.mask import mask
        from shapely.geometry import box
        import geopandas as gpd

        if progress_cb:
            progress_cb(f"Mergování {len(tif_files)} TIF souborů...")

        datasets = [rasterio.open(f) for f in tif_files]
        mosaic, out_transform = merge(datasets)

        out_meta = datasets[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "compress": "lzw",
        })

        for ds in datasets:
            ds.close()

        # Ořízni na bbox pokud je zadán
        if clip_bbox_wgs84 is not None:
            min_lat, min_lon, max_lat, max_lon = clip_bbox_wgs84
            # Ulož dočasně mergovaný soubor
            tmp_path = output_path + "_tmp.tif"
            with rasterio.open(tmp_path, "w", **out_meta) as dest:
                dest.write(mosaic)

            # Ořízni
            with rasterio.open(tmp_path) as src:
                bbox_geom = box(min_lon, min_lat, max_lon, max_lat)
                # Transformuj bbox do CRS dat
                from rasterio.warp import transform_bounds
                from pyproj import Transformer
                t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                xmin, ymin = t.transform(min_lon, min_lat)
                xmax, ymax = t.transform(max_lon, max_lat)
                from shapely.geometry import mapping
                from rasterio.warp import transform_geom
                clip_geom = {
                    "type": "Polygon",
                    "coordinates": [[
                        [xmin, ymin], [xmax, ymin],
                        [xmax, ymax], [xmin, ymax],
                        [xmin, ymin],
                    ]],
                }
                clipped, clip_transform = mask(src, [clip_geom], crop=True, nodata=src.nodata)
                clip_meta = src.meta.copy()
                clip_meta.update({
                    "height": clipped.shape[1],
                    "width": clipped.shape[2],
                    "transform": clip_transform,
                    "compress": "lzw",
                })
                with rasterio.open(output_path, "w", **clip_meta) as out:
                    out.write(clipped)

            os.remove(tmp_path)
        else:
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(mosaic)

        print(f"[austria] Merge hotov → {os.path.basename(output_path)}")
        return True

    except Exception as e:
        print(f"[austria] Chyba při mergování: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_austria(
    bbox: dict,
    out_dir: str,
    progress_cb=None,
) -> dict:
    """
    Hlavní funkce: stáhne DGM (DTM) + DOM (DSM) pro daný bbox z BEV.

    bbox: { min_lat, min_lon, max_lat, max_lon }  (WGS84)
    out_dir: výstupní složka na serveru
    progress_cb: volitelná funkce(msg: str)

    Vrací: { dmr_path, dmp_path, crs }
    """
    os.makedirs(out_dir, exist_ok=True)

    mn_lat = bbox["min_lat"]
    mn_lon = bbox["min_lon"]
    mx_lat = bbox["max_lat"]
    mx_lon = bbox["max_lon"]

    def _cb(msg):
        print(f"[austria] {msg}")
        if progress_cb:
            progress_cb(msg)

    # Najdi překrývající kachle
    _cb("Hledám překrývající kachle BEV...")
    tiles = _find_overlapping_tiles(bbox)

    if not tiles:
        raise ValueError("Žádné BEV kachle pro vybranou oblast. Je oblast v Rakousku?")

    _cb(f"Nalezeno {len(tiles)} kachlí.")

    dtm_raw_dir = os.path.join(out_dir, "dtm_tiles")
    dsm_raw_dir = os.path.join(out_dir, "dsm_tiles")
    os.makedirs(dtm_raw_dir, exist_ok=True)
    os.makedirs(dsm_raw_dir, exist_ok=True)

    dtm_files = []
    dsm_files = []

    # Stáhni DTM kachle
    for i, (tn, te) in enumerate(tiles, 1):
        name = _tile_name(tn, te)
        _cb(f"DTM {i}/{len(tiles)}: {name}")
        url = _tile_url(tn, te, "DTM")
        files = _download_tile(url, dtm_raw_dir, _cb)
        dtm_files.extend(files)

    if not dtm_files:
        raise RuntimeError("Žádné DTM soubory se nepodařilo stáhnout. Zkuste menší oblast.")

    # Stáhni DSM kachle
    for i, (tn, te) in enumerate(tiles, 1):
        name = _tile_name(tn, te)
        _cb(f"DSM {i}/{len(tiles)}: {name}")
        url = _tile_url(tn, te, "DSM")
        files = _download_tile(url, dsm_raw_dir, _cb)
        dsm_files.extend(files)

    # Merguj DTM
    _cb("Mergování DTM...")
    dtm_merged = os.path.join(out_dir, "AT_BEV_DTM_merged.tif")
    bbox_tuple = (mn_lat, mn_lon, mx_lat, mx_lon)
    ok_dtm = merge_tif_files(dtm_files, dtm_merged, clip_bbox_wgs84=bbox_tuple, progress_cb=_cb)
    if not ok_dtm:
        raise RuntimeError("Merge DTM selhal.")

    # Merguj DSM
    dsm_merged = None
    if dsm_files:
        _cb("Mergování DSM...")
        dsm_merged = os.path.join(out_dir, "AT_BEV_DSM_merged.tif")
        merge_tif_files(dsm_files, dsm_merged, clip_bbox_wgs84=bbox_tuple, progress_cb=_cb)

    _cb("Hotovo!")

    return {
        "dmr_path": dtm_merged,
        "dmp_path": dsm_merged,
        "crs": "EPSG:3035",
    }
