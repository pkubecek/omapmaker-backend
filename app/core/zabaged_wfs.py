"""
zabaged_wfs.py — stahování ZABAGED® dat přes WFS službu ČÚZK.

Endpoint: https://ags.cuzk.gov.cz/arcgis/services/ZABAGED_POLOHOPIS/MapServer/WFSServer
- Zdarma, bez registrace, licence CC BY 4.0
- Výchozí CRS: EPSG:5514
- Limit: 1000 prvků na request (CountDefault), paginace přes startindex
- Výstup: GEOJSON
"""
import time
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from pyproj import Transformer

WFS_URL = "https://ags.cuzk.gov.cz/arcgis/services/ZABAGED_POLOHOPIS/MapServer/WFSServer"
PAGE_SIZE = 1000
MAX_RETRIES = 3
RETRY_DELAY = 5  # sekund

# Mapování klíčů (stejné jako v vector_layers.py / pipeline.py ZAB_MAP)
# na přesné názvy vrstev ve WFS (z GetCapabilities)
# Vrstvy bez budov dle požadavku.
ZABAGED_LAYERS = {
    "SilniceDalnice":               "ZABAGED_POLOHOPIS:Silnice__dálnice",
    "Cesta":                        "ZABAGED_POLOHOPIS:Cesta",
    "Pesina":                       "ZABAGED_POLOHOPIS:Pěšina__turistická_stezka",
    "ZeleznicniTrat":               "ZABAGED_POLOHOPIS:Železniční_trať",
    "VodniTok":                     "ZABAGED_POLOHOPIS:Vodní_tok",
    "VodniPlocha":                  "ZABAGED_POLOHOPIS:Vodní_plocha",
    "ElektrickeVedeni":             "ZABAGED_POLOHOPIS:Elektrické_vedení",
    "Zed":                          "ZABAGED_POLOHOPIS:Zeď",
    "Raseliniste":                  "ZABAGED_POLOHOPIS:Rašeliniště",
    "BazinaMocal":                  "ZABAGED_POLOHOPIS:Bažina__močál",
    "TrvalyTravniPorost":           "ZABAGED_POLOHOPIS:Trvalý_travní_porost",
    "VyznamnyNeboOsamelyStromLesik":"ZABAGED_POLOHOPIS:Významný_nebo_osamělý_strom__lesík",
    "OsamelyBalvanSkalaSkalniSuk":  "ZABAGED_POLOHOPIS:Osamělý_balvan__skála__skalní_suk",
    "StupenSraz":                   "ZABAGED_POLOHOPIS:Stupeň__sráz",
    "HradbaValBastaOpevneni":       "ZABAGED_POLOHOPIS:Hradba__val__bašta__opevnění",
    "ZdrojPodzemnichVod":           "ZABAGED_POLOHOPIS:Zdroj_podzemních_vod",
    "MohylaPomnikNahrobek":         "ZABAGED_POLOHOPIS:Mohyla__pomník__náhrobek",
    "LesniPozemek":                 "ZABAGED_POLOHOPIS:Lesní_pozemek",
    "SkalniSraz":                   "ZABAGED_POLOHOPIS:Skalní_sráz__výchoz",
    "Most":                         "ZABAGED_POLOHOPIS:Most",
    "Ohrada":                       "ZABAGED_POLOHOPIS:Ohrada__plot",
    "Krmitko":                      "ZABAGED_POLOHOPIS:Krmítko",
    "Proseka":                      "ZABAGED_POLOHOPIS:Průsek",
    "NasupisteHraze":               "ZABAGED_POLOHOPIS:Násyp__hráz",
    "HustyPorost":                  "ZABAGED_POLOHOPIS:Hustý_porost",
    "OrnaPudaAOstatniDaleNespecifikovanePlochy": "ZABAGED_POLOHOPIS:Orná_půda_a_ostatní_dále_nespecifikované_plochy",
}


def _fetch_page(typename: str, bbox_5514: tuple, startindex: int,
                progress_cb=None) -> dict | None:
    """Stáhne jednu stránku GeoJSON dat z WFS."""
    minx, miny, maxx, maxy = bbox_5514
    # WFS 2.0 s EPSG:5514 — bbox parametr musí být v pořadí miny,minx,maxy,maxx
    # (lat,lon pořadí pro geografické CRS) ale ESRI WFS akceptuje minx,miny,maxx,maxy
    bbox_str = f"{minx},{miny},{maxx},{maxy},urn:ogc:def:crs:EPSG::5514"

    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": typename,
        "outputFormat": "GEOJSON",
        "count": PAGE_SIZE,
        "startindex": startindex,
        "bbox": bbox_str,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(WFS_URL, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                if progress_cb:
                    progress_cb(f"  Retry {attempt+1}/{MAX_RETRIES}: {e}")
                time.sleep(RETRY_DELAY)
            else:
                print(f"[zabaged_wfs] Chyba {typename}: {e}")
                return None


def _download_layer(key: str, typename: str, bbox_5514: tuple,
                    target_crs: str, progress_cb=None) -> gpd.GeoDataFrame | None:
    """Stáhne celou vrstvu s paginací."""
    frames = []
    startindex = 0

    while True:
        data = _fetch_page(typename, bbox_5514, startindex, progress_cb)
        if data is None:
            break

        features = data.get("features", [])
        if not features:
            break

        try:
            gdf_page = gpd.GeoDataFrame.from_features(features, crs="EPSG:5514")
            frames.append(gdf_page)
        except Exception as e:
            print(f"[zabaged_wfs] Parse chyba {key} @{startindex}: {e}")
            break

        if progress_cb:
            progress_cb(f"  {key}: staženo {startindex + len(features)} prvků")

        if len(features) < PAGE_SIZE:
            # Poslední stránka
            break

        startindex += PAGE_SIZE

    if not frames:
        return None

    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:5514")

    # Převod do cílového CRS pokud se liší
    if target_crs and target_crs != "EPSG:5514":
        try:
            gdf = gdf.to_crs(target_crs)
        except Exception as e:
            print(f"[zabaged_wfs] CRS převod {key}: {e}")

    return gdf if not gdf.empty else None


def download_zabaged_wfs(
    bbox_wgs84: tuple,
    target_crs: str = "EPSG:5514",
    progress_cb=None,
) -> dict:
    """
    Stáhne všechny ZABAGED vrstvy (bez budov) pro daný bbox.

    bbox_wgs84: (min_lon, min_lat, max_lon, max_lat)  — WGS84
    target_crs: cílový CRS výsledných GeoDataFrames
    progress_cb: volitelná funkce(msg: str)

    Vrací: dict { klíč: GeoDataFrame } — stejná struktura jako zabaged_gdfs v pipeline
    """
    def cb(msg):
        print(f"[zabaged_wfs] {msg}")
        if progress_cb:
            progress_cb(msg)

    # Převod bbox z WGS84 do EPSG:5514 pro WFS dotaz
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    try:
        t = Transformer.from_crs("EPSG:4326", "EPSG:5514", always_xy=True)
        x1, y1 = t.transform(min_lon, min_lat)
        x2, y2 = t.transform(max_lon, max_lat)
        bbox_5514 = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    except Exception as e:
        cb(f"Varování: CRS transformace bbox selhala ({e}), používám WGS84 bbox")
        bbox_5514 = (min_lon, min_lat, max_lon, max_lat)

    cb(f"Stahuji ZABAGED WFS pro bbox {bbox_5514}")

    result = {}
    total = len(ZABAGED_LAYERS)

    for i, (key, typename) in enumerate(ZABAGED_LAYERS.items(), 1):
        cb(f"[{i}/{total}] {key}...")
        try:
            gdf = _download_layer(key, typename, bbox_5514, target_crs, cb)
            if gdf is not None and not gdf.empty:
                result[key] = gdf
                cb(f"  OK: {key} — {len(gdf)} prvků")
            else:
                cb(f"  Prázdná vrstva: {key}")
        except Exception as e:
            cb(f"  Chyba {key}: {e}")

    cb(f"ZABAGED WFS hotovo: {len(result)}/{total} vrstev staženo")
    return result
