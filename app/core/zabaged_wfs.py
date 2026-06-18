"""
zabaged_wfs.py — stahování ZABAGED® dat přes ArcGIS REST API (ČÚZK).

Layer ID ověřena z:
https://ags.cuzk.gov.cz/arcgis/rest/services/ZABAGED_POLOHOPIS/MapServer
Klíče odpovídají názvům používaným v OMapMaker_v7.py.
"""
import time
import requests
import geopandas as gpd
import pandas as pd
from pyproj import Transformer

REST_BASE = "https://ags.cuzk.gov.cz/arcgis/rest/services/ZABAGED_POLOHOPIS/MapServer"
PAGE_SIZE = 2000
MAX_RETRIES = 3
RETRY_DELAY = 3

ZABAGED_LAYERS = {
    # Terénní reliéf
    "OsamelyBalvanSkalaSkalniSuk":   10,   # Osamělý balvan, skála, skalní suk
    "VstupDoJeskyne":                11,   # Vstup do jeskyně
    "SkupinaBalvanu_b":              12,   # Skupina balvanů (bod)
    "StupenSraz":                    95,   # Stupeň, sráz
    "RokleVymol":                    94,   # Rokle, výmol
    "SkalniUtvary":                  130,  # Skalní útvary

    # Vegetace
    "VyznamnyNeboOsamelyStromLesik": 14,   # Významný nebo osamělý strom, lesík
    "LiniovaVegetace":               15,   # Liniová vegetace
    "LesniPrusek":                   16,   # Lesní průsek
    "Raseliniste":                   18,   # Rašeliniště (plocha)
    "TrvalyTravniPorost":            141,  # Trvalý travní porost
    "LesniPozemek":                  143,  # Lesní půda se stromy
    "LesniPudaSeStromyKategorizovana": 144, # Lesní půda se stromy kategorizovaná
    "OrnaPudaAOstatniDaleNespecifikovanePlochy": 142, # Orná půda a ostatní plochy
    "OkrasnaZahradaPark":            134,  # Udržovaná zeleň (parky, okrasné zahrady)
    "OvocnySadZahrada":              135,  # Ovocný sad, zahrada
    "Vinice":                        136,  # Vinice

    # Vodstvo
    "ZdrojPodzemnichVod":            19,   # Zdroj podzemních vod
    "NasupisteHraze":                22,   # Přehradní hráz, jez
    "VodniTok":                      93,   # Vodní tok
    "VodniPlocha":                   132,  # Vodní plocha
    "BazinaMocal":                   131,  # Bažina, močál

    # Komunikace
    "SilniceDalnice":                79,   # Silnice, dálnice
    "SilniceNeevidovana":            80,   # Silnice neevidovaná
    "SilniceVeVastavbe":             81,   # Silnice ve výstavbě
    "Pesina":                        82,   # Pěšina
    "Cesta":                         83,   # Cesta
    "Ulice":                         84,   # Ulice
    "Most":                          73,   # Most
    "Lavka":                         67,   # Lávka (linie)
    "ZeleznicniTrat":                75,   # Železniční trať
    "LanovaDrahaLyzarskyVlek":       72,   # Lanová dráha, lyžařský vlek
    "LyzarskyMustek":                41,   # Lyžařský můstek
    "ParkovisteOdpocivka":           123,  # Parkoviště, odpočívka

    # Rozvodné sítě
    "ElektrickeVedeni":              88,   # Elektrické vedení

    # Sídla, hospodářské a kulturní objekty
    "BudovaJednotlivaNeboBlokBudov": 99,   # Budova jednotlivá nebo blok budov (plocha)
    "HradbaValBastaOpevneni":        38,   # Hradba, val, bašta, opevnění
    "Zed":                           39,   # Zeď
    "MohylaPomnikNahrobek":          25,   # Mohyla, pomník, náhrobek
    "KrizSloupKulturnihoVyznamu":    24,   # Kříž, sloup kulturního významu
    "RozvalinaZricenina":            103,  # Rozvalina, zřícenina
    "Bunkr":                         37,   # Bunkr
    "PovrchTezbaLom":                118,  # Povrchová těžba, lom
    "ArealUceloveZastavby":          114,  # Areál účelové zástavby
    "Hrbitov":                       116,  # Hřbitov
    "Skladka":                       117,  # Skládka
}


def _fetch_page(layer_id: int, bbox_5514: tuple, offset: int,
                progress_cb=None) -> dict | None:
    minx, miny, maxx, maxy = bbox_5514
    url = f"{REST_BASE}/{layer_id}/query"
    params = {
        "f": "geojson",
        "geometry": f"{minx},{miny},{maxx},{maxy}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "5514",
        "outSR": "5514",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "resultOffset": offset,
        "resultRecordCount": PAGE_SIZE,
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise ValueError(f"ArcGIS error: {data['error']}")
            return data
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                if progress_cb:
                    progress_cb(f"Retry {attempt+1}/{MAX_RETRIES}: {e}")
                time.sleep(RETRY_DELAY)
            else:
                print(f"[zabaged] Chyba layer {layer_id}: {e}")
                return None


def _download_layer(key: str, layer_id: int, bbox_5514: tuple,
                    target_crs: str, progress_cb=None) -> gpd.GeoDataFrame | None:
    frames = []
    offset = 0
    while True:
        data = _fetch_page(layer_id, bbox_5514, offset, progress_cb)
        if data is None:
            break
        features = data.get("features", [])
        if not features:
            break
        try:
            gdf_page = gpd.GeoDataFrame.from_features(features, crs="EPSG:5514")
            frames.append(gdf_page)
        except Exception as e:
            print(f"[zabaged] Parse chyba {key} @{offset}: {e}")
            break
        if progress_cb:
            progress_cb(f"  {key}: {offset + len(features)} prvků")
        if not data.get("exceededTransferLimit", False):
            break
        offset += PAGE_SIZE

    if not frames:
        return None
    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:5514")
    if target_crs and target_crs != "EPSG:5514":
        try:
            gdf = gdf.to_crs(target_crs)
        except Exception as e:
            print(f"[zabaged] CRS převod {key}: {e}")
    return gdf if not gdf.empty else None


def download_zabaged_wfs(
    bbox_wgs84: tuple,
    target_crs: str = "EPSG:5514",
    progress_cb=None,
) -> dict:
    """
    Stáhne ZABAGED vrstvy pro daný bbox přes ArcGIS REST API.
    bbox_wgs84: (min_lon, min_lat, max_lon, max_lat) — WGS84
    """
    def cb(msg):
        print(f"[zabaged] {msg}")
        if progress_cb:
            progress_cb(msg)

    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    try:
        t = Transformer.from_crs("EPSG:4326", "EPSG:5514", always_xy=True)
        x1, y1 = t.transform(min_lon, min_lat)
        x2, y2 = t.transform(max_lon, max_lat)
        bbox_5514 = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    except Exception as e:
        cb(f"Varování: CRS transformace selhala ({e})")
        bbox_5514 = (min_lon, min_lat, max_lon, max_lat)

    cb(f"Stahuji ZABAGED REST bbox={bbox_5514}")
    result = {}
    total = len(ZABAGED_LAYERS)

    for i, (key, layer_id) in enumerate(ZABAGED_LAYERS.items(), 1):
        cb(f"[{i}/{total}] {key} (layer {layer_id})...")
        try:
            gdf = _download_layer(key, layer_id, bbox_5514, target_crs, cb)
            if gdf is not None and not gdf.empty:
                result[key] = gdf
                cb(f"  OK: {key} — {len(gdf)} prvků")
            else:
                cb(f"  Prázdná vrstva: {key}")
        except Exception as e:
            cb(f"  Chyba {key}: {e}")

    cb(f"ZABAGED hotovo: {len(result)}/{total} vrstev staženo")
    return result