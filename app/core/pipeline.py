"""
pipeline.py — tile-based orchestrace analýzy.

Oblast se rozdělí na dlaždice max TILE_SIZE_M × TILE_SIZE_M s překryvem OVERLAP_M.
Každá dlaždice se zpracuje samostatně (DTM, DSM, vegetace, skály, vrstevnice).
Výsledky se sloučí do finálního PNG a GPKG.
"""
import os
import gc
import time
import math
import tempfile
import numpy as np
import rasterio
import rasterio.transform
import rasterio.features
import rasterio.merge
import geopandas as gpd
import osmnx as ox
import fiona
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pyproj import CRS, Transformer
from shapely.geometry import box
from rasterio.features import rasterize
from rasterio.io import MemoryFile

from .processor import (
    load_dmr_grid, load_dmp_grid,
    classify_vegetation, vectorize_rocks,
    find_depressions, find_knolls,
    make_clip_polygon,
)
from .renderer import generate_contour_layers, setup_map_figure
from .symbols import SymbolLibrary
from .exporter import OomCollector

# Max velikost dlaždice v metrech. 1500×1500m při 0.5m pixelu = 9M pixelů = ~700 MB
TILE_SIZE_M = 1500
OVERLAP_M = 150   # překryv kvůli artefaktům na hranicích


def _compute_tiles(minx, maxx, miny, maxy, tile_size, overlap):
    """Vrátí seznam (tx0, tx1, ty0, ty1) dlaždic s překryvem."""
    tiles = []
    x = minx
    while x < maxx:
        x1 = min(x + tile_size, maxx)
        y = miny
        while y < maxy:
            y1 = min(y + tile_size, maxy)
            # S překryvem
            tx0 = max(minx, x - overlap)
            tx1 = min(maxx, x1 + overlap)
            ty0 = max(miny, y - overlap)
            ty1 = min(maxy, y1 + overlap)
            tiles.append((tx0, tx1, ty0, ty1, x, x1, y, y1))
            y = y1
        x = x1
    return tiles


def _process_tile(
    tile_bbox, dmr_path, dmp_path, params, sym_library,
    gdf_osm, zabaged_gdfs, isom_gdfs,
    tile_idx, total_tiles, cb,
):
    """
    Zpracuje jednu dlaždici. Vrátí dict s GeoDataFrames a numpy gridy.
    tx0,tx1,ty0,ty1 = bbox s překryvem (zpracovává se)
    core_x0,x1,y0,y1 = core bbox bez překryvu (vystřihuje se)
    """
    tx0, tx1, ty0, ty1, core_x0, core_x1, core_y0, core_y1 = tile_bbox
    pct_base = int(10 + (tile_idx / total_tiles) * 70)

    CURRENT_CRS = params["crs"]
    SIGMA = params["sigma"]
    SLOPE_THRESHOLD = params["slope_threshold"]
    BINS = params["bins"]
    DEP = params.get("depressions", {})
    KNO = params.get("knolls", {})
    FIXED_PIXEL_SIZE = 0.5

    def tcb(msg):
        cb(pct_base, f"Dlaždice {tile_idx+1}/{total_tiles}: {msg}")

    tcb("Načítám DTM...")
    try:
        dmr_grid_cubic, grid_x, grid_y, extent, dmr_points, dmr_z = load_dmr_grid(
            dmr_path, CURRENT_CRS,
            pixel_size=FIXED_PIXEL_SIZE,
            sigma_smooth=SIGMA,
            bbox_clip=(tx0, tx1, ty0, ty1),
            progress_cb=tcb,
        )
    except Exception as e:
        print(f"[tile {tile_idx}] DTM chyba: {e}")
        return None

    minx, maxx, miny, maxy = extent
    shape = grid_x.shape
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy,
                                                width=shape[0], height=shape[1])
    clip_polygon = make_clip_polygon(dmr_points)

    # Linear DTM
    from scipy.interpolate import griddata
    shift_x = np.mean(dmr_points[:, 0])
    shift_y = np.mean(dmr_points[:, 1])
    pts_sh = dmr_points - np.array([shift_x, shift_y])
    gx_sh = grid_x - shift_x
    gy_sh = grid_y - shift_y
    dmr_grid_linear = griddata(pts_sh, dmr_z, (gx_sh, gy_sh), method="linear")
    if np.isnan(dmr_grid_linear).all():
        dmr_grid_linear = griddata(pts_sh, dmr_z, (gx_sh, gy_sh), method="nearest")

    # DSM
    tcb("Načítám DSM...")
    try:
        dmp_grid = load_dmp_grid(dmp_path, grid_x, grid_y, extent, CURRENT_CRS)
        vegetation_height = np.clip(dmp_grid - dmr_grid_linear, 0, None)
        del dmp_grid
    except Exception as e:
        print(f"[tile {tile_idx}] DSM chyba: {e}")
        vegetation_height = np.zeros_like(dmr_grid_linear)

    # Lesní maska z OSM
    forest_mask = np.zeros(shape, dtype=np.uint8)
    if gdf_osm is not None and not gdf_osm.empty:
        try:
            tile_box = box(minx, miny, maxx, maxy)
            gdf_tile_osm = gpd.clip(gdf_osm, tile_box)
            natural_col = gdf_tile_osm["natural"].fillna("") if "natural" in gdf_tile_osm.columns else ""
            landuse_col = gdf_tile_osm["landuse"].fillna("") if "landuse" in gdf_tile_osm.columns else ""
            forest_polys = gdf_tile_osm[(natural_col == "wood") | (landuse_col == "forest")].geometry
            if not forest_polys.empty:
                fm_t = rasterize(forest_polys, out_shape=(shape[1], shape[0]),
                                  transform=transform, fill=0, default_value=1, dtype=np.uint8)
                forest_mask = np.flipud(fm_t).T
        except Exception:
            pass

    b1 = BINS[0]
    is_clearing = ((vegetation_height < b1) & (vegetation_height >= 0)) & (forest_mask == 1)
    vegetation_height[is_clearing] = -1

    # Clip maska
    if clip_polygon is not None:
        try:
            cm_t = rasterize([(clip_polygon, 1)], out_shape=(shape[1], shape[0]),
                              transform=transform, fill=0, default_value=1, dtype=np.uint8)
            clip_mask = np.flipud(cm_t).T.astype(bool)
            dmr_grid_linear_viz = np.nan_to_num(dmr_grid_linear, nan=0)
            dmr_grid_linear_viz[~clip_mask] = 0
            dmr_grid_cubic_viz = np.nan_to_num(dmr_grid_cubic, nan=0)
            dmr_grid_cubic_viz[~clip_mask] = np.nan
        except Exception:
            dmr_grid_linear_viz = np.nan_to_num(dmr_grid_linear, nan=0)
            dmr_grid_cubic_viz = np.nan_to_num(dmr_grid_cubic, nan=0)
    else:
        dmr_grid_linear_viz = np.nan_to_num(dmr_grid_linear, nan=0)
        dmr_grid_cubic_viz = np.nan_to_num(dmr_grid_cubic, nan=0)

    del dmr_grid_linear, dmr_grid_cubic
    gc.collect()

    # Vegetace, skály, vrstevnice, mikrotvary
    tcb("Klasifikuji vegetaci...")
    gdf_vegetation = classify_vegetation(vegetation_height, BINS, transform, dmr_path)
    if gdf_vegetation is not None and not gdf_vegetation.empty:
        gdf_vegetation = gdf_vegetation.set_crs(CURRENT_CRS, allow_override=True)
    del vegetation_height
    gc.collect()

    tcb("Vektorizuji skály...")
    gdf_rocks = vectorize_rocks(grid_x, grid_y, dmr_grid_linear_viz, transform,
                                  slope_threshold_deg=SLOPE_THRESHOLD)
    if gdf_rocks is not None and not gdf_rocks.empty:
        gdf_rocks = gdf_rocks.set_crs(CURRENT_CRS, allow_override=True)

    tcb("Generuji vrstevnice...")
    contour_layers = generate_contour_layers(grid_x, grid_y, dmr_grid_cubic_viz,
                                              clip_polygon=clip_polygon)
    for k, gdf_c in contour_layers.items():
        if not gdf_c.empty:
            contour_layers[k] = gdf_c.set_crs(CURRENT_CRS, allow_override=True)

    tcb("Mikrotvary...")
    depressions = find_depressions(
        grid_x, grid_y, dmr_grid_linear_viz,
        pixel_size=FIXED_PIXEL_SIZE, current_crs=CURRENT_CRS,
        min_diameter=params.get("dep_min_diameter", 2),
        max_diameter=params.get("dep_max_diameter", 5),
        min_depth=params.get("dep_min_depth", 0.7),
    )
    knolls = find_knolls(
        grid_x, grid_y, dmr_grid_linear_viz,
        pixel_size=FIXED_PIXEL_SIZE, current_crs=CURRENT_CRS,
        min_diameter=params.get("kno_min_diameter", 1.5),
        max_diameter=params.get("kno_max_diameter", 10),
        min_height=params.get("kno_min_height", 0.5),
    )

    # Ořez na core bbox (bez překryvu) — aby se prvky neduplikovaly
    core_box = box(core_x0, core_y0, core_x1, core_y1)

    def clip_to_core(gdf):
        if gdf is None or gdf.empty:
            return gdf
        try:
            return gpd.clip(gdf, core_box)
        except Exception:
            return gdf

    gdf_vegetation = clip_to_core(gdf_vegetation)
    gdf_rocks = clip_to_core(gdf_rocks)
    for k in contour_layers:
        contour_layers[k] = clip_to_core(contour_layers[k])
    if depressions:
        dep_gdf = gpd.GeoDataFrame(geometry=depressions, crs=CURRENT_CRS)
        dep_gdf = clip_to_core(dep_gdf)
        depressions = list(dep_gdf.geometry) if not dep_gdf.empty else []
    if knolls:
        kno_gdf = gpd.GeoDataFrame(geometry=knolls, crs=CURRENT_CRS)
        kno_gdf = clip_to_core(kno_gdf)
        knolls = list(kno_gdf.geometry) if not kno_gdf.empty else []

    del dmr_grid_linear_viz, dmr_grid_cubic_viz, grid_x, grid_y
    gc.collect()

    return {
        "extent": (core_x0, core_x1, core_y0, core_y1),
        "full_extent": extent,
        "gdf_vegetation": gdf_vegetation,
        "gdf_rocks": gdf_rocks,
        "contour_layers": contour_layers,
        "depressions": depressions,
        "knolls": knolls,
        "clip_polygon": clip_polygon,
    }


def _merge_tile_results(tile_results):
    """Sloučí výsledky všech dlaždic do jednoho setu GeoDataFrames."""
    all_veg, all_rocks = [], []
    all_contours = {"base": [], "major": [], "minor": []}
    all_dep, all_kno = [], []
    all_clips = []

    for tr in tile_results:
        if tr is None:
            continue
        if tr["gdf_vegetation"] is not None and not tr["gdf_vegetation"].empty:
            all_veg.append(tr["gdf_vegetation"])
        if tr["gdf_rocks"] is not None and not tr["gdf_rocks"].empty:
            all_rocks.append(tr["gdf_rocks"])
        for k in all_contours:
            gdf_c = tr["contour_layers"].get(k)
            if gdf_c is not None and not gdf_c.empty:
                all_contours[k].append(gdf_c)
        all_dep.extend(tr["depressions"])
        all_kno.extend(tr["knolls"])
        if tr["clip_polygon"] is not None:
            all_clips.append(tr["clip_polygon"])

    import pandas as pd
    from shapely.ops import unary_union

    def safe_concat(frames):
        if not frames:
            return gpd.GeoDataFrame()
        return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)

    merged_contours = {}
    for k, frames in all_contours.items():
        merged_contours[k] = safe_concat(frames)

    clip_union = unary_union(all_clips) if all_clips else None

    return {
        "gdf_vegetation": safe_concat(all_veg),
        "gdf_rocks": safe_concat(all_rocks),
        "contour_layers": merged_contours,
        "depressions": all_dep,
        "knolls": all_kno,
        "clip_polygon": clip_union,
    }


def run_pipeline(job_id: str, params: dict, file_paths: dict,
                 output_dir: str, progress_cb) -> dict:
    start = time.time()

    def cb(pct, msg):
        print(f"[pipeline {job_id}] {pct}% — {msg}")
        progress_cb(pct, msg)

    cb(1, "Spouštím analýzu...")

    CURRENT_CRS = params.get("crs", "EPSG:5514")
    SCALE = int(params.get("scale", 10000))
    PAPER_FORMAT = params.get("paper_format", "A4 (Landscape)")
    SIGMA = float(params.get("sigma", 4))
    SLOPE_THRESHOLD = float(params.get("slope_threshold", 45.0))
    NORTH_ROTATION = float(params.get("north_rotation", 5.0))
    BINS = [float(b) for b in params.get("bins", [1, 2, 6, 12])]
    LAYER_VISIBILITY = params.get("layers", {
        "contours": True, "rocks": True, "water": True,
        "vegetation": True, "roads": True, "buildings": True,
        "man_made": True, "magnetic_lines": False,
    })

    dep_params = params.get("depressions", {})
    DEP_MIN_DIAM = float(dep_params.get("min_diameter", 2))
    DEP_MAX_DIAM = float(dep_params.get("max_diameter", 5))
    DEP_MIN_DEPTH = float(dep_params.get("min_depth", 0.7))

    kno_params = params.get("knolls", {})
    KNO_MIN_DIAM = float(kno_params.get("min_diameter", 1.5))
    KNO_MAX_DIAM = float(kno_params.get("max_diameter", 10))
    KNO_MIN_HEIGHT = float(kno_params.get("min_height", 0.5))

    tile_params = {
        "crs": CURRENT_CRS, "sigma": SIGMA,
        "slope_threshold": SLOPE_THRESHOLD, "bins": BINS,
        "dep_min_diameter": DEP_MIN_DIAM, "dep_max_diameter": DEP_MAX_DIAM,
        "dep_min_depth": DEP_MIN_DEPTH,
        "kno_min_diameter": KNO_MIN_DIAM, "kno_max_diameter": KNO_MAX_DIAM,
        "kno_min_height": KNO_MIN_HEIGHT,
    }

    # Symbols
    sym_xml = f"symbols{10 if SCALE == 10000 else 15}.xml"
    for candidate in [sym_xml,
                       os.path.join(os.path.dirname(__file__), "..", "..", sym_xml),
                       os.path.join(os.path.dirname(__file__), sym_xml)]:
        if os.path.exists(candidate):
            sym_xml = candidate
            break
    sym_library = SymbolLibrary(sym_xml)

    dmr_path = file_paths["dtm"]
    dmp_path = file_paths["dsm"]

    # Zjisti rozsah dat z LAZ hlavičky
    cb(3, "Zjišťuji rozsah dat...")
    import laspy
    with laspy.open(dmr_path) as fh:
        hdr = fh.header
        try:
            src_crs = hdr.parse_crs() or CRS.from_epsg(5514)
        except Exception:
            src_crs = CRS.from_epsg(5514)
        raw_minx, raw_maxx = float(hdr.x_min), float(hdr.x_max)
        raw_miny, raw_maxy = float(hdr.y_min), float(hdr.y_max)

    try:
        dst_crs = CRS.from_string(CURRENT_CRS)
        if src_crs != dst_crs:
            t = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
            xs, ys = t.transform([raw_minx, raw_maxx], [raw_miny, raw_maxy])
            global_minx, global_maxx = min(xs), max(xs)
            global_miny, global_maxy = min(ys), max(ys)
        else:
            global_minx, global_maxx = raw_minx, raw_maxx
            global_miny, global_maxy = raw_miny, raw_maxy
    except Exception:
        global_minx, global_maxx = raw_minx, raw_maxx
        global_miny, global_maxy = raw_miny, raw_maxy

    area_km2 = ((global_maxx - global_minx) / 1000) * ((global_maxy - global_miny) / 1000)
    cb(4, f"Oblast: {(global_maxx-global_minx)/1000:.1f} × {(global_maxy-global_miny)/1000:.1f} km (~{area_km2:.1f} km²)")

    # Výpočet dlaždic
    tiles = _compute_tiles(global_minx, global_maxx, global_miny, global_maxy,
                            TILE_SIZE_M, OVERLAP_M)
    n_tiles = len(tiles)
    cb(5, f"Zpracuji {n_tiles} dlaždic ({TILE_SIZE_M}×{TILE_SIZE_M}m)...")

    # OSM stáhnout jednou pro celou oblast
    cb(6, "Stahuji OSM data...")
    gdf_osm = None
    try:
        ox.settings.use_cache = True
        ox.settings.cache_folder = os.path.join(tempfile.gettempdir(), "OMapMaker_OSM")
        ox.settings.user_agent = "OMapMaker-Web-v7"
        ox.settings.timeout = 300
        to_wgs = Transformer.from_crs(CURRENT_CRS, "EPSG:4326", always_xy=True)
        buf = 300
        mn_lon, mn_lat = to_wgs.transform(global_minx - buf, global_miny - buf)
        mx_lon, mx_lat = to_wgs.transform(global_maxx + buf, global_maxy + buf)
        tags = {
            "highway": True, "building": True, "waterway": True, "natural": True,
            "landuse": True, "leisure": True, "railway": True, "power": True,
            "man_made": True, "barrier": True, "historic": True, "amenity": True,
            "aerialway": True, "water": True, "wetland": True, "military": True,
            "access": True, "bridge": True, "tunnel": True, "surface": True,
            "tracktype": True, "trail_visibility": True, "geological": True,
            "intermittent": True, "covered": True, "place": True, "emergency": True,
        }
        gdf_osm = ox.features_from_bbox((mn_lon, mn_lat, mx_lon, mx_lat), tags=tags)
        gdf_osm = gdf_osm.to_crs(CURRENT_CRS)
        cb(8, f"OSM staženo: {len(gdf_osm)} prvků")
    except Exception as e:
        cb(8, f"Varování OSM: {e}")

    # ZABAGED
    cb(9, "Načítám ZABAGED® soubory...")
    zabaged_gdfs = {}
    target_bbox_geom = box(global_minx, global_miny, global_maxx, global_maxy)
    for path in file_paths.get("zabaged", []):
        fname = os.path.basename(path)
        try:
            # Zjistíme CRS ze souboru — pokud chybí .prj, fallback na EPSG:5514
            try:
                with fiona.open(path) as src:
                    crs_wkt = src.crs_wkt
                file_crs = CRS.from_user_input(crs_wkt) if crs_wkt else CRS.from_epsg(5514)
            except Exception:
                file_crs = CRS.from_epsg(5514)

            crs_dst = CRS.from_user_input(CURRENT_CRS)
            file_bbox = None
            try:
                if file_crs != crs_dst:
                    t2 = Transformer.from_crs(crs_dst, file_crs, always_xy=True)
                    b = target_bbox_geom.bounds
                    tx, ty = t2.transform([b[0], b[2]], [b[1], b[3]])
                    file_bbox = (min(tx), min(ty), max(tx), max(ty))
                else:
                    file_bbox = target_bbox_geom.bounds
            except Exception:
                pass

            gdf_z = gpd.read_file(path, bbox=file_bbox) if file_bbox else gpd.read_file(path)

            # Přiřadíme CRS pokud soubor nemá .prj (gdf_z.crs bude None)
            if gdf_z.crs is None:
                gdf_z = gdf_z.set_crs(file_crs)

            if not gdf_z.empty and gdf_z.crs != crs_dst:
                gdf_z = gdf_z.to_crs(CURRENT_CRS)
            # Klíč bez přípony — ukládáme pod původním názvem i pod
            # normalizovaným (bez číslic na konci, bez podtržítek) aby
            # vector_layers.py mohl najít vrstvu bez ohledu na přesný název souboru.
            key = fname.rsplit(".", 1)[0]
            zabaged_gdfs[key] = gdf_z
        except Exception as e:
            print(f"[pipeline] ZABAGED {fname}: {e}")

    # ISOM
    isom_gdfs = {}
    for path in file_paths.get("isom", []):
        fname = os.path.basename(path)
        try:
            gdf_i = gpd.read_file(path)
            if not gdf_i.empty:
                gdf_i = gdf_i.set_crs(CURRENT_CRS) if gdf_i.crs is None else gdf_i.to_crs(CURRENT_CRS)
            isom_gdfs[fname.rsplit(".", 1)[0]] = gdf_i
            isom_gdfs[fname] = gdf_i
        except Exception as e:
            print(f"[pipeline] ISOM {fname}: {e}")

    # Zpracování dlaždic
    tile_results = []
    for i, tile_bbox in enumerate(tiles):
        result = _process_tile(
            tile_bbox, dmr_path, dmp_path, tile_params, sym_library,
            gdf_osm, zabaged_gdfs, isom_gdfs,
            i, n_tiles, cb,
        )
        tile_results.append(result)
        gc.collect()

    # Sloučení výsledků
    cb(80, "Slučuji výsledky dlaždic...")
    merged = _merge_tile_results(tile_results)
    del tile_results
    gc.collect()

    global_extent = (global_minx, global_maxx, global_miny, global_maxy)

    # Render
    cb(85, "Sestavuji mapu...")
    from .renderer import render_map
    output_png = os.path.join(output_dir, f"{job_id}_OMap.png")
    render_result = render_map(
        grid_x=None, grid_y=None,
        dmr_grid_cubic=None,
        dmr_grid_linear=None,
        gdf_vegetation=merged["gdf_vegetation"],
        gdf_rocks=merged["gdf_rocks"],
        contour_layers=merged["contour_layers"],
        depressions=merged["depressions"],
        knolls=merged["knolls"],
        gdf_osm=gdf_osm,
        zabaged_gdfs=zabaged_gdfs,
        isom_gdfs=isom_gdfs,
        extent=global_extent,
        clip_polygon=merged["clip_polygon"],
        sym_library=sym_library,
        current_crs=CURRENT_CRS,
        scale=SCALE,
        paper_format=PAPER_FORMAT,
        north_rotation=NORTH_ROTATION,
        layer_visibility=LAYER_VISIBILITY,
        output_png_path=output_png,
        progress_cb=lambda msg: cb(90, msg),
    )

    # GPKG
    gpkg_path = None
    cb(95, "Exportuji GPKG...")
    try:
        gpkg_path = os.path.join(output_dir, f"{job_id}_OOM.gpkg")
        collector = OomCollector(current_crs=CURRENT_CRS)

        # Vrstevnice
        for sym_key, layer_key in [
            ("sym101", "base"), ("sym102", "major"), ("sym103", "minor")
        ]:
            gdf_c = merged["contour_layers"].get(layer_key)
            if gdf_c is not None and not gdf_c.empty:
                collector.collect(sym_key, gdf_c)

        # Skály
        if merged["gdf_rocks"] is not None and not merged["gdf_rocks"].empty:
            collector.collect("sym201", merged["gdf_rocks"])

        # Mikrotvary
        if merged["depressions"]:
            collector.collect("sym111", gpd.GeoDataFrame(
                geometry=merged["depressions"], crs=CURRENT_CRS))
        if merged["knolls"]:
            collector.collect("sym109", gpd.GeoDataFrame(
                geometry=merged["knolls"], crs=CURRENT_CRS))

        # Vegetace — všechny třídy
        VEG_SYM = {
            "Paseka": "sym403", "Louka": "sym401",
            "Les": "sym405", "Vysoky_porost": "sym406",
            "Stredni_porost": "sym408", "Nizky_porost": "sym410",
        }
        veg = merged["gdf_vegetation"]
        if veg is not None and not veg.empty and "class_name" in veg.columns:
            for class_name, sym_key in VEG_SYM.items():
                subset = veg[veg["class_name"] == class_name]
                if not subset.empty:
                    collector.collect(sym_key, subset)

        # OSM vrstvy — voda, cesty, budovy, umělé prvky
        if gdf_osm is not None and not gdf_osm.empty:
            def col(c):
                import pandas as pd
                return gdf_osm[c].fillna("") if c in gdf_osm.columns else pd.Series(
                    [""] * len(gdf_osm), index=gdf_osm.index)

            gdf_lines = gdf_osm[gdf_osm.geometry.geom_type.isin(
                ["LineString", "MultiLineString"])].copy()
            gdf_polys = gdf_osm[gdf_osm.geometry.geom_type.isin(
                ["Polygon", "MultiPolygon"])].copy()
            gdf_pts = gdf_osm[gdf_osm.geometry.geom_type == "Point"].copy()

            # Voda
            collector.collect("sym301", gdf_polys[
                col("natural").isin(["lake", "water"]) |
                col("water").isin(["lake", "river", "reservoir"])])
            collector.collect("sym304", gdf_lines[
                col("waterway").isin(["river", "canal"])])
            collector.collect("sym305", gdf_lines[
                col("waterway").isin(["stream", "ditch"])])
            collector.collect("sym307", gdf_polys[col("wetland") == "reedbed"])
            collector.collect("sym308", gdf_polys[col("natural") == "wetland"])
            collector.collect("sym312", gdf_pts[col("natural") == "spring"])
            collector.collect("sym312", gdf_polys[col("natural") == "spring"])

            # Cesty
            collector.collect("sym502Da", gdf_lines[col("highway").isin(["motorway", "trunk"])])
            collector.collect("sym502a", gdf_lines[
                col("highway").isin(["primary", "secondary", "residential", "tertiary"])])
            collector.collect("sym503", gdf_lines[col("highway").isin(["service"])])
            collector.collect("sym504", gdf_lines[col("highway").isin(["track", "unclassified"])])
            collector.collect("sym505", gdf_lines[
                col("highway").isin(["footway", "pedestrian", "bridleway"])])
            collector.collect("sym506", gdf_lines[col("highway") == "path"])
            collector.collect("sym509a", gdf_lines[col("railway").isin(["rail", "narrow_gauge"])])

            # Budovy a umělé prvky
            collector.collect("sym521", gdf_polys[
                col("building").notna() & (col("building") != "")])
            collector.collect("sym510", gdf_lines[col("power").isin(["line", "minor_line"])])
            collector.collect("sym513-1a", gdf_lines[col("barrier") == "wall"])

            # Bodové prvky
            gdf_centroids = gdf_osm.copy()
            gdf_centroids["geometry"] = gdf_osm.geometry.centroid

            collector.collect("sym312", gdf_centroids[
                (col("natural") == "spring") & (col("covered") != "yes")])
            collector.collect("sym417a", gdf_centroids[col("natural") == "tree"])
            collector.collect("sym417b", gdf_centroids[col("natural") == "tree"])
            collector.collect("sym205", gdf_centroids[
                col("natural").isin(["stone", "rock"])])
            collector.collect("sym524a", gdf_centroids[
                col("man_made").isin(["tower", "chimney", "water_tower",
                                       "communications_tower", "mast"])])
            collector.collect("sym526a", gdf_centroids[
                col("historic").isin(["memorial", "boundary_stone", "wayside_cross"])])
            collector.collect("sym311", gdf_centroids[col("natural") == "sinkhole"])

        # ZABAGED vrstvy
        for zab_key, gdf_z in zabaged_gdfs.items():
            if gdf_z is None or gdf_z.empty:
                continue
            # Mapování názvů ZABAGED souborů na ISOM symboly
            ZAB_MAP = {
                "SilniceDalnice": "sym502Da", "Cesta": "sym504",
                "Pesina": "sym506", "ZeleznicniTrat": "sym509a",
                "VodniTok": "sym305", "VodniPlocha": "sym301",
                "BudovaJednotlivaNeboBlokBudov": "sym521",
                "ElektrickeVedeni": "sym510", "Zed": "sym513-1a",
                "Raseliniste": "sym307", "BazinaMocal": "sym308",
                "TrvalyTravniPorost": "sym401",
                "VyznamnyNeboOsamelyStromLesik": "sym417a",
                "OsamelyBalvanSkalaSkalniSuk": "sym205",
                "StupenSraz": "sym104", "HradbaValBastaOpevneni": "sym105-1a",
                "ZdrojPodzemnichVod": "sym312",
                "MohylaPomnikNahrobek": "sym526a",
            }
            sym_key = ZAB_MAP.get(zab_key)
            if sym_key:
                collector.collect(sym_key, gdf_z)

        # ISOM vlastní vrstvy
        for isom_key, gdf_i in isom_gdfs.items():
            if gdf_i is None or gdf_i.empty:
                continue
            # Klíč je název souboru = kód ISOM
            clean_key = isom_key.replace(".shp", "").replace(".SHP", "")
            collector.collect(clean_key, gdf_i)

        collector.export(gpkg_path)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[pipeline] GPKG chyba: {e}")
        gpkg_path = None

    elapsed = time.time() - start
    mins, secs = divmod(int(elapsed), 60)
    cb(100, f"Hotovo! Čas: {mins} min {secs} s · {n_tiles} dlaždic")

    return {
        "png_path": render_result["png_path"],
        "gpkg_path": gpkg_path,
        "world_file_path": render_result.get("world_file_path"),
    }