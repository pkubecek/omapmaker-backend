"""
pipeline.py — orchestrace celé analýzy jako jeden job.
Volá processor → renderer → exporter, reportuje průběh přes callback.
"""
import os
import time
import tempfile
import numpy as np
import rasterio.transform
import rasterio.features
import geopandas as gpd
import osmnx as ox
import fiona
from pyproj import CRS, Transformer
from shapely.geometry import box
from rasterio.features import rasterize

from .processor import (
    load_dmr_grid, load_dmp_grid,
    classify_vegetation, vectorize_rocks,
    find_depressions, find_knolls,
    make_clip_polygon,
)
from .renderer import render_map, generate_contour_layers
from .symbols import SymbolLibrary
from .exporter import export_gpkg


def run_pipeline(job_id: str, params: dict, file_paths: dict,
                 output_dir: str, progress_cb) -> dict:
    """
    Spustí celou analýzu synchronně.

    params: viz JobParams v routes/jobs.py
    file_paths: { dtm, dsm, zabaged: [...], isom: [...] }
    output_dir: složka pro výstupy tohoto jobu
    progress_cb(pct: int, msg: str): callback průběhu

    Vrací: { png_path, gpkg_path, world_file_path }
    """
    start = time.time()

    def cb(pct, msg):
        print(f"[pipeline {job_id}] {pct}% — {msg}")
        progress_cb(pct, msg)

    cb(1, "Spouštím analýzu...")

    CURRENT_CRS = params.get("crs", "EPSG:5514")
    SCALE = int(params.get("scale", 10000))
    PAPER_FORMAT = params.get("paper_format", "A4 (Landscape)")
    SIGMA = float(params.get("sigma", 6.5))
    SLOPE_THRESHOLD = float(params.get("slope_threshold", 45.0))
    NORTH_ROTATION = float(params.get("north_rotation", 5.0))
    BINS = [float(b) for b in params.get("bins", [1, 2, 6, 12])]
    LAYER_VISIBILITY = params.get("layers", {
        "contours": True, "rocks": True, "water": True,
        "vegetation": True, "roads": True, "buildings": True,
        "man_made": True, "magnetic_lines": False,
    })
    FIXED_PIXEL_SIZE = 0.5

    # Načti symbols.xml — hledáme vedle tohoto souboru nebo v CWD
    sym_xml = f"symbols{10 if SCALE == 10000 else 15}.xml"
    for candidate in [
        sym_xml,
        os.path.join(os.path.dirname(__file__), "..", "..", sym_xml),
        os.path.join(os.path.dirname(__file__), sym_xml),
    ]:
        if os.path.exists(candidate):
            sym_xml = candidate
            break
    sym_library = SymbolLibrary(sym_xml)

    dmr_path = file_paths["dtm"]
    dmp_path = file_paths["dsm"]

    # --- DTM ---
    cb(5, "Načítám DTM (bodové mračno)...")
    dmr_grid_cubic, grid_x, grid_y, extent, dmr_points, dmr_z = load_dmr_grid(
        dmr_path, CURRENT_CRS,
        pixel_size=FIXED_PIXEL_SIZE,
        sigma_smooth=SIGMA,
        progress_cb=lambda msg: cb(7, msg),
    )
    minx, maxx, miny, maxy = extent

    # Clip polygon z DTM
    cb(10, "Vytvářím ořezovou masku...")
    clip_polygon = make_clip_polygon(dmr_points)

    # Lineární interpolace DTM pro vizualizaci terénních prvků
    cb(12, "Interpoluji DTM (linear)...")
    from scipy.interpolate import griddata
    shift_x = np.mean(dmr_points[:, 0])
    shift_y = np.mean(dmr_points[:, 1])
    pts_sh = dmr_points - np.array([shift_x, shift_y])
    gx_sh = grid_x - shift_x
    gy_sh = grid_y - shift_y
    dmr_grid_linear = griddata(pts_sh, dmr_z, (gx_sh, gy_sh), method="linear")
    if np.isnan(dmr_grid_linear).all():
        dmr_grid_linear = griddata(pts_sh, dmr_z, (gx_sh, gy_sh), method="nearest")

    # --- DSM ---
    cb(15, "Načítám DSM...")
    dmp_grid = load_dmp_grid(
        dmp_path, grid_x, grid_y, extent, CURRENT_CRS,
        progress_cb=lambda msg: cb(18, msg),
    )

    # Výška vegetace
    cb(22, "Počítám výšku vegetace...")
    vegetation_height = np.clip(dmp_grid - dmr_grid_linear, 0, None)

    # Rasterio transform
    shape = grid_x.shape
    transform = rasterio.transform.from_bounds(
        minx, miny, maxx, maxy, width=shape[0], height=shape[1]
    )

    # OSM lesní maska
    cb(25, "Stahuji OSM data...")
    gdf_osm = None
    try:
        ox.settings.use_cache = True
        ox.settings.cache_folder = os.path.join(tempfile.gettempdir(), "OMapMaker_OSM")
        ox.settings.user_agent = "OMapMaker-Web-v7"
        ox.settings.timeout = 300
        download_buffer = 300
        to_wgs = Transformer.from_crs(CURRENT_CRS, "EPSG:4326", always_xy=True)
        mn_lon, mn_lat = to_wgs.transform(minx - download_buffer, miny - download_buffer)
        mx_lon, mx_lat = to_wgs.transform(maxx + download_buffer, maxy + download_buffer)
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
        if clip_polygon is not None:
            try:
                gdf_osm = gpd.clip(gdf_osm, clip_polygon)
            except Exception:
                pass
        cb(35, f"OSM staženo: {len(gdf_osm)} prvků")
    except Exception as e:
        cb(35, f"Varování OSM: {e}")

    # Lesní maska
    forest_mask = np.zeros(shape, dtype=np.uint8)
    if gdf_osm is not None and not gdf_osm.empty:
        try:
            from .processor import _get_col_safe
        except ImportError:
            def _get_col_safe(df, col):
                return df[col].fillna("") if col in df.columns else ""
        try:
            natural_col = gdf_osm["natural"].fillna("") if "natural" in gdf_osm.columns else ""
            landuse_col = gdf_osm["landuse"].fillna("") if "landuse" in gdf_osm.columns else ""
            forest_polys = gdf_osm[
                (natural_col == "wood") | (landuse_col == "forest")
            ].geometry
            if not forest_polys.empty:
                fm_t = rasterize(forest_polys, out_shape=(shape[1], shape[0]),
                                 transform=transform, fill=0, default_value=1, dtype=np.uint8)
                forest_mask = np.flipud(fm_t).T
        except Exception as e:
            print(f"[pipeline] Lesní maska: {e}")

    # Označení pasek
    b1 = BINS[0]
    is_clearing = ((vegetation_height < b1) & (vegetation_height >= 0)) & (forest_mask == 1)
    vegetation_height[is_clearing] = -1

    # Clip maska pro rastry
    cb(38, "Aplikuji ořezovou masku...")
    if clip_polygon is not None:
        try:
            cm_t = rasterize([(clip_polygon, 1)], out_shape=(shape[1], shape[0]),
                             transform=transform, fill=0, default_value=1, dtype=np.uint8)
            clip_mask_grid = np.flipud(cm_t).T.astype(bool)
            dmr_grid_linear_viz = np.nan_to_num(dmr_grid_linear, nan=0)
            dmr_grid_linear_viz[~clip_mask_grid] = 0
            dmr_grid_cubic_viz = np.nan_to_num(dmr_grid_cubic, nan=0)
            dmr_grid_cubic_viz[~clip_mask_grid] = np.nan
        except Exception as e:
            print(f"[pipeline] Clip maska: {e}")
            dmr_grid_linear_viz = np.nan_to_num(dmr_grid_linear, nan=0)
            dmr_grid_cubic_viz = np.nan_to_num(dmr_grid_cubic, nan=0)
    else:
        dmr_grid_linear_viz = np.nan_to_num(dmr_grid_linear, nan=0)
        dmr_grid_cubic_viz = np.nan_to_num(dmr_grid_cubic, nan=0)

    # Vegetace
    cb(42, "Klasifikuji vegetaci...")
    gdf_vegetation = classify_vegetation(
        vegetation_height, BINS, transform, dmr_path,
        progress_cb=lambda msg: cb(48, msg),
    )
    if gdf_vegetation is not None and not gdf_vegetation.empty:
        gdf_vegetation = gdf_vegetation.set_crs(CURRENT_CRS, allow_override=True)

    cb(52, "Vektorizuji skály...")
    gdf_rocks = vectorize_rocks(
        grid_x, grid_y, dmr_grid_linear_viz, transform,
        slope_threshold_deg=SLOPE_THRESHOLD,
        progress_cb=lambda msg: cb(55, msg),
    )
    if gdf_rocks is not None and not gdf_rocks.empty:
        gdf_rocks = gdf_rocks.set_crs(CURRENT_CRS, allow_override=True)

    cb(58, "Generuji vrstevnice...")
    contour_layers = generate_contour_layers(
        grid_x, grid_y, dmr_grid_cubic_viz,
        clip_polygon=clip_polygon,
        progress_cb=lambda msg: cb(62, msg),
    )
    for k, gdf_c in contour_layers.items():
        if not gdf_c.empty:
            contour_layers[k] = gdf_c.set_crs(CURRENT_CRS, allow_override=True)

    cb(65, "Hledám terénní mikrotvary...")
    depressions = find_depressions(
        grid_x, grid_y, dmr_grid_linear_viz,
        pixel_size=FIXED_PIXEL_SIZE, current_crs=CURRENT_CRS,
        progress_cb=lambda msg: cb(67, msg),
    )
    knolls = find_knolls(
        grid_x, grid_y, dmr_grid_linear_viz,
        pixel_size=FIXED_PIXEL_SIZE, current_crs=CURRENT_CRS,
        progress_cb=lambda msg: cb(69, msg),
    )

    # ZABAGED
    cb(70, "Načítám ZABAGED® soubory...")
    zabaged_gdfs = {}
    target_bbox = box(minx, miny, maxx, maxy)
    for path in file_paths.get("zabaged", []):
        fname = os.path.basename(path)
        try:
            with fiona.open(path) as src:
                file_crs = CRS.from_user_input(src.crs_wkt or "EPSG:5514")
            file_bbox = None
            try:
                crs_dst = CRS.from_user_input(CURRENT_CRS)
                if file_crs != crs_dst:
                    t2 = Transformer.from_crs(crs_dst, file_crs, always_xy=True)
                    b = target_bbox.bounds
                    tx, ty = t2.transform([b[0], b[2]], [b[1], b[3]])
                    file_bbox = (min(tx), min(ty), max(tx), max(ty))
                else:
                    file_bbox = target_bbox.bounds
            except Exception:
                pass
            gdf_z = gpd.read_file(path, bbox=file_bbox) if file_bbox else gpd.read_file(path)
            if not gdf_z.empty:
                gdf_z = gdf_z.to_crs(CURRENT_CRS)
                if clip_polygon:
                    gdf_z = gpd.clip(gdf_z, clip_polygon)
            zabaged_gdfs[fname.rsplit(".", 1)[0]] = gdf_z
            cb(70, f"ZABAGED načteno: {fname}")
        except Exception as e:
            print(f"[pipeline] Chyba ZABAGED {fname}: {e}")

    # ISOM
    isom_gdfs = {}
    for path in file_paths.get("isom", []):
        fname = os.path.basename(path)
        try:
            gdf_i = gpd.read_file(path)
            if not gdf_i.empty:
                if gdf_i.crs is None:
                    gdf_i = gdf_i.set_crs(CURRENT_CRS)
                else:
                    gdf_i = gdf_i.to_crs(CURRENT_CRS)
                if clip_polygon:
                    gdf_i = gpd.clip(gdf_i, clip_polygon)
            key = fname.rsplit(".", 1)[0]
            isom_gdfs[key] = gdf_i
            isom_gdfs[fname] = gdf_i
        except Exception as e:
            print(f"[pipeline] Chyba ISOM {fname}: {e}")

    # Render
    cb(75, "Sestavuji mapu...")
    output_png = os.path.join(output_dir, f"{job_id}_OMap.png")
    render_result = render_map(
        grid_x=grid_x, grid_y=grid_y,
        dmr_grid_cubic=dmr_grid_cubic_viz,
        dmr_grid_linear=dmr_grid_linear_viz,
        gdf_vegetation=gdf_vegetation,
        gdf_rocks=gdf_rocks,
        contour_layers=contour_layers,
        depressions=depressions,
        knolls=knolls,
        gdf_osm=gdf_osm,
        zabaged_gdfs=zabaged_gdfs,
        isom_gdfs=isom_gdfs,
        extent=extent,
        clip_polygon=clip_polygon,
        sym_library=sym_library,
        current_crs=CURRENT_CRS,
        scale=SCALE,
        paper_format=PAPER_FORMAT,
        north_rotation=NORTH_ROTATION,
        layer_visibility=LAYER_VISIBILITY,
        output_png_path=output_png,
        progress_cb=lambda msg: cb(88, msg),
    )

    # GPKG export
    gpkg_path = None
    cb(95, "Exportuji GPKG...")
    try:
        gpkg_path = os.path.join(output_dir, f"{job_id}_OOM.gpkg")
        # Sbírání vrstev pro export
        from .exporter import OomCollector
        collector = OomCollector(current_crs=CURRENT_CRS)
        for sym_key, gdf_c in [
            ("sym101", contour_layers.get("base")),
            ("sym102", contour_layers.get("major")),
            ("sym103", contour_layers.get("minor")),
            ("sym201", gdf_rocks),
            ("sym405", gdf_vegetation[gdf_vegetation["class_name"] == "Les"] if gdf_vegetation is not None and not gdf_vegetation.empty else None),
        ]:
            if gdf_c is not None and not gdf_c.empty:
                collector.collect(sym_key, gdf_c)
        if depressions:
            import geopandas as gpd2
            collector.collect("sym111", gpd2.GeoDataFrame(geometry=depressions, crs=CURRENT_CRS))
        if knolls:
            import geopandas as gpd2
            collector.collect("sym109", gpd2.GeoDataFrame(geometry=knolls, crs=CURRENT_CRS))
        collector.export(gpkg_path)
    except Exception as e:
        print(f"[pipeline] GPKG export chyba: {e}")
        gpkg_path = None

    elapsed = time.time() - start
    mins, secs = divmod(int(elapsed), 60)
    cb(100, f"Hotovo! Čas: {mins} min {secs} s")

    return {
        "png_path": render_result["png_path"],
        "gpkg_path": gpkg_path,
        "world_file_path": render_result.get("world_file_path"),
    }
