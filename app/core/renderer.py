"""
renderer.py — generování vrstevnic, vykreslení vektorových vrstev,
sestavení matplotlib figury a export do PNG + world file.
Přepsáno z OMapMaker_v7.py bez tkinter závislostí.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # bez GUI!
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.transforms import Affine2D
from matplotlib.patches import PathPatch, Polygon as MplPolygon
from skimage import measure
from shapely.geometry import (
    LineString, MultiLineString, Polygon, MultiPolygon,
    box, Point, MultiPoint,
)
from shapely.ops import unary_union
from shapely import affinity
import geopandas as gpd
import pandas as pd
from scipy.ndimage import binary_dilation, gaussian_filter
from pyproj import Transformer

from .symbols import SymbolLibrary, plot_symbol


SCALE_DEFAULT = 10000


def in2m(inch, scale=10_000):
    return inch * 0.0254 * scale


def pt2m(pt, scale=10_000):
    return pt * 0.0003527 * scale


# ---------------------------------------------------------------------------
# Contour generation
# ---------------------------------------------------------------------------

def _generate_contours_for_levels(padded_grid, levels, transform_info, clip_geom):
    """Vrátí GeoDataFrame liniových vrstevnic pro dané výšky."""
    min_x, min_y, px_step_x, px_step_y, pad = transform_info
    lines = []
    for level in levels:
        contours = measure.find_contours(padded_grid, level)
        for contour in contours:
            x_coords = min_x + (contour[:, 0] - pad) * px_step_x
            y_coords = min_y + (contour[:, 1] - pad) * px_step_y
            if len(x_coords) > 2:
                lines.append(LineString(np.column_stack((x_coords, y_coords))))
    if not lines:
        return gpd.GeoDataFrame(geometry=[], crs=None)
    gdf = gpd.GeoDataFrame(geometry=lines)
    if clip_geom is not None:
        try:
            gdf = gpd.clip(gdf, clip_geom)
        except Exception:
            pass
    return gdf


def generate_contour_layers(grid_x, grid_y, dmr_grid, clip_polygon=None,
                             progress_cb=None) -> dict:
    """
    Generuje vrstevnice (základní 5m, hlavní 25m, pomocné 2.5m).
    Vrátí dict: { 'base': GDF, 'major': GDF, 'minor': GDF }
    """
    def _cb(msg):
        if progress_cb:
            progress_cb(msg)

    _cb("Generuji vrstevnice...")

    valid_mask = (dmr_grid > 0) & (~np.isnan(dmr_grid))
    dmr_plot = np.where(valid_mask, dmr_grid, np.nan)

    pad = 100
    dmr_padded = np.pad(dmr_plot, pad_width=pad, mode="edge")

    min_x, max_x = grid_x.min(), grid_x.max()
    min_y, max_y = grid_y.min(), grid_y.max()
    px_step_x = (max_x - min_x) / (grid_x.shape[0] - 1)
    px_step_y = (max_y - min_y) / (grid_y.shape[1] - 1)
    transform_info = (min_x, min_y, px_step_x, px_step_y, pad)

    clip_geom = clip_polygon if clip_polygon is not None else box(min_x, min_y, max_x, max_y)

    min_z = np.nanmin(dmr_plot)
    max_z = np.nanmax(dmr_plot)

    major_levels = np.arange(np.floor(min_z / 25) * 25, np.ceil(max_z / 25) * 25 + 1, 25)
    base_levels = np.arange(np.floor(min_z / 5) * 5, np.ceil(max_z / 5) * 5 + 1, 5)
    base_levels = np.setdiff1d(base_levels, major_levels)

    # Pomocné vrstevnice — jen v oblastech s velkou křivostí a malým sklonem
    gy, gx_g = np.gradient(dmr_grid)
    gxx, _ = np.gradient(gx_g)
    _, gyy = np.gradient(gy)
    curvature = np.abs(gxx + gyy)
    slope = np.hypot(gx_g, gy)
    curvature_thr = np.percentile(curvature[valid_mask], 30)
    gentle_thr = np.percentile(slope[valid_mask], 30)
    combined_mask = (curvature > curvature_thr) & (slope < gentle_thr) & valid_mask
    dilated = binary_dilation(combined_mask, iterations=2)
    dmr_minor = np.where(dilated, dmr_plot, np.nan)
    dmr_padded_minor = np.pad(dmr_minor, pad_width=pad, mode="edge")

    minor_levels = np.arange(np.floor(min_z / 2.5) * 2.5, np.ceil(max_z / 2.5) * 2.5 + 1, 2.5)
    minor_levels = np.setdiff1d(minor_levels, np.union1d(major_levels, base_levels))

    return {
        "base": _generate_contours_for_levels(dmr_padded, base_levels, transform_info, clip_geom),
        "major": _generate_contours_for_levels(dmr_padded, major_levels, transform_info, clip_geom),
        "minor": _generate_contours_for_levels(dmr_padded_minor, minor_levels, transform_info, clip_geom),
    }


# ---------------------------------------------------------------------------
# Map figure setup
# ---------------------------------------------------------------------------

def setup_map_figure(extent, paper_format="A4 (Landscape)", scale=10_000):
    PAPER_SIZES_IN = {
        "A4 (Landscape)": (11.693, 8.268),
        "A4 (Portrait)": (8.268, 11.693),
        "A3 (Landscape)": (16.535, 11.693),
        "A3 (Portrait)": (11.693, 16.535),
    }
    minx_orig, maxx_orig, miny_orig, maxy_orig = extent

    if paper_format == "Data Extent":
        meters_per_inch = 0.0254 * scale
        fig_w = (maxx_orig - minx_orig) / meters_per_inch
        fig_h = (maxy_orig - miny_orig) / meters_per_inch
        minx, maxx, miny, maxy = minx_orig, maxx_orig, miny_orig, maxy_orig
    else:
        fig_w, fig_h = PAPER_SIZES_IN.get(paper_format, PAPER_SIZES_IN["A4 (Landscape)"])
        map_w = in2m(fig_w, scale)
        map_h = in2m(fig_h, scale)
        cx = (minx_orig + maxx_orig) / 2
        cy = (miny_orig + maxy_orig) / 2
        minx, maxx = cx - map_w / 2, cx + map_w / 2
        miny, maxy = cy - map_h / 2, cy + map_h / 2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.axis("off")

    return fig, ax, (minx, maxx, miny, maxy)


# ---------------------------------------------------------------------------
# Magnetic north lines
# ---------------------------------------------------------------------------

def add_magnetic_north_lines(ax, extent, scale, rotation=0.0, spacing_mm=30,
                              zorder=20, sym_library=None, current_crs="EPSG:5514"):
    spacing_m = (spacing_mm / 1000.0) * scale
    minx, maxx, miny, maxy = extent
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    diag = np.hypot(maxx - minx, maxy - miny)
    n = int(diag / spacing_m) + 2

    lines = []
    for i in range(-n, n + 1):
        x = cx + i * spacing_m
        line = LineString([(x, cy - diag), (x, cy + diag)])
        if rotation != 0:
            line = affinity.rotate(line, -rotation, origin=(cx, cy))
        lines.append(line)

    map_box = box(minx, miny, maxx, maxy)
    clipped = MultiLineString(lines).intersection(map_box)
    if clipped.is_empty:
        return
    visible = list(clipped.geoms) if hasattr(clipped, "geoms") else [clipped]
    gdf = gpd.GeoDataFrame(geometry=visible)

    if sym_library and sym_library.has("sym601"):
        plot_symbol(ax, "sym601", gdf, zorder, sym_library, current_crs)
    else:
        gdf.plot(ax=ax, color="#21d1ff", linewidth=0.35, zorder=zorder)


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_map(
    *,
    grid_x, grid_y,
    dmr_grid_cubic,
    dmr_grid_linear,
    gdf_vegetation,
    gdf_rocks,
    contour_layers,
    depressions,
    knolls,
    gdf_osm,
    zabaged_gdfs,
    isom_gdfs,
    extent,
    clip_polygon,
    sym_library,
    current_crs,
    scale,
    paper_format,
    north_rotation,
    layer_visibility,
    output_png_path,
    progress_cb=None,
) -> dict:
    """
    Sestaví matplotlib mapu a uloží ji jako PNG + world file.
    Vrátí { png_path, world_file_path }.
    """
    from .vector_layers import add_vector_layers

    def _cb(msg):
        if progress_cb:
            progress_cb(msg)

    _cb("Sestavuji mapu...")

    fig, ax, map_extent = setup_map_figure(extent, paper_format, scale)
    map_minx, map_maxx, map_miny, map_maxy = map_extent
    clip_box_map = box(map_minx, map_miny, map_maxx, map_maxy)

    # Dummy gridy pro add_vector_layers pokud nejsou k dispozici (tile mode)
    if grid_x is None or grid_y is None:
        _nx = max(10, int((map_maxx - map_minx) / 50))
        _ny = max(10, int((map_maxy - map_miny) / 50))
        grid_x, grid_y = np.mgrid[map_minx:map_maxx:complex(0, _nx),
                                    map_miny:map_maxy:complex(0, _ny)]
    if dmr_grid_linear is None:
        dmr_grid_linear = np.zeros_like(grid_x)
    if dmr_grid_cubic is None:
        dmr_grid_cubic = np.zeros_like(grid_x)

    # Magnetický sever
    if layer_visibility.get("magnetic_lines", False):
        add_magnetic_north_lines(ax, map_extent, scale, rotation=north_rotation,
                                  spacing_mm=30, zorder=50,
                                  sym_library=sym_library, current_crs=current_crs)

    # Clip path
    if clip_polygon:
        try:
            clip_patch = MplPolygon(
                np.array(clip_polygon.exterior.coords),
                transform=ax.transData,
            )
            ax.set_clip_path(clip_patch)
        except Exception:
            pass

    # Vegetace
    if layer_visibility.get("vegetation", True) and gdf_vegetation is not None and not gdf_vegetation.empty:
        _cb("Kreslím vegetaci...")
        VEG_MAP = {
            "Paseka": "sym403", "Louka": "sym401",
            "Les": "sym405", "Vysoky_porost": "sym406",
            "Stredni_porost": "sym408", "Nizky_porost": "sym410",
        }
        try:
            veg_clipped = gpd.clip(gdf_vegetation, clip_box_map)
            for class_name, sym_key in VEG_MAP.items():
                if class_name in veg_clipped.get("class_name", pd.Series()).values:
                    mask = veg_clipped["class_name"] == class_name
                    sub = veg_clipped[mask]
                    if not sub.empty:
                        plot_symbol(ax, sym_key, sub, 1.0 + list(VEG_MAP).index(class_name) * 0.1,
                                    sym_library, current_crs)
        except Exception as e:
            print(f"[renderer] Chyba vegetace: {e}")

    # Skály a vrstevnice
    if layer_visibility.get("rocks", True):
        if gdf_rocks is not None and not gdf_rocks.empty:
            _cb("Kreslím skály...")
            try:
                rocks_cl = gpd.clip(gdf_rocks, clip_box_map)
                if not rocks_cl.empty:
                    rocks_cl.plot(ax=ax, color="black", zorder=26)
            except Exception as e:
                print(f"[renderer] Chyba skal: {e}")

        _cb("Kreslím vrstevnice...")
        for sym_key, layer_key, zo in [
            ("sym101", "base", 25),
            ("sym102", "major", 25),
            ("sym103", "minor", 25),
        ]:
            gdf_c = contour_layers.get(layer_key)
            if gdf_c is not None and not gdf_c.empty:
                plot_symbol(ax, sym_key, gdf_c, zo, sym_library, current_crs)

    # Terénní mikrotvary
    if layer_visibility.get("contours", True):
        if depressions:
            _cb("Kreslím prohlubně...")
            dep_gdf = gpd.GeoDataFrame(geometry=depressions, crs=current_crs)
            plot_symbol(ax, "sym111", dep_gdf, 21, sym_library, current_crs)
        if knolls:
            _cb("Kreslím kupky...")
            kno_gdf = gpd.GeoDataFrame(geometry=knolls, crs=current_crs)
            plot_symbol(ax, "sym109", kno_gdf, 21, sym_library, current_crs)

    # Vektorové vrstvy (OSM + ZABAGED + ISOM)
    _cb("Kreslím OSM a ZABAGED® data...")
    if gdf_osm is not None and not gdf_osm.empty:
        add_vector_layers(
            ax=ax,
            gdf=gdf_osm,
            extent=map_extent,
            zabaged_gdfs=zabaged_gdfs,
            dmr_grid=dmr_grid_linear,
            grid_x=grid_x,
            grid_y=grid_y,
            visibility=layer_visibility,
            isom_gdfs=isom_gdfs,
            sym_library=sym_library,
            current_crs=current_crs,
        )

    # Uložení PNG
    _cb("Ukládám PNG (1000 DPI)...")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    transparent = paper_format == "Data Extent"
    plt.savefig(output_png_path, dpi=1000, bbox_inches="tight",
                pad_inches=0, transparent=transparent)
    plt.close(fig)

    # World file
    world_file_path = os.path.splitext(output_png_path)[0] + ".pgw"
    try:
        from PIL import Image
        with Image.open(output_png_path) as img:
            img_w, img_h = img.width, img.height
        psx = (map_maxx - map_minx) / img_w
        psy = (map_maxy - map_miny) / img_h
        content = f"{psx}\n0.0\n0.0\n{-psy}\n{map_minx + psx/2}\n{map_maxy - psy/2}\n"
        with open(world_file_path, "w") as f:
            f.write(content)
    except Exception as e:
        print(f"[renderer] World file error: {e}")
        world_file_path = None

    _cb("Mapa uložena.")
    return {"png_path": output_png_path, "world_file_path": world_file_path}
