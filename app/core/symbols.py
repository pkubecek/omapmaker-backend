"""
symbols.py — načítání symbols.xml a vykreslování ISOM symbolů.
Přepsáno z OMapMaker_v7.py.
"""
import os
import re
import xml.etree.ElementTree as ET
from ast import literal_eval
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.transforms import Affine2D
from matplotlib.patches import PathPatch
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union

try:
    from svgpath2mpl import parse_path
except ImportError:
    parse_path = None


CUSTOM_PLOT_KEYS = frozenset([
    "dotsize", "dotdistance", "dotcolor", "marker_shape",
    "facecolor_alt", "hatchdistance", "hatchcolor",
    "hatchwidth", "hatchstyle", "d", "path_d",
])


def _strip_custom_keys(props: dict):
    for k in CUSTOM_PLOT_KEYS:
        props.pop(k, None)


def pt2m(pt, scale=10_000):
    return pt * 0.0003527 * scale


class SymbolLibrary:
    def __init__(self, xml_path: str):
        self._lib = {}
        self._load(xml_path)

    def _load(self, xml_path: str):
        if not os.path.exists(xml_path):
            print(f"[symbols] Soubor {xml_path} nenalezen.")
            return
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for symbol in root.findall("symbol"):
                sid = symbol.get("id")
                stype = symbol.get("type")
                if not sid:
                    continue
                style_elem = symbol.find("style")
                props = style_elem.attrib.copy() if style_elem is not None else {}
                ticks_elem = symbol.find("style_ticks")
                if ticks_elem is not None:
                    for k, v in ticks_elem.attrib.items():
                        props[f"tick_{k}"] = v
                clean = {}
                for k, v in props.items():
                    val = str(v).strip()
                    if val.startswith("("):
                        try:
                            clean[k] = literal_eval(val)
                            continue
                        except Exception:
                            pass
                    try:
                        clean[k] = float(val.replace(",", "."))
                        continue
                    except ValueError:
                        pass
                    clean[k] = val

                path_obj = None
                path_d = None
                path_elem = symbol.find("path")
                if path_elem is not None and parse_path is not None:
                    path_d = path_elem.get("d")
                    if path_d:
                        try:
                            path_obj = parse_path(path_d)
                            ext = path_obj.get_extents()
                            center = (
                                (ext.xmin + ext.xmax) / 2,
                                (ext.ymin + ext.ymax) / 2,
                            )
                            path_obj.vertices -= center
                            path_obj.vertices[:, 1] *= -1
                        except Exception as e:
                            print(f"[symbols] Path parse chyba {sid}: {e}")
                            path_obj = None

                self._lib[sid] = {
                    "type": stype,
                    "props": clean,
                    "path": path_obj,
                    "path_d": path_d,
                }
            print(f"[symbols] Načteno {len(self._lib)} symbolů z {xml_path}")
        except Exception as e:
            print(f"[symbols] Chyba načítání XML: {e}")

    def get(self, key: str) -> dict | None:
        return self._lib.get(key)

    def has(self, key: str) -> bool:
        return key in self._lib


def plot_symbol(ax, sym_key: str, gdf: gpd.GeoDataFrame,
                zorder: float, sym_library: SymbolLibrary,
                current_crs: str = "EPSG:5514",
                dmr_grid=None, grid_x=None, grid_y=None):
    """Vykreslí GeoDataFrame pomocí ISOM symbolu ze sym_library."""
    if gdf is None or gdf.empty:
        return

    sym_data = sym_library.get(sym_key) if sym_library else None

    # Fallback — bez symbolu jen nakreslíme čáru/polygon základní barvou
    if sym_data is None:
        try:
            gdf.plot(ax=ax, zorder=zorder)
        except Exception:
            pass
        return

    sym_type = sym_data.get("type")
    sym_path = sym_data.get("path")
    sym_props = sym_data.get("props", {}).copy()

    if "solid_capstyle" in sym_props:
        sym_props["capstyle"] = sym_props.pop("solid_capstyle")

    # Bodové symboly (SVG path) — souřadnice v XML jsou přímo v mapových metrech
    if sym_type == "point" and sym_path is not None:
        _strip_custom_keys(sym_props)
        for geom in gdf.geometry:
            pts_list = []
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == "Point":
                pts_list.append((geom.x, geom.y))
            elif geom.geom_type == "MultiPoint":
                pts_list.extend([(p.x, p.y) for p in geom.geoms])
            for x, y in pts_list:
                t = Affine2D().translate(x, y) + ax.transData
                patch = PathPatch(sym_path, transform=t, zorder=zorder, **sym_props)
                ax.add_patch(patch)
        return

    # Hatch fill (dashed)
    if "hatchdistance" in sym_props:
        _plot_dashed_hatch(ax, gdf, sym_props, zorder)
        return

    # Dot fill
    if "dotdistance" in sym_props:
        _plot_dotted_hatch(ax, gdf, sym_props, zorder)
        return

    # sym510 — elektrické vedení: kolmé tiky ve vertexech
    if sym_key == "sym510":
        _strip_custom_keys(sym_props)
        gdf.plot(ax=ax, zorder=zorder, **sym_props)
        _plot_power_line_ticks(ax, gdf, sym_props, zorder)
        return

    # Tick marks (cliff symbols) — fousky po svahu pomocí DMR
    if "tick_length" in sym_props or sym_key in ("sym104", "sym201", "sym202"):
        _plot_with_ticks(ax, gdf, sym_props, zorder,
                         dmr_grid=dmr_grid, grid_x=grid_x, grid_y=grid_y)
        return

    # Standard line/polygon
    _strip_custom_keys(sym_props)
    try:
        gdf.plot(ax=ax, zorder=zorder, **sym_props)
    except Exception as e:
        print(f"[symbols] Chyba kreslení {sym_key}: {e}")


def _plot_dashed_hatch(ax, gdf, style_props, zorder):
    hatch_distance = style_props.pop("hatchdistance")
    hatch_color = style_props.pop("hatchcolor")
    hatch_width = style_props.pop("hatchwidth")
    hatch_style = style_props.pop("hatchstyle")
    _strip_custom_keys(style_props)
    gdf.plot(ax=ax, zorder=zorder, **style_props)
    try:
        all_geoms = unary_union(gdf.geometry)
    except Exception:
        all_geoms = gdf.geometry.buffer(0).unary_union
    if all_geoms.is_empty:
        return
    minx, miny, maxx, maxy = all_geoms.bounds
    step = pt2m(hatch_distance)
    if not step:
        return
    y_coords = np.arange(np.floor(miny / step) * step, maxy, step)
    h_lines = [LineString([(minx, y), (maxx, y)]) for y in y_coords]
    if not h_lines:
        return
    clipped = MultiLineString(h_lines).intersection(all_geoms)
    if not clipped.is_empty:
        gpd.GeoSeries([clipped]).plot(
            ax=ax, color=hatch_color, linewidth=hatch_width,
            linestyle=hatch_style, zorder=zorder - 0.1,
        )


def _plot_dotted_hatch(ax, gdf, style_props, zorder):
    dot_distance = style_props.pop("dotdistance")
    dot_size = style_props.pop("dotsize")
    dot_color = style_props.pop("dotcolor")
    _strip_custom_keys(style_props)
    gdf.plot(ax=ax, zorder=zorder, **style_props)
    try:
        all_geoms = gdf.geometry.union_all()
    except Exception:
        all_geoms = gdf.geometry.buffer(0).union_all()
    if all_geoms.is_empty:
        return
    minx, miny, maxx, maxy = all_geoms.bounds
    step = pt2m(dot_distance)
    if not step:
        return
    x_coords = np.arange(np.floor(minx / step) * step, maxx, step)
    y_coords = np.arange(np.floor(miny / step) * step, maxy, step)
    if not len(x_coords) or not len(y_coords):
        return
    xx, yy = np.meshgrid(x_coords, y_coords)
    fx, fy = xx.flatten(), yy.flatten()
    from shapely import prepare, contains_xy
    prepare(all_geoms)
    inside = contains_xy(all_geoms, fx, fy)
    if inside.any():
        ax.scatter(fx[inside], fy[inside], marker=".", color=dot_color,
                   s=dot_size, zorder=zorder + 0.1, edgecolors="none")



def _plot_power_line_ticks(ax, gdf, sym_props, zorder):
    """Kolmé tiky ve vertexech linie — pro elektrické vedení (sym510)."""
    tick_len = 10
    tick_segments = []

    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        parts = [geom] if geom.geom_type == "LineString" else list(getattr(geom, "geoms", [geom]))
        for line in parts:
            coords = np.array(line.coords)
            if len(coords) < 2:
                continue
            vectors = np.diff(coords, axis=0)
            norms = np.hypot(vectors[:, 0], vectors[:, 1])
            valid = norms > 0
            vectors = vectors[valid]
            norms = norms[valid]
            coords_clean = np.vstack([coords[0], coords[1:][valid]])
            if len(vectors) == 0:
                continue

            tangents = vectors / norms[:, None]
            for i in range(len(coords_clean)):
                x, y = coords_clean[i]
                if i == 0:
                    t = tangents[0]
                elif i == len(coords_clean) - 1:
                    t = tangents[-1]
                else:
                    t = tangents[i - 1] + tangents[i]
                    nm = np.hypot(t[0], t[1])
                    t = t / nm if nm != 0 else tangents[i - 1]

                nx, ny = -t[1], t[0]
                p1 = (x - nx * tick_len / 2, y - ny * tick_len / 2)
                p2 = (x + nx * tick_len / 2, y + ny * tick_len / 2)
                tick_segments.append([p1, p2])

    if tick_segments:
        lc = LineCollection(tick_segments,
                            colors=sym_props.get("color", "black"),
                            linewidths=sym_props.get("linewidth", 1.0),
                            zorder=zorder)
        ax.add_collection(lc)


def _plot_with_ticks(ax, gdf, sym_props, zorder, dmr_grid=None, grid_x=None, grid_y=None):
    tick_len = float(sym_props.pop("tick_length", 4))
    tick_space = float(sym_props.pop("tick_spacing", 4))
    tick_width = float(sym_props.pop("tick_linewidth", 0.3))
    tick_color = sym_props.pop("tick_color", sym_props.get("color", "black"))
    tick_angle = float(sym_props.pop("tick_angle", 90))  # stupně od tangenty; 90=kolmé, 45=šikmé
    _strip_custom_keys(sym_props)
    gdf.plot(ax=ax, zorder=zorder, **sym_props)

    ticks = []
    epsilon = 0.1

    # Připrav gradient DMR pro směr fousku (dolů po svahu)
    use_dmr = (dmr_grid is not None and grid_x is not None and grid_y is not None)
    if use_dmr:
        grad_x, grad_y = np.gradient(dmr_grid)
        min_x_g, max_x_g = grid_x.min(), grid_x.max()
        min_y_g, max_y_g = grid_y.min(), grid_y.max()
        shape_x, shape_y = grid_x.shape
        px_size_x = (max_x_g - min_x_g) / max(shape_x - 1, 1)
        px_size_y = (max_y_g - min_y_g) / max(shape_y - 1, 1)

    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        parts = [geom] if geom.geom_type == "LineString" else list(getattr(geom, "geoms", [geom]))
        for line in parts:
            if not hasattr(line, "length"):
                continue
            line_len = line.length
            if line_len < tick_space:
                continue
            for dist in np.arange(tick_space / 2, line_len, tick_space):
                pt = line.interpolate(dist)
                pt_ahead = line.interpolate(min(dist + epsilon, line_len))
                dx = pt_ahead.x - pt.x
                dy = pt_ahead.y - pt.y
                tan_len = np.hypot(dx, dy)
                if tan_len == 0:
                    continue
                tx, ty = dx / tan_len, dy / tan_len
                n1x, n1y = ty, -tx   # normála vlevo
                n2x, n2y = -ty, tx   # normála vpravo

                if use_dmr:
                    # Použij gradient DMR — fousky jdou dolů po svahu
                    ix = int((pt.x - min_x_g) / px_size_x)
                    iy = int((pt.y - min_y_g) / px_size_y)
                    if 0 <= ix < shape_x and 0 <= iy < shape_y:
                        gx = grad_x[ix, iy]
                        gy = grad_y[ix, iy]
                        if gx == 0 and gy == 0:
                            final_nx, final_ny = n1x, n1y
                        elif (n1x * gx) + (n1y * gy) < 0:
                            final_nx, final_ny = n1x, n1y
                        else:
                            final_nx, final_ny = n2x, n2y
                    else:
                        final_nx, final_ny = n1x, n1y
                else:
                    # Fallback bez DMR — použij tick_angle
                    a = np.radians(90 - tick_angle)
                    final_nx = tx * np.cos(a) + ty * np.sin(a)
                    final_ny = -tx * np.sin(a) + ty * np.cos(a)

                ticks.append([(pt.x, pt.y),
                               (pt.x + final_nx * tick_len, pt.y + final_ny * tick_len)])
    if ticks:
        lc = LineCollection(ticks, colors=tick_color, linewidths=tick_width, zorder=zorder)
        ax.add_collection(lc)