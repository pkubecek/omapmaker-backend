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
                            # Vycentruj symbol kolem 0,0
                            ext = path_obj.get_extents()
                            center_x = (ext.xmin + ext.xmax) / 2
                            center_y = (ext.ymin + ext.ymax) / 2
                            path_obj.vertices[:, 0] -= center_x
                            path_obj.vertices[:, 1] -= center_y
                            # SVG má Y dolů, matplotlib má Y nahoru
                            # Invertujeme Y ale zachováme sweep flags pro arc
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
                current_crs: str = "EPSG:5514"):
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

    # Bodové symboly (SVG path)
    if sym_type == "point" and sym_path is not None:
        _strip_custom_keys(sym_props)
        # SVG souřadnice jsou v pt (body), potřebujeme převést na display units
        # Používáme ax.transData + scale faktor přes display transformation
        try:
            # Zjisti scale: 1 pt v SVG = kolik pixelů v display space
            # Pro mapu 1:10000 při 300 DPI: 1pt = 0.353mm = 3.53m v mapě
            # Ale jednodušší: použijeme Affine2D se scale
            # Symboly v XML jsou v mm * scale_factor
            # scale=1 znamená surové SVG jednotky jako display pts
            import matplotlib.transforms as mtransforms
            fig = ax.get_figure()
            dpi = fig.dpi
            # 1 pt = 1/72 inch = dpi/72 pixels
            pt_to_display = dpi / 72.0
            display_to_data = ax.transData.inverted()
            origin_display = ax.transData.transform([0, 0])
            one_pt_display = origin_display + np.array([pt_to_display, pt_to_display])
            one_pt_data = display_to_data.transform(one_pt_display) - display_to_data.transform(origin_display)
            sx = abs(one_pt_data[0])
            sy = abs(one_pt_data[1])
        except Exception:
            sx, sy = 1.0, 1.0

        import copy
        for geom in gdf.geometry:
            pts = []
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == "Point":
                pts.append((geom.x, geom.y))
            elif geom.geom_type == "MultiPoint":
                pts.extend([(p.x, p.y) for p in geom.geoms])
            for x, y in pts:
                scaled_path = copy.deepcopy(sym_path)
                scaled_path.vertices[:, 0] = scaled_path.vertices[:, 0] * sx + x
                scaled_path.vertices[:, 1] = scaled_path.vertices[:, 1] * sy + y
                patch = PathPatch(scaled_path, transform=ax.transData,
                                  zorder=zorder, **sym_props)
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

    # Tick marks (cliff symbols)
    if "tick_length" in sym_props or sym_key in ("sym104", "sym201", "sym202"):
        _plot_with_ticks(ax, gdf, sym_props, zorder)
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


def _plot_with_ticks(ax, gdf, sym_props, zorder):
    tick_len = float(sym_props.pop("tick_length", 4))
    tick_space = float(sym_props.pop("tick_spacing", 4))
    tick_width = float(sym_props.pop("tick_linewidth", 0.3))
    tick_color = sym_props.pop("tick_color", sym_props.get("color", "black"))
    _strip_custom_keys(sym_props)
    gdf.plot(ax=ax, zorder=zorder, **sym_props)

    ticks = []
    epsilon = 0.1
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
                n1x, n1y = ty, -tx
                ticks.append([(pt.x, pt.y), (pt.x + n1x * tick_len, pt.y + n1y * tick_len)])
    if ticks:
        lc = LineCollection(ticks, colors=tick_color, linewidths=tick_width, zorder=zorder)
        ax.add_collection(lc)