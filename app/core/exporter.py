"""
exporter.py — export výsledných vrstev do GeoPackage pro OpenOrienteering Mapper.
"""
import os
import re
import geopandas as gpd
import pandas as pd


def _oom_isom_code(sym_key: str) -> str | None:
    raw = sym_key[3:] if sym_key.startswith("sym") else sym_key
    m = re.match(r"^(\d+)", raw)
    return m.group(1) if m else None


class OomCollector:
    """Sbírá GeoDataFramy podle ISOM kódů pro pozdější export do GPKG."""

    def __init__(self, current_crs: str = "EPSG:5514"):
        self._layers: dict = {}
        self._crs = current_crs

    def collect(self, sym_key: str, gdf: gpd.GeoDataFrame):
        if gdf is None or gdf.empty:
            return
        code = _oom_isom_code(sym_key)
        if code is None:
            return
        if code not in self._layers:
            self._layers[code] = {"Point": [], "Line": [], "Polygon": []}
        for geom_type, geom_types in [
            ("Point", ["Point", "MultiPoint"]),
            ("Line", ["LineString", "MultiLineString"]),
            ("Polygon", ["Polygon", "MultiPolygon"]),
        ]:
            mask = gdf.geometry.geom_type.isin(geom_types)
            subset = gdf.loc[mask, ["geometry"]].copy()
            if not subset.empty:
                self._layers[code][geom_type].append(subset)

    def export(self, output_path: str):
        if not self._layers:
            print("[exporter] Žádné vrstvy k exportu.")
            return

        if os.path.exists(output_path):
            os.remove(output_path)

        SUFFIX = {"Point": "_point", "Line": "_line", "Polygon": "_poly"}
        written = 0

        for code in sorted(self._layers.keys(), key=lambda x: int(x)):
            buckets = self._layers[code]
            non_empty = {k: v for k, v in buckets.items() if v}
            if not non_empty:
                continue
            use_suffix = len(non_empty) > 1

            for geom_type, frames in non_empty.items():
                try:
                    merged = gpd.GeoDataFrame(
                        pd.concat(frames, ignore_index=True), crs=self._crs
                    )
                    merged = merged[merged.geometry.notna() & ~merged.geometry.is_empty]
                    if merged.empty:
                        continue
                    if geom_type == "Polygon":
                        merged.geometry = merged.geometry.buffer(0)
                        merged = merged[merged.geometry.is_valid & ~merged.geometry.is_empty]
                        if merged.empty:
                            continue
                    merged = merged.drop_duplicates(subset=["geometry"])
                    layer_name = f"isom_{code}{SUFFIX[geom_type] if use_suffix else ''}"
                    merged = merged[["geometry"]].copy()
                    merged["Layer"] = layer_name
                    merged.to_file(output_path, layer=layer_name, driver="GPKG")
                    print(f"[exporter] {layer_name}: {len(merged)} prvků [{geom_type}]")
                    written += 1
                except Exception as e:
                    print(f"[exporter] Chyba isom_{code} [{geom_type}]: {e}")

        print(f"[exporter] GPKG export: {written} vrstev → {output_path}")


def export_gpkg(layers: dict, output_path: str, current_crs: str = "EPSG:5514"):
    """Zkrácená verze pro přímé volání s dict {sym_key: GeoDataFrame}."""
    collector = OomCollector(current_crs)
    for sym_key, gdf in layers.items():
        collector.collect(sym_key, gdf)
    collector.export(output_path)
