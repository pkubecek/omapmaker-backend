"""
vector_layers.py — mapování OSM / ZABAGED® / vlastních ISOM vrstev na symboly.
Přepsáno z OMapMaker_v7.py.
"""
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from .symbols import SymbolLibrary, plot_symbol


def _get_col(df, col):
    if col in df.columns:
        return df[col].fillna("")
    return pd.Series([""] * len(df), index=df.index)


def _clip(gdf, extent):
    if gdf is None or gdf.empty:
        return gdf
    try:
        return gpd.clip(gdf, box(extent[0], extent[2], extent[1], extent[3]))
    except Exception:
        return gdf


def add_vector_layers(
    ax, gdf, extent, zabaged_gdfs, dmr_grid, grid_x, grid_y,
    visibility, isom_gdfs, sym_library: SymbolLibrary, current_crs: str,
):
    """Vykreslí OSM + ZABAGED + ISOM vrstvy na ax podle visibility."""

    gdf = _clip(gdf, extent)
    for k in list(zabaged_gdfs.keys()):
        zabaged_gdfs[k] = _clip(zabaged_gdfs[k], extent)

    if (gdf is None or gdf.empty) and not zabaged_gdfs and not isom_gdfs:
        return

    def pm(sym_key, zorder, mask, src_gdf, to_mask=True):
        """Pomocná: vybere subset a pošle do plot_symbol."""
        if src_gdf is None or src_gdf.empty:
            return
        if to_mask:
            if mask is None:
                return
            if isinstance(mask, (pd.Series, gpd.GeoSeries)):
                mask = mask.reindex(src_gdf.index).fillna(False)
            subset = src_gdf[mask].copy()
        else:
            subset = src_gdf.copy()
        if subset.empty:
            return
        plot_symbol(ax, sym_key, subset, zorder, sym_library, current_crs)

    # Sloupce OSM
    _c = {col: _get_col(gdf, col) for col in [
        "access", "amenity", "barrier", "bridge", "building", "covered",
        "emergency", "geological", "highway", "historic", "intermittent",
        "landuse", "leisure", "man_made", "military", "natural", "parking",
        "place", "power", "railway", "surface", "tracktype", "tunnel",
        "water", "waterway", "wetland", "aerialway",
    ]} if gdf is not None and not gdf.empty else {}

    def c(col):
        return _c.get(col, pd.Series(dtype=str))

    # Geometry subsets
    if gdf is not None and not gdf.empty:
        gdf_pts = gdf[gdf.geometry.geom_type.isin(["Point"])].copy()
        gdf_lines = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
        gdf_polys = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        gdf_centroids = gdf.copy()
        gdf_centroids["geometry"] = gdf_centroids.geometry.centroid
    else:
        gdf_pts = gdf_lines = gdf_polys = gdf_centroids = gpd.GeoDataFrame()

    def zab(key):
        return zabaged_gdfs.get(key)

    def isom(key):
        return isom_gdfs.get(key)

    # ----------------------------------------------------------------
    # TERRAIN
    # ----------------------------------------------------------------
    if visibility.get("contours", True):
        for code, sym, zo in [("104", "sym104", 21), ("105", "sym105-1a", 21),
                               ("107", "sym107", 20), ("108", "sym108", 21),
                               ("109", "sym109", 21), ("111", "sym111", 21),
                               ("112", "sym112", 21)]:
            cgdf = isom(code)
            if cgdf is not None:
                pm(sym, zo, None, cgdf, to_mask=False)
            elif code == "104" and zab("StupenSraz"):
                pm("sym104", zo, None, zab("StupenSraz"), to_mask=False)
            elif code == "104":
                pm("sym104", zo, c("man_made") == "embankment", gdf_lines)

        if isom("105") is None and zab("HradbaValBastaOpevneni"):
            for s in ["sym105-1a", "sym105-1b"]:
                pm(s, 30, None, zab("HradbaValBastaOpevneni"), to_mask=False)

    # ----------------------------------------------------------------
    # ROCKS
    # ----------------------------------------------------------------
    if visibility.get("rocks", True):
        for code, sym, zo in [("203.1", "sym203-1", 56), ("204", "sym204", 56),
                               ("205", "sym205", 56), ("207", "sym207", 56),
                               ("208", "sym208", 18), ("209", "sym209", 18),
                               ("210", "sym210", 18), ("213", "sym213", 15)]:
            cgdf = isom(code)
            if cgdf is not None:
                pm(sym, zo, None, cgdf, to_mask=False)

        if isom("205") is None and zab("OsamelyBalvanSkalaSkalniSuk"):
            pm("sym205", 56, None, zab("OsamelyBalvanSkalaSkalniSuk"), to_mask=False)
        elif isom("205") is None:
            pm("sym205", 56, c("natural").isin(["stone", "rock"]), gdf_centroids)

        if isom("215") is None:
            mask_ditch = c("barrier").isin(["ditch"]) | c("military").isin(["trench"])
            pm("sym215a", 21, mask_ditch, gdf_lines)
            pm("sym215b", 21, mask_ditch, gdf_lines)

    # ----------------------------------------------------------------
    # WATER
    # ----------------------------------------------------------------
    if visibility.get("water", True):
        # 301 Vodní plocha
        cgdf = isom("301")
        if cgdf is not None:
            pm("sym301", 27, None, cgdf, to_mask=False)
        elif zab("VodniPlocha"):
            pm("sym301", 27, None, zab("VodniPlocha"), to_mask=False)
        else:
            pm("sym301", 27,
               c("natural").isin(["lake", "water"]) | c("water").isin(["lake", "river", "reservoir"]),
               gdf_polys)

        # 304 Řeka
        cgdf = isom("304")
        if cgdf is not None:
            pm("sym304", 26, None, cgdf, to_mask=False)
        elif zab("VodniTok"):
            mask = _get_col(zab("VodniTok"), "typtoku_p").isin(["povrchový splavný"]) & \
                   _get_col(zab("VodniTok"), "vydattok_p").isin(["stálý"])
            pm("sym304", 26, mask, zab("VodniTok"))
        else:
            pm("sym304", 26,
               c("waterway").isin(["river", "canal"]) & ~c("tunnel").isin(["yes", "culvert"]),
               gdf_lines)

        # 305 Potok
        cgdf = isom("305")
        if cgdf is not None:
            pm("sym305", 26, None, cgdf, to_mask=False)
        elif zab("VodniTok"):
            mask = _get_col(zab("VodniTok"), "typtoku_p").isin(["povrchový nesplavný"]) & \
                   _get_col(zab("VodniTok"), "vydattok_p").isin(["stálý"])
            pm("sym305", 26, mask, zab("VodniTok"))
        else:
            pm("sym305", 26,
               c("waterway").isin(["stream", "ditch"]) & ~c("tunnel").isin(["yes", "culvert"]),
               gdf_lines)

        # 307/308 Bažina
        for code, sym, zo in [("307", "sym307", 25), ("308", "sym308", 25)]:
            cgdf = isom(code)
            if cgdf is not None:
                pm(sym, zo, None, cgdf, to_mask=False)

        if isom("307") is None and zab("Raseliniste"):
            pm("sym307", 25, None, zab("Raseliniste"), to_mask=False)
        elif isom("307") is None:
            pm("sym307", 25, c("wetland") == "reedbed", gdf_polys)

        if isom("308") is None and zab("BazinaMocal"):
            pm("sym308", 25, None, zab("BazinaMocal"), to_mask=False)
        elif isom("308") is None:
            pm("sym308", 25, c("natural") == "wetland", gdf_polys)

        # 312 Pramen
        cgdf = isom("312")
        if cgdf is not None:
            pm("sym312", 52, None, cgdf, to_mask=False)
        elif zab("ZdrojPodzemnichVod"):
            pm("sym312", 52, None, zab("ZdrojPodzemnichVod"), to_mask=False)
        else:
            pm("sym312", 52, (c("natural") == "spring") & (c("covered") != "yes"), gdf_centroids)

    # ----------------------------------------------------------------
    # VEGETATION
    # ----------------------------------------------------------------
    if visibility.get("vegetation", True):
        for code, sym, zo in [("401", "sym401", 1.0), ("402", "sym402", 1.0),
                               ("412", "sym412a", 1.9), ("413", "sym413", 1.9),
                               ("417", "sym417a", 54)]:
            cgdf = isom(code)
            if cgdf is not None:
                pm(sym, zo, None, cgdf, to_mask=False)

        if isom("401") is None and zab("TrvalyTravniPorost"):
            pm("sym401", 1.0, None, zab("TrvalyTravniPorost"), to_mask=False)
        elif isom("401") is None:
            pm("sym401", 1.0,
               c("landuse").isin(["grassland", "grass", "meadow"]) | c("natural").isin(["grassland"]),
               gdf_polys)

        if isom("412") is None and zab("OrnaPudaAOstatniDaleNespecifikovanePlochy"):
            mask = _get_col(zab("OrnaPudaAOstatniDaleNespecifikovanePlochy"), "typ_pudy_p").isin(["orná půda"])
            pm("sym412a", 1.9, mask, zab("OrnaPudaAOstatniDaleNespecifikovanePlochy"))
        elif isom("412") is None:
            pm("sym412a", 1.9, c("landuse") == "farmland", gdf_polys)

        if isom("417") is None and zab("VyznamnyNeboOsamelyStromLesik"):
            pm("sym417a", 54, None, zab("VyznamnyNeboOsamelyStromLesik"), to_mask=False)
            pm("sym417b", 55, None, zab("VyznamnyNeboOsamelyStromLesik"), to_mask=False)
        elif isom("417") is None:
            pm("sym417a", 54, c("natural") == "tree", gdf_centroids)
            pm("sym417b", 55, c("natural") == "tree", gdf_centroids)

    # ----------------------------------------------------------------
    # ROADS
    # ----------------------------------------------------------------
    if visibility.get("roads", True):
        # 502D Dálnice
        mask_motorway = c("highway").isin(["motorway", "trunk"]) & \
                        ~c("tunnel").isin(["yes"]) & (c("bridge") != "yes")
        cgdf = isom("502D")
        for sym, zo in [("sym502Da", 45), ("sym502Db", 47), ("sym502Dc", 48)]:
            if cgdf is not None:
                pm(sym, zo, None, cgdf, to_mask=False)
            elif zab("SilniceDalnice"):
                mask = _get_col(zab("SilniceDalnice"), "typsil_k").isin(["D1", "D2", "M"])
                pm(sym, zo, mask, zab("SilniceDalnice"))
            else:
                pm(sym, zo, mask_motorway, gdf_lines)

        # 502 Silnice
        mask_road = c("highway").isin(["primary", "secondary", "residential", "tertiary"]) & \
                    ~c("tunnel").isin(["yes"]) & (c("bridge") != "yes")
        cgdf = isom("502")
        for sym, zo in [("sym502a", 45), ("sym502b", 47)]:
            if cgdf is not None:
                pm(sym, zo, None, cgdf, to_mask=False)
            elif zab("SilniceDalnice"):
                mask = ~_get_col(zab("SilniceDalnice"), "typsil_k").isin(["D1", "D2", "M"])
                pm(sym, zo, mask, zab("SilniceDalnice"))
            else:
                pm(sym, zo, mask_road, gdf_lines)

        # 503 Vozová cesta
        cgdf = isom("503")
        if cgdf is not None:
            pm("sym503", 45, None, cgdf, to_mask=False)
        elif zab("Cesta"):
            mask = _get_col(zab("Cesta"), "povrch_p").isin(
                ["zpevněný (asfalt, beton)", "zpevněný (panel, dlažba)"])
            pm("sym503", 45, mask, zab("Cesta"))
        else:
            pm("sym503", 45,
               c("highway").isin(["service", "tertiary_link"]) & ~c("tunnel").isin(["yes"]),
               gdf_lines)

        # 504 Cesta
        cgdf = isom("504")
        if cgdf is not None:
            pm("sym504", 45, None, cgdf, to_mask=False)
        elif zab("Cesta"):
            mask = _get_col(zab("Cesta"), "typcesty_p").isin(["cesta udržovaná"])
            pm("sym504", 45, mask, zab("Cesta"))
        else:
            pm("sym504", 45,
               c("highway").isin(["track", "unclassified"]) & ~c("tunnel").isin(["yes"]),
               gdf_lines)

        # 505 Pěší cesta
        cgdf = isom("505")
        if cgdf is not None:
            pm("sym505", 45, None, cgdf, to_mask=False)
        else:
            pm("sym505", 45,
               c("highway").isin(["footway", "pedestrian", "bridleway"]) & (c("bridge") != "yes"),
               gdf_lines)

        # 506 Pěšina
        cgdf = isom("506")
        if cgdf is not None:
            pm("sym506", 45, None, cgdf, to_mask=False)
        elif zab("Pesina"):
            pm("sym506", 45, None, zab("Pesina"), to_mask=False)
        else:
            pm("sym506", 45, c("highway") == "path", gdf_lines)

        # 509 Železnice
        mask_rail = c("railway").isin(["rail", "narrow_gauge"]) & ~c("tunnel").isin(["yes"])
        cgdf = isom("509")
        for sym, zo in [("sym509a", 40), ("sym509b", 41)]:
            if cgdf is not None:
                pm(sym, zo, None, cgdf, to_mask=False)
            elif zab("ZeleznicniTrat"):
                pm(sym, zo, None, zab("ZeleznicniTrat"), to_mask=False)
            else:
                pm(sym, zo, mask_rail, gdf_lines)

    # ----------------------------------------------------------------
    # MAN-MADE
    # ----------------------------------------------------------------
    if visibility.get("man_made", True):
        # 510 El. vedení
        cgdf = isom("510")
        if cgdf is not None:
            pm("sym510", 70, None, cgdf, to_mask=False)
        elif zab("ElektrickeVedeni"):
            pm("sym510", 70, None, zab("ElektrickeVedeni"), to_mask=False)
        else:
            pm("sym510", 70, c("power").isin(["line", "minor_line"]), gdf_lines)

        # 513 Zeď
        cgdf = isom("513.1")
        if cgdf is not None:
            for s in ["sym513-1a", "sym513-1b"]:
                pm(s, 30, None, cgdf, to_mask=False)
        elif zab("Zed"):
            for s in ["sym513-1a", "sym513-1b"]:
                pm(s, 30, None, zab("Zed"), to_mask=False)
        else:
            pm("sym513-1a", 30, c("barrier") == "wall", gdf_lines)

        # 521 Budova
        if visibility.get("buildings", True):
            cgdf = isom("521")
            if cgdf is not None:
                pm("sym521", 50, None, cgdf, to_mask=False)
            elif zab("BudovaJednotlivaNeboBlokBudov"):
                pm("sym521", 50, None, zab("BudovaJednotlivaNeboBlokBudov"), to_mask=False)
            else:
                pm("sym521", 50,
                   c("building").notna() & (c("building") != "") & ~c("building").isin(["roof", "ruins"]),
                   gdf_polys)

        # 520 Privátní oblast
        if visibility.get("private", True):
            cgdf = isom("520")
            if cgdf is not None:
                pm("sym520", 1.5, None, cgdf, to_mask=False)
            else:
                for zk in ["Hrbitov", "Letiste", "ArealUceloveZastavby"]:
                    if zab(zk) is not None:
                        pm("sym520", 1.5, None, zab(zk), to_mask=False)
                        break
                else:
                    pm("sym520", 1.5,
                       c("landuse").isin(["residential", "industrial", "commercial",
                                          "cemetery", "military", "quarry"]),
                       gdf_polys)

        # 524 Věž
        cgdf = isom("524")
        if cgdf is not None:
            for s in ["sym524a", "sym524b"]:
                pm(s, 56, None, cgdf, to_mask=False)
        else:
            mask_tower = c("man_made").isin(["tower", "chimney", "water_tower",
                                              "communications_tower", "mast"])
            pm("sym524a", 56, mask_tower, gdf_pts)
            pm("sym524b", 56, mask_tower, gdf_pts)

        # 526 Pomník
        cgdf = isom("526")
        if cgdf is not None:
            for s in ["sym526a", "sym526b"]:
                pm(s, 56, None, cgdf, to_mask=False)
        elif zab("MohylaPomnikNahrobek"):
            for s in ["sym526a", "sym526b"]:
                pm(s, 56, None, zab("MohylaPomnikNahrobek"), to_mask=False)
        else:
            pm("sym526a", 56,
               c("historic").isin(["memorial", "boundary_stone", "wayside_cross"]),
               gdf_centroids)

    print("[vector_layers] Vše vykresleno")
