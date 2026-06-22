"""
vector_layers.py — mapování OSM / ZABAGED® / vlastních ISOM vrstev na symboly.
Přepsáno z OMapMaker_v7.py — logika filtrování odpovídá originálu.
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
    print(f"[vector_layers] extent={extent}")
    print(f"[vector_layers] ZABAGED klíče: {list(zabaged_gdfs.keys())}")
    for k, v in zabaged_gdfs.items():
        if v is not None:
            print(f"[vector_layers]   {k}: {len(v)} prvků, CRS={v.crs}, bounds={v.total_bounds}")
        else:
            print(f"[vector_layers]   {k}: None")
    print(f"[vector_layers] visibility={visibility}")

    gdf = _clip(gdf, extent)
    for k in list(zabaged_gdfs.keys()):
        before = len(zabaged_gdfs[k]) if zabaged_gdfs[k] is not None else 0
        zabaged_gdfs[k] = _clip(zabaged_gdfs[k], extent)
        after = len(zabaged_gdfs[k]) if zabaged_gdfs[k] is not None and not zabaged_gdfs[k].empty else 0
        print(f"[vector_layers]   clip {k}: {before} → {after} prvků")

    if (gdf is None or gdf.empty) and not zabaged_gdfs and not isom_gdfs:
        return

    _CLIFF_SYMS = {"sym104", "sym201", "sym202"}

    def pm(sym_key, zorder, mask, src_gdf, to_mask=True):
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
        # Pro cliff symboly předej DMR grid pro správný směr fousku
        if sym_key in _CLIFF_SYMS:
            plot_symbol(ax, sym_key, subset, zorder, sym_library, current_crs,
                        dmr_grid=dmr_grid, grid_x=grid_x, grid_y=grid_y)
        else:
            plot_symbol(ax, sym_key, subset, zorder, sym_library, current_crs)

    if gdf is not None and not gdf.empty:
        gdf = gdf.reset_index(drop=True)

    _c = {col: _get_col(gdf, col) for col in [
        "access", "amenity", "barrier", "bridge", "building", "covered",
        "emergency", "geological", "highway", "historic", "intermittent",
        "landuse", "leisure", "man_made", "military", "natural", "parking",
        "place", "power", "railway", "surface", "tracktype", "tunnel",
        "water", "waterway", "wetland", "aerialway", "trail_visibility",
    ]} if gdf is not None and not gdf.empty else {}

    def c(col):
        return _c.get(col, pd.Series(dtype=str))

    if gdf is not None and not gdf.empty:
        geom_type = gdf.geometry.geom_type
        gdf_pts       = gdf[geom_type == "Point"].copy()
        gdf_lines     = gdf[geom_type.isin(["LineString", "MultiLineString"])].copy()
        gdf_polys     = gdf[geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        gdf_centroids = gdf.copy()
        gdf_centroids["geometry"] = gdf_centroids.geometry.centroid
    else:
        gdf_pts = gdf_lines = gdf_polys = gdf_centroids = gpd.GeoDataFrame()

    def zab(key):
        gdf_z = zabaged_gdfs.get(key)
        if gdf_z is None or gdf_z.empty:
            return None
        return gdf_z

    def isom(key):
        gdf_i = isom_gdfs.get(key)
        if gdf_i is None or gdf_i.empty:
            return None
        return gdf_i

    def zab_in(*keys):
        return any(zab(k) is not None for k in keys)

    # ----------------------------------------------------------------
    # TERRAIN
    # ----------------------------------------------------------------
    if visibility.get("contours", True):
        # 104 - Zemní sráz
        cgdf = isom("104")
        if cgdf is not None:
            pm("sym104", 21, None, cgdf, to_mask=False)
        elif zab("StupenSraz") is not None:
            pm("sym104", 21, None, zab("StupenSraz"), to_mask=False)
        else:
            pm("sym104", 21, c("man_made") == "embankment", gdf_lines)

        # 105 - Zemní val
        cgdf = isom("105")
        if cgdf is not None:
            pm("sym105-1a", 21, None, cgdf, to_mask=False)
            pm("sym105-1b", 21, None, cgdf, to_mask=False)
        elif zab("HradbaValBastaOpevneni") is not None:
            pm("sym105-1a", 30, None, zab("HradbaValBastaOpevneni"), to_mask=False)
            pm("sym105-1b", 30, None, zab("HradbaValBastaOpevneni"), to_mask=False)

        # 107 - Rokle / výmol
        cgdf = isom("107")
        if cgdf is not None:
            pm("sym107", 20, None, cgdf, to_mask=False)
        elif zab("RokleVymol") is not None:
            pm("sym107", 20, None, zab("RokleVymol"), to_mask=False)

        # 108, 109, 111, 112 - jen ISOM
        for code, sym, zo in [("108", "sym108", 21), ("109", "sym109", 21),
                               ("111", "sym111", 21), ("112", "sym112", 21)]:
            cgdf = isom(code)
            if cgdf is not None:
                pm(sym, zo, None, cgdf, to_mask=False)

    # ----------------------------------------------------------------
    # ROCKS
    # ----------------------------------------------------------------
    if visibility.get("rocks", True):
        # 203-1 - Jeskyně
        cgdf = isom("203.1")
        if cgdf is not None:
            pm("sym203-1", 56, None, cgdf, to_mask=False)
        if zab("VstupDoJeskyne") is not None:
            pm("sym203-1", 56, None, zab("VstupDoJeskyne"), to_mask=False)
        else:
            pm("sym203-1", 56,
               c("natural").isin(["cave_entrance"]) | (c("man_made") == "adit"),
               gdf_centroids)

        # 203-2
        cgdf = isom("203.2")
        if cgdf is not None:
            pm("sym203-2", 56, None, cgdf, to_mask=False)

        # 204 - Malý balvan
        cgdf = isom("204")
        if cgdf is not None:
            pm("sym204", 56, None, cgdf, to_mask=False)

        # 205 - Balvan
        cgdf = isom("205")
        if cgdf is not None:
            pm("sym205", 56, None, cgdf, to_mask=False)
        elif zab("OsamelyBalvanSkalaSkalniSuk") is not None:
            pm("sym205", 56, None, zab("OsamelyBalvanSkalaSkalniSuk"), to_mask=False)
        else:
            pm("sym205", 56,
               c("natural").isin(["stone", "rock"]) | (c("geological") == "glacial_erratic"),
               gdf_centroids)

        # 207 - Skupina balvanů
        cgdf = isom("207")
        if cgdf is not None:
            pm("sym207", 56, None, cgdf, to_mask=False)
        elif zab("SkupinaBalvanu_b") is not None:
            pm("sym207", 56, None, zab("SkupinaBalvanu_b"), to_mask=False)

        # 208, 209
        for code, sym in [("208", "sym208"), ("209", "sym209")]:
            cgdf = isom(code)
            if cgdf is not None:
                pm(sym, 18, None, cgdf, to_mask=False)

        # 210 - Suťoviště
        cgdf = isom("210")
        if cgdf is not None:
            pm("sym210", 18, None, cgdf, to_mask=False)
        else:
            pm("sym210", 18, c("natural") == "blockfield", gdf_polys)

        # 211, 212 → sym210
        for code in ["211", "212"]:
            cgdf = isom(code)
            if cgdf is not None:
                pm("sym210", 18, None, cgdf, to_mask=False)

        # 213 - Písek
        cgdf = isom("213")
        if cgdf is not None:
            pm("sym213", 15, None, cgdf, to_mask=False)
        else:
            pm("sym213", 15, c("natural").isin(["sand", "dune"]), gdf_polys)

        # 215 - Příkop
        cgdf = isom("215")
        mask_ditch = (c("barrier") == "ditch") | (c("military") == "trench")
        if cgdf is not None:
            pm("sym215a", 21, None, cgdf, to_mask=False)
            pm("sym215b", 21, None, cgdf, to_mask=False)
        else:
            pm("sym215a", 21, mask_ditch, gdf_lines)
            pm("sym215b", 21, mask_ditch, gdf_lines)

    # ----------------------------------------------------------------
    # WATER
    # ----------------------------------------------------------------
    if visibility.get("water", True):
        # 301 - Vodní plocha
        cgdf = isom("301")
        if cgdf is not None:
            pm("sym301", 27, None, cgdf, to_mask=False)
        elif zab_in("VodniPlocha", "PozemniNadrz"):
            for zk in ["VodniPlocha", "PozemniNadrz"]:
                if zab(zk) is not None:
                    pm("sym301", 27, None, zab(zk), to_mask=False)
        else:
            pm("sym301", 27,
               c("natural").isin(["lake", "water", "canal"]) |
               c("water").isin(["lake", "river", "basin", "bay", "reservoir"]) |
               (c("landuse") == "basin") | (c("leisure") == "swimming_pool"),
               gdf_polys)

        # 302 - Mělká voda
        cgdf = isom("302")
        if cgdf is not None:
            pm("sym302", 27, None, cgdf, to_mask=False)
        else:
            pm("sym302", 27, c("water") == "stream", gdf_polys)

        # 303
        cgdf = isom("303")
        if cgdf is not None:
            pm("sym303", 27, None, cgdf, to_mask=False)

        # 304 - Řeka
        cgdf = isom("304")
        if cgdf is not None:
            pm("sym304", 26, None, cgdf, to_mask=False)
        elif zab("VodniTok") is not None:
            mask = (_get_col(zab("VodniTok"), "typtoku_p").isin(["povrchový splavný"]) &
                    _get_col(zab("VodniTok"), "vydattok_p").isin(["stálý"]))
            pm("sym304", 26, mask, zab("VodniTok"))
        else:
            pm("sym304", 26,
               (c("waterway").isin(["river", "canal"])) &
               (~c("tunnel").isin(["culvert", "yes", "pipe", "covered", "cave"])) &
               (~c("intermittent").isin(["yes", "dry"])),
               gdf_lines)

        # 305 - Potok
        cgdf = isom("305")
        if cgdf is not None:
            pm("sym305", 26, None, cgdf, to_mask=False)
        elif zab("VodniTok") is not None:
            mask = (_get_col(zab("VodniTok"), "typtoku_p").isin(["povrchový nesplavný"]) &
                    _get_col(zab("VodniTok"), "vydattok_p").isin(["stálý"]))
            pm("sym305", 26, mask, zab("VodniTok"))
        else:
            pm("sym305", 26,
               (c("waterway").isin(["stream", "ditch"])) &
               (~c("tunnel").isin(["culvert", "yes", "pipe", "covered", "cave"])) &
               (~c("intermittent").isin(["yes", "dry"])),
               gdf_lines)

        # 306 - Občasný tok
        cgdf = isom("306")
        if cgdf is not None:
            pm("sym306", 26, None, cgdf, to_mask=False)
        elif zab("VodniTok") is not None:
            mask = (_get_col(zab("VodniTok"), "typtoku_p").isin(["povrchový splavný", "povrchový nesplavný"]) &
                    _get_col(zab("VodniTok"), "vydattok_p").isin(["občasný"]))
            pm("sym306", 26, mask, zab("VodniTok"))
        else:
            pm("sym306", 26,
               ((c("waterway") == "drain") |
                (c("waterway").isin(["stream", "ditch"]) & c("intermittent").isin(["yes", "dry"]))) &
               (~c("tunnel").isin(["culvert", "yes", "pipe", "covered", "cave"])),
               gdf_lines)

        # 307 - Neprůchodná bažina
        cgdf = isom("307")
        if cgdf is not None:
            pm("sym307", 25, None, cgdf, to_mask=False)
        elif zab("Raseliniste") is not None:
            pm("sym307", 25, None, zab("Raseliniste"), to_mask=False)
        else:
            pm("sym307", 25, c("wetland") == "reedbed", gdf_polys)

        # 308 - Bažina
        cgdf = isom("308")
        if cgdf is not None:
            pm("sym308", 25, None, cgdf, to_mask=False)
        elif zab("BazinaMocal") is not None:
            pm("sym308", 25, None, zab("BazinaMocal"), to_mask=False)
        else:
            pm("sym308", 25,
               (c("natural") == "wetland") & (~c("wetland").isin(["marsh", "wet_meadow", "reedbed"])),
               gdf_polys)

        # 309
        cgdf = isom("309")
        if cgdf is not None:
            pm("sym309", 25, None, cgdf, to_mask=False)

        # 310 - Nezřetelná bažina
        cgdf = isom("310")
        if cgdf is not None:
            pm("sym308", 25, None, cgdf, to_mask=False)
        else:
            pm("sym308", 25,
               (c("wetland") == "marsh") | (c("water") == "wet_meadow"),
               gdf_polys)

        # 311 - Studna / nádrž
        cgdf = isom("311")
        if cgdf is not None:
            pm("sym311", 52, None, cgdf, to_mask=False)
        else:
            pm("sym311", 52,
               (c("man_made") == "water_well") | (c("amenity") == "fountain") |
               (c("natural") == "geyser"),
               gdf_centroids)

        # 312 - Pramen
        cgdf = isom("312")
        if cgdf is not None:
            pm("sym312", 52, None, cgdf, to_mask=False)
        elif zab("ZdrojPodzemnichVod") is not None:
            pm("sym312", 52, None, zab("ZdrojPodzemnichVod"), to_mask=False)
        else:
            pm("sym312", 52,
               (c("natural") == "spring") & (c("covered") != "yes"),
               gdf_centroids)

        # 313
        cgdf = isom("313")
        if cgdf is not None:
            pm("sym312", 52, None, cgdf, to_mask=False)

    # ----------------------------------------------------------------
    # VEGETATION
    # ----------------------------------------------------------------
    if visibility.get("vegetation", True):
        # 401 - Otevřený prostor
        cgdf = isom("401")
        if cgdf is not None:
            pm("sym401", 1.0, None, cgdf, to_mask=False)
        elif zab("TrvalyTravniPorost") is not None:
            pm("sym401", 1.0, None, zab("TrvalyTravniPorost"), to_mask=False)
        else:
            pm("sym401", 1.0,
               c("landuse").isin(["grassland", "grass", "meadow"]) |
               c("natural").isin(["grassland", "fell", "heath"]),
               gdf_polys)

        # 402 - Park
        cgdf = isom("402")
        if cgdf is not None:
            pm("sym402", 1.0, None, cgdf, to_mask=False)
        elif zab("OkrasnaZahradaPark") is not None:
            pm("sym402", 1.0, None, zab("OkrasnaZahradaPark"), to_mask=False)
        else:
            pm("sym402", 1.0, c("leisure") == "park", gdf_polys)

        # 403, 404 - jen ISOM
        for code, sym in [("403", "sym403"), ("404", "sym404")]:
            cgdf = isom(code)
            if cgdf is not None:
                pm(sym, 1.0, None, cgdf, to_mask=False)

        # 408 - Živý plot / alej
        cgdf = isom("408")
        if cgdf is not None:
            pm("sym408l", 19, None, cgdf, to_mask=False)
        elif zab("LiniovaVegetace") is not None:
            mask = _get_col(zab("LiniovaVegetace"), "typveg_p").isin(["živý plot"])
            pm("sym408l", 19, mask, zab("LiniovaVegetace"))
        else:
            pm("sym408l", 19, c("natural") == "tree_row", gdf_polys)

        # 412 - Orná půda
        cgdf = isom("412")
        if cgdf is not None:
            pm("sym412a", 1.9, None, cgdf, to_mask=False)
            pm("sym412b", 15, None, cgdf, to_mask=False)
        elif zab("OrnaPudaAOstatniDaleNespecifikovanePlochy") is not None:
            mask = _get_col(zab("OrnaPudaAOstatniDaleNespecifikovanePlochy"), "typ_pudy_p").isin(["orná půda"])
            pm("sym412a", 1.9, mask, zab("OrnaPudaAOstatniDaleNespecifikovanePlochy"))
            pm("sym412b", 15, mask, zab("OrnaPudaAOstatniDaleNespecifikovanePlochy"))
        else:
            pm("sym412a", 1.9, c("landuse") == "farmland", gdf_polys)
            pm("sym412b", 15, c("landuse") == "farmland", gdf_polys)

        # 413 - Sad
        cgdf = isom("413")
        if cgdf is not None:
            pm("sym413", 1.9, None, cgdf, to_mask=False)
        else:
            pm("sym413", 1.9, c("landuse") == "orchard", gdf_polys)

        # 414 - Vinice / Chmelnice
        cgdf = isom("414")
        if cgdf is not None:
            pm("sym414", 1.9, None, cgdf, to_mask=False)
        elif zab_in("Vinice", "Chmelnice"):
            for zk in ["Vinice", "Chmelnice"]:
                if zab(zk) is not None:
                    pm("sym414", 1.9, None, zab(zk), to_mask=False)
        else:
            pm("sym414", 1.9, c("landuse").isin(["plant_nursery", "vineyard"]), gdf_polys)

        # 415 - Hranice kultivace
        cgdf = isom("415")
        if cgdf is not None:
            pm("sym216l", 15, None, cgdf, to_mask=False)

        # 416 - Hranice vegetace
        cgdf = isom("416")
        if cgdf is not None:
            pm("sym416p", 1.8, None, cgdf, to_mask=False)
        elif zab("LesniPudaSeStromyKategorizovana") is not None:
            mask = (_get_col(zab("LesniPudaSeStromyKategorizovana"), "druh_k").isin(["J"]) &
                    _get_col(zab("LesniPudaSeStromyKategorizovana"), "vyska_k").isin(["3"]))
            pm("sym416p", 1.8, mask, zab("LesniPudaSeStromyKategorizovana"))

        # 417 - Výrazný strom
        cgdf = isom("417")
        if cgdf is not None:
            pm("sym417a", 54, None, cgdf, to_mask=False)
            pm("sym417b", 54, None, cgdf, to_mask=False)
        elif zab("VyznamnyNeboOsamelyStromLesik") is not None:
            pm("sym417a", 54, None, zab("VyznamnyNeboOsamelyStromLesik"), to_mask=False)
            pm("sym417b", 54, None, zab("VyznamnyNeboOsamelyStromLesik"), to_mask=False)
        else:
            pm("sym417a", 54, c("natural") == "tree", gdf_centroids)
            pm("sym417b", 55, c("natural") == "tree", gdf_centroids)

        # 418 - Výrazný keř
        cgdf = isom("418")
        if cgdf is not None:
            pm("sym418a", 54, None, cgdf, to_mask=False)
            pm("sym418b", 54, None, cgdf, to_mask=False)
        else:
            pm("sym418a", 54, c("natural") == "shrub", gdf_centroids)
            pm("sym418b", 55, c("natural") == "shrub", gdf_centroids)

        # 419 - Výrazný vegetační objekt
        cgdf = isom("419")
        if cgdf is not None:
            pm("sym419", 54, None, cgdf, to_mask=False)
        else:
            pm("sym419", 54, c("natural") == "tree_stump", gdf_centroids)

    # ----------------------------------------------------------------
    # ROADS
    # ----------------------------------------------------------------
    if visibility.get("roads", True):
        # 501 - Parkoviště
        cgdf = isom("501")
        if cgdf is not None:
            pm("sym501", 49, None, cgdf, to_mask=False)
        elif zab_in("ParkovisteOdpocivka", "ArealUceloveZastavby"):
            if zab("ParkovisteOdpocivka") is not None:
                pm("sym501", 49, None, zab("ParkovisteOdpocivka"), to_mask=False)
            if zab("ArealUceloveZastavby") is not None:
                mask = _get_col(zab("ArealUceloveZastavby"), "typzast_k").isin(["408"])
                pm("sym501", 49, mask, zab("ArealUceloveZastavby"))
        else:
            pm("sym501", 49,
               (c("amenity") == "parking") | (c("place") == "square"),
               gdf_polys)

        # 502D - Dálnice
        cgdf = isom("502D")
        mask_motorway = (c("highway").isin(["motorway", "trunk"]) &
                         ~c("tunnel").isin(["yes", "avalanche_protector", "building_passage"]) &
                         (c("bridge") != "yes") & (c("access") != "private"))
        for sym, z in [("sym502Da", 45), ("sym502Db", 47), ("sym502Dc", 48)]:
            if cgdf is not None:
                pm(sym, z, None, cgdf, to_mask=False)
            elif zab("SilniceDalnice") is not None:
                mask = _get_col(zab("SilniceDalnice"), "typsil_k").isin(["D1", "D2", "M"])
                pm(sym, z, mask, zab("SilniceDalnice"))
            else:
                pm(sym, z, mask_motorway, gdf_lines)

        # 502 - Hlavní silnice
        cgdf = isom("502")
        mask_road = (c("highway").isin(["highway_link", "trunk_link", "primary", "primary_link",
                                          "secondary", "secondary_link", "residential", "tertiary",
                                          "living_street"]) &
                     ~c("tunnel").isin(["yes", "avalanche_protector", "building_passage"]) &
                     (c("bridge") != "yes") & (c("access") != "private"))
        for sym, z in [("sym502a", 45), ("sym502b", 47)]:
            if cgdf is not None:
                pm(sym, z, None, cgdf, to_mask=False)
            elif zab_in("SilniceDalnice", "Ulice", "SilniceVeVastavbe"):
                if zab("SilniceDalnice") is not None:
                    mask = ~_get_col(zab("SilniceDalnice"), "typsil_k").isin(["D1", "D2", "M"])
                    pm(sym, z, mask, zab("SilniceDalnice"))
                if zab("Ulice") is not None:
                    mask = _get_col(zab("Ulice"), "typulice_k").isin(["026", "926"])
                    pm(sym, z, mask, zab("Ulice"))
                if zab("SilniceVeVastavbe") is not None:
                    pm(sym, z, None, zab("SilniceVeVastavbe"), to_mask=False)
            else:
                pm(sym, z, mask_road, gdf_lines)

        # 503 - Silnice / zpevněná cesta
        cgdf = isom("503")
        mask_service = (c("highway").isin(["tertiary_link", "service"]) | c("highway").isin(["track", "road", "cycleway", "unclassified"]) &
                        (c("surface").isin(["concrete", "asphalt"])) | c("tracktype" == "grade1") &
                        ~c("tunnel").isin(["yes", "avalanche_protector", "building_passage"]) &
                        (c("bridge") != "yes") & (c("access") != "private"))
        if cgdf is not None:
            pm("sym503", 45, None, cgdf, to_mask=False)
        elif zab_in("SilniceNeevidovana", "Cesta", "LyzarskyMustek"):
            if zab("SilniceNeevidovana") is not None:
                pm("sym503", 45, None, zab("SilniceNeevidovana"), to_mask=False)
            if zab("Cesta") is not None:
                mask = (_get_col(zab("Cesta"), "povrch_p").isin(
                    ["zpevněný (panel, dlažba)", "zpevněný (asfalt, beton)"]) &
                        _get_col(zab("Cesta"), "typcesty_p").isin(["cesta udržovaná"]))
                pm("sym503", 45, mask, zab("Cesta"))
            if zab("LyzarskyMustek") is not None:
                pm("sym503", 46, None, zab("LyzarskyMustek"), to_mask=False)
        else:
            pm("sym503", 45, mask_service, gdf_lines)

        # 504 - Vozová cesta
        cgdf = isom("504")
        mask_track = (c("highway").isin(["cycleway", "unclassified"]) &
                      (~c("surface").isin(["concrete", "asphalt"])) & (c("tracktype") != "grade1") &
                      ~c("tunnel").isin(["yes", "avalanche_protector", "building_passage"]) &
                      (c("bridge") != "yes") & (c("access") != "private"))
        if cgdf is not None:
            pm("sym504", 45, None, cgdf, to_mask=False)
        elif zab("Cesta") is not None:
            mask = (_get_col(zab("Cesta"), "povrch_p").isin([
                "zpevněný (nosný terén, štěrk, kalený povrch)",
                "nedostatečně zpevněný (tráva, hlína, písek, kamení)", "neurčeno", "NULL"]) &
                    _get_col(zab("Cesta"), "typcesty_p").isin(["cesta udržovaná"]))
            pm("sym504", 45, mask, zab("Cesta"))
        else:
            pm("sym504", 45, mask_track, gdf_lines)

        # 505 - Pěší cesta
        cgdf = isom("505")
        mask_footway = (c("highway").isin(["pedestrian", "road", "footway", "track", "bridleway"]) |
                        ((c("highway") == "cycleway") & (~c("surface").isin(["concrete", "asphalt"])) &
                        (c("tracktype") != "grade1")) &
                        (c("bridge") != "yes") & (c("access") != "private"))
        if cgdf is not None:
            pm("sym505", 45, None, cgdf, to_mask=False)
        elif zab_in("Ulice", "Cesta"):
            if zab("Ulice") is not None:
                mask = ~_get_col(zab("Ulice"), "typulice_k").isin(["925", "025"])
                pm("sym505", 45, mask, zab("Ulice"))
            if zab("Cesta") is not None:
                mask = _get_col(zab("Cesta"), "typcesty_p").isin(["cesta neudržovaná"])
                pm("sym505", 45, mask, zab("Cesta"))
        else:
            pm("sym505", 45, mask_footway, gdf_lines)

        # 506 - Pěšina
        cgdf = isom("506")
        mask_path = ((c("highway") == "path") &
                     ~c("trail_visibility").isin(["low", "poor", "bad", "very_bad", "horrible", "no"]) &
                     (c("bridge") != "yes") & (c("access") != "private"))
        if cgdf is not None:
            pm("sym506", 45, None, cgdf, to_mask=False)
        elif zab("Pesina") is not None:
            pm("sym506", 45, None, zab("Pesina"), to_mask=False)
        else:
            pm("sym506", 45, mask_path, gdf_lines)

        # 507 - Nezřetelná pěšina
        cgdf = isom("507")
        if cgdf is not None:
            pm("sym507", 45, None, cgdf, to_mask=False)
        else:
            pm("sym507", 45,
               (c("highway") == "path") &
               c("trail_visibility").isin(["low", "poor", "bad", "very_bad", "horrible"]) &
               (c("bridge") != "yes"),
               gdf_lines)

        # 508 - Průsek
        cgdf = isom("508")
        if cgdf is not None:
            pm("sym508", 38, None, cgdf, to_mask=False)
        elif zab("LesniPrusek") is not None:
            pm("sym508", 38, None, zab("LesniPrusek"), to_mask=False)
        else:
            pm("sym508", 38, c("man_made") == "cutline", gdf_lines)

        # Mosty - dálnice
        mask_bridge_double = (c("highway").isin(["motorway", "trunk"]) &
                              ~c("tunnel").isin(["yes", "avalanche_protector", "building_passage"]) &
                              (c("bridge") == "yes") & (c("access") != "private"))
        for sym, z in zip(["sym502DBa", "sym502DBb", "sym502Da", "sym502Db", "sym502Dc"],
                          [65, 66, 67, 68, 69]):
            pm(sym, z, mask_bridge_double, gdf_lines)

        # Mosty - hlavní silnice
        mask_bridge_major = (c("highway").isin(["highway_link", "trunk_link", "primary", "primary_link",
                                                 "secondary", "secondary_link", "residential", "tertiary",
                                                 "living_street"]) &
                             ~c("tunnel").isin(["yes", "avalanche_protector", "building_passage"]) &
                             (c("bridge") == "yes") & (c("access") != "private"))
        for sym, z in zip(["sym502Ba", "sym502Bb", "sym502a", "sym502b"], [65, 66, 67, 68]):
            pm(sym, z, mask_bridge_major, gdf_lines)

        # Mosty - vedlejší silnice
        mask_bridge_minor = (c("highway").isin(["tertiary_link", "service", "track", "road", "unclassified"]) &
                             ~c("tunnel").isin(["yes", "avalanche_protector", "building_passage"]) &
                             (c("bridge") == "yes") & (c("access") != "private"))
        for sym, z in zip(["sym503Ba", "sym503Bb", "sym503"], [65, 66, 67]):
            pm(sym, z, mask_bridge_minor, gdf_lines)

        # Lávka
        if zab("Lavka") is not None:
            pm("sym503", 40, None, zab("Lavka"), to_mask=False)
        else:
            pm("sym503", 67,
               c("highway").isin(["path", "cycleway", "footway", "bridleway"]) &
               (c("bridge") == "yes") & (c("access") != "private"),
               gdf_lines)

        # 509 - Železnice
        cgdf = isom("509")
        mask_railway = (c("railway").isin(["rail", "disused", "funicular", "narrow_gauge"]) &
                        ~c("tunnel").isin(["yes", "avalanche_protector", "building_passage"]) &
                        (c("bridge") != "yes"))
        for sym, z in [("sym509a", 40), ("sym509b", 41)]:
            if cgdf is not None:
                pm(sym, z, None, cgdf, to_mask=False)
            elif zab_in("ZeleznicniTrat", "ZeleznicniVlecka"):
                for zk in ["ZeleznicniTrat", "ZeleznicniVlecka"]:
                    if zab(zk) is not None:
                        pm(sym, z, None, zab(zk), to_mask=False)
            else:
                pm(sym, z, mask_railway, gdf_lines)

        # Železniční most
        mask_bridge_railway = (c("railway").isin(["rail", "disused", "funicular", "narrow_gauge"]) &
                               ~c("tunnel").isin(["yes", "avalanche_protector", "building_passage"]) &
                               (c("bridge") == "yes"))
        for sym, z in zip(["sym509Ba", "sym509Bb", "sym509a", "sym509b"], [60, 61, 62, 63]):
            pm(sym, z, mask_bridge_railway, gdf_lines)

    # ----------------------------------------------------------------
    # MAN MADE
    # ----------------------------------------------------------------
    if visibility.get("man_made", True):
        # 510 - El. vedení (nízké napětí / lanovky)
        cgdf = isom("510")
        if cgdf is not None:
            pm("sym510", 70, None, cgdf, to_mask=False)
        elif zab("ElektrickeVedeni") is not None:
            mask = ~_get_col(zab("ElektrickeVedeni"), "napeti").isin(["400", "110"])
            pm("sym510", 70, mask, zab("ElektrickeVedeni"))
        elif zab("LanovaDrahaLyzarskyVlek") is not None:
            pm("sym510", 70, None, zab("LanovaDrahaLyzarskyVlek"), to_mask=False)
        else:
            pm("sym510", 70,
               c("power").isin(["line", "minor_line"]) | c("aerialway").isin([
                   "cable_car", "gondola", "chair_lift", "drag_lift", "t-bar", "j-bar"]),
               gdf_lines)

        # 511 - VVN (vysoké napětí)
        if zab("ElektrickeVedeni") is not None:
            mask = _get_col(zab("ElektrickeVedeni"), "napeti").isin(["400", "110"])
            pm("sym510", 70, mask, zab("ElektrickeVedeni"))
        else:
            pm("sym510", 70, c("power").isin(["line"]), gdf_lines)

        # 513.1 - Zeď
        cgdf = isom("513.1")
        if cgdf is not None:
            pm("sym513-1a", 30, None, cgdf, to_mask=False)
            pm("sym513-1b", 30, None, cgdf, to_mask=False)
        elif zab_in("Zed", "NasupisteHraze"):
            for zk in ["Zed", "NasupisteHraze"]:
                if zab(zk) is not None:
                    pm("sym513-1a", 30, None, zab(zk), to_mask=False)
                    pm("sym513-1b", 30, None, zab(zk), to_mask=False)
        else:
            pm("sym513-1a", 30, c("barrier") == "wall", gdf_lines)
            pm("sym513-1b", 30, c("barrier") == "wall", gdf_lines)

        # 515 - Nepřekonatelná zeď
        cgdf = isom("515")
        if cgdf is not None:
            pm("sym515a", 30, None, cgdf, to_mask=False)
            pm("sym515b", 30, None, cgdf, to_mask=False)
        elif zab("Zed") is not None:
            mask = _get_col(zab("Zed"), "typzed_p").isin(
                ["protihluková stěna", "zeď vodního díla", "zeď ostatní"])
            pm("sym515a", 30, mask, zab("Zed"))
        elif zab("HradbaValBastaOpevneni") is not None:
            pm("sym515b", 30, None, zab("HradbaValBastaOpevneni"), to_mask=False)
        else:
            pm("sym515a", 30, c("barrier") == "city_wall", gdf_lines)
            pm("sym515b", 30, c("barrier") == "city_wall", gdf_lines)

        # 520 - Privátní oblast
        if visibility.get("private", True):
            cgdf = isom("520")
            if cgdf is not None:
                pm("sym520", 1.5, None, cgdf, to_mask=False)
            else:
                garden_layers = ["Hrbitov", "Letiste", "OvocnySadZahrada",
                                 "PovrchTezbaLom", "ArealUceloveZastavby", "Skladka"]
                found = False
                for gl in garden_layers:
                    if zab(gl) is not None:
                        pm("sym520", 1.5, None, zab(gl), to_mask=False)
                        found = True
                if not found:
                    pm("sym520", 1.5,
                       c("landuse").isin([
                           "residential", "allotments", "military", "commercial",
                           "construction", "industrial", "cemetery", "landfill", "quarry"
                       ]) | c("leisure").isin(["pitch", "sports_centre"]),
                       gdf_polys)

        # 521 - Budova
        if visibility.get("buildings", True):
            cgdf = isom("521")
            if cgdf is not None:
                pm("sym521", 50, None, cgdf, to_mask=False)
            elif zab("BudovaJednotlivaNeboBlokBudov") is not None:
                pm("sym521", 50, None, zab("BudovaJednotlivaNeboBlokBudov"), to_mask=False)
            else:
                pm("sym521", 50,
                   c("building").notna() & (c("building") != "") & ~c("building").isin(["roof", "ruins"]),
                   gdf_polys)

        # 522 - Zastřešení
        cgdf = isom("522")
        if cgdf is not None:
            pm("sym522", 36, None, cgdf, to_mask=False)
        else:
            pm("sym522", 36, c("building") == "roof", gdf_polys)

        # 523 - Zřícenina / Bunkr
        cgdf = isom("523")
        if cgdf is not None:
            pm("sym523", 35, None, cgdf, to_mask=False)
        elif zab("RozvalinaZricenina") is not None:
            pm("sym523", 35, None, zab("RozvalinaZricenina"), to_mask=False)
        elif zab("Bunkr") is not None:
            pm("sym523", 56, None, zab("Bunkr"), to_mask=False)
        else:
            pm("sym523", 35,
               (c("building") == "ruins") | (c("historic") == "ruins") | (c("military") == "bunker"),
               gdf_polys)

        # 524 - Věž
        cgdf = isom("524")
        if cgdf is not None:
            pm("sym524a", 56, None, cgdf, to_mask=False)
            pm("sym524b", 56, None, cgdf, to_mask=False)
        else:
            tower_layers = ["Silo", "TezniVez", "TovarniKomin", "VetrnyMotor",
                            "VetrnyMlyn", "VodojemVezovy", "VezovitaStavba"]
            found = False
            for tl in tower_layers:
                if zab(tl) is not None:
                    pm("sym524a", 56, None, zab(tl), to_mask=False)
                    pm("sym524b", 56, None, zab(tl), to_mask=False)
                    found = True
            if not found:
                pm("sym524a", 56,
                   c("man_made").isin(["tower", "water_tower", "communications_tower",
                                       "mast", "chimney", "crane"]) |
                   (c("building") == "clock_tower"),
                   gdf_pts)
                pm("sym524b", 56,
                   c("man_made").isin(["tower", "water_tower", "communications_tower",
                                       "mast", "chimney", "crane"]) |
                   (c("building") == "clock_tower"),
                   gdf_pts)

        # 525 - Malá věž / Posed
        cgdf = isom("525")
        if cgdf is not None:
            pm("sym525", 56, None, cgdf, to_mask=False)
        else:
            pm("sym525", 56, c("amenity") == "hunting_stand", gdf_pts)

        # 526 - Pomník
        cgdf = isom("526")
        if cgdf is not None:
            pm("sym526a", 56, None, cgdf, to_mask=False)
            pm("sym526b", 56, None, cgdf, to_mask=False)
        elif zab("MohylaPomnikNahrobek") is not None:
            pm("sym526a", 56, None, zab("MohylaPomnikNahrobek"), to_mask=False)
            pm("sym526b", 56, None, zab("MohylaPomnikNahrobek"), to_mask=False)
        else:
            pm("sym526a", 56,
               c("historic").isin(["boundary_stone", "memorial", "wayside_cross"]) |
               c("man_made").isin(["cross", "survey_point", "obelisk"]),
               gdf_centroids)
            pm("sym526b", 56,
               c("historic").isin(["boundary_stone", "memorial", "wayside_cross"]) |
               c("man_made").isin(["cross", "survey_point", "obelisk"]),
               gdf_centroids)

        # 531 - Výrazný umělý objekt / Kříž
        cgdf = isom("531")
        if cgdf is not None:
            pm("sym531", 56, None, cgdf, to_mask=False)
        elif zab("KrizSloupKulturnihoVyznamu") is not None:
            pm("sym531", 56, None, zab("KrizSloupKulturnihoVyznamu"), to_mask=False)
        else:
            pm("sym531", 56, c("man_made").isin(["insect_hotel", "street_cabinet"]), gdf_centroids)

        # 532 - Schody
        cgdf = isom("532")
        if cgdf is not None:
            for sym, z in [("sym532a", 49), ("sym532b", 50), ("sym532c", 51)]:
                pm(sym, z, None, cgdf, to_mask=False)
        else:
            for sym, z in [("sym532a", 49), ("sym532b", 50), ("sym532c", 51)]:
                pm(sym, z, c("highway") == "steps", gdf_lines)