"""
vector_layers.py — mapování OSM / ZABAGED® / vlastních ISOM vrstev na symboly.

Názvy ZABAGED® klíčů odpovídají názvům SHP souborů dle Katalogu objektů
ZABAGED® v4.6 (Zeměměřický úřad, leden 2026). Přehled použitých klíčů:

  Kategorie 1 — Sídla, hospodářské a kulturní objekty:
    BudovaJednotlivaNeboBlokBudov  (1.02)
    PovrchTezbaLom                  (1.06)
    MohylaPomnikNahrobek            (1.20)
    HradbaValBastaOpevneni          (1.22)
    Zed                             (1.23)
    RozvalinazRicenina              (1.19)

  Kategorie 2 — Komunikace:
    SilniceDalnice                  (2.01)
    Cesta                           (2.03)
    Pesina                          (2.04)
    Most                            (2.08)
    ZeleznicniTrat                  (2.17)

  Kategorie 3 — Rozvodné sítě:
    ElektrickeVedeni                (3.03)

  Kategorie 4 — Vodstvo:
    ZdrojPodzemnichVod              (4.01)
    VodniTok                        (4.02)
    VodniPlocha                     (4.10)
    BazinaMocal                     (4.12)

  Kategorie 6 — Vegetace a povrch:
    TrvalyTravniPorost              (6.06)
    LesniPudaSeStromy               (6.07)
    LesniPudaSKrovinatymPorostem    (6.08)
    VyznamnyNeboOsamelyStromLesik   (6.11)
    LesniPrusek                     (6.13)
    Raseliniste                     (6.14)

  Kategorie 7 — Terénní reliéf:
    SkalniUtvary                    (7.06)
    OsamelyBalvanSkalaSkalniSuk     (7.10)
    StupeSraz                       (7.12)
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

    # Resetujeme index celého gdf jednou
    if gdf is not None and not gdf.empty:
        gdf = gdf.reset_index(drop=True)

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
        """Vrátí ZABAGED® GDF podle klíče (název SHP souboru bez přípony)."""
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
            elif code == "104" and zab("StupeSraz"):
                # ZABAGED® 7.12 Stupeň, sráz
                pm("sym104", zo, None, zab("StupeSraz"), to_mask=False)
            elif code == "104":
                pm("sym104", zo, c("man_made") == "embankment", gdf_lines)

        # 105 Hradba, val (ZABAGED® 1.22)
        if isom("105") is None and zab("HradbaValBastaOpevneni"):
            for s in ["sym105-1a", "sym105-1b"]:
                pm(s, 30, None, zab("HradbaValBastaOpevneni"), to_mask=False)

        # 106 Zřícená zemní zeď
        cgdf = isom("106")
        if cgdf is not None:
            pm("sym106", 21, None, cgdf, to_mask=False)
        else:
            pm("sym106", 21, c("barrier").isin(["ditch", "hedge"]) & (c("historic") == "yes"), gdf_lines)

        # 110 Podlouhlý kopeček
        cgdf = isom("110")
        if cgdf is not None:
            pm("sym110", 21, None, cgdf, to_mask=False)

        # 113 Nerovný terén — ZABAGED® nemá přímý ekvivalent, fallback OSM
        cgdf = isom("113")
        if cgdf is not None:
            pm("sym113", 18, None, cgdf, to_mask=False)
        else:
            pm("sym113", 18, c("natural").isin(["earth_bank", "embankment"]), gdf_lines)

        # 114 Velmi nerovný terén
        cgdf = isom("114")
        if cgdf is not None:
            pm("sym114", 18, None, cgdf, to_mask=False)

        # 115 Výrazný terénní útvar
        cgdf = isom("115")
        if cgdf is not None:
            pm("sym115", 56, None, cgdf, to_mask=False)

    # ----------------------------------------------------------------
    # ROCKS
    # ----------------------------------------------------------------
    if visibility.get("rocks", True):
        # 201 Nepřekonatelná skála/sráz — ZABAGED® 7.12 StupeSraz (strmý)
        cgdf = isom("201")
        if cgdf is not None:
            pm("sym201", 57, None, cgdf, to_mask=False)
        elif zab("StupeSraz"):
            pm("sym201", 57, None, zab("StupeSraz"), to_mask=False)
        else:
            pm("sym201", 57,
               c("natural").isin(["cliff"]) & ~c("access").isin(["yes", "permissive"]),
               gdf_lines)

        # 202 Překonatelný sráz/lom — ZABAGED® 1.06 PovrchTezbaLom
        cgdf = isom("202")
        if cgdf is not None:
            pm("sym202", 56, None, cgdf, to_mask=False)
        elif zab("PovrchTezbaLom"):
            pm("sym202", 56, None, zab("PovrchTezbaLom"), to_mask=False)
        else:
            pm("sym202", 56,
               c("natural").isin(["cliff"]) | c("man_made").isin(["embankment", "cutting"]),
               gdf_lines)

        for code, sym, zo in [("203.1", "sym203-1", 56), ("204", "sym204", 56),
                               ("205", "sym205", 56), ("207", "sym207", 56),
                               ("208", "sym208", 18), ("209", "sym209", 18),
                               ("210", "sym210", 18), ("213", "sym213", 15)]:
            cgdf = isom(code)
            if cgdf is not None:
                pm(sym, zo, None, cgdf, to_mask=False)

        # 205 Osamělý balvan — ZABAGED® 7.10 OsamelyBalvanSkalaSkalniSuk
        if isom("205") is None and zab("OsamelyBalvanSkalaSkalniSuk"):
            pm("sym205", 56, None, zab("OsamelyBalvanSkalaSkalniSuk"), to_mask=False)
        elif isom("205") is None:
            pm("sym205", 56, c("natural").isin(["stone", "rock"]), gdf_centroids)

        # 203.2 Nebezpečná jáma
        cgdf = isom("203.2")
        if cgdf is not None:
            pm("sym203-2", 56, None, cgdf, to_mask=False)

        # 206 Obrovský balvan — ZABAGED® 7.10, pouze velmi velké objekty
        cgdf = isom("206")
        if cgdf is not None:
            pm("sym206", 56, None, cgdf, to_mask=False)
        elif zab("OsamelyBalvanSkalaSkalniSuk"):
            mask = _get_col(zab("OsamelyBalvanSkalaSkalniSuk"), "vyska_p") > 5
            if mask.any():
                pm("sym206", 56, mask, zab("OsamelyBalvanSkalaSkalniSuk"))

        # 211 Kamenité – chůze
        cgdf = isom("211")
        if cgdf is not None:
            pm("sym211", 18, None, cgdf, to_mask=False)
        else:
            pm("sym211", 18,
               c("surface").isin(["rocks", "rock"]) | c("natural").isin(["scree"]),
               gdf_polys)

        # 212 Kamenité – boj
        cgdf = isom("212")
        if cgdf is not None:
            pm("sym212", 18, None, cgdf, to_mask=False)

        # 214 Holá skála — ZABAGED® 7.06 SkalniUtvary
        cgdf = isom("214")
        if cgdf is not None:
            pm("sym214", 18, None, cgdf, to_mask=False)
        elif zab("SkalniUtvary"):
            pm("sym214", 18, None, zab("SkalniUtvary"), to_mask=False)
        else:
            pm("sym214", 18, c("natural") == "bare_rock", gdf_polys)

        mask_ditch = c("barrier").isin(["ditch"]) | c("military").isin(["trench"])
        pm("sym215a", 21, mask_ditch, gdf_lines)
        pm("sym215b", 21, mask_ditch, gdf_lines)

    # ----------------------------------------------------------------
    # WATER
    # ----------------------------------------------------------------
    if visibility.get("water", True):
        # 301 Vodní plocha — ZABAGED® 4.10 VodniPlocha
        cgdf = isom("301")
        if cgdf is not None:
            pm("sym301", 27, None, cgdf, to_mask=False)
        elif zab("VodniPlocha"):
            pm("sym301", 27, None, zab("VodniPlocha"), to_mask=False)
        else:
            pm("sym301", 27,
               c("natural").isin(["lake", "water"]) | c("water").isin(["lake", "river", "reservoir"]),
               gdf_polys)

        # 304 Řeka — ZABAGED® 4.02 VodniTok (splavný, stálý)
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

        # 305 Potok — ZABAGED® 4.02 VodniTok (nesplavný, stálý)
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

        # 306 Přerušovaný potok — ZABAGED® 4.02 VodniTok (přerušovaný)
        cgdf = isom("306")
        if cgdf is not None:
            pm("sym306", 26, None, cgdf, to_mask=False)
        elif zab("VodniTok"):
            mask = _get_col(zab("VodniTok"), "vydattok_p").isin(["přerušovaný"])
            pm("sym306", 26, mask, zab("VodniTok"))
        else:
            pm("sym306", 26,
               c("waterway").isin(["stream", "ditch"]) & c("intermittent").isin(["yes"]),
               gdf_lines)

        # 307 Rákosová bažina — ZABAGED® 6.14 Raseliniste
        for code, sym, zo in [("307", "sym307", 25), ("308", "sym308", 25)]:
            cgdf = isom(code)
            if cgdf is not None:
                pm(sym, zo, None, cgdf, to_mask=False)

        if isom("307") is None and zab("Raseliniste"):
            pm("sym307", 25, None, zab("Raseliniste"), to_mask=False)
        elif isom("307") is None:
            pm("sym307", 25, c("wetland") == "reedbed", gdf_polys)

        # 308 Bažina, močál — ZABAGED® 4.12 BazinaMocal
        if isom("308") is None and zab("BazinaMocal"):
            pm("sym308", 25, None, zab("BazinaMocal"), to_mask=False)
        elif isom("308") is None:
            pm("sym308", 25, c("natural") == "wetland", gdf_polys)

        # 309 Úzká bažina — ZABAGED® nemá přímý ekvivalent, OSM fallback
        cgdf = isom("309")
        if cgdf is not None:
            pm("sym309", 25, None, cgdf, to_mask=False)
        else:
            pm("sym309", 25,
               (c("natural") == "wetland") & c("waterway").isin(["ditch", "drain"]),
               gdf_lines)

        # 310 Neurčitá bažina — OSM fallback
        cgdf = isom("310")
        if cgdf is not None:
            pm("sym310", 24, None, cgdf, to_mask=False)
        else:
            pm("sym310", 24,
               (c("natural") == "wetland") & c("wetland").isin(["bog", "fen", "marsh"]),
               gdf_polys)

        # 311 Studna / fontána — OSM fallback (ZABAGED® nemá samostatnou vrstvu)
        cgdf = isom("311")
        if cgdf is not None:
            pm("sym311", 52, None, cgdf, to_mask=False)
        else:
            pm("sym311", 52,
               c("amenity").isin(["fountain", "water_point", "drinking_water"]) |
               c("man_made").isin(["water_well", "water_works", "reservoir_covered"]),
               gdf_pts)

        # 312 Pramen — ZABAGED® 4.01 ZdrojPodzemnichVod
        cgdf = isom("312")
        if cgdf is not None:
            pm("sym312", 52, None, cgdf, to_mask=False)
        elif zab("ZdrojPodzemnichVod"):
            pm("sym312", 52, None, zab("ZdrojPodzemnichVod"), to_mask=False)
        else:
            pm("sym312", 52, (c("natural") == "spring") & (c("covered") != "yes"), gdf_centroids)

        # 302 Mělká vodní plocha
        cgdf = isom("302")
        if cgdf is not None:
            pm("sym302", 27, None, cgdf, to_mask=False)
        else:
            pm("sym302", 27,
               c("natural").isin(["water"]) & c("water").isin(["pond", "stream"]) |
               c("waterway").isin(["riverbank"]),
               gdf_polys)

        # 303 Napajedlo
        cgdf = isom("303")
        if cgdf is not None:
            pm("sym303", 52, None, cgdf, to_mask=False)
        else:
            pm("sym303", 52,
               c("amenity").isin(["watering_place"]) | c("natural").isin(["waterhole"]),
               gdf_centroids)

        # 313 Výrazný vodní prvek
        cgdf = isom("313")
        if cgdf is not None:
            pm("sym313", 52, None, cgdf, to_mask=False)

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

        # 401 Trvalý travní porost — ZABAGED® 6.06 TrvalyTravniPorost
        if isom("401") is None and zab("TrvalyTravniPorost"):
            pm("sym401", 1.0, None, zab("TrvalyTravniPorost"), to_mask=False)
        elif isom("401") is None:
            pm("sym401", 1.0,
               c("landuse").isin(["grassland", "grass", "meadow"]) | c("natural").isin(["grassland"]),
               gdf_polys)

        # 403 Drsná otevřená plocha — OSM fallback (ZABAGED® nemá přímý ekvivalent)
        cgdf = isom("403")
        if cgdf is not None:
            pm("sym403", 1.0, None, cgdf, to_mask=False)
        else:
            pm("sym403", 1.0,
               c("natural").isin(["heath", "fell"]),
               gdf_polys)

        # 404 Drsná plocha s rozptýlenými stromy
        cgdf = isom("404")
        if cgdf is not None:
            pm("sym404", 1.0, None, cgdf, to_mask=False)
        else:
            pm("sym404", 1.0,
               c("natural").isin(["heath"]) & (c("landuse") == "wood"),
               gdf_polys)

        # 405 Les: dobrá průchodnost — ZABAGED® 6.07 LesniPudaSeStromy
        cgdf = isom("405")
        if cgdf is not None:
            pm("sym405", 1.1, None, cgdf, to_mask=False)
        elif zab("LesniPudaSeStromy"):
            mask = _get_col(zab("LesniPudaSeStromy"), "druhporost_p").isin(
                ["jehličnatý", "listnatý", "smíšený"])
            pm("sym405", 1.1, mask, zab("LesniPudaSeStromy"))
        else:
            pm("sym405", 1.1,
               c("landuse").isin(["forest"]) | c("natural").isin(["wood"]),
               gdf_polys)

        # 406 Vegetace: pomalý běh
        cgdf = isom("406")
        if cgdf is not None:
            pm("sym406", 1.2, None, cgdf, to_mask=False)

        # 407 Vegetace: pomalý běh, dobrá viditelnost
        cgdf = isom("407")
        if cgdf is not None:
            pm("sym407", 1.3, None, cgdf, to_mask=False)

        # 408 Vegetace: chůze
        cgdf = isom("408")
        if cgdf is not None:
            pm("sym408", 1.5, None, cgdf, to_mask=False)

        # 409 Vegetace: chůze, dobrá viditelnost
        cgdf = isom("409")
        if cgdf is not None:
            pm("sym409", 1.6, None, cgdf, to_mask=False)

        # 410 Vegetace: boj
        cgdf = isom("410")
        if cgdf is not None:
            pm("sym410", 1.7, None, cgdf, to_mask=False)

        # 411 Vegetace: neprostupná — ZABAGED® 6.08 LesniPudaSKrovinatymPorostem
        cgdf = isom("411")
        if cgdf is not None:
            pm("sym411", 1.8, None, cgdf, to_mask=False)
        elif zab("LesniPudaSKrovinatymPorostem"):
            pm("sym411", 1.8, None, zab("LesniPudaSKrovinatymPorostem"), to_mask=False)

        # 412 Orná půda — ZABAGED® nemá přímý ekvivalent (6.02 je centroid), OSM fallback
        cgdf = isom("412")
        if cgdf is not None:
            pm("sym412a", 1.9, None, cgdf, to_mask=False)
        else:
            pm("sym412a", 1.9, c("landuse") == "farmland", gdf_polys)

        # 413 Sad
        cgdf = isom("413")
        if cgdf is not None:
            pm("sym413", 1.9, None, cgdf, to_mask=False)
        else:
            pm("sym413", 1.9, c("landuse") == "orchard", gdf_polys)

        # 414 Vinice
        cgdf = isom("414")
        if cgdf is not None:
            pm("sym414", 1.9, None, cgdf, to_mask=False)
        else:
            pm("sym414", 1.9, c("landuse") == "vineyard", gdf_polys)

        # 415 Hranice kultivace
        cgdf = isom("415")
        if cgdf is not None:
            pm("sym415", 2.0, None, cgdf, to_mask=False)

        # 416 Hranice vegetace
        cgdf = isom("416")
        if cgdf is not None:
            pm("sym416l", 2.0, None, cgdf, to_mask=False)

        # 417 Významný strom — ZABAGED® 6.11 VyznamnyNeboOsamelyStromLesik
        if isom("417") is None and zab("VyznamnyNeboOsamelyStromLesik"):
            pm("sym417a", 54, None, zab("VyznamnyNeboOsamelyStromLesik"), to_mask=False)
            pm("sym417b", 55, None, zab("VyznamnyNeboOsamelyStromLesik"), to_mask=False)
        elif isom("417") is None:
            pm("sym417a", 54, c("natural") == "tree", gdf_centroids)
            pm("sym417b", 55, c("natural") == "tree", gdf_centroids)

        # 418 Výrazný keř
        cgdf = isom("418")
        if cgdf is not None:
            pm("sym418a", 54, None, cgdf, to_mask=False)
            pm("sym418b", 55, None, cgdf, to_mask=False)
        else:
            mask_shrub = (c("natural") == "shrub")
            pm("sym418a", 54, mask_shrub, gdf_pts)
            pm("sym418b", 55, mask_shrub, gdf_pts)

        # 419 Výrazný vegetační prvek
        cgdf = isom("419")
        if cgdf is not None:
            pm("sym419", 56, None, cgdf, to_mask=False)

    # ----------------------------------------------------------------
    # ROADS
    # ----------------------------------------------------------------
    if visibility.get("roads", True):
        # 501 Zpevněná plocha
        cgdf = isom("501")
        if cgdf is not None:
            pm("sym501", 44, None, cgdf, to_mask=False)
        else:
            pm("sym501", 44,
               (c("highway").isin(["pedestrian"]) & (c("area") == "yes")) |
               c("amenity").isin(["parking"]) |
               c("landuse").isin(["garages"]),
               gdf_polys)

        # 502D Dálnice/rychlostní komunikace — ZABAGED® 2.01 SilniceDalnice
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

        # 502 Silnice — ZABAGED® 2.01 SilniceDalnice (ostatní)
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

        # 503 Vozová cesta — ZABAGED® 2.03 Cesta (zpevněná)
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

        # 504 Cesta — ZABAGED® 2.03 Cesta (udržovaná)
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

        # 506 Pěšina — ZABAGED® 2.04 Pesina
        cgdf = isom("506")
        if cgdf is not None:
            pm("sym506", 45, None, cgdf, to_mask=False)
        elif zab("Pesina"):
            pm("sym506", 45, None, zab("Pesina"), to_mask=False)
        else:
            pm("sym506", 45, c("highway") == "path", gdf_lines)

        # 507 Méně zřetelná stezka
        cgdf = isom("507")
        if cgdf is not None:
            pm("sym507", 45, None, cgdf, to_mask=False)
        else:
            pm("sym507", 45,
               c("highway").isin(["path", "footway"]) & c("trail_visibility").isin(["bad", "horrible"]),
               gdf_lines)

        # 508 Průsek — ZABAGED® 6.13 LesniPrusek
        cgdf = isom("508")
        if cgdf is not None:
            pm("sym508", 45, None, cgdf, to_mask=False)
        elif zab("LesniPrusek"):
            pm("sym508", 45, None, zab("LesniPrusek"), to_mask=False)
        else:
            pm("sym508", 45,
               c("man_made").isin(["cutline"]) | (c("highway") == "track") & (c("operator") == "forestry"),
               gdf_lines)

        # 509 Železnice — ZABAGED® 2.17 ZeleznicniTrat
        mask_rail = c("railway").isin(["rail", "narrow_gauge"]) & ~c("tunnel").isin(["yes"])
        cgdf = isom("509")
        for sym, zo in [("sym509a", 40), ("sym509b", 41)]:
            if cgdf is not None:
                pm(sym, zo, None, cgdf, to_mask=False)
            elif zab("ZeleznicniTrat"):
                pm(sym, zo, None, zab("ZeleznicniTrat"), to_mask=False)
            else:
                pm(sym, zo, mask_rail, gdf_lines)

        # 511 Stožár el. vedení
        cgdf = isom("511")
        if cgdf is not None:
            pm("sym511P", 71, None, cgdf, to_mask=False)
        else:
            pm("sym511P", 71,
               c("power").isin(["tower", "pole"]),
               gdf_pts)

        # 512 Most — ZABAGED® 2.08 Most
        cgdf = isom("512")
        if cgdf is not None:
            pm("sym512", 46, None, cgdf, to_mask=False)
        elif zab("Most"):
            pm("sym512", 46, None, zab("Most"), to_mask=False)
        else:
            pm("sym512", 46,
               c("bridge").isin(["yes", "viaduct"]) & c("highway").notna() & (c("highway") != ""),
               gdf_lines)

        # 514 Zřícená zeď — OSM fallback (ZABAGED® nemá samostatnou vrstvu)
        cgdf = isom("514")
        if cgdf is not None:
            pm("sym514", 30, None, cgdf, to_mask=False)
        else:
            pm("sym514", 30,
               (c("barrier") == "wall") & c("historic").isin(["yes", "ruins"]),
               gdf_lines)

        # 515 Nepřekonatelná zeď — ZABAGED® 1.23 Zed (doplněno přes OSM access)
        cgdf = isom("515")
        if cgdf is not None:
            for s in ["sym515a", "sym515b"]:
                pm(s, 30, None, cgdf, to_mask=False)
        else:
            mask_imp_wall = (c("barrier") == "wall") & c("access").isin(["no", "private"])
            pm("sym515a", 30, mask_imp_wall, gdf_lines)
            pm("sym515b", 30, mask_imp_wall, gdf_lines)

        # 516 Plot — OSM fallback
        cgdf = isom("516")
        if cgdf is not None:
            pm("sym516", 30, None, cgdf, to_mask=False)
        else:
            pm("sym516", 30,
               c("barrier").isin(["fence", "railing", "wire_fence", "chain_link_fence"]),
               gdf_lines)

        # 517 Zřícený plot — OSM fallback
        cgdf = isom("517")
        if cgdf is not None:
            pm("sym517", 30, None, cgdf, to_mask=False)
        else:
            pm("sym517", 30,
               c("barrier").isin(["fence"]) & c("historic").isin(["yes", "ruins"]),
               gdf_lines)

        # 518 Nepřekonatelný plot — OSM fallback
        cgdf = isom("518")
        if cgdf is not None:
            pm("sym518", 30, None, cgdf, to_mask=False)
        else:
            pm("sym518", 30,
               c("barrier").isin(["fence", "wall"]) & c("access").isin(["no", "private"]) &
               ~(c("barrier") == "wall"),
               gdf_lines)

        # 519 Průchod plotem/zdí
        cgdf = isom("519")
        if cgdf is not None:
            pm("sym519", 56, None, cgdf, to_mask=False)
        else:
            pm("sym519", 56,
               c("barrier").isin(["gate", "kissing_gate", "stile", "lift_gate"]),
               gdf_pts)

    # ----------------------------------------------------------------
    # MAN_MADE
    # ----------------------------------------------------------------
    if visibility.get("man_made", True):
        # 510 El. vedení — ZABAGED® 3.03 ElektrickeVedeni
        cgdf = isom("510")
        if cgdf is not None:
            pm("sym510", 70, None, cgdf, to_mask=False)
        elif zab("ElektrickeVedeni"):
            pm("sym510", 70, None, zab("ElektrickeVedeni"), to_mask=False)
        else:
            pm("sym510", 70, c("power").isin(["line", "minor_line"]), gdf_lines)

        # 513 Zeď — ZABAGED® 1.23 Zed
        cgdf = isom("513.1")
        if cgdf is not None:
            for s in ["sym513-1a", "sym513-1b"]:
                pm(s, 30, None, cgdf, to_mask=False)
        elif zab("Zed"):
            for s in ["sym513-1a", "sym513-1b"]:
                pm(s, 30, None, zab("Zed"), to_mask=False)
        else:
            pm("sym513-1a", 30, c("barrier") == "wall", gdf_lines)

        # 521 Budova — ZABAGED® 1.02 BudovaJednotlivaNeboBlokBudov
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

        # 520 Privátní oblast — OSM fallback (ZABAGED® 1.27 ArealUceloveZastavby jako bonus)
        if visibility.get("private", True):
            cgdf = isom("520")
            if cgdf is not None:
                pm("sym520", 1.5, None, cgdf, to_mask=False)
            else:
                # ZABAGED® 1.27 ArealUceloveZastavby
                if zab("ArealUceloveZastavby") is not None:
                    pm("sym520", 1.5, None, zab("ArealUceloveZastavby"), to_mask=False)
                else:
                    pm("sym520", 1.5,
                       c("landuse").isin(["residential", "industrial", "commercial", "allotments",
                                          "cemetery", "military", "quarry"]),
                       gdf_polys)

        # 523 Ruina — ZABAGED® 1.19 RozvalinazRicenina
        cgdf = isom("523")
        if cgdf is not None:
            pm("sym523", 50, None, cgdf, to_mask=False)
        elif zab("RozvalinazRicenina"):
            pm("sym523", 50, None, zab("RozvalinazRicenina"), to_mask=False)
        else:
            pm("sym523", 50,
               c("building").isin(["ruins"]) | c("historic").isin(["ruins"]),
               gdf_centroids)

        # 524 Věž — OSM fallback (ZABAGED® 1.03 VezVezovitaNastavba jen body)
        cgdf = isom("524")
        if cgdf is not None:
            for s in ["sym524a", "sym524b"]:
                pm(s, 56, None, cgdf, to_mask=False)
        else:
            mask_tower = c("man_made").isin(["tower", "chimney", "water_tower",
                                              "communications_tower", "mast"])
            pm("sym524a", 56, mask_tower, gdf_pts)
            pm("sym524b", 56, mask_tower, gdf_pts)

        # 522 Přístřešek
        cgdf = isom("522")
        if cgdf is not None:
            pm("sym522", 50, None, cgdf, to_mask=False)
        else:
            pm("sym522", 50,
               c("building").isin(["roof", "canopy", "carport"]) | (c("amenity") == "shelter"),
               gdf_polys)

        # 525 Malá věž
        cgdf = isom("525")
        if cgdf is not None:
            pm("sym525", 56, None, cgdf, to_mask=False)
        else:
            pm("sym525", 56,
               c("man_made").isin(["surveillance", "flagpole"]) |
               c("historic").isin(["milestone"]) |
               (c("amenity") == "hunting_stand"),
               gdf_pts)

        # 526 Pomník — ZABAGED® 1.20 MohylaPomnikNahrobek
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

        # 527 Krmítko — OSM fallback (ZABAGED® nemá samostatnou vrstvu)
        cgdf = isom("527")
        if cgdf is not None:
            pm("sym527", 56, None, cgdf, to_mask=False)
        else:
            pm("sym527", 56,
               c("man_made").isin(["feeding_place", "wildlife_feeding_place"]),
               gdf_pts)

        # 528–531 Výrazné prvky
        for code, sym in [("528", "sym528"), ("529", "sym529"), ("530", "sym530")]:
            cgdf = isom(code)
            if cgdf is not None:
                pm(sym, 30, None, cgdf, to_mask=False)

        cgdf = isom("531")
        if cgdf is not None:
            pm("sym531", 56, c("artwork_type").isin(["statue"]) | c("playground").isin(["structure"]),
               gdf_pts)

        # 532 Schody — OSM fallback (ZABAGED® nemá samostatnou vrstvu)
        cgdf = isom("532")
        if cgdf is not None:
            for s in ["sym532a", "sym532b", "sym532c"]:
                pm(s, 46, None, cgdf, to_mask=False)
        else:
            mask_steps = (c("highway") == "steps")
            for s in ["sym532a", "sym532b", "sym532c"]:
                pm(s, 46, mask_steps, gdf_lines)