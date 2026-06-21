"""
poland_downloader.py — stahování LiDAR dlaždic z polského GUGiK geoportálu.

Polský geoportal (GUGiK) poskytuje LiDAR point cloudy (LAZ) a NMT/NMPT (GeoTIFF)
přes WFS službu. Dlaždice se vybírají podle bbox v EPSG:4326, atribut
url_do_pobrania obsahuje přímý odkaz ke stažení.

DTM (NMT - Numeryczny Model Terenu):
  WFS: https://mapy.geoportal.gov.pl/wss/service/PZGIK/NumerycznyModelTerenuEVRF2007/WFS/Skorowidze
  TypeName: gugik:SkorowidzNumerycznegoModeluTerenu{YEAR}
  Formát: GeoTIFF (ARC/INFO ASCII Grid), CRS EPSG:2180

DSM (NMPT - Numeryczny Model Powierzchni Terenu):
  WFS: https://mapy.geoportal.gov.pl/wss/service/PZGIK/NumerycznyModelPowierzchniEVRF2007/WFS/Skorowidze
  TypeName: gugik:SkorowidzNumerycznegoModeluPowierzchniTerenu{YEAR}
  Formát: GeoTIFF, CRS EPSG:2180

LiDAR point cloudy (alternativa k NMT):
  WFS: https://mapy.geoportal.gov.pl/wss/service/PZGIK/DanePomiaroweLidarEVRF2007/WFS/Skorowidze
  TypeName: gugik:SkorowidzDanychPomiarowychLIDAR{YEAR}
  Formát: LAZ, CRS EPSG:2180

Výstupní CRS dat: EPSG:2180 (PL-2000 PUWG 1992 / GRS80)
"""
import os
import ssl
import zipfile
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
import numpy as np
from pyproj import CRS, Transformer

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

# -------------------------------------------------------------------------
# WFS endpoints
# -------------------------------------------------------------------------
_WFS_LIDAR = (
    "https://mapy.geoportal.gov.pl/wss/service/PZGIK"
    "/DanePomiaroweLidarEVRF2007/WFS/Skorowidze"
)
_WFS_NMT = (
    "https://mapy.geoportal.gov.pl/wss/service/PZGIK"
    "/NumerycznyModelTerenuEVRF2007/WFS/Skorowidze"
)
_WFS_NMT_KRON86 = (
    "https://mapy.geoportal.gov.pl/wss/service/PZGIK"
    "/NumerycznyModelTerenuKRON86/WFS/Skorowidze"
)
_WFS_NMPT = (
    "https://mapy.geoportal.gov.pl/wss/service/PZGIK"
    "/NumerycznyModelPowierzchniEVRF2007/WFS/Skorowidze"
)

# Roky dostupných dat (od nejnovějšího) — WFS má vrstvy per rok
# EVRF2007: LiDAR 2018-2022, NMT 2018-2020, NMPT 2018-2021
_LIDAR_YEARS = [2022, 2021, 2020, 2019, 2018]
_NMT_YEARS   = [2020, 2019, 2018]
_NMPT_YEARS  = [2021, 2020, 2019, 2018]
# KRON86 fallback pro oblasti bez EVRF2007 pokryti (2000-2019)
_NMT_KRON86_YEARS = [2019, 2018, 2017, 2016, 2015]

_HEADERS = {"User-Agent": "OMapMaker/7 (orienteering map tool)"}


def _bbox_wgs84_to_2180(bbox_wgs84: tuple) -> tuple:
    """Transformuje bbox z WGS84 na EPSG:2180 (PL-1992)."""
    mn_lat, mn_lon, mx_lat, mx_lon = bbox_wgs84
    t = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
    x0, y0 = t.transform(mn_lon, mn_lat)
    x1, y1 = t.transform(mx_lon, mx_lat)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _wfs_get_feature(wfs_url: str, type_name: str,
                     bbox_wgs84: tuple, max_features: int = 500) -> list[dict]:
    """
    Volá WFS GetFeature a vrátí seznam dlaždic jako dict s klíčem 'url'.
    bbox_wgs84: (min_lat, min_lon, max_lat, max_lon)
    Vrátí: [{"url": "https://...", "name": "..."}]

    GUGiK WFS vyžaduje bbox v EPSG:2180. Zkouší WFS 2.0.0, fallback na 1.1.0.
    """
    mn_lat, mn_lon, mx_lat, mx_lon = bbox_wgs84

    # Transformuj bbox do EPSG:2180 — GUGiK WFS odmítá WGS84 bbox (vrací 400)
    try:
        bx0, by0, bx1, by1 = _bbox_wgs84_to_2180(bbox_wgs84)
    except Exception as e:
        print(f"[pl_downloader] Transformace bbox selhala: {e}, zkouším WGS84")
        bx0, by0, bx1, by1 = mn_lon, mn_lat, mx_lon, mx_lat

    # Malý buffer 200m pro edge dlaždice
    BUFFER = 200
    bx0 -= BUFFER; by0 -= BUFFER; bx1 += BUFFER; by1 += BUFFER

    _DL_EXTS = (".laz", ".las", ".zip", ".tif", ".tiff", ".asc", ".xyz")

    # Skutečný clip bbox bez bufferu pro filtrování GML features
    try:
        clip_x0, clip_y0, clip_x1, clip_y1 = _bbox_wgs84_to_2180(bbox_wgs84)
    except Exception:
        clip_x0, clip_y0, clip_x1, clip_y1 = bx0, by0, bx1, by1

    def _bbox_from_gml_coords(coords_text: str, swap_xy: bool = False):
        """Parsuj souřadnice z gml:coordinates nebo gml:posList, vrať (x0,y0,x1,y1)."""
        try:
            # gml:coordinates: "x,y x,y ..." nebo "x,y z x,y z ..."
            pairs = coords_text.strip().split()
            xs, ys = [], []
            for p in pairs:
                parts = p.split(",")
                if len(parts) >= 2:
                    a, b = float(parts[0]), float(parts[1])
                    if swap_xy:
                        xs.append(b); ys.append(a)
                    else:
                        xs.append(a); ys.append(b)
            if xs and ys:
                return min(xs), min(ys), max(xs), max(ys)
        except Exception:
            pass
        try:
            # gml:posList: "x y x y ..." (párové hodnoty)
            nums = list(map(float, coords_text.strip().split()))
            if len(nums) >= 4 and len(nums) % 2 == 0:
                if swap_xy:
                    ys = nums[0::2]; xs = nums[1::2]
                else:
                    xs = nums[0::2]; ys = nums[1::2]
                return min(xs), min(ys), max(xs), max(ys)
        except Exception:
            pass
        return None

    def _feature_overlaps_bbox(feature_el) -> bool:
        """Zkontroluj jestli GML feature překrývá clip bbox."""
        GML = "http://www.opengis.net/gml"
        GML32 = "http://www.opengis.net/gml/3.2"

        def _test_bbox(fx0, fy0, fx1, fy1) -> bool | None:
            """Vrátí True/False pokud souřadnice vypadají jako EPSG:2180, jinak None."""
            # EPSG:2180: X (easting) 150000–950000, Y (northing) 100000–850000
            # Obě souřadnice musí být v realistickém rozsahu pro Polsko
            x_ok = 100000 < fx0 < 1000000 and 100000 < fx1 < 1000000
            y_ok = 50000 < fy0 < 900000 and 50000 < fy1 < 900000
            if x_ok and y_ok:
                return (fx1 >= clip_x0 and fx0 <= clip_x1 and
                        fy1 >= clip_y0 and fy0 <= clip_y1)
            return None

        for ns in [GML, GML32]:
            # posList (GML 3.x)
            for el in feature_el.findall(f".//{{{ns}}}posList"):
                if not el.text: continue
                for swap in [False, True]:
                    b = _bbox_from_gml_coords(el.text, swap_xy=swap)
                    if b:
                        r = _test_bbox(*b)
                        if r is not None: return r

            # coordinates (GML 2/3.1)
            for el in feature_el.findall(f".//{{{ns}}}coordinates"):
                if not el.text: continue
                b = _bbox_from_gml_coords(el.text, swap_xy=False)
                if b:
                    r = _test_bbox(*b)
                    if r is not None: return r

            # Envelope lowerCorner/upperCorner
            lc = feature_el.find(f".//{{{ns}}}lowerCorner")
            uc = feature_el.find(f".//{{{ns}}}upperCorner")
            if lc is not None and uc is not None and lc.text and uc.text:
                try:
                    lp = list(map(float, lc.text.split()))
                    up = list(map(float, uc.text.split()))
                    for fx0, fy0, fx1, fy1 in [
                        (lp[0], lp[1], up[0], up[1]),
                        (lp[1], lp[0], up[1], up[0]),  # swap
                    ]:
                        r = _test_bbox(fx0, fy0, fx1, fy1)
                        if r is not None: return r
                except Exception:
                    pass

        # Zkus i bez namespace (WFS 1.0 někdy vrací holé tagy)
        for tag in ["coordinates", "posList"]:
            for el in feature_el.findall(f".//{tag}"):
                if not el.text: continue
                b = _bbox_from_gml_coords(el.text, swap_xy=False)
                if b:
                    r = _test_bbox(*b)
                    if r is not None: return r

        # Geometrie nenalezena — vyřaď (safe default: raději přeskočit než stáhnout 30 souborů)
        print(f"[pl_downloader]   WARNING: geometrie nenalezena v feature, přeskakuji")
        return False

    def _parse_tiles(raw: bytes, apply_bbox_filter: bool = True) -> list[dict]:
        raw_str = raw.decode("utf-8", errors="replace")
        print(f"[pl_downloader] WFS odpověď ({type_name}): {raw_str[:300]}")
        try:
            root = ET.fromstring(raw)
        except ET.ParseError as e:
            print(f"[pl_downloader] XML parse chyba: {e}")
            return []

        tiles = []
        skipped = 0

        # Najdi feature members — každý obsahuje geometrii + URL
        _MEMBER_TAGS = [
            "{http://www.opengis.net/wfs}featureMember",
            "{http://www.opengis.net/wfs/2.0}member",
            "featureMember", "member",
        ]
        members = []
        for tag in _MEMBER_TAGS:
            members = root.findall(f".//{tag}")
            if members:
                break

        # Debug: vypiš první feature jako XML
        if members:
            import xml.etree.ElementTree as ET2
            first_xml = ET2.tostring(members[0], encoding="unicode")
            print(f"[pl_downloader] První feature XML: {first_xml[:800]}")

        if members:
            for i, member in enumerate(members):
                overlaps = _feature_overlaps_bbox(member)
                if apply_bbox_filter and not overlaps:
                    skipped += 1
                    if i < 3:
                        print(f"[pl_downloader]   feature {i} přeskočena (overlaps={overlaps})")
                    continue
                # Hledej URL
                url = None
                for el in member.iter():
                    text = (el.text or "").strip()
                    if text.startswith("http") and any(text.lower().endswith(ext) for ext in _DL_EXTS):
                        url = text
                        break
                if not url:
                    for el in member.iter():
                        href = el.get("{http://www.w3.org/1999/xlink}href", "") or el.get("href", "")
                        if href.startswith("http") and any(href.lower().endswith(ext) for ext in _DL_EXTS):
                            url = href
                            break
                if url:
                    name = os.path.basename(url)
                    tiles.append({"url": url, "name": name})
                    print(f"[pl_downloader]   dlaždice: {name}")
        else:
            # Fallback bez member struktury
            for el in root.iter():
                text = (el.text or "").strip()
                if text.startswith("http") and any(text.lower().endswith(ext) for ext in _DL_EXTS):
                    tiles.append({"url": text, "name": os.path.basename(text)})
            if not tiles:
                for el in root.iter():
                    href = el.get("{http://www.w3.org/1999/xlink}href", "") or el.get("href", "")
                    if href.startswith("http") and any(href.lower().endswith(ext) for ext in _DL_EXTS):
                        tiles.append({"url": href, "name": os.path.basename(href)})

        if skipped:
            print(f"[pl_downloader]   přeskočeno mimo bbox: {skipped} dlaždic")
        return tiles

    # Pokus 1: WFS 2.0.0 s EPSG:2180 bbox
    bbox_str_2180 = f"{bx0:.2f},{by0:.2f},{bx1:.2f},{by1:.2f},urn:ogc:def:crs:EPSG::2180"
    url_v2 = (
        f"{wfs_url}?SERVICE=WFS&REQUEST=GetFeature&VERSION=2.0.0"
        f"&TYPENAMES={type_name}"
        f"&BBOX={bbox_str_2180}"
        f"&COUNT={max_features}"
    )
    print(f"[pl_downloader] WFS 2.0 URL: {url_v2}")
    try:
        req = urllib.request.Request(url_v2, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as resp:
            raw = resp.read()
        tiles = _parse_tiles(raw, apply_bbox_filter=True)  # vždy filtruj
        if tiles:
            return tiles
    except urllib.error.HTTPError as e:
        print(f"[pl_downloader] WFS 2.0 chyba ({type_name}): {e}")
    except Exception as e:
        print(f"[pl_downloader] WFS 2.0 chyba ({type_name}): {e}")
        return []

    # Pokus 2: WFS 1.1.0 s EPSG:2180 bbox
    bbox_str_v11 = f"{bx0:.2f},{by0:.2f},{bx1:.2f},{by1:.2f},EPSG:2180"
    url_v11 = (
        f"{wfs_url}?SERVICE=WFS&REQUEST=GetFeature&VERSION=1.1.0"
        f"&TYPENAME={type_name}"
        f"&BBOX={bbox_str_v11}"
        f"&MAXFEATURES={max_features}"
    )
    print(f"[pl_downloader] WFS 1.1 URL: {url_v11}")
    try:
        req = urllib.request.Request(url_v11, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as resp:
            raw = resp.read()
        tiles = _parse_tiles(raw, apply_bbox_filter=True)
        if tiles:
            return tiles
    except Exception as e:
        print(f"[pl_downloader] WFS 1.1 chyba ({type_name}): {e}")

    # Pokus 3: WFS 1.0.0 — nefiltruje bbox, použijeme post-processing filtr
    url_v10 = (
        f"{wfs_url}?SERVICE=WFS&REQUEST=GetFeature&VERSION=1.0.0"
        f"&TYPENAME={type_name}"
        f"&BBOX={bx0:.2f},{by0:.2f},{bx1:.2f},{by1:.2f}"
        f"&MAXFEATURES={max_features}"
    )
    print(f"[pl_downloader] WFS 1.0 URL: {url_v10}")
    try:
        req = urllib.request.Request(url_v10, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as resp:
            raw = resp.read()
        return _parse_tiles(raw, apply_bbox_filter=True)  # filtruj GML geometrie
    except Exception as e:
        print(f"[pl_downloader] WFS 1.0 chyba ({type_name}): {e}")

    return []



def _query_tiles(wfs_url: str, years: list[int],
                 type_prefix: str, bbox_wgs84: tuple,
                 progress_cb=None) -> list[dict]:
    """
    Prochází roky od nejnovějšího a sbírá dlaždice pokrývající bbox.
    Každá dlaždice se započítá jen jednou (preferuje novější rok).
    Jakmile najde dlaždice v daném roce, přestane (nejnovější data stačí).
    """
    seen_names = set()
    all_tiles = []

    for year in years:
        type_name = f"gugik:{type_prefix}{year}"
        if progress_cb:
            progress_cb(f"Dotazuji WFS {type_name}...")
        tiles = _wfs_get_feature(wfs_url, type_name, bbox_wgs84)
        new_tiles = []
        for t in tiles:
            if t["name"] not in seen_names:
                seen_names.add(t["name"])
                new_tiles.append(t)
        if new_tiles:
            all_tiles.extend(new_tiles)
            if progress_cb:
                progress_cb(f"  rok {year}: {len(new_tiles)} nových dlaždic")
            # Máme data z nejnovějšího dostupného roku — stop
            break

    return all_tiles


def _download_file(url: str, dest_path: str, progress_cb=None) -> bool:
    """Stáhne jeden soubor. Vrátí True při úspěchu."""
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=300, context=_SSL_CTX) as r, \
             open(dest_path, "wb") as f:
            f.write(r.read())
        return True
    except Exception as e:
        if progress_cb:
            progress_cb(f"  Chyba stahování {os.path.basename(url)}: {e}")
        return False


def _extract_if_zip(zip_path: str, dest_dir: str) -> list[str]:
    """Rozbalí ZIP a vrátí seznam extrahovaných souborů."""
    extracted = []
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if name.lower().endswith((".laz", ".las", ".tif", ".tiff", ".asc")):
                    zf.extract(name, dest_dir)
                    extracted.append(os.path.join(dest_dir, name))
        os.remove(zip_path)
    except Exception as e:
        print(f"[pl_downloader] ZIP chyba {zip_path}: {e}")
    return extracted


def _merge_tif_to_laz(tif_paths: list[str], output_laz: str,
                       progress_cb=None) -> bool:
    """
    Konvertuje rastrové GeoTIFF/ASC soubory NMT do LAZ formátu kompatibilního
    s pipeline.py (klasifikace bodů jako ground class=2).
    """
    try:
        import rasterio
        import laspy

        all_x, all_y, all_z = [], [], []
        src_crs = None

        for tif_path in tif_paths:
            if progress_cb:
                progress_cb(f"  Konvertuji {os.path.basename(tif_path)}...")
            with rasterio.open(tif_path) as src:
                if src_crs is None:
                    src_crs = src.crs
                data = src.read(1)
                nodata = src.nodata
                rows, cols = np.where(data != nodata if nodata is not None else np.ones_like(data, dtype=bool))
                xs, ys = rasterio.transform.xy(src.transform, rows, cols)
                zs = data[rows, cols].astype(np.float64)
                valid = np.isfinite(zs) & (zs > -9000)
                all_x.append(np.array(xs)[valid])
                all_y.append(np.array(ys)[valid])
                all_z.append(zs[valid])

        if not all_x:
            return False

        x = np.concatenate(all_x)
        y = np.concatenate(all_y)
        z = np.concatenate(all_z)

        header = laspy.LasHeader(point_format=0, version="1.2")
        header.scales = np.array([0.01, 0.01, 0.01])
        header.offsets = np.array([x.min(), y.min(), z.min()])

        # Uložíme CRS jako WKT do VLR
        if src_crs is not None:
            try:
                wkt = src_crs.to_wkt()
                header.vlrs.append(laspy.LasAppender.make_vlr(
                    user_id="LASF_Projection",
                    record_id=2112,
                    description="OGC Coordinate System WKT",
                    record_data=wkt.encode("utf-8"),
                ))
            except Exception:
                pass

        las = laspy.LasData(header=header)
        las.x = x
        las.y = y
        las.z = z
        las.classification = np.full(len(x), 2, dtype=np.uint8)  # ground
        las.write(output_laz)

        if progress_cb:
            progress_cb(f"  Zapsáno {len(x):,} bodů → {os.path.basename(output_laz)}")
        return True

    except Exception as e:
        print(f"[pl_downloader] Konverze TIF→LAZ chyba: {e}")
        import traceback; traceback.print_exc()
        return False


def _merge_laz_epsg2180(input_paths: list, output_path: str,
                         bbox_wgs84: tuple, progress_cb=None) -> bool:
    """
    Merge LAZ/LAS souborů v EPSG:2180 s ořezem podle WGS84 bbox.
    Na rozdíl od ČÚZK merge_laz_files: předpokládá EPSG:2180 vstup,
    transformuje clip bbox do EPSG:2180 přímo (bez čtení CRS z LAZ headeru).
    """
    import gc
    import laspy
    import numpy as np
    from pyproj import Transformer

    if not input_paths:
        return False

    # Transformuj clip bbox WGS84 → EPSG:2180
    mn_lat, mn_lon, mx_lat, mx_lon = bbox_wgs84
    t = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
    xs, ys = t.transform([mn_lon, mx_lon, mn_lon, mx_lon],
                          [mn_lat, mn_lat, mx_lat, mx_lat])
    bx0, bx1 = min(xs), max(xs)
    by0, by1 = min(ys), max(ys)
    # Malý buffer 100m pro jistotu
    bx0 -= 100; by0 -= 100; bx1 += 100; by1 += 100

    if progress_cb:
        progress_cb(f"  Clip bbox EPSG:2180: {bx0:.0f},{by0:.0f} .. {bx1:.0f},{by1:.0f}")

    try:
        with laspy.open(input_paths[0]) as fh_tmp:
            header_ref = fh_tmp.header

        out_header = laspy.LasHeader(
            point_format=header_ref.point_format,
            version=header_ref.version,
        )
        out_header.scales = np.array([0.01, 0.01, 0.01])

        global_min_x, global_min_y, global_min_z = np.inf, np.inf, np.inf
        for path in input_paths:
            with laspy.open(path) as fh:
                hdr = fh.header
                global_min_x = min(global_min_x, float(hdr.x_min))
                global_min_y = min(global_min_y, float(hdr.y_min))
                global_min_z = min(global_min_z, float(hdr.z_min))
        out_header.offsets = np.array([global_min_x, global_min_y, global_min_z])

        total_written = 0
        CHUNK_SIZE = 200_000

        with laspy.open(output_path, mode="w", header=out_header) as out_fh:
            for path in input_paths:
                if progress_cb:
                    progress_cb(f"  Mergování: {os.path.basename(path)}")
                with laspy.open(path) as fh:
                    for chunk in fh.chunk_iterator(CHUNK_SIZE):
                        cx = np.array(chunk.x)
                        cy = np.array(chunk.y)
                        cz = np.array(chunk.z)
                        cc = np.array(chunk.classification)
                        m = (cx >= bx0) & (cx <= bx1) & (cy >= by0) & (cy <= by1)
                        if not np.any(m):
                            continue
                        cx, cy, cz, cc = cx[m], cy[m], cz[m], cc[m]
                        out_chunk = laspy.ScaleAwarePointRecord.zeros(len(cx), header=out_header)
                        out_chunk.x = cx
                        out_chunk.y = cy
                        out_chunk.z = cz
                        out_chunk.classification = cc
                        out_fh.write_points(out_chunk)
                        total_written += len(cx)
                        del cx, cy, cz, cc, out_chunk
                gc.collect()

        if total_written == 0:
            print("[pl_downloader] Varování: po ořezu nezůstaly žádné body!")
            # Zkus merge bez ořezu jako fallback
            if progress_cb:
                progress_cb("  Clip selhal, zkouším merge bez ořezu...")
            with laspy.open(output_path, mode="w", header=out_header) as out_fh:
                for path in input_paths:
                    with laspy.open(path) as fh:
                        for chunk in fh.chunk_iterator(CHUNK_SIZE):
                            out_chunk = laspy.ScaleAwarePointRecord.zeros(
                                len(chunk.x), header=out_header)
                            out_chunk.x = np.array(chunk.x)
                            out_chunk.y = np.array(chunk.y)
                            out_chunk.z = np.array(chunk.z)
                            out_chunk.classification = np.array(chunk.classification)
                            out_fh.write_points(out_chunk)
                            total_written += len(chunk.x)
            if progress_cb:
                progress_cb(f"  Merge bez ořezu: {total_written:,} bodů")

        print(f"[pl_downloader] Merge: {total_written:,} bodů → {os.path.basename(output_path)}")
        return total_written > 0

    except Exception as e:
        print(f"[pl_downloader] Merge chyba: {e}")
        import traceback; traceback.print_exc()
        return False



def _merge_laz_dsm_epsg2180(input_paths: list, output_path: str,
                              bbox_wgs84: tuple, progress_cb=None) -> bool:
    """
    Vytvoří DSM LAZ ze stejných LiDAR souborů jako DTM,
    ale vybere všechny body KROMĚ země (class != 2) a noise (class != 7).
    Výsledek je kompatibilní s pipeline.py jako DSM vstup.
    """
    import gc
    import laspy
    import numpy as np
    from pyproj import Transformer

    if not input_paths:
        return False

    mn_lat, mn_lon, mx_lat, mx_lon = bbox_wgs84
    t = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
    xs, ys = t.transform([mn_lon, mx_lon, mn_lon, mx_lon],
                          [mn_lat, mn_lat, mx_lat, mx_lat])
    bx0, bx1 = min(xs) - 100, max(xs) + 100
    by0, by1 = min(ys) - 100, max(ys) + 100

    try:
        with laspy.open(input_paths[0]) as fh_tmp:
            header_ref = fh_tmp.header

        out_header = laspy.LasHeader(
            point_format=header_ref.point_format,
            version=header_ref.version,
        )
        out_header.scales = np.array([0.01, 0.01, 0.01])

        global_min_x, global_min_y, global_min_z = np.inf, np.inf, np.inf
        for path in input_paths:
            with laspy.open(path) as fh:
                hdr = fh.header
                global_min_x = min(global_min_x, float(hdr.x_min))
                global_min_y = min(global_min_y, float(hdr.y_min))
                global_min_z = min(global_min_z, float(hdr.z_min))
        out_header.offsets = np.array([global_min_x, global_min_y, global_min_z])

        total_written = 0
        CHUNK_SIZE = 200_000

        with laspy.open(output_path, mode="w", header=out_header) as out_fh:
            for path in input_paths:
                if progress_cb:
                    progress_cb(f"  DSM merge: {os.path.basename(path)}")
                with laspy.open(path) as fh:
                    for chunk in fh.chunk_iterator(CHUNK_SIZE):
                        cx = np.array(chunk.x)
                        cy = np.array(chunk.y)
                        cz = np.array(chunk.z)
                        cc = np.array(chunk.classification)
                        # bbox filter
                        m = (cx >= bx0) & (cx <= bx1) & (cy >= by0) & (cy <= by1)
                        # DSM: vše kromě noise (7) a unclassified který je pod zemí
                        # Ponecháme: 1 (unclass), 3 (low veg), 4 (med veg), 5 (high veg),
                        #             6 (building), 9 (water), 2 (ground) jako podádní body
                        # Vynecháme: 7 (noise), 18 (high noise)
                        m &= ~np.isin(cc, [7, 18])
                        if not np.any(m):
                            continue
                        cx, cy, cz, cc = cx[m], cy[m], cz[m], cc[m]
                        out_chunk = laspy.ScaleAwarePointRecord.zeros(len(cx), header=out_header)
                        out_chunk.x = cx
                        out_chunk.y = cy
                        out_chunk.z = cz
                        out_chunk.classification = cc
                        out_fh.write_points(out_chunk)
                        total_written += len(cx)
                        del cx, cy, cz, cc, out_chunk
                gc.collect()

        print(f"[pl_downloader] DSM merge: {total_written:,} bodů → {os.path.basename(output_path)}")
        return total_written > 0

    except Exception as e:
        print(f"[pl_downloader] DSM merge chyba: {e}")
        import traceback; traceback.print_exc()
        return False



def download_poland(
    bbox: dict,
    out_dir: str,
    use_lidar_point_cloud: bool = True,
    progress_cb=None,
) -> dict:
    """
    Hlavní funkce: stáhne DTM a DSM pro daný bbox z polského GUGiK.

    bbox: { min_lat, min_lon, max_lat, max_lon }  (WGS84)
    out_dir: výstupní složka
    use_lidar_point_cloud: True = stáhne LiDAR LAZ (lepší kvalita),
                           False = stáhne NMT rastr (rychlejší, menší)

    Vrací: { dmr_path, dmp_path, crs }
      crs = "EPSG:2180" (PL-2000 PUWG 1992)
      dmr_path = LAZ soubor s DTM
      dmp_path = LAZ nebo TIF soubor s DSM (nebo "" pokud nedostupný)
    """
    os.makedirs(out_dir, exist_ok=True)

    mn_lat = bbox["min_lat"]
    mn_lon = bbox["min_lon"]
    mx_lat = bbox["max_lat"]
    mx_lon = bbox["max_lon"]
    bbox_wgs84 = (mn_lat, mn_lon, mx_lat, mx_lon)

    def cb(msg):
        print(f"[pl_downloader] {msg}")
        if progress_cb:
            progress_cb(msg)

    # -------------------------------------------------------------------------
    # DTM
    # -------------------------------------------------------------------------
    dtm_raw_dir = os.path.join(out_dir, "dtm_tiles")
    os.makedirs(dtm_raw_dir, exist_ok=True)
    dtm_files = []

    if use_lidar_point_cloud:
        cb("Hledám LiDAR dlaždice (DTM)...")
        lidar_tiles = _query_tiles(
            _WFS_LIDAR, _LIDAR_YEARS,
            "SkorowidzDanychPomiarowychLIDAR",
            bbox_wgs84, progress_cb=cb,
        )
        if lidar_tiles:
            cb(f"Stahuju {len(lidar_tiles)} LiDAR dlaždic...")
            for i, tile in enumerate(lidar_tiles, 1):
                cb(f"  LiDAR {i}/{len(lidar_tiles)}: {tile['name']}")
                ext = os.path.splitext(tile["url"])[1].lower()
                dest = os.path.join(dtm_raw_dir, tile["name"] + (ext if ext else ".laz"))
                if _download_file(tile["url"], dest, cb):
                    if ext == ".zip":
                        dtm_files.extend(_extract_if_zip(dest, dtm_raw_dir))
                    else:
                        dtm_files.append(dest)
        else:
            cb("LiDAR dlaždice nenalezeny, zkouším NMT rastr...")
            use_lidar_point_cloud = False

    if not use_lidar_point_cloud or not dtm_files:
        cb("Hledám NMT (rastr DTM) dlaždice [EVRF2007]...")
        nmt_tiles = _query_tiles(
            _WFS_NMT, _NMT_YEARS,
            "SkorowidzNumerycznegoModeluTerenu",
            bbox_wgs84, progress_cb=cb,
        )

        # Fallback na KRON86 WFS pokud EVRF2007 nemá pokryti
        if not nmt_tiles:
            cb("EVRF2007 NMT nenalezen, zkouším KRON86 fallback...")
            nmt_tiles = _query_tiles(
                _WFS_NMT_KRON86, _NMT_KRON86_YEARS,
                "SkorowidzNumerycznegoModeluTerenu",
                bbox_wgs84, progress_cb=cb,
            )

        cb(f"Stahuju {len(nmt_tiles)} NMT dlaždic...")
        nmt_tif_files = []
        for i, tile in enumerate(nmt_tiles, 1):
            cb(f"  NMT {i}/{len(nmt_tiles)}: {tile['name']}")
            ext = os.path.splitext(tile["url"])[1].lower()
            dest = os.path.join(dtm_raw_dir, tile["name"] + (ext if ext else ".tif"))
            if _download_file(tile["url"], dest, cb):
                if ext == ".zip":
                    extracted = _extract_if_zip(dest, dtm_raw_dir)
                    nmt_tif_files.extend(extracted)
                elif ext in (".tif", ".tiff", ".asc"):
                    nmt_tif_files.append(dest)

        if not nmt_tif_files:
            raise RuntimeError("Žádné DTM dlaždice pro zadanou oblast. Je oblast v Polsku?")

        # Konvertuj TIF → LAZ (pipeline.py očekává LAZ jako vstup DTM)
        cb("Konvertuji NMT rastr → LAZ...")
        dtm_laz = os.path.join(out_dir, "PL_NMT_merged.laz")
        ok = _merge_tif_to_laz(nmt_tif_files, dtm_laz, cb)
        if not ok:
            raise RuntimeError("Konverze NMT TIF → LAZ selhala.")
        dtm_files = [dtm_laz]

    # Merge LAZ dlaždic do jednoho souboru (vždy s bbox ořezem)
    dtm_merged = os.path.join(out_dir, "PL_LiDAR_DTM_merged.laz")
    if dtm_files:
        cb("Merguji/ořezávám DTM dlaždice...")
        ok = _merge_laz_epsg2180(dtm_files, dtm_merged, bbox_wgs84, cb)
        if not ok:
            raise RuntimeError("Merge DTM LAZ selhal.")
    else:
        raise RuntimeError("Žádné DTM soubory k mergování.")

    # -------------------------------------------------------------------------
    # DSM — z LiDAR non-ground bodů (NMPT WFS vyžaduje autorizaci)
    # -------------------------------------------------------------------------
    dmp_merged = ""

    if dtm_files and use_lidar_point_cloud:
        cb("Vytvářím DSM z LiDAR non-ground bodů...")
        dmp_laz = os.path.join(out_dir, "PL_LiDAR_DSM_merged.laz")
        ok = _merge_laz_dsm_epsg2180(dtm_files, dmp_laz, bbox_wgs84, cb)
        if ok:
            dmp_merged = dmp_laz
        else:
            cb("DSM z LiDAR selhal — mapa bude bez vegetace.")
    else:
        cb("LiDAR nedostupný — DSM nebude k dispozici.")

    cb("Hotovo!")
    return {
        "dmr_path": dtm_merged,
        "dmp_path": dmp_merged,
        "crs": "EPSG:2180",
    }