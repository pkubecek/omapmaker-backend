"""
downloader.py — stahování LiDAR dlaždic z ČÚZK ATOM feedu a merge do jednoho LAZ souboru.
Přepsáno z OMapMaker_v7.py bez tkinter závislostí.
"""
import os
import ssl
import zipfile
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
import numpy as np
import laspy
from pyproj import CRS, Transformer

_SSL_CTX = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE
CUZK_ATOM_DMR5G = "https://atom.cuzk.gov.cz/DMR5G-SJTSK/DMR5G-SJTSK.xml"
CUZK_ATOM_DMP1G = "https://atom.cuzk.gov.cz/DMP1G-SJTSK/DMP1G-SJTSK.xml"
CUZK_ATOM_DMPOK = "https://atom.cuzk.gov.cz/DMPOK-SJTSK-LAZ/DMPOK-SJTSK-LAZ.xml"


def _parse_atom_feed_tiles(atom_url: str) -> list:
    """Parsuje ATOM feed a vrátí seznam (tile_id, sub_url, min_lat, min_lon, max_lat, max_lon)."""
    tiles = []
    try:
        req = urllib.request.Request(atom_url, headers={"User-Agent": "OMapMaker/7"})
        with urllib.request.urlopen(req, timeout=300, context=_SSL_CTX) as resp:
            raw = resp.read()
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "georss": "http://www.georss.org/georss",
        }
        root_el = ET.fromstring(raw)
        for entry in root_el.findall("atom:entry", ns):
            id_el = entry.find("atom:id", ns)
            tile_id = id_el.text.strip() if id_el is not None else ""
            link_el = entry.find("atom:link[@rel='alternate']", ns)
            if link_el is None:
                link_el = entry.find("atom:link", ns)
            sub_url = link_el.get("href", "") if link_el is not None else ""
            poly_el = entry.find("georss:polygon", ns)
            bbox = None
            if poly_el is not None and poly_el.text:
                coords = list(map(float, poly_el.text.strip().split()))
                lats = coords[0::2]
                lons = coords[1::2]
                bbox = (min(lats), min(lons), max(lats), max(lons))
            else:
                box_el = entry.find("georss:box", ns)
                if box_el is not None and box_el.text:
                    p = list(map(float, box_el.text.strip().split()))
                    bbox = (p[0], p[1], p[2], p[3])
            if sub_url and bbox:
                tiles.append((tile_id, sub_url, bbox[0], bbox[1], bbox[2], bbox[3]))
    except Exception as e:
        print(f"[downloader] Chyba při parsování ATOM feedu: {e}")
    return tiles


def _get_download_url_from_subfeed(sub_feed_url: str) -> str | None:
    """Načte sub-feed dlaždice a vrátí URL ZIP souboru."""
    try:
        req = urllib.request.Request(sub_feed_url, headers={"User-Agent": "OMapMaker/7"})
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
            raw = resp.read()
        root_el = ET.fromstring(raw)
        for link_el in root_el.iter("{http://www.w3.org/2005/Atom}link"):
            href = link_el.get("href", "")
            if href.lower().endswith(".zip"):
                return href
    except Exception as e:
        print(f"[downloader] Chyba sub-feedu ({sub_feed_url}): {e}")
    return None


def _download_tile(tile_id: str, sub_url: str, dest_dir: str, file_list: list,
                   progress_cb=None) -> None:
    """Stáhne jednu dlaždici (ZIP → LAZ/LAS) do dest_dir."""
    zip_url = _get_download_url_from_subfeed(sub_url)
    if not zip_url:
        print(f"[downloader] Nelze najít ZIP pro {tile_id}")
        return
    name = tile_id.split("_")[-1] if "_" in tile_id else tile_id.replace("/", "_")
    zip_path = os.path.join(dest_dir, f"{name}.zip")
    try:
        req = urllib.request.Request(zip_url, headers={"User-Agent": "OMapMaker/7"})
        with urllib.request.urlopen(req, timeout=120, context=_SSL_CTX) as r, open(zip_path, "wb") as f:
            f.write(r.read())
        with zipfile.ZipFile(zip_path) as zf:
            for n2 in zf.namelist():
                if n2.lower().endswith((".laz", ".las")):
                    zf.extract(n2, dest_dir)
                    file_list.append(os.path.join(dest_dir, n2))
        os.remove(zip_path)
        print(f"[downloader] OK: {name}")
        if progress_cb:
            progress_cb(f"Staženo: {name}")
    except Exception as e:
        print(f"[downloader] Chyba {name}: {e}")


def merge_laz_files(input_paths: list, output_path: str,
                    clip_bbox_wgs84: tuple | None = None) -> bool:
    """
    Sloučí více LAZ/LAS souborů do jednoho. Volitelně ořízne na WGS84 bbox.
    clip_bbox_wgs84: (min_lat, min_lon, max_lat, max_lon)
    """
    if not input_paths:
        return False

    clip_bounds_native = None
    if clip_bbox_wgs84 is not None:
        try:
            mn_lat, mn_lon, mx_lat, mx_lon = clip_bbox_wgs84
            with laspy.open(input_paths[0]) as fh_tmp:
                try:
                    src_crs = fh_tmp.header.parse_crs()
                    if src_crs is None:
                        raise ValueError("No CRS")
                except Exception:
                    src_crs = CRS.from_epsg(5514)
            wgs84 = CRS.from_epsg(4326)
            if src_crs != wgs84:
                t = Transformer.from_crs(wgs84, src_crs, always_xy=True)
                cx, cy = t.transform(
                    [mn_lon, mx_lon, mn_lon, mx_lon],
                    [mn_lat, mn_lat, mx_lat, mx_lat],
                )
                clip_bounds_native = (min(cx), max(cx), min(cy), max(cy))
            else:
                clip_bounds_native = (mn_lon, mx_lon, mn_lat, mx_lat)
        except Exception as e:
            print(f"[downloader] Varování: clip bbox selhal ({e}), merge bez ořezu.")

    if len(input_paths) == 1 and clip_bounds_native is None:
        import shutil
        shutil.copy2(input_paths[0], output_path)
        return True

    try:
        all_x, all_y, all_z, all_cls = [], [], [], []
        header_ref = None
        for path in input_paths:
            with laspy.open(path) as fh:
                if header_ref is None:
                    header_ref = fh.header
                for chunk in fh.chunk_iterator(500_000):
                    cx = np.array(chunk.x)
                    cy = np.array(chunk.y)
                    cz = np.array(chunk.z)
                    cc = np.array(chunk.classification)
                    if clip_bounds_native is not None:
                        bx0, bx1, by0, by1 = clip_bounds_native
                        m = (cx >= bx0) & (cx <= bx1) & (cy >= by0) & (cy <= by1)
                        cx, cy, cz, cc = cx[m], cy[m], cz[m], cc[m]
                    if len(cx):
                        all_x.append(cx)
                        all_y.append(cy)
                        all_z.append(cz)
                        all_cls.append(cc)

        if not all_x:
            print("[downloader] Varování: po ořezu nezůstaly žádné body!")
            return False

        x = np.concatenate(all_x)
        y = np.concatenate(all_y)
        z = np.concatenate(all_z)
        cls = np.concatenate(all_cls)

        las_out = laspy.LasData(
            header=laspy.LasHeader(
                point_format=header_ref.point_format,
                version=header_ref.version,
            )
        )
        las_out.header.offsets = np.array([x.min(), y.min(), z.min()])
        las_out.header.scales = np.array([0.01, 0.01, 0.01])
        try:
            if header_ref.parse_crs() is not None:
                las_out.header.set_crs(header_ref.parse_crs())
        except Exception:
            pass
        las_out.x = x
        las_out.y = y
        las_out.z = z
        las_out.classification = cls
        las_out.write(output_path)
        print(f"[downloader] Merge hotov: {len(x):,} bodů → {os.path.basename(output_path)}")
        return True
    except Exception as e:
        print(f"[downloader] Chyba při merge: {e}")
        return False


def download_cuzk(
    bbox: dict,
    dsm_type: str,
    out_dir: str,
    progress_cb=None,
) -> dict:
    """
    Hlavní funkce: stáhne DMR 5G + DMP (1G nebo OK) pro daný bbox.

    bbox: { min_lat, min_lon, max_lat, max_lon }  (WGS84)
    dsm_type: 'DMPOK' nebo 'DMP1G'
    out_dir: výstupní složka na serveru
    progress_cb: volitelná funkce(msg: str) pro reportování průběhu

    Vrací: { dmr_path, dmp_path }
    """
    os.makedirs(out_dir, exist_ok=True)
    mn_lat = bbox["min_lat"]
    mn_lon = bbox["min_lon"]
    mx_lat = bbox["max_lat"]
    mx_lon = bbox["max_lon"]

    use_dmpok = dsm_type == "DMPOK"
    dmp_atom_url = CUZK_ATOM_DMPOK if use_dmpok else CUZK_ATOM_DMP1G
    dmp_merged_name = "DMPOK_merged.laz" if use_dmpok else "DMP1G_merged.laz"

    def _cb(msg):
        print(f"[downloader] {msg}")
        if progress_cb:
            progress_cb(msg)

    _cb("Načítám ATOM feed DMR 5G...")
    dmr_tiles = _parse_atom_feed_tiles(CUZK_ATOM_DMR5G)
    _cb(f"Načítám ATOM feed {dsm_type}...")
    dmp_tiles = _parse_atom_feed_tiles(dmp_atom_url)

    def _overlap(tiles):
        return [
            (tid, su)
            for (tid, su, tla, tlo, txa, txo) in tiles
            if txa >= mn_lat and tla <= mx_lat and txo >= mn_lon and tlo <= mx_lon
        ]

    dmr_t = _overlap(dmr_tiles)
    dmp_t = _overlap(dmp_tiles)

    if not dmr_t:
        raise ValueError("Žádné DMR 5G dlaždice pro vybranou oblast. Je oblast v ČR?")

    dmr_raw_dir = os.path.join(out_dir, "dmr_tiles")
    dmp_raw_dir = os.path.join(out_dir, "dmp_tiles")
    os.makedirs(dmr_raw_dir, exist_ok=True)
    os.makedirs(dmp_raw_dir, exist_ok=True)

    dmr_files, dmp_files = [], []

    _cb(f"Stahuji {len(dmr_t)} DMR 5G dlaždic...")
    for i, (tid, su) in enumerate(dmr_t, 1):
        _cb(f"DMR {i}/{len(dmr_t)}: {tid}")
        _download_tile(tid, su, dmr_raw_dir, dmr_files)

    _cb(f"Stahuji {len(dmp_t)} {dsm_type} dlaždic...")
    for i, (tid, su) in enumerate(dmp_t, 1):
        _cb(f"{dsm_type} {i}/{len(dmp_t)}: {tid}")
        _download_tile(tid, su, dmp_raw_dir, dmp_files)

    _cb("Mergování DMR dlaždic...")
    dmr_merged = os.path.join(out_dir, "DMR5G_merged.laz")
    bbox_tuple = (mn_lat, mn_lon, mx_lat, mx_lon)
    ok_dmr = merge_laz_files(dmr_files, dmr_merged, clip_bbox_wgs84=bbox_tuple)
    if not ok_dmr:
        raise RuntimeError("Merge DMR selhal. Zkuste menší oblast.")

    dmp_merged = None
    if dmp_files:
        _cb(f"Mergování {dsm_type} dlaždic...")
        dmp_merged = os.path.join(out_dir, dmp_merged_name)
        merge_laz_files(dmp_files, dmp_merged, clip_bbox_wgs84=bbox_tuple)

    _cb("Hotovo!")
    return {
        "dmr_path": dmr_merged,
        "dmp_path": dmp_merged or "",
    }
