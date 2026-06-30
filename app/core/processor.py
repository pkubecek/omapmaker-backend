"""
processor.py — zpracování bodových mračen, interpolace DTM/DSM,
klasifikace vegetace, vektorizace skal, terénní mikrotvary.
Přepsáno z OMapMaker_v7.py bez tkinter závislostí.
"""
import os
import tempfile
import numpy as np
import laspy
import rasterio
import rasterio.features
import rasterio.transform
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from scipy.interpolate import griddata
from scipy.ndimage import (
    binary_dilation, binary_erosion, gaussian_filter, label,
    find_objects, minimum_filter, binary_opening, binary_closing,
)
from shapely.geometry import MultiPoint, Point, shape
from shapely.ops import unary_union
import geopandas as gpd
from pyproj import CRS, Transformer


# ---------------------------------------------------------------------------
# DTM (DMR) loading
# ---------------------------------------------------------------------------

def load_dmr_grid(dmr_path: str, target_crs_code: str,
                  pixel_size: float = 0.5, sigma_smooth: float = 4,
                  bbox_clip: tuple | None = None,
                  progress_cb=None) -> tuple:
    """
    Načte bodové mračno DTM (.las/.laz), interpoluje na pravidelnou mřížku.
    bbox_clip: (minx, maxx, miny, maxy) pro ořez na dlaždici
    Vrací: (dmr_grid_cubic, grid_x, grid_y, extent, points, z)
    """
    def _cb(msg):
        print(f"[processor] {msg}")
        if progress_cb:
            progress_cb(msg)

    _cb(f"Načítám DTM: {os.path.basename(dmr_path)}")
    MAX_POINTS = 2_500_000
    xs, ys, zs = [], [], []
    transformer = None

    with laspy.open(dmr_path) as fh:
        try:
            source_crs = fh.header.parse_crs()
            if source_crs is None:
                raise ValueError("No CRS")
        except Exception:
            source_crs = CRS.from_epsg(5514)

        # Detekce CRS z rozsahu souřadnic — GUGiK LAZ nemá CRS v headeru
        # EPSG:5514 má záporné souřadnice; EPSG:2180 má kladné ~140k-900k
        hdr = fh.header
        x_min = float(hdr.x_min)
        if source_crs.equals(CRS.from_epsg(5514)) and x_min > 0:
            source_crs = CRS.from_epsg(2180)

        try:
            target_crs_obj = CRS.from_string(target_crs_code)
            if source_crs != target_crs_obj:
                transformer = Transformer.from_crs(source_crs, target_crs_obj, always_xy=True)
        except Exception as e:
            print(f"[processor] Varování transformace DTM: {e}")

        total_points = fh.header.point_count
        fraction = min(1.0, MAX_POINTS / total_points) if total_points > 0 else 1.0

        # Automatická detekce záměny os (cx/cy) — udělá se jen JEDNOU za celý
        # soubor na základě mediánu z prvního neprázdného chunku, ne podle
        # jediného bodu opakovaně pro každý chunk/dlaždici (to vedlo k tomu,
        # že se některé dlaždice nesprávně ořízly na prázdno).
        cx_is_easting = None
        if bbox_clip is not None:
            bx0, bx1, by0, by1 = bbox_clip
            e_mid = (bx0 + bx1) / 2
            n_mid = (by0 + by1) / 2

        for chunk in fh.chunk_iterator(1_000_000):
            clas = np.array(chunk.classification)
            ground_mask = (clas == 2) | (clas == 8)
            if not np.any(ground_mask):
                continue
            cx = np.array(chunk.x[ground_mask])
            cy = np.array(chunk.y[ground_mask])
            cz = np.array(chunk.z[ground_mask])
            if fraction < 1.0:
                rnd = np.random.rand(len(cx)) < fraction
                cx, cy, cz = cx[rnd], cy[rnd], cz[rnd]
            if len(cx) == 0:
                continue
            if transformer:
                cx, cy = transformer.transform(cx, cy)
            # Ořez na bbox dlaždice
            if bbox_clip is not None:
                if cx_is_easting is None:
                    n_sample = min(2000, len(cx))
                    med_x = np.median(cx[:n_sample])
                    cx_is_easting = abs(med_x - e_mid) < abs(med_x - n_mid)
                if cx_is_easting:
                    m = (cx >= bx0) & (cx <= bx1) & (cy >= by0) & (cy <= by1)
                else:
                    m = (cx >= by0) & (cx <= by1) & (cy >= bx0) & (cy <= bx1)
                cx, cy, cz = cx[m], cy[m], cz[m]
            if len(cx) == 0:
                continue
            xs.append(cx)
            ys.append(cy)
            zs.append(cz)

    if not xs:
        raise ValueError("DTM neobsahuje žádné body klasifikované jako terén (třídy 2, 8).")

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)
    _cb(f"DTM načteno: {len(x):,} bodů")

    buffer_dist = pixel_size
    min_x, max_x = x.min() - buffer_dist, x.max() + buffer_dist
    min_y, max_y = y.min() - buffer_dist, y.max() + buffer_dist
    extent = (min_x, max_x, min_y, max_y)

    grid_x, grid_y = np.mgrid[min_x:max_x:pixel_size, min_y:max_y:pixel_size]

    _cb("Interpoluji DTM (cubic)...")
    points = np.vstack((x, y)).T
    valid = np.isfinite(points).all(axis=1) & np.isfinite(z)
    points = points[valid]
    z = z[valid]

    shift_x = np.mean(points[:, 0])
    shift_y = np.mean(points[:, 1])
    pts_shifted = points - np.array([shift_x, shift_y])
    gx_shifted = grid_x - shift_x
    gy_shifted = grid_y - shift_y

    dmr_grid = griddata(pts_shifted, z, (gx_shifted, gy_shifted), method="cubic")
    mask_nan = np.isnan(dmr_grid)
    if np.any(mask_nan):
        dmr_grid_nearest = griddata(pts_shifted, z, (gx_shifted, gy_shifted), method="nearest")
        dmr_grid[mask_nan] = dmr_grid_nearest[mask_nan]
    dmr_grid = gaussian_filter(dmr_grid, sigma=sigma_smooth)

    return dmr_grid, grid_x, grid_y, extent, points, z


# ---------------------------------------------------------------------------
# DSM (DMP) loading
# ---------------------------------------------------------------------------

def load_dmp_grid(dmp_path: str, grid_x: np.ndarray, grid_y: np.ndarray,
                  extent: tuple, target_crs_code: str,
                  progress_cb=None) -> np.ndarray:
    """Načte DSM (.las/.laz nebo .tif) a interpoluje na stejnou mřížku jako DTM."""
    def _cb(msg):
        print(f"[processor] {msg}")
        if progress_cb:
            progress_cb(msg)

    _cb(f"Načítám DSM: {os.path.basename(dmp_path)}")
    MAX_POINTS = 2_500_000
    ext = os.path.splitext(dmp_path)[1].lower()

    if ext in (".las", ".laz"):
        xs, ys, zs = [], [], []
        transformer = None
        with laspy.open(dmp_path) as fh:
            try:
                source_crs = fh.header.parse_crs()
                if source_crs is None:
                    source_crs = CRS.from_epsg(5514)
            except Exception:
                source_crs = CRS.from_epsg(5514)

            # Detekce CRS z souřadnic
            if source_crs.equals(CRS.from_epsg(5514)) and float(fh.header.x_min) > 0:
                source_crs = CRS.from_epsg(2180)

            try:
                target_crs_obj = CRS.from_string(target_crs_code)
                if source_crs != target_crs_obj:
                    transformer = Transformer.from_crs(source_crs, target_crs_obj, always_xy=True)
            except Exception:
                pass

            total = fh.header.point_count
            fraction = min(1.0, MAX_POINTS / total) if total > 0 else 1.0

            for chunk in fh.chunk_iterator(500_000):
                cx = np.array(chunk.x)
                cy = np.array(chunk.y)
                cz = np.array(chunk.z)
                cc = np.array(chunk.classification)
                valid_mask = cc != 7
                if fraction < 1.0:
                    valid_mask &= np.random.rand(len(cx)) < fraction
                if np.any(valid_mask):
                    cx, cy, cz = cx[valid_mask], cy[valid_mask], cz[valid_mask]
                    if transformer:
                        cx, cy = transformer.transform(cx, cy)
                    xs.append(cx)
                    ys.append(cy)
                    zs.append(cz)

        if not xs:
            raise ValueError("DSM neobsahuje platná data.")

        x = np.concatenate(xs)
        y = np.concatenate(ys)
        z = np.concatenate(zs)
        pts = np.vstack((x, y)).T
        valid = np.isfinite(pts).all(axis=1) & np.isfinite(z)
        pts, z = pts[valid], z[valid]

    elif ext in (".tif", ".tiff"):
        with rasterio.open(dmp_path) as src:
            total_px = src.width * src.height
            if total_px > MAX_POINTS:
                scale = (MAX_POINTS / total_px) ** 0.5
                nw, nh = int(src.width * scale), int(src.height * scale)
                data = src.read(1, out_shape=(nh, nw), resampling=Resampling.bilinear)
                transform = src.transform * src.transform.scale(src.width / nw, src.height / nh)
            else:
                data = src.read(1)
                transform = src.transform
            rows, cols = np.indices(data.shape)
            xs2, ys2 = rasterio.transform.xy(transform, rows.flatten(), cols.flatten())
            z = data.flatten()
            if src.nodata is not None:
                mask = z != src.nodata
                x, y = np.array(xs2)[mask], np.array(ys2)[mask]
                z = z[mask]
            else:
                x, y = np.array(xs2), np.array(ys2)
        pts = np.vstack((x, y)).T
        valid = np.isfinite(pts).all(axis=1) & np.isfinite(z)
        pts, z = pts[valid], z[valid]
    else:
        raise ValueError(f"Nepodporovaný formát DSM: {ext}")

    _cb("Interpoluji DSM...")
    shift_x = np.mean(pts[:, 0])
    shift_y = np.mean(pts[:, 1])
    pts_shifted = pts - np.array([shift_x, shift_y])
    gx_shifted = grid_x - shift_x
    gy_shifted = grid_y - shift_y

    dmp_grid = griddata(pts_shifted, z, (gx_shifted, gy_shifted), method="linear")
    if np.isnan(dmp_grid).all():
        dmp_grid = griddata(pts_shifted, z, (gx_shifted, gy_shifted), method="nearest")

    return dmp_grid


# ---------------------------------------------------------------------------
# Vegetation classification
# ---------------------------------------------------------------------------

def classify_vegetation(vegetation_height: np.ndarray, bins: list,
                         transform, dmr_path: str,
                         progress_cb=None) -> gpd.GeoDataFrame:
    """
    Klasifikuje výšku vegetace do tříd a vektorizuje polygony.
    bins: [b1, b2, b3, b4] — hranice výšek v metrech
    """
    def _cb(msg):
        print(f"[processor] {msg}")
        if progress_cb:
            progress_cb(msg)

    _cb("Klasifikuji vegetaci...")
    full_bins = [-1, 0] + list(bins)
    class_names = {
        1: "Paseka", 2: "Louka", 3: "Nizky_porost",
        4: "Stredni_porost", 5: "Vysoky_porost", 6: "Les",
    }

    classified_raster_raw = np.digitize(
        np.nan_to_num(vegetation_height, nan=-9999), full_bins
    ).astype(np.int32)

    cleaned = classified_raster_raw.copy()
    struct = np.ones((3, 3), dtype=bool)
    for c in np.unique(cleaned):
        if c == 0:
            continue
        mask = cleaned == c
        mask = binary_opening(mask, structure=struct)
        mask = binary_closing(mask, structure=struct)
        cleaned[cleaned == c] = 0
        cleaned[mask] = c

    classified_raster = np.flipud(cleaned.T)
    pixel_area = abs(transform.a * transform.e)
    min_area = 50 * pixel_area
    mask = classified_raster != 0

    try:
        results = rasterio.features.shapes(classified_raster, mask=mask, transform=transform)
        features = []
        for geom, value in results:
            class_id = int(value)
            if class_id == 0:
                continue
            features.append({
                "geometry": shape(geom),
                "class_id": class_id,
                "class_name": class_names.get(class_id, "Neznama"),
            })
        if not features:
            return gpd.GeoDataFrame(columns=["class_name", "class_id", "geometry"])

        gdf = gpd.GeoDataFrame(features)
        gdf = gdf[gdf.geometry.area >= min_area]
        gdf.geometry = gdf.geometry.simplify(0.5, preserve_topology=True)
        dissolved = gdf.dissolve(by="class_name", aggfunc="first").reset_index()
        gdf.geometry = gdf.geometry.buffer(0.8).buffer(0)
        _cb(f"Vegetace vektorizována: {len(dissolved)} tříd")
        return dissolved
    except Exception as e:
        _cb(f"Chyba vektorizace vegetace: {e}")
        return gpd.GeoDataFrame(columns=["class_name", "class_id", "geometry"])


# ---------------------------------------------------------------------------
# Rock / cliff detection
# ---------------------------------------------------------------------------

def vectorize_rocks(grid_x: np.ndarray, grid_y: np.ndarray,
                    dmr_grid: np.ndarray, transform,
                    slope_threshold_deg: float = 54,
                    progress_cb=None) -> gpd.GeoDataFrame:
    """Detekuje skalní srázy podle sklonu terénu."""
    def _cb(msg):
        print(f"[processor] {msg}")
        if progress_cb:
            progress_cb(msg)

    _cb("Vektorizuji skály...")
    pixel_size_x = abs(transform.a)
    pixel_size_y = abs(transform.e)

    dy, dx = np.gradient(dmr_grid, pixel_size_y, pixel_size_x)
    slope = np.rad2deg(np.arctan(np.hypot(dx, dy)))

    valid_data_mask = (dmr_grid > 0) & (~np.isnan(dmr_grid))
    safe_mask = binary_erosion(valid_data_mask, iterations=7)
    rock_mask_raw = (slope > slope_threshold_deg) & safe_mask

    rock_area = rock_mask_raw.astype(np.int32).T
    rock_area = np.flipud(rock_area)

    pixel_area = pixel_size_x * pixel_size_y
    min_area = 10 * pixel_area

    if not np.any(rock_area):
        return gpd.GeoDataFrame(columns=["class_name", "geometry"])

    try:
        results = rasterio.features.shapes(rock_area, mask=(rock_area != 0), transform=transform)
        features = [{"geometry": shape(geom), "class_name": "Skala"} for geom, _ in results]
        if not features:
            return gpd.GeoDataFrame(columns=["class_name", "geometry"])
        gdf = gpd.GeoDataFrame(features)
        gdf = gdf[gdf.geometry.area >= min_area]
        gdf.geometry = gdf.geometry.buffer(0.4).simplify(0.3)
        dissolved = gdf.dissolve(by="class_name").reset_index()
        _cb("Skály vektorizovány")
        return dissolved
    except Exception as e:
        _cb(f"Chyba vektorizace skal: {e}")
        return gpd.GeoDataFrame(columns=["class_name", "geometry"])


# ---------------------------------------------------------------------------
# Fill sinks (depressions + knolls)
# ---------------------------------------------------------------------------

def _fill_depressions_numpy(dem: np.ndarray, iterations: int = 50) -> np.ndarray:
    """Numpy implementace Fill Sinks bez externích závislostí."""
    from scipy.ndimage import minimum_filter
    filled = dem.copy()
    border_val = np.nanmin(dem) - 1.0
    filled[0, :] = border_val
    filled[-1, :] = border_val
    filled[:, 0] = border_val
    filled[:, -1] = border_val
    for _ in range(iterations):
        neighbor_min = minimum_filter(filled, size=3, mode="nearest")
        new_filled = np.maximum(dem, neighbor_min)
        new_filled[0, :] = border_val
        new_filled[-1, :] = border_val
        new_filled[:, 0] = border_val
        new_filled[:, -1] = border_val
        if np.allclose(new_filled, filled, atol=1e-6):
            break
        filled = new_filled
    return filled


def _compute_depth_grid(fill_input: np.ndarray, grid_x: np.ndarray,
                         grid_y: np.ndarray, current_crs: str,
                         invert: bool = False) -> np.ndarray:
    """Spočítá depth/height grid přes pysheds nebo numpy fallback."""
    data = (-fill_input) if invert else fill_input
    try:
        from pysheds.grid import Grid as PyshedsGrid
        min_x, max_x = grid_x.min(), grid_x.max()
        min_y, max_y = grid_y.min(), grid_y.max()
        tform = from_bounds(min_x, min_y, max_x, max_y, data.shape[0], data.shape[1])
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name
        dem_for_write = data.T
        with rasterio.open(
            tmp_path, "w", driver="GTiff",
            height=dem_for_write.shape[0], width=dem_for_write.shape[1],
            count=1, dtype=dem_for_write.dtype, crs=current_crs,
            transform=tform, nodata=-9999,
        ) as dst:
            dst.write(dem_for_write, 1)
        grid = PyshedsGrid.from_raster(tmp_path)
        dem_rd = grid.read_raster(tmp_path)
        pit_filled = grid.fill_pits(dem_rd)
        dem_filled = grid.fill_depressions(pit_filled)
        depth = (np.array(dem_filled) - np.array(dem_rd)).T
        os.unlink(tmp_path)
        return depth
    except Exception:
        filled = _fill_depressions_numpy(data.T.astype(np.float64))
        return (filled - data.T).T


def find_depressions(grid_x, grid_y, dmr_grid, pixel_size=0.5,
                     min_diameter=1, max_diameter=5, min_depth=0.5,
                     current_crs="EPSG:5514", progress_cb=None) -> list:
    """Vrátí seznam Point geometrií pro malé prohlubně."""
    def _cb(msg):
        if progress_cb:
            progress_cb(msg)

    _cb("Hledám prohlubně...")
    valid_mask = (dmr_grid > 0) & (~np.isnan(dmr_grid))
    safe_mask = binary_erosion(valid_mask, iterations=3)
    fill_mean = np.nanmean(dmr_grid[valid_mask])
    fill_input = np.where(np.isnan(dmr_grid) | ~valid_mask, fill_mean, dmr_grid)

    depth_grid = _compute_depth_grid(fill_input, grid_x, grid_y, current_crs, invert=False)
    depression_mask = (depth_grid > min_depth) & safe_mask

    labeled, _ = label(depression_mask)
    slices = find_objects(labeled)
    pts = []
    for slc in slices:
        region = labeled[slc]
        region_mask = region > 0
        ny, nx = region_mask.shape
        diameter = max(ny, nx) * pixel_size
        if not (min_diameter <= diameter <= max_diameter):
            continue
        cy, cx = np.argwhere(region_mask).mean(axis=0)
        i0 = int(slc[0].start + cy)
        j0 = int(slc[1].start + cx)
        if i0 < dmr_grid.shape[0] and j0 < dmr_grid.shape[1]:
            pts.append(Point(float(grid_x[i0, j0]), float(grid_y[i0, j0])))
    _cb(f"Nalezeno {len(pts)} prohlubní")
    return pts


def find_knolls(grid_x, grid_y, dmr_grid, pixel_size=0.5,
                min_diameter=1.5, max_diameter=10, min_height=0.5,
                current_crs="EPSG:5514", progress_cb=None) -> list:
    """Vrátí seznam Point geometrií pro kupky."""
    def _cb(msg):
        if progress_cb:
            progress_cb(msg)

    _cb("Hledám kupky...")
    valid_mask = (dmr_grid > 0) & (~np.isnan(dmr_grid))
    safe_mask = binary_erosion(valid_mask, iterations=3)
    fill_mean = np.nanmean(dmr_grid[valid_mask])
    fill_input = np.where(np.isnan(dmr_grid) | ~valid_mask, fill_mean, dmr_grid)

    height_grid = _compute_depth_grid(fill_input, grid_x, grid_y, current_crs, invert=True)
    knoll_mask = (height_grid > min_height) & safe_mask

    labeled, _ = label(knoll_mask)
    slices = find_objects(labeled)
    pts = []
    for slc in slices:
        region = labeled[slc]
        region_mask = region > 0
        ny, nx = region_mask.shape
        diameter = max(ny, nx) * pixel_size
        if not (min_diameter <= diameter <= max_diameter):
            continue
        cy, cx = np.argwhere(region_mask).mean(axis=0)
        i0 = int(slc[0].start + cy)
        j0 = int(slc[1].start + cx)
        if i0 < dmr_grid.shape[0] and j0 < dmr_grid.shape[1]:
            pts.append(Point(float(grid_x[i0, j0]), float(grid_y[i0, j0])))
    _cb(f"Nalezeno {len(pts)} kupek")
    return pts


# ---------------------------------------------------------------------------
# Clip polygon from DTM points
# ---------------------------------------------------------------------------

def make_clip_polygon(points: np.ndarray):
    """Vytvoří konvexní obal z DTM bodů pro ořez vrstev."""
    try:
        safe = points[np.isfinite(points).all(axis=1)]
        poly = MultiPoint(safe).convex_hull
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly
    except Exception as e:
        print(f"[processor] Clip polygon error: {e}")
        return None