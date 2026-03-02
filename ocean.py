from typing import Dict, Tuple
import cairosvg
import io
import numpy as np
from PIL import Image
from scipy import ndimage
from models import BasinStats

def rasterize_svg(svg_path: str, out_w: int, out_h: int) -> np.ndarray:
    png_bytes = cairosvg.svg2png(url=svg_path, output_width=out_w, output_height=out_h)
    img = Image.open(io.BytesIO(png_bytes)).convert('RGBA')
    return np.array(img)

def ocean_mask_from_rgba(rgba: np.ndarray) -> np.ndarray:
    r = rgba[:, :, 0].astype(np.int16)
    g = rgba[:, :, 1].astype(np.int16)
    b = rgba[:, :, 2].astype(np.int16)
    a = rgba[:, :, 3].astype(np.int16)
    ocean = (a > 0) & (r < 70) & (g < 70) & (b > 170)
    return ocean

def mountain_mask_from_svg(height_svg: str, out_w: int, out_h: int) -> np.ndarray:
    """Rasterize a height SVG and return a boolean mask where
    rgb(255,0,0) pixels mark high-mountain regions."""
    rgba = rasterize_svg(height_svg, out_w=out_w, out_h=out_h)
    r = rgba[:, :, 0].astype(np.int16)
    g = rgba[:, :, 1].astype(np.int16)
    b = rgba[:, :, 2].astype(np.int16)
    return (r > 200) & (g < 60) & (b < 60)

def label_basins(ocean: np.ndarray) -> Tuple[np.ndarray, int]:
    labels, n = ndimage.label(ocean)
    if n <= 1:
        return (labels, n)
    parent = list(range(n + 1))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra
    h, w = ocean.shape
    if w >= 2:
        for y in range(h):
            la = int(labels[y, 0])
            if la == 0:
                continue
            for dy in (-1, 0, 1):
                yy = y + dy
                if yy < 0 or yy >= h:
                    continue
                lb = int(labels[yy, w - 1])
                if lb != 0:
                    union(la, lb)
    out = labels.copy()
    mapping: Dict[int, int] = {}
    next_id = 1
    nonzero = out > 0
    roots = np.zeros_like(out, dtype=np.int32)
    roots[nonzero] = np.vectorize(find)(out[nonzero])
    for root in np.unique(roots[nonzero]):
        mapping[int(root)] = next_id
        next_id += 1
    for root, new_id in mapping.items():
        out[roots == root] = new_id
    return (out, next_id - 1)

def itcz_offset(month: int) -> float:
    """Saisonale ITCZ-Verschiebung in Breitengraden (positiv = nordwärts).

    Januar: ~-5° (ITCZ südlich des Äquators)
    Juli:   ~+5° (ITCZ nördlich, Monsun-Maximum)
    """
    import math
    month_norm = (month - 1) % 12
    return -5.0 * math.cos(2.0 * math.pi * month_norm / 12.0)

def windstress_curl_forcing(lat: float, month: int=4) -> float:
    offset = itcz_offset(month)
    lat_shifted = lat - offset
    a = abs(lat_shifted)
    if a < 3.0 or a > 78.0:
        return 0.0
    hemi = 1.0 if lat_shifted >= 0.0 else -1.0
    subtropic = np.exp(-((a - 30.0) / 14.0) ** 2)
    subpolar = np.exp(-((a - 55.0) / 14.0) ** 2)
    return hemi * (1.25 * subtropic - 0.8 * subpolar)

def find_gyre_boundaries_px(h: int, h_px: int, month: int=4) -> list[float]:
    profile = [windstress_curl_forcing(lat_from_y(float(y), h_px), month=month) for y in range(h)]
    boundaries: list[float] = []
    for y in range(1, h):
        f0, f1 = (profile[y - 1], profile[y])
        if f0 * f1 < 0 and abs(f0) + abs(f1) > 0.05:
            t = abs(f0) / (abs(f0) + abs(f1))
            boundaries.append(float(y - 1) + t)
    return boundaries

def find_western_boundary_strips(basin_mask: np.ndarray, ocean_full: np.ndarray, min_span_frac: float=0.2) -> np.ndarray:
    h, w = basin_mask.shape
    ys_occ = np.where(np.any(basin_mask, axis=1))[0]
    if len(ys_occ) == 0:
        return np.zeros_like(basin_mask, dtype=bool)
    basin_h = int(ys_occ[-1] - ys_occ[0] + 1)
    min_span = max(3, int(basin_h * min_span_frac))
    strip_w = max(1, w // 50)
    wb = np.zeros_like(basin_mask, dtype=bool)
    for x in range(w):
        x_prev = (x - 1) % w
        boundary_pixels = basin_mask[:, x] & ~ocean_full[:, x_prev]
        if not np.any(boundary_pixels):
            continue
        if np.sum(boundary_pixels) < min_span:
            continue
        for dx in range(strip_w):
            wb[:, (x + dx) % w] |= basin_mask[:, (x + dx) % w]
    return wb

def build_streamfunction_currents(labels: np.ndarray, basins: Dict[int, BasinStats], h_px: int, coarsen: int=6, beta_eff: float=0.15, month: int=4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve
    h, w = labels.shape
    ch = max(4, h // coarsen)
    cw = max(4, w // coarsen)
    ocean_full = labels > 0
    h_blk = ch * coarsen
    w_blk = cw * coarsen
    ocean_work = np.zeros((h_blk, w_blk), dtype=np.float32)
    ocean_work[:min(h, h_blk), :min(w, w_blk)] = ocean_full[:min(h, h_blk), :min(w, w_blk)].astype(np.float32)
    labels_work = np.zeros((h_blk, w_blk), dtype=np.int32)
    labels_work[:min(h, h_blk), :min(w, w_blk)] = labels[:min(h, h_blk), :min(w, w_blk)]
    blocks_ocean = ocean_work.reshape(ch, coarsen, cw, coarsen)
    ocean_frac = blocks_ocean.mean(axis=(1, 3))
    ocean_c = ocean_frac > 0.65
    sub_labels_c, n_sub = ndimage.label(ocean_c, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    forcing = np.zeros((ch, cw), dtype=np.float64)
    for cy in range(ch):
        lat = lat_from_y(cy * h / ch + h / (2 * ch), h_px)
        forcing[cy, :] = windstress_curl_forcing(lat, month=month)
    psi_c = np.zeros((ch, cw), dtype=np.float64)
    half_beta = beta_eff / 2.0
    for lab in range(1, n_sub + 1):
        mask_c = sub_labels_c == lab
        if not np.any(mask_c):
            continue
        pts = [(int(r), int(c)) for r, c in np.argwhere(mask_c)]
        n = len(pts)
        if n < 16:
            continue
        k_map: Dict[Tuple[int, int], int] = {pt: k for k, pt in enumerate(pts)}
        A = lil_matrix((n, n), dtype=np.float64)
        b = np.zeros(n, dtype=np.float64)
        for k, (y, x) in enumerate(pts):
            nbrs_laplacian = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
            nbrs_laplacian = [(ny, nx) for ny, nx in nbrs_laplacian if 0 <= ny < ch and 0 <= nx < cw]
            A[k, k] = float(-len(nbrs_laplacian))
            b[k] = -forcing[y, x]
            for ny, nx in nbrs_laplacian:
                j = k_map.get((ny, nx), -1)
                if j < 0:
                    continue
                if ny == y:
                    is_east = nx == x + 1
                    is_west = nx == x - 1
                    if is_east and (not is_west):
                        A[k, j] += 1.0 + half_beta
                    elif is_west and (not is_east):
                        A[k, j] += 1.0 - half_beta
                    else:
                        A[k, j] += 1.0
                else:
                    A[k, j] += 1.0
        try:
            psi_loc = spsolve(A.tocsr(), b)
            if not (np.any(np.isnan(psi_loc)) or np.any(np.isinf(psi_loc))):
                for k, (y, x) in enumerate(pts):
                    psi_c[y, x] = float(psi_loc[k])
        except Exception:
            pass
    psi_c = ndimage.gaussian_filter(psi_c, sigma=1.5)
    psi = ndimage.zoom(psi_c, (h / ch, w / cw), order=1)
    y_arr = np.arange(h, dtype=np.float64)
    sigma_y = max(6.0, h * 0.067)
    gyre_ys = find_gyre_boundaries_px(h, h_px, month=month)
    sub_labels_full, n_sub_full = ndimage.label(ocean_full, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    for sub_lab in range(1, n_sub_full + 1):
        basin_mask = sub_labels_full == sub_lab
        ys_occ = np.where(np.any(basin_mask, axis=1))[0]
        if len(ys_occ) < 10:
            continue
        lat_top = lat_from_y(float(ys_occ[0]), h_px)
        lat_bot = lat_from_y(float(ys_occ[-1]), h_px)
        if lat_top - lat_bot < 40.0:
            continue
        xs_occ = np.where(np.any(basin_mask, axis=0))[0]
        basin_lon_span = 360.0 * len(xs_occ) / max(1, w)
        amoc_scale = 1.0
        if basin_lon_span > 100.0:
            amoc_scale = 0.08
        elif lat_top < 30.0:
            amoc_scale = 0.15
        wb_mask = find_western_boundary_strips(basin_mask, ocean_full)
        psi_range_basin = float(np.max(np.abs(psi[basin_mask])))
        psi_range_wb = float(np.max(np.abs(psi[wb_mask]))) if np.any(wb_mask) else 0.0
        psi_range = max(psi_range_basin, psi_range_wb)
        forcing_peak = float(np.max(np.abs(forcing))) if forcing.size > 0 else 1.0
        amoc_amp = max(0.4 * psi_range, 0.18 * forcing_peak) * amoc_scale
        pert_1d = np.zeros(h, dtype=np.float64)
        for yb in gyre_ys:
            pert_1d += 0.5 * amoc_amp * np.tanh((y_arr - yb) / sigma_y)
        sigma_broad = max(8.0, h * 0.028)
        basin_smooth = ndimage.gaussian_filter(basin_mask.astype(np.float64), sigma=sigma_broad)
        if np.any(wb_mask):
            wb_boost = ndimage.gaussian_filter(wb_mask.astype(np.float64), sigma=max(4.0, h * 0.014))
            weight = basin_smooth * (1.0 + 2.5 * wb_boost)
        else:
            weight = basin_smooth
        psi += pert_1d[:, np.newaxis] * weight
    dpsi_dy = np.gradient(psi, axis=0)
    dpsi_dx = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / 2.0
    u = dpsi_dy
    v = -dpsi_dx
    v_sst = -dpsi_dx
    u[~ocean_full] = 0.0
    v[~ocean_full] = 0.0
    v_sst[~ocean_full] = 0.0
    mag = np.sqrt(u * u + v * v)
    if np.any(ocean_full):
        p95 = float(np.percentile(mag[ocean_full], 95))
        if p95 > 1e-09:
            u /= p95
            v /= p95
            v_sst /= p95
    return (u, v, v_sst)

def compute_basin_stats(labels: np.ndarray, n_labels: int, min_area: int) -> Dict[int, BasinStats]:
    stats: Dict[int, BasinStats] = {}
    objects = ndimage.find_objects(labels)
    for i, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        region = labels[sl] == i
        area = int(region.sum())
        if area < min_area:
            continue
        ys, xs = np.where(region)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        stats[i] = BasinStats(label=i, area=area, x_min=int(xs.min()), x_max=int(xs.max()), x_mean=float(xs.mean()), y_mean=float(ys.mean()))
    return stats

def lat_from_y(y_px: float, h_px: int) -> float:
    return 90.0 - 180.0 * (y_px / max(1, h_px - 1))

def lon_from_x(x_px: float, w_px: int) -> float:
    """Längengrad in [-180, 180] für äquidistantes Raster."""
    return -180.0 + 360.0 * (x_px / max(1, w_px - 1))
