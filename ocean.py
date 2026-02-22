from typing import Dict, Tuple

import cairosvg
import io
import numpy as np
from PIL import Image
from scipy import ndimage

from models import BasinStats


def rasterize_svg(svg_path: str, out_w: int, out_h: int) -> np.ndarray:
    png_bytes = cairosvg.svg2png(
        url=svg_path,
        output_width=out_w,
        output_height=out_h,
    )
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    return np.array(img)


def ocean_mask_from_rgba(rgba: np.ndarray) -> np.ndarray:
    r = rgba[:, :, 0].astype(np.int16)
    g = rgba[:, :, 1].astype(np.int16)
    b = rgba[:, :, 2].astype(np.int16)
    a = rgba[:, :, 3].astype(np.int16)

    ocean = (a > 0) & (r < 70) & (g < 70) & (b > 170)
    return ocean


def label_basins(ocean: np.ndarray) -> Tuple[np.ndarray, int]:
    labels, n = ndimage.label(ocean)
    if n <= 1:
        return labels, n

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

    return out, next_id - 1


def windstress_curl_forcing(lat: float) -> float:
    a = abs(lat)
    if a < 3.0 or a > 78.0:
        return 0.0

    hemi = 1.0 if lat >= 0.0 else -1.0
    subtropic = np.exp(-((a - 30.0) / 14.0) ** 2)
    subpolar = np.exp(-((a - 55.0) / 14.0) ** 2)

    return hemi * (1.25 * subtropic - 0.80 * subpolar)


def build_streamfunction_currents(
    labels: np.ndarray,
    basins: Dict[int, BasinStats],
    h_px: int,
    coarsen: int = 6,
    beta_eff: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve

    h, w = labels.shape
    ch = max(4, h // coarsen)
    cw = max(4, w // coarsen)
    ocean_full = labels > 0

    h_blk = ch * coarsen
    w_blk = cw * coarsen
    ocean_work = np.zeros((h_blk, w_blk), dtype=np.float32)
    ocean_work[: min(h, h_blk), : min(w, w_blk)] = ocean_full[
        : min(h, h_blk), : min(w, w_blk)
    ].astype(np.float32)
    labels_work = np.zeros((h_blk, w_blk), dtype=np.int32)
    labels_work[: min(h, h_blk), : min(w, w_blk)] = labels[
        : min(h, h_blk), : min(w, w_blk)
    ]

    blocks_ocean = ocean_work.reshape(ch, coarsen, cw, coarsen)
    ocean_frac = blocks_ocean.mean(axis=(1, 3))
    ocean_c = ocean_frac > 0.65

    blocks_labels = labels_work.reshape(ch, coarsen, cw, coarsen)
    labels_c = np.zeros((ch, cw), dtype=np.int32)
    ocean_flat = ocean_c.ravel()
    labels_flat = labels_c.ravel()
    blocks_flat = blocks_labels.reshape(ch * cw, coarsen * coarsen)
    for i in range(ch * cw):
        if ocean_flat[i]:
            valid = blocks_flat[i][blocks_flat[i] > 0]
            if len(valid) > 0:
                labels_flat[i] = int(np.argmax(np.bincount(valid)))
    valid_labs = list(basins.keys())
    if valid_labs:
        labels_c[~np.isin(labels_c, valid_labs)] = 0

    forcing = np.zeros((ch, cw), dtype=np.float64)
    for cy in range(ch):
        lat = lat_from_y(cy * h / ch + h / (2 * ch), h_px)
        forcing[cy, :] = windstress_curl_forcing(lat)

    psi_c = np.zeros((ch, cw), dtype=np.float64)

    half_beta = beta_eff / 2.0

    for lab in valid_labs:
        mask_c = labels_c == lab
        if not np.any(mask_c):
            continue
        pts = [(int(r), int(c)) for r, c in np.argwhere(mask_c)]
        n = len(pts)
        if n < 16:
            continue

        k_map: Dict[Tuple[int, int], int] = {
            pt: k for k, pt in enumerate(pts)
        }

        A = lil_matrix((n, n), dtype=np.float64)
        b = np.zeros(n, dtype=np.float64)

        for k, (y, x) in enumerate(pts):
            nbrs_laplacian = [
                (y - 1, x),
                (y + 1, x),
                (y, (x - 1) % cw),
                (y, (x + 1) % cw),
            ]
            nbrs_laplacian = [
                (ny, nx) for ny, nx in nbrs_laplacian if 0 <= ny < ch
            ]

            A[k, k] = float(-len(nbrs_laplacian))
            b[k] = -forcing[y, x]

            for ny, nx in nbrs_laplacian:
                j = k_map.get((ny, nx), -1)
                if j < 0:
                    continue

                if ny == y:
                    is_east = nx == (x + 1) % cw
                    is_west = nx == (x - 1) % cw
                    if is_east and not is_west:
                        A[k, j] += 1.0 + half_beta
                    elif is_west and not is_east:
                        A[k, j] += 1.0 - half_beta
                    else:
                        A[k, j] += 1.0
                else:
                    A[k, j] += 1.0

        try:
            psi_loc = spsolve(A.tocsr(), b)
            if not (
                np.any(np.isnan(psi_loc)) or np.any(np.isinf(psi_loc))
            ):
                for k, (y, x) in enumerate(pts):
                    psi_c[y, x] = float(psi_loc[k])
        except Exception:
            pass

    psi_c = ndimage.gaussian_filter(psi_c, sigma=1.5)
    psi = ndimage.zoom(psi_c, (h / ch, w / cw), order=1)

    dpsi_dy_base = np.gradient(psi, axis=0)
    dpsi_dx_base = (
        np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)
    ) / 2.0
    v_sst = -dpsi_dx_base
    v_sst[~ocean_full] = 0.0

    y_arr = np.arange(h, dtype=np.float64)
    sigma_y = 12.0 * (h - 1) / 180.0
    y_42n = (90.0 - 42.0) * (h - 1) / 180.0
    y_42s = (90.0 + 42.0) * (h - 1) / 180.0

    for lab in valid_labs:
        basin_mask = labels == lab
        ys_occ = np.where(np.any(basin_mask, axis=1))[0]
        if len(ys_occ) < 10:
            continue
        lat_top = lat_from_y(float(ys_occ[0]), h_px)
        lat_bot = lat_from_y(float(ys_occ[-1]), h_px)
        if lat_top - lat_bot < 40.0:
            continue

        basin_psi = psi[basin_mask]
        psi_range = float(np.max(np.abs(basin_psi)))
        if psi_range < 1e-12:
            continue

        amoc_amp = 0.40 * psi_range
        pert_1d = 0.5 * amoc_amp * (
            np.tanh((y_arr - y_42n) / sigma_y)
            + np.tanh((y_arr - y_42s) / sigma_y)
        )
        basin_smooth = ndimage.gaussian_filter(
            basin_mask.astype(np.float64), sigma=25.0
        )
        psi += pert_1d[:, np.newaxis] * basin_smooth

    dpsi_dy = np.gradient(psi, axis=0)
    dpsi_dx = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / 2.0
    u = dpsi_dy
    v = -dpsi_dx
    u[~ocean_full] = 0.0
    v[~ocean_full] = 0.0

    mag = np.sqrt(u * u + v * v)
    if np.any(ocean_full):
        p95 = float(np.percentile(mag[ocean_full], 95))
        if p95 > 1e-9:
            u /= p95
            v /= p95
            v_sst /= p95

    return u, v, v_sst


def compute_basin_stats(
    labels: np.ndarray, n_labels: int, min_area: int
) -> Dict[int, BasinStats]:
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

        stats[i] = BasinStats(
            label=i,
            area=area,
            x_min=int(xs.min()),
            x_max=int(xs.max()),
            x_mean=float(xs.mean()),
            y_mean=float(ys.mean()),
        )
    return stats


def lat_from_y(y_px: float, h_px: int) -> float:
    return 90.0 - 180.0 * (y_px / max(1, h_px - 1))
