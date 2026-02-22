import io
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

from constants import SST_STOPS
from ocean import lat_from_y


def _sst_color_vec(
    abs_lat_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    last_c = SST_STOPS[-1][1]
    r = np.full(len(abs_lat_arr), float(last_c[0]), dtype=np.float32)
    g = np.full(len(abs_lat_arr), float(last_c[1]), dtype=np.float32)
    b = np.full(len(abs_lat_arr), float(last_c[2]), dtype=np.float32)
    for i in range(len(SST_STOPS) - 1):
        a0, c0 = SST_STOPS[i]
        a1, c1 = SST_STOPS[i + 1]
        in_range = (abs_lat_arr >= a0) & (abs_lat_arr < a1)
        if not np.any(in_range):
            continue
        t = (abs_lat_arr[in_range] - a0) / (a1 - a0)
        r[in_range] = c0[0] + t * (c1[0] - c0[0])
        g[in_range] = c0[1] + t * (c1[1] - c0[1])
        b[in_range] = c0[2] + t * (c1[2] - c0[2])
    return r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8)


def sst_color(lat: float) -> Tuple[int, int, int]:
    a = abs(lat)
    for i in range(len(SST_STOPS) - 1):
        a0, c0 = SST_STOPS[i]
        a1, c1 = SST_STOPS[i + 1]
        if a0 <= a <= a1:
            t = (a - a0) / (a1 - a0)
            r = int(c0[0] + t * (c1[0] - c0[0]))
            g = int(c0[1] + t * (c1[1] - c0[1]))
            b = int(c0[2] + t * (c1[2] - c0[2]))
            return (r, g, b)
    return SST_STOPS[-1][1]


def build_sst_image(
    ocean: np.ndarray,
    raster_h: int,
    alpha: int = 155,
    coast_window_px: int = 180,
    coast_max_shift: float = 8.0,
    stream_v: Optional[np.ndarray] = None,
    current_max_shift: float = 14.0,
    enclosure_max_shift: float = 12.0,
    debug_mode: Optional[str] = None,
) -> bytes:
    from scipy.ndimage import (
        binary_dilation,
        binary_erosion,
        label,
        uniform_filter1d,
    )

    h, w = ocean.shape
    land_f = (~ocean).astype(np.float32)

    def _box2d_xwrap(arr: np.ndarray, win: int) -> np.ndarray:
        win = max(3, int(win))
        tmp = uniform_filter1d(arr, size=win, axis=1, mode="wrap")
        out = uniform_filter1d(tmp, size=win, axis=0, mode="nearest")
        return out

    half = coast_window_px // 2
    land_to_east = uniform_filter1d(
        np.roll(land_f, -half, axis=1), size=coast_window_px, axis=1, mode="wrap"
    )
    land_to_west = uniform_filter1d(
        np.roll(land_f, half, axis=1), size=coast_window_px, axis=1, mode="wrap"
    )

    lat_arr = np.array([lat_from_y(float(y), raster_h) for y in range(h)])
    a_arr = np.abs(lat_arr)
    lat_w = np.exp(-((a_arr - 30.0) ** 2) / (2.0 * 22.0 ** 2))
    lat_w[a_arr < 5.0] = 0.0
    lat_w[a_arr > 72.0] = 0.0

    coastal_shift = (land_to_east - land_to_west) * lat_w[:, np.newaxis] * coast_max_shift

    enclosure_window = max(45, int(coast_window_px * 0.50))
    land_frac_2d = _box2d_xwrap(land_f, win=enclosure_window)
    enc_lr = np.sqrt(np.clip(land_to_east * land_to_west, 0.0, 1.0))
    enc = np.clip(0.55 * land_frac_2d + 0.45 * enc_lr, 0.0, 1.0)
    enc_strength = np.clip((enc - 0.20) / 0.35, 0.0, 1.0)

    strait_r = 4
    pad = strait_r + 2
    o_x = np.pad(ocean, ((0, 0), (pad, pad)), mode="wrap")
    o_xy = np.pad(o_x, ((pad, pad), (0, 0)), mode="constant", constant_values=False)

    o_open = binary_dilation(
        binary_erosion(o_xy, iterations=strait_r),
        iterations=strait_r,
    )

    o_open = o_open[pad : pad + h, pad : pad + w]

    labs, n = label(o_open)
    if n > 0:
        areas = np.bincount(labs.ravel())
        areas[0] = 0
        main_lab = int(np.argmax(areas))

        semi = ((labs != 0) & (labs != main_lab)) & ocean

        semi_strength = ndimage.gaussian_filter(
            semi.astype(np.float32),
            sigma=(max(1.0, strait_r * 0.8), max(1.0, strait_r * 0.8)),
            mode=("nearest", "wrap"),
        )
        semi_strength = np.clip(semi_strength * 1.8, 0.0, 1.0)

        gate = np.clip((land_frac_2d - 0.18) / 0.22, 0.0, 1.0)
        semi_strength *= gate

        enc_strength2 = np.maximum(enc_strength, semi_strength)
    else:
        enc_strength2 = enc_strength

    coastal_shift = coastal_shift * (1.0 - 0.90 * enc_strength2)
    if debug_mode == "B":
        coastal_shift[:] = 0.0

    enc_lat_w = np.exp(-((a_arr - 38.0) ** 2) / (2.0 * 24.0 ** 2))
    enclosure_shift = -enc_strength2 * enc_lat_w[:, np.newaxis] * enclosure_max_shift
    enclosure_shift[~ocean] = 0.0
    if debug_mode == "C":
        enclosure_shift[:] = 0.0

    total_shift = coastal_shift + enclosure_shift

    if stream_v is not None and debug_mode != "A":
        sv_smooth = ndimage.gaussian_filter(stream_v.astype(np.float64), sigma=10.0)
        hemi_arr = np.sign(lat_arr)[:, np.newaxis]
        current_shift = sv_smooth * hemi_arr * current_max_shift
        current_shift[~ocean] = 0.0
        current_shift *= lat_w[:, np.newaxis]
        current_shift *= (1.0 - 0.70 * enc_strength2)
        current_shift = np.clip(current_shift, -6.0, 6.0)
        total_shift += current_shift

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        mask = ocean[y]
        if not np.any(mask):
            continue

        adj_lat = np.clip(np.abs(lat_arr[y]) + total_shift[y, mask], 0.0, 90.0)

        r_px, g_px, b_px = _sst_color_vec(adj_lat)
        rgba[y, mask, 0] = r_px
        rgba[y, mask, 1] = g_px
        rgba[y, mask, 2] = b_px
        rgba[y, mask, 3] = alpha

    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()
