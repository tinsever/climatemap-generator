import io
import math
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

from constants import SST_STOPS, TROPIC_LAT, SUBTROPIC_MAX_LAT, POLAR_CIRCLE_LAT
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
    coast_max_shift: float = 5.0,
    stream_v: Optional[np.ndarray] = None,
    stream_u: Optional[np.ndarray] = None,
    current_max_shift: float = 14.0,
    enclosure_max_shift: float = 12.0,
    debug_mode: Optional[str] = None,
    month: int = 1,
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
    _lat_w_sigma = (POLAR_CIRCLE_LAT - TROPIC_LAT) / 2.0
    lat_w = np.exp(-((a_arr - SUBTROPIC_MAX_LAT) ** 2) / (2.0 * _lat_w_sigma ** 2))
    lat_w[a_arr < TROPIC_LAT * 0.25] = 0.0
    lat_w[a_arr > POLAR_CIRCLE_LAT] = 0.0

    coastal_shift = (land_to_east - land_to_west) * lat_w[:, np.newaxis] * coast_max_shift

    upwelling_max_shift = 6.0
    dist_ocean = ndimage.distance_transform_edt(ocean).astype(np.float32)
    grad_oy, grad_ox = np.gradient(dist_ocean)
    east_boundary = np.zeros((h, w), dtype=np.float32)
    grad_mag = np.sqrt(grad_ox ** 2 + grad_oy ** 2)
    valid = ocean & (grad_mag > 1e-6)
    east_boundary[valid] = np.clip(-grad_ox[valid] / grad_mag[valid], 0.0, 1.0)
    offshore_decay = np.exp(-dist_ocean / max(8.0, w * 0.035))
    trade_lat_w = np.exp(-((a_arr - 22.0) / 12.0) ** 2)
    trade_lat_w[a_arr < 5.0] = 0.0
    trade_lat_w[a_arr > 40.0] = 0.0
    upwelling_shift = (
        east_boundary
        * offshore_decay
        * trade_lat_w[:, np.newaxis]
        * upwelling_max_shift
    )
    upwelling_shift[~ocean] = 0.0
    upwelling_shift = ndimage.gaussian_filter(
        upwelling_shift, sigma=(3.0, 3.0), mode=("nearest", "wrap")
    )
    upwelling_shift[~ocean] = 0.0
    coastal_shift += upwelling_shift

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

    _enc_peak = (SUBTROPIC_MAX_LAT + POLAR_CIRCLE_LAT) / 2.0
    _enc_sigma = (POLAR_CIRCLE_LAT - SUBTROPIC_MAX_LAT) / 2.0
    enc_lat_w = np.exp(-((a_arr - _enc_peak) ** 2) / (2.0 * _enc_sigma ** 2))
    enclosure_shift = -enc_strength2 * enc_lat_w[:, np.newaxis] * enclosure_max_shift
    enclosure_shift[~ocean] = 0.0
    if debug_mode == "C":
        enclosure_shift[:] = 0.0

    total_shift = coastal_shift + enclosure_shift

    if (stream_v is not None or stream_u is not None) and debug_mode != "A":
        from scipy.ndimage import map_coordinates

        su = (stream_u if stream_u is not None else np.zeros((h, w))).astype(np.float64)
        sv = (stream_v if stream_v is not None else np.zeros((h, w))).astype(np.float64)

        su = ndimage.gaussian_filter(su, sigma=8.0)
        sv = ndimage.gaussian_filter(sv, sigma=8.0)
        su[~ocean] = 0.0
        sv[~ocean] = 0.0

        mag = np.sqrt(su ** 2 + sv ** 2)
        if np.any(ocean):
            p95 = float(np.percentile(mag[ocean], 95))
            if p95 > 1e-9:
                su /= p95
                sv /= p95
                mag /= p95

        abs_lat_2d = a_arr[:, np.newaxis] * np.ones(w)
        yy = np.arange(h, dtype=np.float64)[:, np.newaxis] * np.ones(w)
        xx = np.ones((h, 1)) * np.arange(w, dtype=np.float64)

        n_steps = 60
        ds_step = max(1.0, h * 0.016)

        src_y = yy.copy()
        src_x = xx.copy()
        frozen = ~ocean.copy()

        ocean_f = ocean.astype(np.float32)

        for _ in range(n_steps):
            su_src = map_coordinates(
                su, [src_y.ravel(), src_x.ravel()], order=1, mode="nearest"
            ).reshape(h, w)
            sv_src = map_coordinates(
                sv, [src_y.ravel(), src_x.ravel()], order=1, mode="nearest"
            ).reshape(h, w)

            new_y = np.clip(src_y - sv_src * ds_step, 0.0, h - 1.0)
            new_x = np.clip(src_x - su_src * ds_step, 0.0, w - 1.0)

            on_land = map_coordinates(
                ocean_f, [new_y.ravel(), new_x.ravel()], order=0, mode="nearest"
            ).reshape(h, w) < 0.5
            frozen |= on_land

            src_y = np.where(frozen, src_y, new_y)
            src_x = np.where(frozen, src_x, new_x)

            src_y[~ocean] = yy[~ocean]
            src_x[~ocean] = xx[~ocean]
            frozen[~ocean] = True

        eff = map_coordinates(
            abs_lat_2d, [src_y.ravel(), src_x.ravel()], order=1, mode="nearest"
        ).reshape(h, w)
        eff[~ocean] = abs_lat_2d[~ocean]

        travel_dist = np.sqrt((src_y - yy) ** 2 + (src_x - xx) ** 2)
        decay_scale = max(20.0, h * 0.12)
        survival = np.exp(-travel_dist / decay_scale)
        eff = abs_lat_2d + (eff - abs_lat_2d) * survival

        adv_shift = eff - abs_lat_2d
        adv_shift[~ocean] = 0.0
        adv_shift *= (1.0 - 0.60 * enc_strength2)

        if np.any(ocean):
            adv_p95 = float(np.percentile(np.abs(adv_shift[ocean]), 95))
            if adv_p95 > 1e-9:
                adv_shift = np.clip(
                    adv_shift / adv_p95 * current_max_shift,
                    -current_max_shift,
                    current_max_shift * 0.8,
                )

        total_shift += adv_shift

    A_annual = 1.0 + 7.0 * np.exp(-((a_arr - 40.0) / 18.0) ** 2)
    month_norm = (month - 1) % 12
    annual_shift_1d = (
        A_annual
        * math.cos(2.0 * math.pi * month_norm / 12.0)
        * np.sign(lat_arr)
    )
    annual_shift = np.broadcast_to(annual_shift_1d[:, np.newaxis], (h, w))

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        mask = ocean[y]
        if not np.any(mask):
            continue

        base_adj = np.abs(lat_arr[y]) + total_shift[y, mask] + annual_shift[y, mask]
        adj_lat = np.clip(base_adj, 0.0, 90.0)

        r_px, g_px, b_px = _sst_color_vec(adj_lat)
        rgba[y, mask, 0] = r_px
        rgba[y, mask, 1] = g_px
        rgba[y, mask, 2] = b_px
        rgba[y, mask, 3] = alpha

    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()
