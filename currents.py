import math
from typing import Dict, Optional, Tuple
import numpy as np
from scipy import ndimage
from constants import COAST_REPEL_GAIN, COAST_REPEL_SCALE, COAST_STEER_DIST, TROPIC_LAT, SUBTROPIC_MAX_LAT, POLAR_CIRCLE_LAT
from models import BasinStats
from ocean import lat_from_y, windstress_curl_forcing

def climate_zone(lat: float) -> str:
    a = abs(lat)
    if a <= TROPIC_LAT:
        return 'tropisch'
    if a <= SUBTROPIC_MAX_LAT:
        return 'subtropisch'
    if a <= POLAR_CIRCLE_LAT:
        return 'gemaessigt'
    return 'polar'

def zonal_current_u(lat: float) -> float:
    zone = climate_zone(lat)
    if zone == 'tropisch':
        return -1.0
    if zone == 'subtropisch':
        return -0.5
    if zone == 'gemaessigt':
        return 0.9
    return -0.25

def equatorial_countercurrent_u(lat: float) -> float:
    if 3.0 <= lat <= 13.0:
        return math.exp(-((lat - 7.0) / 2.5) ** 2)
    return 0.0

def circumpolar_boost_u(lat: float) -> float:
    if -65.0 <= lat <= -45.0:
        return 1.2
    return 0.0

def gyre_k(lat: float) -> float:
    zone = climate_zone(lat)
    if zone == 'tropisch':
        return 0.0
    if zone == 'subtropisch':
        return 0.9 if lat > 0 else -0.9
    if zone == 'gemaessigt':
        return -0.6 if lat > 0 else 0.6
    if zone == 'polar':
        return -0.2 if lat > 0 else 0.2
    return 0.0

def coastal_damping(dist_to_land: float, scale: float) -> float:
    return math.tanh(dist_to_land / max(1e-06, scale))

def coastal_repulsion_strength(dist_to_land: float, scale: float) -> float:
    return math.exp(-dist_to_land / max(1e-06, scale))

def steer_away_from_coast(u: float, v: float, dist_to_land: float, gx: float, gy: float) -> Tuple[float, float]:
    if dist_to_land >= COAST_STEER_DIST:
        return (u, v)
    gnorm = math.sqrt(gx * gx + gy * gy)
    if gnorm <= 1e-08:
        return (u, v)
    nx = gx / gnorm
    ny = gy / gnorm
    u0, v0 = (u, v)
    inward = u * nx + v * ny
    if inward < 0.0:
        u -= inward * nx
        v -= inward * ny
    repel = COAST_REPEL_GAIN * coastal_repulsion_strength(dist_to_land, scale=COAST_REPEL_SCALE)
    u += repel * nx
    v += repel * ny
    alpha = coastal_repulsion_strength(dist_to_land, scale=COAST_STEER_DIST)
    u = (1.0 - alpha) * u0 + alpha * u
    v = (1.0 - alpha) * v0 + alpha * v
    return (u, v)

def western_intensification_factor(x: float, basin: BasinStats, strength: float, scale: float) -> float:
    dist_from_west = x - basin.x_min
    return 1.0 + strength * math.exp(-dist_from_west / max(1e-06, scale))

def _bilinear_sample(field: np.ndarray, x: float, y: float) -> float:
    h, w = field.shape
    if y < 0.0 or y > h - 1.0:
        return 0.0
    x0 = int(math.floor(x)) % w
    y0 = int(math.floor(y))
    x1 = (x0 + 1) % w
    y1 = min(h - 1, y0 + 1)
    tx = x - math.floor(x)
    ty = y - math.floor(y)
    f00 = float(field[y0, x0])
    f10 = float(field[y0, x1])
    f01 = float(field[y1, x0])
    f11 = float(field[y1, x1])
    a = f00 * (1.0 - tx) + f10 * tx
    b = f01 * (1.0 - tx) + f11 * tx
    return a * (1.0 - ty) + b * ty

def split_at_dateline(pts: list[tuple[float, float]], w: int, jump_frac: float=0.45) -> list[list[tuple[float, float]]]:
    if len(pts) < 2:
        return []
    jump = w * jump_frac
    out: list[list[tuple[float, float]]] = [[pts[0]]]
    for (px, py), (x, y) in zip(pts[:-1], pts[1:]):
        if abs(x - px) > jump:
            out.append([(x, y)])
        else:
            out[-1].append((x, y))
    return [seg for seg in out if len(seg) >= 2]

def trace_streamline(x0: float, y0: float, u: np.ndarray, v: np.ndarray, ocean: np.ndarray, dist: np.ndarray, ds: float=2.0, max_steps: int=400, min_speed: float=0.03, min_dist_px: float=1.0) -> list[tuple[float, float]]:
    h, w = ocean.shape

    def march(direction: float) -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = []
        x, y = (x0, y0)
        for _ in range(max_steps):
            xi = int(round(x)) % w
            yi = int(round(y))
            if yi < 0 or yi >= h:
                break
            if not ocean[yi, xi]:
                break
            if float(dist[yi, xi]) < min_dist_px:
                break
            uu = _bilinear_sample(u, x, y)
            vv = _bilinear_sample(v, x, y)
            sp = math.sqrt(uu * uu + vv * vv)
            if sp < min_speed:
                break
            pts.append((x, y))
            x = (x + direction * (uu / sp) * ds) % w
            y = y + direction * (vv / sp) * ds
        return pts
    fwd = march(+1.0)
    bwd = march(-1.0)
    bwd.reverse()
    out = bwd + fwd[1:] if fwd else bwd
    if len(out) < 5:
        return []
    return out

def build_streamlines(ocean: np.ndarray, dist: np.ndarray, u: np.ndarray, v: np.ndarray, seed_spacing_px: int=55, min_dist_px: float=1.0) -> list[list[tuple[float, float]]]:
    h, w = ocean.shape
    uu = ndimage.gaussian_filter(u.astype(np.float64), sigma=2.0)
    vv = ndimage.gaussian_filter(v.astype(np.float64), sigma=2.0)
    uu[~ocean] = 0.0
    vv[~ocean] = 0.0
    lines: list[list[tuple[float, float]]] = []
    occ = np.zeros((h // 6 + 1, w // 6 + 1), dtype=bool)
    for y in range(seed_spacing_px // 2, h, seed_spacing_px):
        for x in range(seed_spacing_px // 2, w, seed_spacing_px):
            if not ocean[y, x]:
                continue
            if float(dist[y, x]) < min_dist_px:
                continue
            oy = y // 6
            ox = x // 6
            if occ[oy, ox]:
                continue
            line = trace_streamline(x0=float(x), y0=float(y), u=uu, v=vv, ocean=ocean, dist=dist, ds=2.0, max_steps=400, min_speed=0.006, min_dist_px=min_dist_px)
            if not line:
                continue
            for px, py in line[::6]:
                occ[int(py) // 6, int(px) // 6] = True
            lines.append(line)
    return lines

def _subtropical_lat_bands(month: int=4) -> list[tuple[float, float]]:
    """Subtropische Breitengradstreifen aus dem Windantriebsprofil.

    Gibt [(lat_min_N, lat_max_N), (lat_min_S, lat_max_S)] zurück.
    Vollständig aus windstress_curl_forcing abgeleitet – keine Festwerte.
    """
    lats = np.linspace(-85.0, 85.0, 500)
    curl = np.array([windstress_curl_forcing(float(lat), month=month) for lat in lats])
    th = 0.15
    bands: list[tuple[float, float]] = []
    nh = np.where((lats > 3.0) & (curl > th))[0]
    if len(nh) > 0:
        bands.append((float(lats[nh[0]]) - 4.0, float(lats[nh[-1]]) + 4.0))
    sh = np.where((lats < -3.0) & (curl < -th))[0]
    if len(sh) > 0:
        bands.append((float(lats[sh[0]]) - 4.0, float(lats[sh[-1]]) + 4.0))
    if not bands:
        bands = [(10.0, 50.0), (-50.0, -10.0)]
    return bands

def _local_western_boundary_strip(basin_mask: np.ndarray, ocean: np.ndarray, lat_ok_col: np.ndarray, w: int, strip_width_frac: float=0.1) -> np.ndarray:
    """Findet Westrand-Streifen zeilenweise – topologisch, ohne x_min-Festwerte.

    Für jede Zeile im Zielbreitengrad-Band werden zusammenhängende Ozean-
    Segmente gesucht. Der linke Rand jedes Segments wird als Westrand markiert,
    sofern die Zelle links davon **Land** ist (kein Ozean). Durch Nutzung von
    `ocean` statt `basin_mask` werden Datumslinie-Artefakte vermieden.
    """
    h = basin_mask.shape[0]
    strip_w = max(4, int(w * strip_width_frac))
    wb = np.zeros_like(basin_mask, dtype=bool)
    for y in range(h):
        if not lat_ok_col[y]:
            continue
        row = basin_mask[y, :]
        if not np.any(row):
            continue
        x = 0
        while x < w:
            if not row[x]:
                x += 1
                continue
            seg_start = x
            x_prev = (seg_start - 1) % w
            is_west_boundary = not ocean[y, x_prev]
            while x < w and row[x]:
                x += 1
            if is_west_boundary:
                n_mark = min(strip_w, x - seg_start)
                wb[y, seg_start:seg_start + n_mark] = True
    return wb

def build_major_driftlines(ocean: np.ndarray, labels: np.ndarray, basins: Dict[int, BasinStats], dist: np.ndarray, stream_u: np.ndarray, stream_v: np.ndarray, h_px: int, min_dist_px: float=1.2, month: int=4) -> list[list[tuple[float, float]]]:
    h, w = ocean.shape
    uu = ndimage.gaussian_filter(stream_u.astype(np.float64), sigma=6.0)
    vv = ndimage.gaussian_filter(stream_v.astype(np.float64), sigma=6.0)
    uu[~ocean] = 0.0
    vv[~ocean] = 0.0
    speed = np.sqrt(uu * uu + vv * vv)

    def pick_seed(mask: np.ndarray) -> Optional[tuple[int, int]]:
        if not np.any(mask):
            return None
        idx = int(np.argmax(speed[mask]))
        ys, xs = np.where(mask)
        return (int(ys[idx]), int(xs[idx]))
    lines: list[list[tuple[float, float]]] = []
    for lab, basin in basins.items():
        basin_mask = (labels == lab) & ocean
        if not np.any(basin_mask):
            continue
        for lat_min, lat_max in _subtropical_lat_bands(month=month):
            lat_ok = np.zeros((h, 1), dtype=bool)
            for y in range(h):
                lat = lat_from_y(float(y), h_px)
                lat_ok[y, 0] = lat_min <= lat <= lat_max
            west_strip = _local_western_boundary_strip(basin_mask, ocean, lat_ok[:, 0], w)
            mask = basin_mask & west_strip & lat_ok & (dist >= min_dist_px)
            seed = pick_seed(mask)
            if seed is None:
                continue
            sy, sx = seed
            line = trace_streamline(x0=float(sx), y0=float(sy), u=uu, v=vv, ocean=ocean, dist=dist, ds=2.0, max_steps=600, min_speed=0.008, min_dist_px=min_dist_px)
            if line:
                lines.append(line)
    return lines

def clip_arrow_to_ocean(x1: float, y1: float, ux: float, vy: float, length: float, ocean: np.ndarray, dist: np.ndarray, min_dist_px: float) -> Optional[Tuple[float, float, float]]:
    h, w = ocean.shape
    steps = max(1, int(math.ceil(length)))
    last_t = 0.0
    for i in range(1, steps + 1):
        t = length * (i / steps)
        xi = int(round(x1 + ux * t))
        yi = int(round(y1 + vy * t))
        if xi < 0 or xi >= w or yi < 0 or (yi >= h):
            break
        if not ocean[yi, xi]:
            break
        if dist[yi, xi] < min_dist_px * 0.6:
            break
        last_t = t
    if last_t <= 0.0:
        return None
    return (x1 + ux * last_t, y1 + vy * last_t, last_t)

def build_arrows(ocean: np.ndarray, labels: np.ndarray, dist: np.ndarray, grad_x: np.ndarray, grad_y: np.ndarray, basins: Dict[int, BasinStats], stream_u: np.ndarray, stream_v: np.ndarray, raster_w: int, raster_h: int, spacing_px: int=24, min_dist_px: float=0.8, base_length: float=18.0) -> list[tuple[float, float, float, float, float]]:
    h, w = ocean.shape
    arrows: list[tuple[float, float, float, float, float]] = []
    for y in range(spacing_px // 2, h, spacing_px):
        for x in range(spacing_px // 2, w, spacing_px):
            if not ocean[y, x]:
                continue
            if float(dist[y, x]) < min_dist_px:
                continue
            u, v, speed = vector_at_point(x, y, w, h, labels, dist, grad_x, grad_y, basins, stream_u, stream_v)
            if speed < 0.02:
                continue
            sp = math.sqrt(u * u + v * v)
            ux, vy = (u / sp, v / sp)
            result = clip_arrow_to_ocean(x1=float(x), y1=float(y), ux=ux, vy=vy, length=base_length, ocean=ocean, dist=dist, min_dist_px=min_dist_px)
            if result is None:
                continue
            x2, y2, _ = result
            stroke_w = 0.6 + min(speed * 8, 2.0)
            arrows.append((float(x), float(y), x2, y2, stroke_w))
    return arrows

def vector_at_point(x: float, y: float, w_px: int, h_px: int, labels: np.ndarray, dist: np.ndarray, grad_x: np.ndarray, grad_y: np.ndarray, basins: Dict[int, BasinStats], stream_u: np.ndarray, stream_v: np.ndarray) -> Tuple[float, float, float]:
    lat = lat_from_y(y, h_px)
    lab = int(labels[int(round(y)), int(round(x))])
    if lab == 0:
        return (0.0, 0.0, 0.0)
    yi = int(round(y))
    xi = int(round(x))
    u = float(stream_u[yi, xi])
    v = float(stream_v[yi, xi])
    stream_mag = math.sqrt(u * u + v * v)
    a_lat = abs(lat)
    if stream_mag < 0.02:
        fb = 0.25
        if a_lat < 25.0:
            trade_wind = -0.8 * math.exp(-((a_lat - 10.0) / 8.0) ** 2)
            u += fb * trade_wind
        u += fb * 0.45 * equatorial_countercurrent_u(lat)
        u += fb * 0.3 * circumpolar_boost_u(lat)
    d = float(dist[yi, xi])
    gx = float(grad_x[yi, xi])
    gy = float(grad_y[yi, xi])
    gn = math.sqrt(gx * gx + gy * gy)
    if gn > 1e-08:
        nx, ny = (gx / gn, gy / gn)
        tx, ty = (-ny, nx)
        un = u * nx + v * ny
        ut = u * tx + v * ty
        damp_n = coastal_damping(d, scale=6.0)
        un *= damp_n
        u = un * nx + ut * tx
        v = un * ny + ut * ty
    u, v = steer_away_from_coast(u=u, v=v, dist_to_land=d, gx=gx, gy=gy)
    speed = math.sqrt(u * u + v * v)
    return (u, v, speed)
