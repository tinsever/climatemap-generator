from __future__ import annotations
from collections import Counter
import numpy as np
from land_climate import _koppen_zone_indices_monthly
from land_climate_physics import simulate_monthly_climate
from ocean import ocean_mask_from_rgba, rasterize_svg
CLASS_NAMES = {0: 'tropical_wet', 1: 'tropical_dry', 2: 'desert_hot', 3: 'steppe', 4: 'mediterranean', 5: 'humid_sub', 6: 'oceanic', 7: 'continental', 8: 'boreal', 9: 'tundra', 10: 'ice'}

def _region_mask(land: np.ndarray, lats: np.ndarray, lons: np.ndarray, lat_north: float, lat_south: float, lon_west: float, lon_east: float) -> np.ndarray:
    yi = np.where((lats <= lat_north) & (lats >= lat_south))[0]
    if yi.size == 0:
        return np.zeros_like(land, dtype=bool)
    mask = np.zeros_like(land, dtype=bool)
    if lon_west <= lon_east:
        xi = np.where((lons >= lon_west) & (lons <= lon_east))[0]
        if xi.size > 0:
            mask[np.ix_(yi, xi)] = True
    else:
        xi1 = np.where(lons >= lon_west)[0]
        xi2 = np.where(lons <= lon_east)[0]
        if xi1.size > 0:
            mask[np.ix_(yi, xi1)] = True
        if xi2.size > 0:
            mask[np.ix_(yi, xi2)] = True
    return mask & land

def _top_classes(values: np.ndarray, top_n: int=4) -> list[tuple[str, float]]:
    if values.size == 0:
        return []
    cnt = Counter(values.tolist())
    total = float(values.size)
    out: list[tuple[str, float]] = []
    for cls, n in cnt.most_common(top_n):
        out.append((CLASS_NAMES[int(cls)], n / total))
    return out

def run_diagnostics(svg_path: str='welt.svg', raster_w: int=1800) -> dict:
    raster_h = int(raster_w / 2)
    rgba = rasterize_svg(svg_path, out_w=raster_w, out_h=raster_h)
    ocean = ocean_mask_from_rgba(rgba)
    sim = simulate_monthly_climate(ocean=ocean, raster_h=raster_h, coast_window_px=270)
    z = _koppen_zone_indices_monthly(sim)
    land = sim['land']
    t_mon = sim['temp_month']
    p_mon = sim['precip_month']
    t_ann = np.mean(t_mon, axis=0)
    t_hot = np.max(t_mon, axis=0)
    t_cold = np.min(t_mon, axis=0)
    p_ann = np.sum(p_mon, axis=0)
    h, w = land.shape
    lats = np.linspace(90.0, -90.0, h, dtype=np.float32)
    lons = np.linspace(-180.0, 180.0, w, dtype=np.float32)
    abs_lat = np.abs(sim['lat_2d'])
    global_classes = {CLASS_NAMES[k]: v / float(np.sum(land)) for k, v in Counter(z[land].tolist()).items()}
    lat_bands = {'tropics_0_23.5': (abs_lat <= 23.5) & land, 'subtropics_23.5_40': (abs_lat > 23.5) & (abs_lat <= 40.0) & land, 'midlat_40_66.5': (abs_lat > 40.0) & (abs_lat <= 66.5) & land, 'polar_66.5_90': (abs_lat > 66.5) & land}
    band_stats = {}
    for key, m in lat_bands.items():
        band_stats[key] = {'t_ann_mean_c': float(np.mean(t_ann[m])) if np.any(m) else float('nan'), 'p_ann_mean_mm': float(np.mean(p_ann[m])) if np.any(m) else float('nan'), 'top_classes': _top_classes(z[m], top_n=3)}
    regions = [('Northern Germany', 55.0, 47.0, 5.0, 15.0), ('British Isles', 59.0, 50.0, -10.0, 2.0), ('Eastern USA', 45.0, 30.0, -90.0, -70.0), ('Southwestern USA', 38.0, 28.0, -122.0, -104.0), ('Mediterranean Basin', 43.0, 30.0, -10.0, 40.0), ('Sahara', 30.0, 15.0, -15.0, 30.0), ('Amazon', 5.0, -10.0, -75.0, -50.0), ('India Monsoon Core', 28.0, 8.0, 72.0, 90.0), ('East China', 40.0, 20.0, 108.0, 123.0), ('Australia Interior', -20.0, -32.0, 120.0, 145.0), ('Patagonia', -40.0, -52.0, -75.0, -64.0)]
    region_stats = {}
    for name, lat_n, lat_s, lon_w, lon_e in regions:
        m = _region_mask(land, lats, lons, lat_n, lat_s, lon_w, lon_e)
        if not np.any(m):
            continue
        region_stats[name] = {'t_ann_mean_c': float(np.mean(t_ann[m])), 't_hot_mean_c': float(np.mean(t_hot[m])), 't_cold_mean_c': float(np.mean(t_cold[m])), 'p_ann_mean_mm': float(np.mean(p_ann[m])), 'top_classes': _top_classes(z[m], top_n=4)}
    diagnostics = {'global_class_shares': global_classes, 'lat_band_stats': band_stats, 'region_stats': region_stats}
    return diagnostics

def _fmt_class_list(items: list[tuple[str, float]]) -> str:
    return ', '.join((f'{k}:{v * 100:.1f}%' for k, v in items))
if __name__ == '__main__':
    result = run_diagnostics(svg_path='welt.svg', raster_w=1800)
    print('=== Global Class Shares ===')
    for k, v in sorted(result['global_class_shares'].items(), key=lambda kv: kv[1], reverse=True):
        print(f'{k:15s} {v * 100:6.2f}%')
    print('\n=== Lat Band Stats ===')
    for band, stats in result['lat_band_stats'].items():
        print(f"{band:20s} Tann={stats['t_ann_mean_c']:.2f}C  Pann={stats['p_ann_mean_mm']:.1f}mm  Top={_fmt_class_list(stats['top_classes'])}")
    print('\n=== Region Checks ===')
    for name, stats in result['region_stats'].items():
        print(f"{name:20s} Tann={stats['t_ann_mean_c']:.2f}C  Thot={stats['t_hot_mean_c']:.2f}C  Tcold={stats['t_cold_mean_c']:.2f}C  Pann={stats['p_ann_mean_mm']:.1f}mm  Top={_fmt_class_list(stats['top_classes'])}")
