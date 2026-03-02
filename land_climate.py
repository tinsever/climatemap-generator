import io
import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, label
from constants import LAND_CLIMATE_COLORS, POLAR_CIRCLE_LAT, TREWARTHA_CLIMATE_COLORS
from land_climate_physics import simulate_monthly_climate
_WIKI_ZONE_LIST: list[str] = ['eiswueste', 'tundra', 'nadelwald', 'mischwald', 'gem_regenwald', 'westseite', 'steppe', 'winterkalt_trocken', 'heiss_trocken', 'dornsavanne', 'ostseite', 'trockensavanne', 'feuchtsavanne', 'trop_regenwald']
_TREWARTHA_ZONE_LIST: list[str] = ['Ar', 'Aw', 'BW', 'BS', 'Cf', 'Cs', 'Do', 'Dc', 'E', 'FT', 'Fi']
_LABEL_CONNECTIVITY = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

def _render_zone_image(zone_indices: np.ndarray, land: np.ndarray, zone_list: list[str], colors: dict[str, tuple[int, int, int]], alpha: int) -> bytes:
    lut_r = np.array([colors[key][0] for key in zone_list], dtype=np.uint8)
    lut_g = np.array([colors[key][1] for key in zone_list], dtype=np.uint8)
    lut_b = np.array([colors[key][2] for key in zone_list], dtype=np.uint8)
    rgba = np.zeros((land.shape[0], land.shape[1], 4), dtype=np.uint8)
    rgba[land, 0] = lut_r[zone_indices[land]]
    rgba[land, 1] = lut_g[zone_indices[land]]
    rgba[land, 2] = lut_b[zone_indices[land]]
    rgba[land, 3] = alpha
    image = Image.fromarray(rgba, 'RGBA')
    buffer = io.BytesIO()
    image.save(buffer, format='PNG', optimize=False)
    return buffer.getvalue()

def _seasonal_precip_from_monthly(precip_month: np.ndarray, lat_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nh_warm_idx = np.array([3, 4, 5, 6, 7, 8], dtype=np.int32)
    nh_cool_idx = np.array([9, 10, 11, 0, 1, 2], dtype=np.int32)
    sh_warm_idx = nh_cool_idx
    sh_cool_idx = nh_warm_idx
    p_nh_warm = np.sum(precip_month[nh_warm_idx], axis=0)
    p_nh_cool = np.sum(precip_month[nh_cool_idx], axis=0)
    p_sh_warm = np.sum(precip_month[sh_warm_idx], axis=0)
    p_sh_cool = np.sum(precip_month[sh_cool_idx], axis=0)
    warm = np.where(lat_2d >= 0.0, p_nh_warm, p_sh_warm).astype(np.float32)
    cool = np.where(lat_2d >= 0.0, p_nh_cool, p_sh_cool).astype(np.float32)
    return (warm, cool)

def _summer_winter_monthly_extremes(precip_month: np.ndarray, lat_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nh_summer_idx = np.array([3, 4, 5, 6, 7, 8], dtype=np.int32)
    nh_winter_idx = np.array([9, 10, 11, 0, 1, 2], dtype=np.int32)
    sh_summer_idx = nh_winter_idx
    sh_winter_idx = nh_summer_idx
    s_dry_nh = np.min(precip_month[nh_summer_idx], axis=0)
    w_wet_nh = np.max(precip_month[nh_winter_idx], axis=0)
    s_dry_sh = np.min(precip_month[sh_summer_idx], axis=0)
    w_wet_sh = np.max(precip_month[sh_winter_idx], axis=0)
    s_dry = np.where(lat_2d >= 0.0, s_dry_nh, s_dry_sh).astype(np.float32)
    w_wet = np.where(lat_2d >= 0.0, w_wet_nh, w_wet_sh).astype(np.float32)
    return (s_dry, w_wet)

def _trewartha_zone_indices_monthly(sim: dict[str, np.ndarray]) -> np.ndarray:
    land = sim['land']
    lat_2d = sim['lat_2d']
    continentality = sim['continentality']
    t_mon = sim['temp_month']
    p_mon = sim['precip_month']
    t_ann = np.mean(t_mon, axis=0)
    t_hot = np.max(t_mon, axis=0)
    t_cold = np.min(t_mon, axis=0)
    p_ann = np.sum(p_mon, axis=0)
    p_dry = np.min(p_mon, axis=0)
    p_warm, _p_cool = _seasonal_precip_from_monthly(p_mon, lat_2d)
    s_dry, w_wet = _summer_winter_monthly_extremes(p_mon, lat_2d)
    warm_share = p_warm / np.maximum(p_ann, 1.0)
    p_thr = 20.0 * t_ann + 140.0
    p_thr = np.where(warm_share >= 0.7, 20.0 * t_ann + 280.0, p_thr)
    p_thr = np.where(warm_share <= 0.3, 20.0 * t_ann + 0.0, p_thr)
    p_thr = np.clip(p_thr, 20.0, None)
    is_arid = p_ann < p_thr
    is_desert = p_ann < 0.5 * p_thr
    n10 = np.sum(t_mon >= 10.0, axis=0)
    AR = 0
    AW = 1
    BW = 2
    BS = 3
    CF = 4
    CS = 5
    DO = 6
    DC = 7
    E = 8
    FT = 9
    FI = 10
    z = np.full(land.shape, DC, dtype=np.uint8)
    lm = land
    z[lm & (t_hot < 0.0)] = FI
    z[lm & (t_hot >= 0.0) & (t_hot < 10.0)] = FT
    z[lm & is_arid & is_desert & (t_hot >= 10.0)] = BW
    z[lm & is_arid & ~is_desert & (t_hot >= 10.0)] = BS
    tropical = lm & ~is_arid & (t_cold >= 18.0)
    z[tropical & (p_dry >= 60.0)] = AR
    z[tropical & (p_dry < 60.0)] = AW
    non_trop = lm & ~is_arid & ~tropical & (t_hot >= 10.0)
    dry_summer = (s_dry < 40.0) & (s_dry < w_wet / 3.0)
    subtropical = non_trop & (n10 >= 8)
    z[subtropical & dry_summer] = CS
    z[subtropical & ~dry_summer] = CF
    temperate = non_trop & (n10 >= 4) & (n10 <= 7)
    do = temperate & ((continentality < 0.42) | (t_cold > -2.0))
    z[do] = DO
    z[temperate & ~do] = DC
    boreal = non_trop & (n10 >= 1) & (n10 <= 3)
    z[boreal] = E
    z[lm & (n10 == 0) & (t_hot >= 0.0)] = FT
    z[lm & (t_hot < 0.0)] = FI
    return z

def _wiki_zone_indices_monthly(sim: dict[str, np.ndarray]) -> np.ndarray:
    land = sim['land']
    lat_2d = sim['lat_2d']
    continentality = sim['continentality']
    west_ocean = sim['west_ocean']
    east_ocean = sim['east_ocean']
    u_sfc = sim['u_surface']
    t_mon = sim['temp_month']
    p_mon = sim['precip_month']
    abs_lat = np.abs(lat_2d)
    t_ann = np.mean(t_mon, axis=0)
    t_hot = np.max(t_mon, axis=0)
    t_cold = np.min(t_mon, axis=0)
    p_ann = np.sum(p_mon, axis=0)
    p_dry = np.min(p_mon, axis=0)
    n10 = np.sum(t_mon >= 10.0, axis=0)
    p_warm, _p_cool = _seasonal_precip_from_monthly(p_mon, lat_2d)
    s_dry, w_wet = _summer_winter_monthly_extremes(p_mon, lat_2d)
    warm_share = p_warm / np.maximum(p_ann, 1.0)
    p_thr = 20.0 * t_ann + 140.0
    p_thr = np.where(warm_share >= 0.7, 20.0 * t_ann + 280.0, p_thr)
    p_thr = np.where(warm_share <= 0.3, 20.0 * t_ann + 0.0, p_thr)
    p_thr = np.clip(p_thr, 20.0, None)
    is_arid = p_ann < p_thr
    is_desert = p_ann < 0.5 * p_thr
    is_semi_arid = is_arid & ~is_desert
    west_windward = u_sfc > 0.0
    west_dominant = west_ocean > east_ocean * 0.85
    east_dominant = east_ocean > west_ocean * 0.85
    EISWUESTE = 0
    TUNDRA_Z = 1
    NADELWALD = 2
    MISCHWALD = 3
    GEM_REGENWALD = 4
    WESTSEITE = 5
    STEPPE = 6
    WINTERKALT_TROCKEN = 7
    HEISS_TROCKEN = 8
    DORNSAVANNE = 9
    OSTSEITE = 10
    TROCKENSAVANNE = 11
    FEUCHTSAVANNE = 12
    TROP_REGENWALD = 13
    z = np.full(land.shape, MISCHWALD, dtype=np.uint8)
    lm = land
    z[lm & (t_hot < 0.0)] = EISWUESTE
    z[lm & (t_hot >= 0.0) & (t_hot < 10.0)] = TUNDRA_Z
    z[lm & is_desert & (t_hot >= 10.0) & (t_cold > -5.0)] = HEISS_TROCKEN
    z[lm & is_desert & (t_hot >= 10.0) & (t_cold <= -5.0)] = WINTERKALT_TROCKEN
    z[lm & is_semi_arid & (t_hot >= 10.0) & (t_cold >= 12.0)] = DORNSAVANNE
    z[lm & is_semi_arid & (t_hot >= 10.0) & (t_cold > -5.0) & (t_cold < 12.0)] = STEPPE
    z[lm & is_semi_arid & (t_hot >= 10.0) & (t_cold <= -5.0)] = WINTERKALT_TROCKEN
    tropical = lm & ~is_arid & (t_cold >= 18.0)
    z[tropical & (p_dry >= 60.0)] = TROP_REGENWALD
    z[tropical & (p_dry >= 20.0) & (p_dry < 60.0)] = FEUCHTSAVANNE
    z[tropical & (p_dry < 20.0)] = TROCKENSAVANNE
    temperate_core = lm & ~is_arid & ~tropical & (t_hot >= 10.0)
    dry_summer = (s_dry < 40.0) & (s_dry < w_wet / 3.0)
    westseite = temperate_core & dry_summer & (abs_lat >= 24.0) & (abs_lat <= 45.0) & (t_cold > -3.0) & (west_windward | west_dominant) & (west_ocean > east_ocean)
    z[westseite] = WESTSEITE
    ostseite = temperate_core & ~westseite & (t_hot >= 22.0) & (t_cold > 0.0) & (p_ann >= 550.0) & (abs_lat >= 18.0) & (abs_lat <= 42.0) & (east_dominant | ~west_windward & (east_ocean > 0.15))
    z[ostseite] = OSTSEITE
    gem_regen = temperate_core & ~westseite & ~ostseite & (t_cold > 2.0) & (continentality < 0.3) & (p_ann >= 700.0) & (n10 >= 4) & (west_windward & (west_ocean > 0.15))
    z[gem_regen] = GEM_REGENWALD
    nadelwald = temperate_core & ~westseite & ~ostseite & ~gem_regen & ((t_cold <= -15.0) | (n10 <= 4) & (continentality > 0.3) | (t_hot < 18.0) & (t_cold < -5.0))
    z[nadelwald] = NADELWALD
    z[temperate_core & ~westseite & ~ostseite & ~gem_regen & ~nadelwald] = MISCHWALD
    z[lm & (abs_lat > POLAR_CIRCLE_LAT) & (t_hot < 10.0)] = TUNDRA_Z
    z[lm & (abs_lat > 78.0) & (t_hot < 1.0)] = EISWUESTE
    return z

def build_land_climate_image_with_data(ocean: np.ndarray, raster_h: int, alpha: int=230, coast_window_px: int=270, month: int=1, mountain_mask: np.ndarray | None=None) -> tuple[bytes, np.ndarray, list[str], np.ndarray]:
    del month
    sim = simulate_monthly_climate(ocean=ocean, raster_h=raster_h, coast_window_px=coast_window_px, mountain_mask=mountain_mask)
    zone_indices = _wiki_zone_indices_monthly(sim)
    land = np.asarray(sim['land'], dtype=bool)
    png = _render_zone_image(zone_indices, land, _WIKI_ZONE_LIST, LAND_CLIMATE_COLORS, alpha)
    return (png, zone_indices, _WIKI_ZONE_LIST, land)

def build_trewartha_climate_image_with_data(ocean: np.ndarray, raster_h: int, alpha: int=230, coast_window_px: int=270, month: int=1, mountain_mask: np.ndarray | None=None) -> tuple[bytes, np.ndarray, list[str], np.ndarray]:
    del month
    sim = simulate_monthly_climate(ocean=ocean, raster_h=raster_h, coast_window_px=coast_window_px, mountain_mask=mountain_mask)
    zone_indices = _trewartha_zone_indices_monthly(sim)
    land = np.asarray(sim['land'], dtype=bool)
    png = _render_zone_image(zone_indices, land, _TREWARTHA_ZONE_LIST, TREWARTHA_CLIMATE_COLORS, alpha)
    return (png, zone_indices, _TREWARTHA_ZONE_LIST, land)

def build_land_climate_image(ocean: np.ndarray, raster_h: int, alpha: int=230, coast_window_px: int=270, month: int=1, mountain_mask: np.ndarray | None=None) -> bytes:
    png, _zone_indices, _zone_list, _land = build_land_climate_image_with_data(ocean=ocean, raster_h=raster_h, alpha=alpha, coast_window_px=coast_window_px, month=month, mountain_mask=mountain_mask)
    return png

def build_trewartha_climate_image(ocean: np.ndarray, raster_h: int, alpha: int=230, coast_window_px: int=270, month: int=1, mountain_mask: np.ndarray | None=None) -> bytes:
    png, _zone_indices, _zone_list, _land = build_trewartha_climate_image_with_data(ocean=ocean, raster_h=raster_h, alpha=alpha, coast_window_px=coast_window_px, month=month, mountain_mask=mountain_mask)
    return png

def compute_zone_label_positions(z: np.ndarray, land: np.ndarray, zone_list: list[str], label_map: dict[str, str], raster_w: int, raster_h: int, vb_min_x: float, vb_min_y: float, vb_width: float, vb_height: float, min_area_px: int=350) -> list[tuple[float, float, str]]:
    del raster_w, raster_h
    h, w = z.shape
    sx = vb_width / w
    sy = vb_height / h
    positions: list[tuple[float, float, str]] = []
    for zone_idx, zone_key in enumerate(zone_list):
        label_text = label_map.get(zone_key, zone_key)
        mask = ((z == zone_idx) & land).astype(np.uint8)
        if int(mask.sum()) < min_area_px:
            continue
        labeled, n_comp = label(mask, structure=_LABEL_CONNECTIVITY)
        for comp_id in range(1, n_comp + 1):
            comp_mask = labeled == comp_id
            if int(comp_mask.sum()) < min_area_px:
                continue
            center_y, center_x = center_of_mass(comp_mask)
            svg_x = vb_min_x + float(center_x) * sx
            svg_y = vb_min_y + float(center_y) * sy
            positions.append((svg_x, svg_y, label_text))
    return positions
