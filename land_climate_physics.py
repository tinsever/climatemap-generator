import math
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.ndimage import uniform_filter1d
from constants import TROPIC_LAT, SUBTROPIC_MAX_LAT, ITCZ_CORE_LAT, MONSOON_CORE_LAT, EAST_ONSHORE_CORE_LAT, SUBTROPICAL_DRY_CORE_LAT, FRONTAL_CORE_LAT, WESTERLY_CORE_LAT, STORM_TRACK_CORE_LAT, SUBPOLAR_LOW_CORE_LAT, POLAR_EASTERLY_CORE_LAT, POLAR_HIGH_CORE_LAT
from ocean import lat_from_y, itcz_offset

def _daily_mean_insolation_wm2(lat_rad: np.ndarray, day_of_year: int) -> np.ndarray:
    """Spencer (1971) daily mean insolation [W m-2]."""
    s0 = 1361.0
    gamma = 2.0 * math.pi * (day_of_year - 1) / 365.0
    decl = 0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma) - 0.006758 * np.cos(2.0 * gamma) + 0.000907 * np.sin(2.0 * gamma) - 0.002697 * np.cos(3.0 * gamma) + 0.00148 * np.sin(3.0 * gamma)
    x = np.clip(-np.tan(lat_rad) * np.tan(decl), -1.0, 1.0)
    h0 = np.arccos(x)
    q = s0 / math.pi * (h0 * np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.sin(h0))
    return np.maximum(q, 0.0).astype(np.float32)

def _q_sat_kgkg(temp_c: np.ndarray) -> np.ndarray:
    """Saturation specific humidity [kg/kg] at near-surface pressure."""
    t = np.asarray(temp_c, dtype=np.float32)
    es_hpa = 6.112 * np.exp(17.67 * t / np.maximum(t + 243.5, 1.0))
    qsat = 0.622 * es_hpa / np.maximum(1013.25 - 0.378 * es_hpa, 1.0)
    return np.clip(qsat, 0.0, 0.045).astype(np.float32)

def _flux_divergence(scalar: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Divergence of horizontal scalar flux (u*q, v*q)."""
    fx = (u * scalar).astype(np.float32)
    fy = (v * scalar).astype(np.float32)
    div_x = ((np.roll(fx, -1, axis=1) - np.roll(fx, 1, axis=1)) * 0.5).astype(np.float32)
    div_y = np.zeros_like(scalar, dtype=np.float32)
    div_y[1:-1] = (fy[2:] - fy[:-2]) * 0.5
    div_y[0] = fy[1] - fy[0]
    div_y[-1] = fy[-1] - fy[-2]
    return (div_x + div_y).astype(np.float32)

def _precipitation_from_moisture(abs_lat: np.ndarray, hemi: np.ndarray, land: np.ndarray, continentality: np.ndarray, ocean_proximity: np.ndarray, west_ocean: np.ndarray, east_ocean: np.ndarray, t: np.ndarray, u: np.ndarray, v: np.ndarray, p_sfc: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One monthly moisture step: evaporation + advection + condensation."""
    seconds_month = 30.0 * 24.0 * 3600.0
    land_f = land.astype(np.float32)
    ocean_f = (~land).astype(np.float32)
    qsat = _q_sat_kgkg(t)
    wind = np.sqrt(u ** 2 + v ** 2).astype(np.float32)
    evap_pot = np.maximum(qsat - q, 0.0).astype(np.float32)
    coast_land = np.clip(1.0 - continentality, 0.15, 1.0).astype(np.float32)
    evap_ocean = (1.45e-05 * (0.3 + wind) * evap_pot * ocean_f).astype(np.float32)
    evap_land = (2.1e-06 * (0.25 + wind) * evap_pot * land_f * coast_land).astype(np.float32)
    adv_q = _flux_divergence(q, u * 0.024, v * 0.032)
    diff_q = (_laplacian(q) * 0.12).astype(np.float32)
    q_next = (q - adv_q + diff_q + (evap_ocean + evap_land) * seconds_month).astype(np.float32)
    q_next = np.maximum(q_next, 0.0).astype(np.float32)
    supersat = np.maximum(q_next - qsat, 0.0).astype(np.float32)
    convergence = np.maximum(-adv_q, 0.0).astype(np.float32)
    tau_cond = 6.0 * 24.0 * 3600.0
    precip_base = (supersat / tau_cond + 0.045 * convergence / seconds_month).astype(np.float32)
    dtdy = np.abs(_gradient_y(t)).astype(np.float32)
    storm_band = np.exp(-((abs_lat - STORM_TRACK_CORE_LAT) / 11.0) ** 2).astype(np.float32)
    storm_rate = (2.4e-06 * storm_band * np.maximum(u, 0.0) * dtdy * (0.2 + 0.8 * west_ocean) * land_f).astype(np.float32)
    frontal_band = np.exp(-((abs_lat - FRONTAL_CORE_LAT) / 13.0) ** 2).astype(np.float32)
    frontal_rate = (2.2e-06 * frontal_band * np.maximum(u, 0.0) * (0.25 + 0.75 * west_ocean) * (0.35 + 0.65 * ocean_proximity) * land_f).astype(np.float32)
    east_onshore_band = np.exp(-((abs_lat - EAST_ONSHORE_CORE_LAT) / 12.0) ** 2).astype(np.float32)
    east_onshore_rate = (1.4e-06 * east_onshore_band * np.maximum(-u, 0.0) * (0.3 + 0.7 * east_ocean) * (0.3 + 0.7 * ocean_proximity) * land_f).astype(np.float32)
    monsoon_band = np.exp(-((abs_lat - MONSOON_CORE_LAT) / 11.0) ** 2).astype(np.float32)
    onshore_v = np.maximum(v * hemi, 0.0).astype(np.float32)
    monsoon_rate = (3.8e-06 * monsoon_band * onshore_v * (0.25 + 0.75 * east_ocean) * (1.0 - 0.55 * continentality) * land_f).astype(np.float32)
    high_supp = np.clip(p_sfc / 6.0, 0.0, 1.0).astype(np.float32)
    rel_h = np.clip(q_next / np.maximum(qsat, 0.0001), 0.0, 1.5).astype(np.float32)
    itcz_band = np.exp(-((abs_lat - ITCZ_CORE_LAT) / 14.0) ** 2).astype(np.float32)
    itcz_rate = (2.8e-06 * itcz_band * np.clip(-p_sfc / 4.0, 0.0, 1.0) * (0.3 + 0.7 * ocean_proximity) * np.clip(0.45 + 0.55 * rel_h, 0.0, 1.4) * land_f).astype(np.float32)
    subsidence_sink = (6.5e-07 * high_supp * (0.45 + 0.55 * continentality) * land_f).astype(np.float32)
    q_next = np.maximum(q_next - subsidence_sink * seconds_month, 0.0).astype(np.float32)
    low_boost = (1.0 + 0.35 * np.clip(-p_sfc / 5.0, 0.0, 1.0)).astype(np.float32)
    precip_rate = (precip_base * np.maximum(1.0 - (0.38 + 0.35 * continentality) * high_supp, 0.18) * low_boost + storm_rate + frontal_rate + east_onshore_rate + itcz_rate + monsoon_rate).astype(np.float32)
    precip_rate = np.clip(precip_rate, 0.0, 0.0006).astype(np.float32)
    q_next = np.maximum(q_next - precip_rate * seconds_month, 0.0).astype(np.float32)
    q_next = np.minimum(q_next, 1.2 * qsat).astype(np.float32)
    dominant_fetch = np.maximum(west_ocean, east_ocean).astype(np.float32)
    coastal_factor = np.maximum(0.8 + 0.55 * dominant_fetch, 0.7 + 0.45 * ocean_proximity).astype(np.float32)
    inland_factor = (0.6 + 0.65 * np.clip(1.0 - continentality, 0.0, 1.0)).astype(np.float32)
    p_mm = (precip_rate * seconds_month * 32.0 * coastal_factor * inland_factor).astype(np.float32)
    subtrop_band = np.exp(-((abs_lat - SUBTROPICAL_DRY_CORE_LAT) / 10.0) ** 2).astype(np.float32)
    p_mm *= (1.0 - 0.25 * subtrop_band * high_supp * land_f).astype(np.float32)
    p_mm = gaussian_filter(p_mm, sigma=(1.0, 1.2), mode=('nearest', 'wrap')).astype(np.float32)
    p_mm[~land] = 0.0
    p_mm = np.clip(p_mm, 0.0, 450.0).astype(np.float32)
    return (p_mm, q_next)

def _zonal_wind_u(lat_deg: np.ndarray, month: int | None=None) -> np.ndarray:
    """Background zonal wind (positive = westerly).

    When *month* is given the 3-cell pattern shifts with the ITCZ,
    consistent with ``windstress_curl_forcing`` in ``ocean.py``.
    """
    if month is not None:
        lat_eff = lat_deg - itcz_offset(month)
    else:
        lat_eff = lat_deg
    a = np.abs(lat_eff)
    trade_center = TROPIC_LAT * 0.65
    trades = -1.0 * np.exp(-((a - trade_center) / 11.0) ** 2)
    westerlies = 1.4 * np.exp(-((a - WESTERLY_CORE_LAT) / 13.0) ** 2)
    polar_east = -0.5 * np.exp(-((a - POLAR_EASTERLY_CORE_LAT) / 10.0) ** 2)
    return (trades + westerlies + polar_east).astype(np.float32)

def _meridional_wind_v(lat_deg: np.ndarray, month: int) -> np.ndarray:
    """Background meridional wind (positive = poleward)."""
    itcz = TROPIC_LAT * math.sin(2.0 * math.pi * (month - 3.0) / 12.0)
    dy = lat_deg - itcz
    hadley_decay = max(12.0, SUBTROPIC_MAX_LAT - 2.0)
    hadley = -0.35 * np.tanh(dy / 14.0) * np.exp(-(np.abs(lat_deg) / hadley_decay) ** 2)
    ferrel = 0.15 * np.sin(np.deg2rad(lat_deg * 2.0)) * np.exp(-((np.abs(lat_deg) - WESTERLY_CORE_LAT) / 20.0) ** 2)
    return (hadley + ferrel).astype(np.float32)

def _surface_pressure_zonal(lat_deg: np.ndarray, month: int) -> np.ndarray:
    """Zonal-mean surface pressure anomaly [hPa] from the 3-cell circulation.

    Shifts seasonally with ``itcz_offset``, consistent with the ocean
    wind stress and the zonal wind.
    """
    offset = itcz_offset(month)
    lat_s = lat_deg - offset
    abs_s = np.abs(lat_s)
    sub_high = 6.0 * np.exp(-((abs_s - SUBTROPICAL_DRY_CORE_LAT) / 10.0) ** 2)
    subpol_low = -5.0 * np.exp(-((abs_s - SUBPOLAR_LOW_CORE_LAT) / 10.0) ** 2)
    polar_high = 3.0 * np.exp(-((abs_s - POLAR_HIGH_CORE_LAT) / 8.0) ** 2)
    itcz_low = -3.0 * np.exp(-(lat_s / max(6.0, ITCZ_CORE_LAT)) ** 2)
    return (sub_high + subpol_low + polar_high + itcz_low).astype(np.float32)

def _laplacian(field: np.ndarray) -> np.ndarray:
    return (np.roll(field, -1, axis=1) + np.roll(field, 1, axis=1) + np.roll(field, -1, axis=0) + np.roll(field, 1, axis=0) - 4.0 * field).astype(np.float32)

def _gradient_x(field: np.ndarray) -> np.ndarray:
    return ((np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) * 0.5).astype(np.float32)

def _gradient_y(field: np.ndarray) -> np.ndarray:
    out = np.zeros_like(field, dtype=np.float32)
    out[1:-1] = (field[2:] - field[:-2]) * 0.5
    out[0] = field[1] - field[0]
    out[-1] = field[-1] - field[-2]
    return out

def _advect(field: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for upwind advection."""
    return _advect_upwind(field, u, v)

def _advect_upwind(field: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """First-order upwind advection tendency."""
    dx_back = (field - np.roll(field, 1, axis=1)).astype(np.float32)
    dx_fwd = (np.roll(field, -1, axis=1) - field).astype(np.float32)
    north = np.empty_like(field, dtype=np.float32)
    south = np.empty_like(field, dtype=np.float32)
    north[1:] = field[:-1]
    north[0] = field[0]
    south[:-1] = field[1:]
    south[-1] = field[-1]
    dy_back = (field - north).astype(np.float32)
    dy_fwd = (south - field).astype(np.float32)
    adv_x = np.where(u >= 0.0, u * dx_back, u * dx_fwd).astype(np.float32)
    adv_y = np.where(v >= 0.0, v * dy_back, v * dy_fwd).astype(np.float32)
    return (adv_x + adv_y).astype(np.float32)

def simulate_monthly_climate(ocean: np.ndarray, raster_h: int, coast_window_px: int=270, mountain_mask: np.ndarray | None=None, mountain_dt: float=-10.0) -> dict[str, np.ndarray]:
    h, w = ocean.shape
    land = ~ocean
    lat_1d = np.array([lat_from_y(float(y), raster_h) for y in range(h)], dtype=np.float32)
    lat_2d = lat_1d[:, np.newaxis] * np.ones((1, w), dtype=np.float32)
    lat_rad = np.deg2rad(lat_2d).astype(np.float32)
    abs_lat = np.abs(lat_2d)
    hemi = np.where(lat_2d >= 0.0, 1.0, -1.0).astype(np.float32)
    dist_ocean = distance_transform_edt(land).astype(np.float32)
    scale_px = max(30.0, w * 0.15)
    continentality = np.clip(dist_ocean / scale_px, 0.0, 1.0)
    ocean_proximity = np.exp(-dist_ocean / (scale_px * 0.25)).astype(np.float32)
    ocean_f = ocean.astype(np.float32)
    half = coast_window_px // 2
    west_ocean = uniform_filter1d(np.roll(ocean_f, half, axis=1), size=coast_window_px, axis=1, mode='wrap').astype(np.float32)
    east_ocean = uniform_filter1d(np.roll(ocean_f, -half, axis=1), size=coast_window_px, axis=1, mode='wrap').astype(np.float32)
    a_olr = 209.0
    b_olr = 2.15
    albedo = np.where(ocean, 0.08, 0.23).astype(np.float32)
    c_eff = np.where(ocean, 4.2, 1.5).astype(np.float32)
    t = (27.0 - 0.3 * abs_lat - 0.001 * abs_lat ** 2).astype(np.float32)
    t = gaussian_filter(t, sigma=(1.0, 1.0), mode=('nearest', 'wrap')).astype(np.float32)
    temp_month = np.zeros((12, h, w), dtype=np.float32)
    precip_month = np.zeros((12, h, w), dtype=np.float32)
    u_month = np.zeros((12, h, w), dtype=np.float32)
    v_month = np.zeros((12, h, w), dtype=np.float32)
    p_month = np.zeros((12, h, w), dtype=np.float32)
    q_month = np.zeros((12, h, w), dtype=np.float32)
    q = (0.66 * _q_sat_kgkg(t)).astype(np.float32)
    q = np.where(ocean, np.maximum(q, 0.78 * _q_sat_kgkg(t)), q).astype(np.float32)
    month_days = [15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
    for month_idx, day in enumerate(month_days):
        month = month_idx + 1
        q_toa = _daily_mean_insolation_wm2(lat_rad, day)
        absorbed = q_toa * (1.0 - albedo)
        t_eq = (absorbed - a_olr) / b_olr
        u_bg = np.broadcast_to(_zonal_wind_u(lat_1d, month=month)[:, np.newaxis], (h, w)).astype(np.float32)
        v_bg = np.broadcast_to(_meridional_wind_v(lat_1d, month)[:, np.newaxis], (h, w)).astype(np.float32)
        t_zm = t.mean(axis=1, keepdims=True).astype(np.float32)
        t_pri = (t - t_zm).astype(np.float32)
        u_pg = (0.08 * _gradient_x(t_pri)).astype(np.float32)
        v_pg = (-0.05 * _gradient_y(t_pri)).astype(np.float32)
        u = (u_bg + u_pg).astype(np.float32)
        v = (v_bg + v_pg).astype(np.float32)
        relax = 0.2 / c_eff
        adv_t = _advect(t, u * 0.018, v * 0.03)
        diff_t = _laplacian(t) * 0.06
        t = (t + relax * (t_eq - t) - adv_t + diff_t).astype(np.float32)
        t = gaussian_filter(t, sigma=(0.6, 0.6), mode=('nearest', 'wrap')).astype(np.float32)
        t = np.clip(t, -55.0, 42.0).astype(np.float32)
        if mountain_mask is not None:
            t = np.where(mountain_mask, t + mountain_dt, t).astype(np.float32)
        p_zonal = np.broadcast_to(_surface_pressure_zonal(lat_1d, month)[:, np.newaxis], (h, w)).astype(np.float32)
        p_thermal = (-1.3 * t_pri * land.astype(np.float32)).astype(np.float32)
        p_sfc = (p_zonal + p_thermal).astype(np.float32)
        p_mm, q = _precipitation_from_moisture(abs_lat=abs_lat, hemi=hemi, land=land, continentality=continentality, ocean_proximity=ocean_proximity, west_ocean=west_ocean, east_ocean=east_ocean, t=t, u=u, v=v, p_sfc=p_sfc, q=q)
        temp_month[month_idx] = t
        precip_month[month_idx] = p_mm
        u_month[month_idx] = u
        v_month[month_idx] = v
        p_month[month_idx] = p_sfc
        q_month[month_idx] = q
    u_surface = np.mean(u_month, axis=0).astype(np.float32)
    return {'land': land, 'lat_2d': lat_2d, 'continentality': continentality, 'west_ocean': west_ocean, 'east_ocean': east_ocean, 'u_surface': u_surface, 'u_month': u_month, 'v_month': v_month, 'p_month': p_month, 'humidity_month': q_month, 'temp_month': temp_month, 'precip_month': precip_month}
