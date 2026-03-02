import numpy as np
from ocean import rasterize_svg, ocean_mask_from_rgba
from land_climate_physics import simulate_monthly_climate, _q_sat_kgkg
rgba = rasterize_svg('welt.svg', out_w=1800, out_h=900)
ocean = ocean_mask_from_rgba(rgba)
h, w = ocean.shape
land = ~ocean
from land_climate_physics import *
lat_1d = np.array([lat_from_y(float(y), h) for y in range(h)], dtype=np.float32)
lat_2d = np.broadcast_to(lat_1d[:, np.newaxis], (h, w))
lat_rad = np.deg2rad(lat_2d).astype(np.float32)
dist_to_ocean = distance_transform_edt(land).astype(np.float32)
continentality = np.clip(dist_to_ocean / max(20.0, w * 0.2), 0.0, 1.0)
ocean_f = ocean.astype(np.float32)
half = 270 // 2
west_ocean = uniform_filter1d(np.roll(ocean_f, half, axis=1), size=270, axis=1, mode='wrap').astype(np.float32)
east_ocean = uniform_filter1d(np.roll(ocean_f, -half, axis=1), size=270, axis=1, mode='wrap').astype(np.float32)
coastal_fetch = np.clip(0.55 * west_ocean + 0.45 * east_ocean, 0.0, 1.0)
c_eff = np.where(ocean, 3.6, 1.2).astype(np.float32)
albedo = np.where(ocean, 0.08, 0.2).astype(np.float32)
a_olr = 240.0
b_olr = 4.3
t = (24.0 - 0.35 * np.abs(lat_2d)).astype(np.float32)
t += np.where(ocean, 1.5, -1.5).astype(np.float32)
t = gaussian_filter(t, sigma=(1.2, 1.2), mode=('nearest', 'wrap')).astype(np.float32)
q = (0.6 * _q_sat_kgkg(t)).astype(np.float32)
month_days = [15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
seconds_month = 30.0 * 24.0 * 3600.0
for month_idx, day in enumerate(month_days[:2]):
    month = month_idx + 1
    q_toa = _daily_mean_insolation_wm2(lat_rad, day)[:, :]
    absorbed = q_toa * (1.0 - albedo)
    t_rad_eq = (absorbed - a_olr) / b_olr
    u_row = _zonal_wind_u(lat_1d)
    v_row = _meridional_wind_v(lat_1d, month=month)
    u_bg = np.broadcast_to(u_row[:, np.newaxis], (h, w)).astype(np.float32)
    v_bg = np.broadcast_to(v_row[:, np.newaxis], (h, w)).astype(np.float32)
    t_zonal_mean = np.mean(t, axis=1, keepdims=True).astype(np.float32)
    t_prime = (t - t_zonal_mean) * (0.75 + 0.25 * land.astype(np.float32))
    p_prime = (-1.3 * t_prime).astype(np.float32)
    dp_dy = np.gradient(p_prime, axis=0).astype(np.float32)
    dp_dx = ((np.roll(p_prime, -1, axis=1) - np.roll(p_prime, 1, axis=1)) * 0.5).astype(np.float32)
    u_pg = (-0.11 * dp_dx).astype(np.float32)
    v_pg = (-0.11 * dp_dy).astype(np.float32)
    hemi = np.where(lat_2d >= 0.0, 1.0, -1.0).astype(np.float32)
    season = math.sin(2.0 * math.pi * (month - 3.0) / 12.0)
    monsoon_band = np.exp(-((np.abs(lat_2d) - 18.0) / 10.0) ** 2).astype(np.float32)
    v_monsoon = (0.95 * season * hemi * monsoon_band * land.astype(np.float32) * (0.35 + 0.65 * np.clip(1.0 - continentality, 0.0, 1.0))).astype(np.float32)
    u = (u_bg + u_pg).astype(np.float32)
    v = (v_bg + v_pg + v_monsoon).astype(np.float32)
    adv_t = _advect_upwind(t, u=u * 0.014, v=v * 0.026)
    diff_t = _laplacian(t) * 0.08
    relax = 0.16 / c_eff
    t = (t + relax * (t_rad_eq - t) - 0.18 * adv_t + diff_t).astype(np.float32)
    t -= (0.1 * continentality * (np.abs(lat_2d) / 90.0)).astype(np.float32)
    t = gaussian_filter(t, sigma=(0.8, 0.8), mode=('nearest', 'wrap')).astype(np.float32)
    t = np.clip(t, -45.0, 42.0).astype(np.float32)
    qsat = _q_sat_kgkg(t)
    wind = np.sqrt(u ** 2 + v ** 2).astype(np.float32)
    evap_ocean = (1.2e-05 * wind * np.maximum(qsat - q, 0.0) * ocean).astype(np.float32)
    evap_land = (1.2e-06 * wind * np.maximum(qsat - q, 0.0) * land * np.clip(1.0 - continentality, 0.15, 1.0)).astype(np.float32)
    evap = evap_ocean + evap_land
    uq = (u * 0.024).astype(np.float32)
    vq = (v * 0.03).astype(np.float32)
    adv_q = _flux_divergence(q, uq, vq)
    diff_q = _laplacian(q) * 0.14
    q_before = q.copy()
    q = (q - adv_q + diff_q + evap * seconds_month).astype(np.float32)
    q = np.maximum(q, 0.0).astype(np.float32)
    supersat = np.maximum(q - qsat, 0.0)
    convergence = np.maximum(-adv_q, 0.0)
    tau_cond = 10.0 * 24.0 * 3600.0
    precip_base = (supersat / tau_cond + 0.018 * convergence / seconds_month).astype(np.float32)
    dtdy = np.gradient(t, axis=0).astype(np.float32)
    storm_band = np.exp(-((np.abs(lat_2d) - 52.0) / 11.0) ** 2).astype(np.float32)
    storm_rate = (2.2e-06 * storm_band * np.maximum(u_bg, 0.0) * np.abs(dtdy) * (0.25 + 0.75 * west_ocean) * land.astype(np.float32)).astype(np.float32)
    monsoon_rate = (2.5e-06 * monsoon_band * np.maximum(v * hemi, 0.0) * (0.3 + 0.7 * east_ocean) * land.astype(np.float32)).astype(np.float32)
    precip_rate = precip_base + storm_rate + monsoon_rate
    precip_rate = np.clip(precip_rate, 0.0, 0.00018).astype(np.float32)
    q = np.maximum(q - precip_rate * seconds_month, 0.0).astype(np.float32)
    q = np.minimum(q, 1.18 * qsat).astype(np.float32)
    p_mm = (precip_rate * seconds_month * 9.0).astype(np.float32)
    p_mm *= (0.7 + 0.8 * coastal_fetch) * (0.7 + 0.5 * np.clip(1.0 - continentality, 0.0, 1.0))
    p_mm = gaussian_filter(p_mm, sigma=(1.0, 1.3), mode=('nearest', 'wrap')).astype(np.float32)
    p_mm[~land] = 0.0
    print(f'Month {month}:')
    print(f'  q mean: {q[land].mean():.5f}, max: {q[land].max():.5f}')
    print(f'  qsat mean: {qsat[land].mean():.5f}')
    print(f'  evap mean: {(evap * seconds_month)[land].mean():.5f}')
    print(f'  adv_q mean: {adv_q[land].mean():.5f}')
    print(f'  precip_base mean: {(precip_base * seconds_month * 9.0)[land].mean():.5f}')
    print(f'  storm_rate mean: {(storm_rate * seconds_month * 9.0)[land].mean():.5f}')
    print(f'  monsoon_rate mean: {(monsoon_rate * seconds_month * 9.0)[land].mean():.5f}')
    print(f'  p_mm mean: {p_mm[land].mean():.5f}, max: {p_mm[land].max():.5f}')
