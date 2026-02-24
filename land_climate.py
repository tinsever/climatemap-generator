import io
import math

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter1d, distance_transform_edt

from constants import LAND_CLIMATE_COLORS, TREWARTHA_CLIMATE_COLORS, TROPIC_LAT, POLAR_CIRCLE_LAT
from ocean import lat_from_y


def build_land_climate_image(
    ocean: np.ndarray,
    raster_h: int,
    alpha: int = 230,
    coast_window_px: int = 270,
    month: int = 1,
) -> bytes:
    h, w = ocean.shape
    land = ~ocean

    dist_to_ocean = distance_transform_edt(land)
    max_dist = float(dist_to_ocean.max()) if dist_to_ocean.max() > 0 else 1.0
    continentality = (dist_to_ocean / max_dist).astype(np.float32)

    ocean_f = ocean.astype(np.float32)
    half = coast_window_px // 2
    west_ocean = uniform_filter1d(
        np.roll(ocean_f, half, axis=1), size=coast_window_px, axis=1, mode="wrap"
    ).astype(np.float32)
    east_ocean = uniform_filter1d(
        np.roll(ocean_f, -half, axis=1), size=coast_window_px, axis=1, mode="wrap"
    ).astype(np.float32)

    lat_arr = np.array(
        [lat_from_y(float(y), raster_h) for y in range(h)], dtype=np.float32
    )
    a_lat = np.broadcast_to(np.abs(lat_arr)[:, np.newaxis], (h, w)).copy()

    month_norm = (month - 1) % 12
    cos_season = math.cos(2.0 * math.pi * month_norm / 12.0)
    itcz_shift = -5.0 * cos_season * np.sign(lat_arr)[:, np.newaxis]
    a_lat_eff = np.clip(a_lat - itcz_shift, 0.0, 90.0)
    monsoon_factor = 0.7 + 0.3 * (-cos_season) * np.sign(lat_arr)[:, np.newaxis]
    monsoon_factor = np.clip(monsoon_factor, 0.5, 1.0)
    east_ocean_thresh = 0.30 * (1.5 - np.broadcast_to(monsoon_factor, (h, w)))
    med_factor = 1.0 + 0.15 * (-cos_season) * np.maximum(0, np.sign(lat_arr))[:, np.newaxis]
    west_ocean_thresh_med = 0.30 * np.broadcast_to(med_factor, (h, w))

    TROPICAL_WET = 0
    TROPICAL_DRY = 1
    DESERT_HOT = 2
    STEPPE = 3
    MEDITERRANEAN = 4
    HUMID_SUB = 5
    OCEANIC = 6
    CONTINENTAL = 7
    BOREAL = 8
    TUNDRA = 9
    ICE = 10

    z = np.full((h, w), CONTINENTAL, dtype=np.uint8)
    lm = land

    z[lm & (a_lat_eff < 10.0)] = TROPICAL_WET
    z[lm & (a_lat_eff >= 10.0) & (a_lat_eff < TROPIC_LAT)] = TROPICAL_DRY

    z[lm & (a_lat_eff >= 14.0) & (a_lat_eff < 45.0) & (continentality > 0.42)] = STEPPE

    z[lm & (a_lat_eff >= 18.0) & (a_lat_eff < 35.0)
      & ((west_ocean > 0.28) | (continentality > 0.52))] = DESERT_HOT

    z[lm & (a_lat_eff >= 22.0) & (a_lat_eff < 38.0)
      & (east_ocean > east_ocean_thresh) & (continentality < 0.50)] = HUMID_SUB

    z[lm & (a_lat_eff >= 30.0) & (a_lat_eff < 44.0)
      & (west_ocean > west_ocean_thresh_med) & (continentality < 0.36)] = MEDITERRANEAN

    z[lm & (a_lat_eff >= 44.0) & (a_lat_eff < 62.0)
      & (west_ocean > 0.30) & (continentality < 0.38)] = OCEANIC

    z[lm & (a_lat_eff > 52.0) & (continentality > 0.40)] = BOREAL

    z[lm & (a_lat_eff > POLAR_CIRCLE_LAT)] = TUNDRA
    z[lm & (a_lat_eff > 75.0)] = ICE

    zone_list = [
        "tropical_wet", "tropical_dry", "desert_hot", "steppe",
        "mediterranean", "humid_sub", "oceanic", "continental",
        "boreal", "tundra", "ice",
    ]
    lut_r = np.array([LAND_CLIMATE_COLORS[k][0] for k in zone_list], dtype=np.uint8)
    lut_g = np.array([LAND_CLIMATE_COLORS[k][1] for k in zone_list], dtype=np.uint8)
    lut_b = np.array([LAND_CLIMATE_COLORS[k][2] for k in zone_list], dtype=np.uint8)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[lm, 0] = lut_r[z[lm]]
    rgba[lm, 1] = lut_g[z[lm]]
    rgba[lm, 2] = lut_b[z[lm]]
    rgba[lm, 3] = alpha

    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def build_trewartha_climate_image(
    ocean: np.ndarray,
    raster_h: int,
    alpha: int = 230,
    coast_window_px: int = 270,
    month: int = 1,
) -> bytes:
    """Klimakarte nach Trewartha-Klassifikation (11 Zonen: Ar, Aw, BW, BS, Cf, Cs, Do, Dc, E, FT, Fi).

    Verwendet dieselben Proxy-Variablen wie die Köppen-Karte:
    effektive Breite, Kontinentalität, West-/Ostküsten-Einfluss und saisonale Korrekturen.
    """
    h, w = ocean.shape
    land = ~ocean

    dist_to_ocean = distance_transform_edt(land)
    max_dist = float(dist_to_ocean.max()) if dist_to_ocean.max() > 0 else 1.0
    continentality = (dist_to_ocean / max_dist).astype(np.float32)

    ocean_f = ocean.astype(np.float32)
    half = coast_window_px // 2
    west_ocean = uniform_filter1d(
        np.roll(ocean_f, half, axis=1), size=coast_window_px, axis=1, mode="wrap"
    ).astype(np.float32)
    east_ocean = uniform_filter1d(
        np.roll(ocean_f, -half, axis=1), size=coast_window_px, axis=1, mode="wrap"
    ).astype(np.float32)

    lat_arr = np.array(
        [lat_from_y(float(y), raster_h) for y in range(h)], dtype=np.float32
    )
    a_lat = np.broadcast_to(np.abs(lat_arr)[:, np.newaxis], (h, w)).copy()

    month_norm = (month - 1) % 12
    cos_season = math.cos(2.0 * math.pi * month_norm / 12.0)
    itcz_shift = -5.0 * cos_season * np.sign(lat_arr)[:, np.newaxis]
    a_lat_eff = np.clip(a_lat - itcz_shift, 0.0, 90.0)

    monsoon_factor = 0.7 + 0.3 * (-cos_season) * np.sign(lat_arr)[:, np.newaxis]
    monsoon_factor = np.clip(monsoon_factor, 0.5, 1.0)
    east_ocean_thresh = 0.30 * (1.5 - np.broadcast_to(monsoon_factor, (h, w)))

    med_factor = 1.0 + 0.15 * (-cos_season) * np.maximum(0, np.sign(lat_arr))[:, np.newaxis]
    west_ocean_thresh_med = 0.30 * np.broadcast_to(med_factor, (h, w))

    AR = 0
    AW = 1
    BW = 2
    BS = 3
    CF = 4
    CS = 5
    DO = 6
    DC = 7
    E  = 8
    FT = 9
    FI = 10

    lm = land
    z = np.full((h, w), DC, dtype=np.uint8)
    z[lm & (a_lat_eff < TROPIC_LAT)] = AW
    z[lm & (a_lat_eff < 12.0)] = AR
    z[lm & (a_lat_eff < 12.0) & (continentality > 0.45)] = AW

    z[lm & (a_lat_eff >= 14.0) & (a_lat_eff < 46.0) & (continentality > 0.42)] = BS
    z[lm & (a_lat_eff >= 18.0) & (a_lat_eff < 35.0)
      & ((west_ocean > 0.28) | (continentality > 0.52))] = BW

    z[lm & (a_lat_eff >= 22.0) & (a_lat_eff < 38.0)
      & (east_ocean > east_ocean_thresh) & (continentality < 0.50)] = CF
    z[lm & (a_lat_eff >= 28.0) & (a_lat_eff < 44.0)
      & (west_ocean > west_ocean_thresh_med) & (continentality < 0.36)] = CS

    z[lm & (a_lat_eff >= 44.0) & (a_lat_eff < 62.0)
      & (west_ocean > 0.30) & (continentality < 0.38)] = DO
    z[lm & (a_lat_eff > 52.0) & (continentality > 0.40)] = E

    z[lm & (a_lat_eff > POLAR_CIRCLE_LAT)] = FT
    z[lm & (a_lat_eff > 78.0)] = FI

    zone_list = ["Ar", "Aw", "BW", "BS", "Cf", "Cs", "Do", "Dc", "E", "FT", "Fi"]
    lut_r = np.array([TREWARTHA_CLIMATE_COLORS[k][0] for k in zone_list], dtype=np.uint8)
    lut_g = np.array([TREWARTHA_CLIMATE_COLORS[k][1] for k in zone_list], dtype=np.uint8)
    lut_b = np.array([TREWARTHA_CLIMATE_COLORS[k][2] for k in zone_list], dtype=np.uint8)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[lm, 0] = lut_r[z[lm]]
    rgba[lm, 1] = lut_g[z[lm]]
    rgba[lm, 2] = lut_b[z[lm]]
    rgba[lm, 3] = alpha

    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()
