"""Landklima – vereinfachte Köppen-Geiger-Klassifikation für Landpixel."""

import io

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter1d, distance_transform_edt

from constants import LAND_CLIMATE_COLORS, TROPIC_LAT, POLAR_CIRCLE_LAT
from ocean import lat_from_y


def build_land_climate_image(
    ocean: np.ndarray,
    raster_h: int,
    alpha: int = 230,
    coast_window_px: int = 270,
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

    lat_col = np.abs(
        np.array([lat_from_y(float(y), raster_h) for y in range(h)], dtype=np.float32)
    )
    a_lat = np.broadcast_to(lat_col[:, np.newaxis], (h, w)).copy()

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

    z[lm & (a_lat < 10.0)] = TROPICAL_WET
    z[lm & (a_lat >= 10.0) & (a_lat < TROPIC_LAT)] = TROPICAL_DRY

    z[lm & (a_lat >= 14.0) & (a_lat < 45.0) & (continentality > 0.42)] = STEPPE

    z[lm & (a_lat >= 18.0) & (a_lat < 35.0)
      & ((west_ocean > 0.28) | (continentality > 0.52))] = DESERT_HOT

    z[lm & (a_lat >= 22.0) & (a_lat < 38.0)
      & (east_ocean > 0.30) & (continentality < 0.50)] = HUMID_SUB

    z[lm & (a_lat >= 30.0) & (a_lat < 44.0)
      & (west_ocean > 0.30) & (continentality < 0.36)] = MEDITERRANEAN

    z[lm & (a_lat >= 44.0) & (a_lat < 62.0)
      & (west_ocean > 0.30) & (continentality < 0.38)] = OCEANIC

    z[lm & (a_lat > 52.0) & (continentality > 0.40)] = BOREAL

    z[lm & (a_lat > POLAR_CIRCLE_LAT)] = TUNDRA
    z[lm & (a_lat > 75.0)] = ICE

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
