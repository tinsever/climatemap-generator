TROPIC_LAT = 23.5
SUBTROPIC_MAX_LAT = 40.0
POLAR_CIRCLE_LAT = 66.5

COAST_REPEL_GAIN = 0.05
COAST_REPEL_SCALE = 9.0
COAST_STEER_DIST = 10.0

OCEAN_GRAY = "#676d74"

SST_STOPS: list = [
    ( 0.0, (210,  55,  20)),
    (15.0, (220, 130,  25)),
    (30.0, (200, 200,  40)),
    (45.0, ( 60, 180, 140)),
    (60.0, ( 50, 130, 210)),
    (90.0, ( 25,  60, 160)),
]

LAND_CLIMATE_COLORS: dict = {
    "tropical_wet":  (  5, 130,  45),
    "tropical_dry":  ( 90, 195,  60),
    "desert_hot":    (240, 200,  80),
    "steppe":        (215, 155,  50),
    "mediterranean": (255, 115,  30),
    "humid_sub":     (190, 230,  80),
    "oceanic":       ( 80, 185, 105),
    "continental":   (140,  80, 175),
    "boreal":        (175, 135, 210),
    "tundra":        (175, 195, 195),
    "ice":           (220, 235, 255),
}

TREWARTHA_CLIMATE_COLORS: dict = {
    "Ar": (140,  26,  20),
    "Aw": (218,  68,  18),
    "BW": (232, 210,  58),
    "BS": (218, 155,  38),
    "Cf": ( 92, 122,  30),
    "Cs": (142, 162,  42),
    "Do": (118, 198, 118),
    "Dc": (132, 178, 222),
    "E":  ( 78, 118, 196),
    "FT": (174, 180, 180),
    "Fi": (200, 232, 242),
}
