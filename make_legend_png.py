"""Standalone-Legende als PNG – exakte Farben, keine Interpolation."""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from constants import SST_STOPS, LAND_CLIMATE_COLORS

OUT_PATH  = "out/legende.png"
W         = 1800
PADDING   = 60
GAP       = 48
HEADING_H = 52
BAR_H     = 90
TICK_H    = 14
LABEL_H   = 36
SWATCH    = 88
SWATCH_GAP= 16
SWATCH_LBL= 30
BG_COLOR  = (255, 255, 255)
FG_COLOR  = (30,  30,  30)

LAND_ITEMS = [
    ("tropical_wet",  "Regenwald"),
    ("tropical_dry",  "Savanne"),
    ("desert_hot",    "Wüste"),
    ("steppe",        "Steppe"),
    ("mediterranean", "Mediterran"),
    ("humid_sub",     "Feuchte\nSubtropen"),
    ("oceanic",       "Ozeanisch"),
    ("continental",   "Kontinental"),
    ("boreal",        "Taiga"),
    ("tundra",        "Tundra"),
    ("ice",           "Polareis"),
]

SST_TEMP_LABELS = [
    (0.0,  "28 °C"),
    (15.0, "22 °C"),
    (30.0, "16 °C"),
    (45.0, "10 °C"),
    (60.0, "4 °C"),
    (90.0, "−2 °C"),
]

def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates_bold = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    candidates_reg = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in (candidates_bold if bold else candidates_reg):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


n_swatches    = len(LAND_ITEMS)
n_cols        = math.ceil(n_swatches / 2)
swatch_row_w  = n_cols * (SWATCH + SWATCH_GAP) - SWATCH_GAP
swatch_rows   = 2

swatch_block_h = (
    HEADING_H
    + swatch_rows * SWATCH
    + swatch_rows * (SWATCH_LBL + SWATCH_GAP)
)
sst_block_h = HEADING_H + BAR_H + TICK_H + LABEL_H

H = PADDING + sst_block_h + GAP + swatch_block_h + PADDING

img  = Image.new("RGB", (W, H), BG_COLOR)
draw = ImageDraw.Draw(img)

font_head  = _font(32, bold=True)
font_label = _font(26)
font_small = _font(22)

bar_inner_w = W - 2 * PADDING

y = PADDING

draw.text(
    (W // 2, y + HEADING_H // 2),
    "Meeresoberflächentemperatur",
    font=font_head, fill=FG_COLOR, anchor="mm",
)
y += HEADING_H

for px in range(bar_inner_w):
    t = px / max(1, bar_inner_w - 1)
    lat = t * 90.0
    col = SST_STOPS[-1][1]
    for i in range(len(SST_STOPS) - 1):
        l0, c0 = SST_STOPS[i]
        l1, c1 = SST_STOPS[i + 1]
        if l0 <= lat <= l1:
            f = (lat - l0) / (l1 - l0)
            col = (
                int(c0[0] + f * (c1[0] - c0[0])),
                int(c0[1] + f * (c1[1] - c0[1])),
                int(c0[2] + f * (c1[2] - c0[2])),
            )
            break
    x = PADDING + px
    draw.line([(x, y), (x, y + BAR_H)], fill=col)

draw.rectangle([PADDING, y, PADDING + bar_inner_w, y + BAR_H],
               outline=FG_COLOR, width=2)

y_tick_top = y + BAR_H
for lat, lbl in SST_TEMP_LABELS:
    tx = PADDING + int(lat / 90.0 * bar_inner_w)
    draw.line([(tx, y_tick_top), (tx, y_tick_top + TICK_H)],
              fill=FG_COLOR, width=2)
    anchor = "mm"
    if lat == 0.0:   anchor = "lm"
    if lat == 90.0:  anchor = "rm"
    draw.text(
        (tx, y_tick_top + TICK_H + LABEL_H // 2),
        lbl, font=font_label, fill=FG_COLOR, anchor=anchor,
    )

y += BAR_H + TICK_H + LABEL_H

y += GAP

draw.text(
    (W // 2, y + HEADING_H // 2),
    "Klimazonen an Land",
    font=font_head, fill=FG_COLOR, anchor="mm",
)
y += HEADING_H

total_swatch_w = n_cols * SWATCH + (n_cols - 1) * SWATCH_GAP
x0_sw = (W - total_swatch_w) // 2

for i, (key, lbl) in enumerate(LAND_ITEMS):
    row = i // n_cols
    col = i % n_cols
    bx = x0_sw + col * (SWATCH + SWATCH_GAP)
    by = y + row * (SWATCH + SWATCH_LBL + SWATCH_GAP)
    color = LAND_CLIMATE_COLORS[key]

    draw.rectangle([bx, by, bx + SWATCH, by + SWATCH], fill=color)
    draw.rectangle([bx, by, bx + SWATCH, by + SWATCH], outline=(160, 160, 160), width=1)

    lines = lbl.split("\n")
    lbl_y = by + SWATCH + 6
    for line in lines:
        draw.text(
            (bx + SWATCH // 2, lbl_y),
            line, font=font_small, fill=FG_COLOR, anchor="mt",
        )
        lbl_y += 24

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
img.save(OUT_PATH)
print(f"Legende gespeichert: {OUT_PATH}  ({W}×{H} px)")
