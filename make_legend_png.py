import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from constants import LAND_CLIMATE_COLORS, SST_STOPS
OUT_PATH = 'out/legende.png'
WIDTH = 1800
PADDING = 60
GAP = 48
HEADING_H = 52
BAR_H = 90
TICK_H = 14
LABEL_H = 36
SWATCH = 88
SWATCH_GAP = 16
SWATCH_LABEL_H = 30
BG_COLOR = (255, 255, 255)
FG_COLOR = (30, 30, 30)
LAND_ITEMS = [('eiswueste', 'Eiswüste'), ('tundra', 'Tundra'), ('nadelwald', 'Nadelwald'), ('mischwald', 'Mischwald'), ('gem_regenwald', 'Gem. Regenw.'), ('westseite', 'Westseite'), ('steppe', 'Steppe'), ('winterkalt_trocken', 'Winterk. Trock.'), ('heiss_trocken', 'Heiße Wüste'), ('dornsavanne', 'Dornsavanne'), ('ostseite', 'Ostseite'), ('trockensavanne', 'Trockensav.'), ('feuchtsavanne', 'Feuchtsav.'), ('trop_regenwald', 'Trop. Regenw.')]
SST_TEMP_LABELS = [(0.0, '28 °C'), (15.0, '22 °C'), (30.0, '16 °C'), (45.0, '10 °C'), (60.0, '4 °C'), (90.0, '−2 °C')]

def _font(size: int, bold: bool=False) -> ImageFont.FreeTypeFont:
    candidates_bold = ['/System/Library/Fonts/Helvetica.ttc', '/System/Library/Fonts/SFNSDisplay.ttf', '/Library/Fonts/Arial Bold.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf']
    candidates_regular = ['/System/Library/Fonts/Helvetica.ttc', '/System/Library/Fonts/SFNS.ttf', '/Library/Fonts/Arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    candidates = candidates_bold if bold else candidates_regular
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def _sst_color_at_lat(lat: float) -> tuple[int, int, int]:
    color = SST_STOPS[-1][1]
    for idx in range(len(SST_STOPS) - 1):
        lat_min, color_min = SST_STOPS[idx]
        lat_max, color_max = SST_STOPS[idx + 1]
        if lat_min <= lat <= lat_max:
            frac = (lat - lat_min) / (lat_max - lat_min)
            color = (int(color_min[0] + frac * (color_max[0] - color_min[0])), int(color_min[1] + frac * (color_max[1] - color_min[1])), int(color_min[2] + frac * (color_max[2] - color_min[2])))
            break
    return color

def main() -> None:
    n_swatches = len(LAND_ITEMS)
    n_cols = math.ceil(n_swatches / 2)
    swatch_rows = 2
    swatch_block_h = HEADING_H + swatch_rows * SWATCH + swatch_rows * (SWATCH_LABEL_H + SWATCH_GAP)
    sst_block_h = HEADING_H + BAR_H + TICK_H + LABEL_H
    height = PADDING + sst_block_h + GAP + swatch_block_h + PADDING
    image = Image.new('RGB', (WIDTH, height), BG_COLOR)
    draw = ImageDraw.Draw(image)
    font_head = _font(32, bold=True)
    font_label = _font(26)
    font_small = _font(22)
    bar_inner_w = WIDTH - 2 * PADDING
    y = PADDING
    draw.text((WIDTH // 2, y + HEADING_H // 2), 'Meeresoberflächentemperatur', font=font_head, fill=FG_COLOR, anchor='mm')
    y += HEADING_H
    for px in range(bar_inner_w):
        lat = px / max(1, bar_inner_w - 1) * 90.0
        x = PADDING + px
        draw.line([(x, y), (x, y + BAR_H)], fill=_sst_color_at_lat(lat))
    draw.rectangle([PADDING, y, PADDING + bar_inner_w, y + BAR_H], outline=FG_COLOR, width=2)
    y_tick_top = y + BAR_H
    for lat, label in SST_TEMP_LABELS:
        tick_x = PADDING + int(lat / 90.0 * bar_inner_w)
        draw.line([(tick_x, y_tick_top), (tick_x, y_tick_top + TICK_H)], fill=FG_COLOR, width=2)
        anchor = 'mm'
        if lat == 0.0:
            anchor = 'lm'
        if lat == 90.0:
            anchor = 'rm'
        draw.text((tick_x, y_tick_top + TICK_H + LABEL_H // 2), label, font=font_label, fill=FG_COLOR, anchor=anchor)
    y += BAR_H + TICK_H + LABEL_H + GAP
    draw.text((WIDTH // 2, y + HEADING_H // 2), 'Klimazonen an Land', font=font_head, fill=FG_COLOR, anchor='mm')
    y += HEADING_H
    total_swatch_w = n_cols * SWATCH + (n_cols - 1) * SWATCH_GAP
    swatch_x0 = (WIDTH - total_swatch_w) // 2
    for idx, (zone_key, zone_label) in enumerate(LAND_ITEMS):
        row = idx // n_cols
        col = idx % n_cols
        swatch_x = swatch_x0 + col * (SWATCH + SWATCH_GAP)
        swatch_y = y + row * (SWATCH + SWATCH_LABEL_H + SWATCH_GAP)
        draw.rectangle([swatch_x, swatch_y, swatch_x + SWATCH, swatch_y + SWATCH], fill=LAND_CLIMATE_COLORS[zone_key])
        draw.rectangle([swatch_x, swatch_y, swatch_x + SWATCH, swatch_y + SWATCH], outline=(160, 160, 160), width=1)
        label_y = swatch_y + SWATCH + 6
        for line in zone_label.split('\n'):
            draw.text((swatch_x + SWATCH // 2, label_y), line, font=font_small, fill=FG_COLOR, anchor='mt')
            label_y += 24
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT_PATH)
    print(f'Legende gespeichert: {OUT_PATH} ({WIDTH}×{height} px)')
if __name__ == '__main__':
    main()
