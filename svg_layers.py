import math
from typing import Tuple
from xml.etree import ElementTree as ET
from constants import LAND_CLIMATE_COLORS, TREWARTHA_CLIMATE_COLORS, SST_STOPS, TROPIC_LAT, SUBTROPIC_MAX_LAT, POLAR_CIRCLE_LAT
from models import ExportSpec, ViewBox
from svg_utils import _qname, get_or_create_defs, rgb_to_hex
_CLIMATE_ZONE_LINES = [(0.0, 'Äquator', None, '4,0', '#ffffff', 0.55), (TROPIC_LAT, 'Wendekreis des Krebses', 'Wendekreis des Steinbocks', '6,5', '#ffe070', 0.6), (SUBTROPIC_MAX_LAT, 'Subtropen-Grenze', 'Subtropen-Grenze', '4,6', '#90e070', 0.45), (POLAR_CIRCLE_LAT, 'Nördlicher Polarkreis', 'Südlicher Polarkreis', '3,7', '#80c8ff', 0.55)]
_SST_LEGEND_LABELS: list[tuple[float, str]] = [(0.0, '28'), (11.25, '24'), (22.5, '20'), (33.75, '16'), (45.0, '12'), (56.25, '8'), (67.5, '4'), (78.75, '0'), (90.0, '−2')]
_LEGEND_FONT = 'serif'
_LEGEND_FILL = '#000000'
_LEGEND_ARROW_FILL = '#2d3748'
_LAND_LEGEND_ITEMS: list[tuple[str, str]] = [('eiswueste', 'Eiswüste'), ('tundra', 'Tundra'), ('nadelwald', 'Nadelwald'), ('mischwald', 'Mischwald'), ('gem_regenwald', 'Gem. Regenw.'), ('westseite', 'Westseite'), ('steppe', 'Steppe'), ('winterkalt_trocken', 'Winterk. Trock.'), ('heiss_trocken', 'Heiße Wüste'), ('dornsavanne', 'Dornsavanne'), ('ostseite', 'Ostseite'), ('trockensavanne', 'Trockensav.'), ('feuchtsavanne', 'Feuchtsav.'), ('trop_regenwald', 'Trop. Regenw.')]
_TREWARTHA_LEGEND_ITEMS: list[tuple[str, str]] = [('Ar', 'Ar  Regenwald'), ('Aw', 'Aw  Savanne'), ('BW', 'BW  Wüste'), ('BS', 'BS  Steppe'), ('Cf', 'Cf  Subtr. humid'), ('Cs', 'Cs  Mediterran'), ('Do', 'Do  Gem. ozeanisch'), ('Dc', 'Dc  Gem. kontinental'), ('E', 'E    Boreales Kl.'), ('FT', 'FT  Tundra'), ('Fi', 'Fi   Polareis')]

def append_streamlines_layer(root: ET.Element, lines_svg: list[list[tuple[float, float]]]) -> None:
    g = ET.SubElement(root, _qname('g'), {'id': 'WAZ_Stromlinien', 'fill': 'none', 'stroke': '#ffffff', 'stroke-opacity': '0.62', 'stroke-width': '1.15', 'stroke-linecap': 'round', 'stroke-linejoin': 'round'})
    for line in lines_svg:
        if len(line) < 2:
            continue
        d = 'M ' + ' L '.join((f'{x:.2f},{y:.2f}' for x, y in line))
        ET.SubElement(g, _qname('path'), {'d': d})

def append_currents_layer(root: ET.Element, arrows: list[Tuple[float, float, float, float, float]], marker_id: str) -> None:
    g = ET.SubElement(root, _qname('g'), {'id': 'WAZ_Meeresstroemungen', 'fill': '#ffffff', 'fill-opacity': '0.72', 'stroke': 'none'})
    for x1, y1, x2, y2, stroke_w in arrows:
        dx, dy = (x2 - x1, y2 - y1)
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-06:
            continue
        ux, uy = (dx / length, dy / length)
        px, py = (-uy, ux)
        head_len = min(stroke_w * 6.0, length * 0.55)
        head_half = head_len * 0.45
        shaft_half = stroke_w * 0.45
        bx = x2 - ux * head_len
        by = y2 - uy * head_len
        pts = [(x1 + px * shaft_half, y1 + py * shaft_half), (bx + px * shaft_half, by + py * shaft_half), (bx + px * head_half, by + py * head_half), (x2, y2), (bx - px * head_half, by - py * head_half), (bx - px * shaft_half, by - py * shaft_half), (x1 - px * shaft_half, y1 - py * shaft_half)]
        d = 'M ' + ' L '.join((f'{px_:.2f},{py_:.2f}' for px_, py_ in pts)) + ' Z'
        ET.SubElement(g, _qname('path'), {'d': d})

def append_climate_zone_labels(root: ET.Element, map_vb: ViewBox, labels: list[tuple[float, float, str]], min_dist: float=12.0) -> None:
    """Add zone name labels (e.g. Savanne, Wüste) onto the map with good positioning."""
    if not labels:
        return
    kept: list[tuple[float, float, str]] = []
    for x, y, text in labels:
        too_close = False
        for ox, oy, _ in kept:
            if (x - ox) ** 2 + (y - oy) ** 2 < min_dist ** 2:
                too_close = True
                break
        if not too_close:
            kept.append((x, y, text))
    g = ET.SubElement(root, _qname('g'), {'id': 'WAZ_Klimazonen_Labels', 'font-family': 'sans-serif', 'font-size': '9', 'font-weight': '600', 'fill': '#1a1a1a', 'fill-opacity': '0.92', 'text-anchor': 'middle', 'dominant-baseline': 'middle', 'stroke': '#ffffff', 'stroke-width': '1.5', 'stroke-opacity': '0.85', 'paint-order': 'stroke fill'})
    for x, y, text in kept:
        t = ET.SubElement(g, _qname('text'), {'x': f'{x:.2f}', 'y': f'{y:.2f}'})
        t.text = text

def append_climate_zone_lines(root: ET.Element, vb: ViewBox) -> None:
    g = ET.SubElement(root, _qname('g'), {'id': 'WAZ_Klimazonen', 'font-family': 'sans-serif'})

    def lat_to_y(lat: float) -> float:
        return vb.min_y + vb.height * (90.0 - lat) / 180.0
    label_x_left = vb.min_x + vb.width * 0.012
    label_x_right = vb.min_x + vb.width * 0.988
    for abs_lat, label_n, label_s, dash, color, opacity in _CLIMATE_ZONE_LINES:
        lats = [abs_lat] if abs_lat == 0.0 else [abs_lat, -abs_lat]
        for lat in lats:
            y = lat_to_y(lat)
            ET.SubElement(g, _qname('line'), {'x1': f'{vb.min_x:.2f}', 'y1': f'{y:.2f}', 'x2': f'{vb.min_x + vb.width:.2f}', 'y2': f'{y:.2f}', 'stroke': color, 'stroke-width': '1.1', 'stroke-dasharray': dash, 'stroke-opacity': str(opacity)})
            label = label_n if lat >= 0.0 else label_s or label_n
            if label:
                ET.SubElement(g, _qname('text'), {'x': f'{label_x_left:.2f}', 'y': f'{y - 2.5:.2f}', 'fill': color, 'font-size': '6.5', 'font-style': 'italic', 'opacity': str(opacity * 1.2), 'text-anchor': 'start'}).text = label

def append_title(root: ET.Element, map_vb: ViewBox, evb: ViewBox, title: str, subtitle: str | None=None, month_label: str | None=None) -> None:
    pad_h = map_vb.min_y - evb.min_y
    cx = evb.min_x + evb.width / 2.0
    fs = pad_h * 0.52
    g = ET.SubElement(root, _qname('g'), {'id': 'WAZ_Titel'})
    ET.SubElement(g, _qname('rect'), {'x': f'{evb.min_x:.3f}', 'y': f'{evb.min_y:.3f}', 'width': f'{evb.width:.3f}', 'height': f'{pad_h:.3f}', 'fill': '#000000', 'fill-opacity': '0.45'})
    cy = evb.min_y + pad_h * 0.45 if subtitle else evb.min_y + pad_h * 0.55
    ET.SubElement(g, _qname('text'), {'x': f'{cx:.3f}', 'y': f'{cy:.3f}', 'font-family': 'sans-serif', 'font-size': f'{fs:.3f}', 'font-weight': 'bold', 'fill': '#ffffff', 'fill-opacity': '0.95', 'text-anchor': 'middle', 'dominant-baseline': 'middle'}).text = title
    if subtitle:
        ET.SubElement(g, _qname('text'), {'x': f'{cx:.3f}', 'y': f'{evb.min_y + pad_h * 0.72:.3f}', 'font-family': 'sans-serif', 'font-size': f'{fs * 0.55:.3f}', 'fill': '#e8f4f8', 'fill-opacity': '0.9', 'text-anchor': 'middle', 'dominant-baseline': 'middle'}).text = subtitle
    if month_label:
        pad_x = evb.width * 0.012
        ET.SubElement(g, _qname('text'), {'x': f'{evb.min_x + evb.width - pad_x:.3f}', 'y': f'{evb.min_y + pad_h * 0.55:.3f}', 'font-family': 'sans-serif', 'font-size': f'{fs * 0.9:.3f}', 'font-weight': 'bold', 'fill': '#ffe680', 'fill-opacity': '0.97', 'text-anchor': 'end', 'dominant-baseline': 'middle'}).text = month_label

def _append_sst_legend_block(root: ET.Element, evb: ViewBox, defs: ET.Element, y: float, block_h: float) -> None:
    bar_w = evb.width * 0.65
    bar_x = evb.min_x + (evb.width - bar_w) / 2.0
    heading_fs = block_h * 0.2
    bar_h = block_h * 0.32
    label_fs = block_h * 0.18
    bar_top = y + heading_fs * 2.2
    grad_id = 'WAZ_SST_grad'
    g = ET.SubElement(root, _qname('g'), {'id': 'WAZ_Legende_SST'})
    ET.SubElement(g, _qname('text'), {'x': f'{bar_x + bar_w / 2:.3f}', 'y': f'{y + heading_fs:.3f}', 'font-family': _LEGEND_FONT, 'font-size': f'{heading_fs:.3f}', 'font-weight': 'normal', 'fill': _LEGEND_FILL, 'text-anchor': 'middle'}).text = 'Meeresoberflächentemperatur in °C'
    grad = ET.SubElement(defs, _qname('linearGradient'), {'id': grad_id, 'x1': '0', 'y1': '0', 'x2': '1', 'y2': '0', 'gradientUnits': 'objectBoundingBox'})
    for lat, c in SST_STOPS:
        ET.SubElement(grad, _qname('stop'), {'offset': f'{lat / 90.0 * 100.0:.1f}%', 'stop-color': rgb_to_hex(*c), 'stop-opacity': '1'})
    ET.SubElement(g, _qname('rect'), {'x': f'{bar_x:.3f}', 'y': f'{bar_top:.3f}', 'width': f'{bar_w:.3f}', 'height': f'{bar_h:.3f}', 'fill': f'url(#{grad_id})'})
    tick_y = bar_top + bar_h + label_fs * 1.8
    for lat, label in _SST_LEGEND_LABELS:
        tx = bar_x + lat / 90.0 * bar_w
        anchor = 'middle'
        if lat == 0.0:
            anchor = 'start'
        elif lat == 90.0:
            anchor = 'end'
        ET.SubElement(g, _qname('line'), {'x1': f'{tx:.3f}', 'y1': f'{bar_top + bar_h:.3f}', 'x2': f'{tx:.3f}', 'y2': f'{bar_top + bar_h + label_fs * 0.4:.3f}', 'stroke': _LEGEND_FILL, 'stroke-opacity': '0.8', 'stroke-width': '0.6'})
        ET.SubElement(g, _qname('text'), {'x': f'{tx:.3f}', 'y': f'{tick_y:.3f}', 'font-family': _LEGEND_FONT, 'font-size': f'{label_fs:.3f}', 'fill': _LEGEND_FILL, 'font-weight': 'normal', 'text-anchor': anchor}).text = label

def _append_land_legend_block(root: ET.Element, evb: ViewBox, y: float, block_h: float) -> None:
    n = len(_LAND_LEGEND_ITEMS)
    n_row1 = math.ceil(n / 2)
    heading_fs = block_h * 0.14
    swatch = block_h * 0.16
    label_fs = max(block_h * 0.12, 8.0)
    row_h = swatch + label_fs * 1.2
    col_w = evb.width * 0.9 / n_row1
    x0 = evb.min_x + evb.width * 0.05
    g = ET.SubElement(root, _qname('g'), {'id': 'WAZ_Legende_Land'})
    ET.SubElement(g, _qname('text'), {'x': f'{evb.min_x + evb.width / 2:.3f}', 'y': f'{y + heading_fs:.3f}', 'font-family': _LEGEND_FONT, 'font-size': f'{heading_fs:.3f}', 'font-weight': 'normal', 'fill': _LEGEND_FILL, 'text-anchor': 'middle'}).text = 'Klimazonen an Land'
    items_top = y + heading_fs * 1.8
    for i, (key, label) in enumerate(_LAND_LEGEND_ITEMS):
        row = i // n_row1
        col = i % n_row1
        bx = x0 + col * col_w + (col_w - swatch) / 2.0
        by = items_top + row * row_h
        rc, gc, bc = LAND_CLIMATE_COLORS[key]
        ET.SubElement(g, _qname('rect'), {'x': f'{bx:.3f}', 'y': f'{by:.3f}', 'width': f'{swatch:.3f}', 'height': f'{swatch:.3f}', 'fill': rgb_to_hex(rc, gc, bc), 'rx': f'{swatch * 0.12:.3f}', 'fill-opacity': '0.95', 'stroke': '#e0e0e0', 'stroke-width': '0.5'})
        ET.SubElement(g, _qname('text'), {'x': f'{bx + swatch / 2:.3f}', 'y': f'{by + swatch + label_fs * 1.0:.3f}', 'font-family': _LEGEND_FONT, 'font-size': f'{label_fs:.3f}', 'fill': _LEGEND_FILL, 'font-weight': 'normal', 'text-anchor': 'middle'}).text = label

def _append_currents_legend_block(root: ET.Element, evb: ViewBox, y: float, block_h: float) -> None:
    label_fs = block_h * 0.3
    arrow_len = evb.width * 0.06
    head_len = arrow_len * 0.42
    head_half = head_len * 0.45
    shaft_half = block_h * 0.07
    cx = evb.min_x + evb.width / 2.0
    ay = y + block_h * 0.38
    ax1 = cx - arrow_len / 2.0
    ax2 = cx + arrow_len / 2.0
    bx = ax2 - head_len
    pts = [(ax1, ay + shaft_half), (bx, ay + shaft_half), (bx, ay + head_half), (ax2, ay), (bx, ay - head_half), (bx, ay - shaft_half), (ax1, ay - shaft_half)]
    d = 'M ' + ' L '.join((f'{px:.2f},{py:.2f}' for px, py in pts)) + ' Z'
    g = ET.SubElement(root, _qname('g'), {'id': 'WAZ_Legende_Stroem'})
    ET.SubElement(g, _qname('path'), {'d': d, 'fill': _LEGEND_ARROW_FILL, 'fill-opacity': '0.9', 'stroke': 'none'})
    ET.SubElement(g, _qname('text'), {'x': f'{cx:.3f}', 'y': f'{ay + block_h * 0.33:.3f}', 'font-family': _LEGEND_FONT, 'font-size': f'{label_fs:.3f}', 'fill': _LEGEND_FILL, 'font-weight': 'normal', 'text-anchor': 'middle'}).text = 'Meeresströmung'

def _append_trewartha_legend_block(root: ET.Element, evb: ViewBox, y: float, block_h: float) -> None:
    """Legende für die Trewartha-Klassifikation (11 Zonen, 2 Reihen à ~6/5 Zonen)."""
    n = len(_TREWARTHA_LEGEND_ITEMS)
    n_row1 = math.ceil(n / 2)
    heading_fs = block_h * 0.14
    swatch = block_h * 0.16
    label_fs = max(block_h * 0.105, 7.5)
    row_h = swatch + label_fs * 1.3
    col_w = evb.width * 0.9 / n_row1
    x0 = evb.min_x + evb.width * 0.05
    g = ET.SubElement(root, _qname('g'), {'id': 'WAZ_Legende_Trewartha'})
    ET.SubElement(g, _qname('text'), {'x': f'{evb.min_x + evb.width / 2:.3f}', 'y': f'{y + heading_fs:.3f}', 'font-family': _LEGEND_FONT, 'font-size': f'{heading_fs:.3f}', 'font-weight': 'normal', 'fill': _LEGEND_FILL, 'text-anchor': 'middle'}).text = 'Klimazonen nach Trewartha'
    items_top = y + heading_fs * 1.8
    for i, (key, label) in enumerate(_TREWARTHA_LEGEND_ITEMS):
        row = i // n_row1
        col = i % n_row1
        bx = x0 + col * col_w + (col_w - swatch) / 2.0
        by = items_top + row * row_h
        rc, gc, bc = TREWARTHA_CLIMATE_COLORS[key]
        ET.SubElement(g, _qname('rect'), {'x': f'{bx:.3f}', 'y': f'{by:.3f}', 'width': f'{swatch:.3f}', 'height': f'{swatch:.3f}', 'fill': rgb_to_hex(rc, gc, bc), 'rx': f'{swatch * 0.12:.3f}', 'fill-opacity': '0.95', 'stroke': '#e0e0e0', 'stroke-width': '0.5'})
        ET.SubElement(g, _qname('text'), {'x': f'{bx + swatch / 2:.3f}', 'y': f'{by + swatch + label_fs * 1.0:.3f}', 'font-family': _LEGEND_FONT, 'font-size': f'{label_fs:.3f}', 'fill': _LEGEND_FILL, 'font-weight': 'normal', 'text-anchor': 'middle'}).text = label

def append_legend(root: ET.Element, map_vb: ViewBox, evb: ViewBox, spec: ExportSpec) -> None:
    defs = get_or_create_defs(root)
    legend_top = map_vb.min_y + map_vb.height
    legend_h = evb.min_y + evb.height - legend_top
    margin_v = legend_h * 0.1
    g_bg = ET.SubElement(root, _qname('g'), {'id': 'WAZ_Legende_BG'})
    ET.SubElement(g_bg, _qname('rect'), {'x': f'{evb.min_x:.3f}', 'y': f'{legend_top:.3f}', 'width': f'{evb.width:.3f}', 'height': f'{legend_h:.3f}', 'fill': '#ffffff', 'fill-opacity': '1', 'stroke': '#c8c8c8', 'stroke-width': '0.4'})
    n_active = sum([spec.currents, spec.sst, spec.land, spec.trewartha])
    if n_active == 0:
        return
    block_h = (legend_h - 2.0 * margin_v) / n_active
    y = legend_top + margin_v
    if spec.currents:
        _append_currents_legend_block(root, evb, y, block_h)
        y += block_h
    if spec.sst:
        _append_sst_legend_block(root, evb, defs, y, block_h)
        y += block_h
    if spec.land:
        _append_land_legend_block(root, evb, y, block_h)
        y += block_h
    if spec.trewartha:
        _append_trewartha_legend_block(root, evb, y, block_h)
