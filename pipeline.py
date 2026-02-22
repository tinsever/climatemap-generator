import base64
import copy
from typing import Optional

import cairosvg
import numpy as np
from scipy import ndimage
from xml.etree import ElementTree as ET

from constants import OCEAN_GRAY
from models import ExportSpec, ViewBox
from ocean import (
    rasterize_svg,
    ocean_mask_from_rgba,
    label_basins,
    compute_basin_stats,
    build_streamfunction_currents,
)
from currents import (
    build_arrows,
    build_streamlines,
    build_major_driftlines,
    split_at_dateline,
)
from sst import build_sst_image
from land_climate import build_land_climate_image
from svg_utils import (
    get_viewbox,
    recolor_ocean_background,
    recolor_land_fills_to_transparent,
    get_or_create_defs,
    _set_viewbox,
    _insert_raster_layer,
)
from svg_layers import (
    append_currents_layer,
    append_streamlines_layer,
    append_climate_zone_lines,
    append_title,
    append_legend,
)


def build_all_layers(
    svg_path: str,
    raster_w: int = 1800,
    spacing_px: int = 24,
    min_dist_px: float = 0.8,
    sst_debug: Optional[str] = None,
) -> dict:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    vb = get_viewbox(root)

    recolor_ocean_background(root, gray=OCEAN_GRAY)
    recolor_land_fills_to_transparent(root)

    raster_h = int(round(raster_w * (vb.height / vb.width)))
    rgba = rasterize_svg(svg_path, out_w=raster_w, out_h=raster_h)

    ocean = ocean_mask_from_rgba(rgba)
    labels, n_labels = label_basins(ocean)
    basins = compute_basin_stats(labels, n_labels, min_area=800)
    dist = ndimage.distance_transform_edt(ocean)
    grad_y, grad_x = np.gradient(dist)

    print("  Berechne Stromfunktion …")
    stream_u, stream_v, stream_v_sst = build_streamfunction_currents(
        labels=labels, basins=basins, h_px=raster_h, coarsen=4,
    )

    print("  Berechne Strömungspfeile …")
    arrows_raster = build_arrows(
        ocean=ocean,
        labels=labels,
        dist=dist,
        grad_x=grad_x,
        grad_y=grad_y,
        basins=basins,
        stream_u=stream_u,
        stream_v=stream_v,
        raster_w=raster_w,
        raster_h=raster_h,
        spacing_px=spacing_px,
        min_dist_px=min_dist_px,
    )

    print("  Trace Stromlinien …")
    lines_raster = build_streamlines(
        ocean=ocean,
        dist=dist,
        u=stream_u,
        v=stream_v,
        seed_spacing_px=55,
        min_dist_px=min_dist_px,
    )

    print("  Trace Major-Driftlinien (Golfstrom → Europa) …")
    major_lines_r = build_major_driftlines(
        ocean=ocean,
        labels=labels,
        basins=basins,
        dist=dist,
        stream_u=stream_u,
        stream_v=stream_v,
        h_px=raster_h,
        min_dist_px=min_dist_px,
    )

    sx = vb.width / float(raster_w)
    sy = vb.height / float(raster_h)

    arrows_svg: list[tuple[float, float, float, float, float]] = []
    for x1, y1, x2, y2, stroke_w in arrows_raster:
        arrows_svg.append(
            (
                vb.min_x + x1 * sx,
                vb.min_y + y1 * sy,
                vb.min_x + x2 * sx,
                vb.min_y + y2 * sy,
                stroke_w,
            )
        )

    lines_svg: list[list[tuple[float, float]]] = []
    for line in lines_raster:
        for seg in split_at_dateline(line, w=raster_w):
            lines_svg.append(
                [(vb.min_x + x * sx, vb.min_y + y * sy) for x, y in seg]
            )
    for line in major_lines_r:
        for seg in split_at_dateline(line, w=raster_w):
            lines_svg.append(
                [(vb.min_x + x * sx, vb.min_y + y * sy) for x, y in seg]
            )

    print("  Berechne Meeresoberflächentemperatur …")
    sst_png = build_sst_image(
        ocean, raster_h, stream_v=stream_v_sst, debug_mode=sst_debug
    )
    sst_b64 = base64.b64encode(sst_png).decode("ascii")

    print("  Berechne Klimazonen …")
    land_png = build_land_climate_image(ocean, raster_h)
    land_b64 = base64.b64encode(land_png).decode("ascii")

    return {
        "base_root": copy.deepcopy(root),
        "vb": vb,
        "ocean": ocean,
        "raster_h": raster_h,
        "raster_w": raster_w,
        "stream_u": stream_u,
        "stream_v": stream_v,
        "arrows_svg": arrows_svg,
        "lines_svg": lines_svg,
        "sst_b64": sst_b64,
        "land_b64": land_b64,
    }


def write_export(
    data: dict,
    spec: ExportSpec,
    out_path: str,
    with_title: bool = True,
    with_legend: bool = True,
    currents_style: str = "arrows",
) -> None:
    root = copy.deepcopy(data["base_root"])
    map_vb: ViewBox = data["vb"]

    if with_title or with_legend:
        title_h  = map_vb.height * 0.065
        legend_h = map_vb.height * 0.22
        evb = ViewBox(
            min_x=map_vb.min_x,
            min_y=map_vb.min_y - (title_h if with_title else 0),
            width=map_vb.width,
            height=map_vb.height
            + (title_h if with_title else 0)
            + (legend_h if with_legend else 0),
        )
    else:
        evb = map_vb

    _set_viewbox(root, evb)

    idx = 1
    if spec.sst:
        idx = _insert_raster_layer(root, "WAZ_SST", map_vb, data["sst_b64"], idx)
    if spec.land:
        idx = _insert_raster_layer(root, "WAZ_Landklima", map_vb, data["land_b64"], idx)

    append_climate_zone_lines(root, map_vb)

    if spec.currents:
        if currents_style == "lines":
            append_streamlines_layer(root, data["lines_svg"])
        else:
            append_currents_layer(
                root,
                arrows=data["arrows_svg"],
                marker_id="waz_arrowhead",
            )

    if with_title:
        append_title(root, map_vb, evb, spec.title)
    if with_legend:
        append_legend(root, map_vb, evb, spec)

    ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)


def write_legend_png(
    data: dict, spec: ExportSpec, out_path: str, width_px: int = 1200
) -> None:
    map_vb = data["vb"]
    content_w = map_vb.width
    content_h = map_vb.height * 0.22
    pad_x = content_w * 0.12
    pad_y = content_h * 0.18
    total_w = content_w + 2 * pad_x
    total_h = content_h + 2 * pad_y

    if sum([spec.currents, spec.sst, spec.land]) == 0:
        return

    map_vb_legend = ViewBox(0, -1, total_w, 1 + pad_y)
    evb_legend = ViewBox(0, 0, total_w, total_h)

    from svg_utils import _qname

    root = ET.Element(
        _qname("svg"),
        {
            "viewBox": f"0 0 {total_w:.1f} {total_h:.1f}",
            "width": f"{total_w:.0f}",
            "height": f"{total_h:.0f}",
        },
    )
    append_legend(root, map_vb_legend, evb_legend, spec)

    height_px = max(100, int(width_px * total_h / total_w))
    svg_bytes = ET.tostring(root, encoding="unicode").encode("utf-8")
    png_bytes = cairosvg.svg2png(
        bytestring=svg_bytes,
        output_width=width_px,
        output_height=height_px,
    )
    with open(out_path, "wb") as f:
        f.write(png_bytes)


def generate_exports(
    svg_path: str,
    out_prefix: str,
    raster_w: int = 1800,
    spacing_px: int = 24,
    min_dist_px: float = 0.8,
    selected_keys: Optional[set] = None,
    split_legend: bool = False,
    legend_png_width: int = 1200,
    sst_debug: Optional[str] = None,
) -> None:
    from models import ALL_EXPORTS

    if selected_keys is None:
        selected_keys = {s.key for s in ALL_EXPORTS}

    print("Berechne Kartendaten …")
    data = build_all_layers(
        svg_path,
        raster_w=raster_w,
        spacing_px=spacing_px,
        min_dist_px=min_dist_px,
        sst_debug=sst_debug,
    )

    for spec in ALL_EXPORTS:
        if spec.key not in selected_keys:
            continue
        out_base = f"{out_prefix}_{spec.slug}"
        if split_legend:
            svg_path_out = f"{out_base}.svg"
            print(f"  Exportiere Karte: {spec.title}  →  {svg_path_out}")
            write_export(
                data,
                spec,
                svg_path_out,
                with_title=False,
                with_legend=False,
                currents_style="arrows",
            )
            if spec.currents:
                svg_path_lines = f"{out_base}_linien.svg"
                print(f"  Exportiere Karte (Linien): {spec.title}  →  {svg_path_lines}")
                write_export(
                    data,
                    spec,
                    svg_path_lines,
                    with_title=False,
                    with_legend=False,
                    currents_style="lines",
                )
            png_path = f"{out_base}_legende.png"
            if sum([spec.currents, spec.sst, spec.land]) > 0:
                print(f"  Exportiere Legende:  →  {png_path}")
                write_legend_png(data, spec, png_path, width_px=legend_png_width)
        else:
            out_path = f"{out_base}.svg"
            print(f"  Exportiere: {spec.title}  →  {out_path}")
            write_export(
                data,
                spec,
                out_path,
                with_legend=False,
                currents_style="arrows",
            )

            if spec.currents:
                out_path_lines = f"{out_base}_linien.svg"
                print(f"  Exportiere (Linien): {spec.title}  →  {out_path_lines}")
                write_export(
                    data,
                    spec,
                    out_path_lines,
                    with_legend=False,
                    currents_style="lines",
                )

        if spec.slug == "ozeankarte":
            png_path = f"{out_base}.png"
            print(f"  Exportiere PNG:  →  {png_path}")
            with open(f"{out_base}.svg", "rb") as f:
                svg_bytes = f.read()
            png_bytes = cairosvg.svg2png(bytestring=svg_bytes)
            with open(png_path, "wb") as f:
                f.write(png_bytes)
            if spec.currents:
                png_path_lines = f"{out_base}_linien.png"
                print(f"  Exportiere PNG (Linien):  →  {png_path_lines}")
                with open(f"{out_base}_linien.svg", "rb") as f:
                    svg_bytes = f.read()
                png_bytes = cairosvg.svg2png(bytestring=svg_bytes)
                with open(png_path_lines, "wb") as f:
                    f.write(png_bytes)

    print("Fertig.")
