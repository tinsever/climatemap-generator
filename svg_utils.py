import re
from typing import Optional, Tuple

from xml.etree import ElementTree as ET

from constants import OCEAN_GRAY
from models import ViewBox


SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


def _qname(tag: str) -> str:
    return f"{{{SVG_NS}}}{tag}"


def parse_float(value: str) -> float:
    value = value.strip()
    value = re.sub(r"[a-zA-Z]+$", "", value)
    return float(value)


def get_viewbox(root: ET.Element) -> ViewBox:
    vb = root.get("viewBox")
    if vb:
        parts = [p for p in vb.replace(",", " ").split() if p]
        if len(parts) == 4:
            return ViewBox(
                min_x=float(parts[0]),
                min_y=float(parts[1]),
                width=float(parts[2]),
                height=float(parts[3]),
            )

    width_attr = root.get("width")
    height_attr = root.get("height")
    if width_attr and height_attr:
        w = parse_float(width_attr)
        h = parse_float(height_attr)
        return ViewBox(min_x=0.0, min_y=0.0, width=w, height=h)

    raise ValueError(
        "SVG braucht viewBox oder width/height, damit Koordinaten stabil sind."
    )


def _parse_hex_color(value: str) -> Optional[Tuple[int, int, int]]:
    s = value.strip().lower()
    if s.startswith("#"):
        raw = s[1:]
        if len(raw) == 3 and all(c in "0123456789abcdef" for c in raw):
            r = int(raw[0] * 2, 16)
            g = int(raw[1] * 2, 16)
            b = int(raw[2] * 2, 16)
            return r, g, b
        if len(raw) == 6 and all(c in "0123456789abcdef" for c in raw):
            r = int(raw[0:2], 16)
            g = int(raw[2:4], 16)
            b = int(raw[4:6], 16)
            return r, g, b
    if s == "blue":
        return 0, 0, 255
    return None


def _parse_rgb_color(value: str) -> Optional[Tuple[int, int, int]]:
    s = value.strip().lower()
    m = re.match(
        r"^rgba?\(\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})(?:\s*,\s*[0-9\.]+\s*)?\)$",
        s,
    )
    if not m:
        return None
    r = max(0, min(255, int(m.group(1))))
    g = max(0, min(255, int(m.group(2))))
    b = max(0, min(255, int(m.group(3))))
    return r, g, b


def _is_oceanish_blue(value: str) -> bool:
    rgb = _parse_hex_color(value)
    if rgb is None:
        rgb = _parse_rgb_color(value)
    if rgb is None:
        return False
    r, g, b = rgb
    return (r < 90) and (g < 110) and (b > 150)


def _is_white_or_light(value: str) -> bool:
    s = value.strip().lower()
    if s in {"white", "#fff", "#ffffff"}:
        return True
    rgb = _parse_hex_color(s)
    if rgb is None:
        rgb = _parse_rgb_color(s)
    if rgb is None:
        return False
    r, g, b = rgb
    return r > 200 and g > 200 and b > 200


def _recolor_style_blue_to_gray(style_value: str, gray: str) -> str:
    parts = [p.strip() for p in style_value.split(";") if p.strip()]
    out: list[str] = []
    for part in parts:
        if ":" not in part:
            out.append(part)
            continue
        k, v = part.split(":", 1)
        key = k.strip().lower()
        val = v.strip()
        if key in {"fill", "stroke"} and _is_oceanish_blue(val):
            out.append(f"{k.strip()}:{gray}")
        else:
            out.append(f"{k.strip()}:{val}")
    return ";".join(out)


def recolor_ocean_background(root: ET.Element, gray: str = OCEAN_GRAY) -> None:
    for elem in root.iter():
        fill = elem.get("fill")
        if fill and _is_oceanish_blue(fill):
            elem.set("fill", gray)

        stroke = elem.get("stroke")
        if stroke and _is_oceanish_blue(stroke):
            elem.set("stroke", gray)

        style = elem.get("style")
        if style:
            elem.set("style", _recolor_style_blue_to_gray(style, gray=gray))


def recolor_land_fills_to_transparent(root: ET.Element) -> None:
    for elem in root.iter():
        if elem.tag == _qname("svg"):
            continue
        fill = elem.get("fill", "")
        if fill and _is_white_or_light(fill):
            elem.set("fill", "none")
        style = elem.get("style", "")
        if "fill" in style:
            parts = [p.strip() for p in style.split(";") if p.strip()]
            new_parts: list[str] = []
            for part in parts:
                if ":" in part:
                    k, v = part.split(":", 1)
                    if k.strip().lower() == "fill" and _is_white_or_light(v.strip()):
                        new_parts.append("fill:none")
                        continue
                new_parts.append(part)
            elem.set("style", ";".join(new_parts))


def get_or_create_defs(root: ET.Element) -> ET.Element:
    defs = root.find(_qname("defs"))
    if defs is None:
        defs = ET.Element(_qname("defs"))
        root.insert(0, defs)
    return defs


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _set_viewbox(root: ET.Element, vb: ViewBox) -> None:
    root.set("viewBox", f"{vb.min_x:.3f} {vb.min_y:.3f} {vb.width:.3f} {vb.height:.3f}")
    if root.get("width"):
        root.set("width", f"{vb.width:.3f}")
    if root.get("height"):
        root.set("height", f"{vb.height:.3f}")


def _insert_raster_layer(
    root: ET.Element, layer_id: str, vb: ViewBox, b64_png: str, insert_index: int
) -> int:
    img = ET.Element(
        _qname("image"),
        {
            "id": layer_id,
            "x": f"{vb.min_x:.2f}", "y": f"{vb.min_y:.2f}",
            "width": f"{vb.width:.2f}", "height": f"{vb.height:.2f}",
            "href": f"data:image/png;base64,{b64_png}",
            "preserveAspectRatio": "none",
        },
    )
    root.insert(insert_index, img)
    return insert_index + 1
