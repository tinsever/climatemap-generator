from dataclasses import dataclass


@dataclass(frozen=True)
class ViewBox:
    min_x: float
    min_y: float
    width: float
    height: float


@dataclass
class BasinStats:
    label: int
    area: int
    x_min: int
    x_max: int
    x_mean: float
    y_mean: float


@dataclass
class ExportSpec:
    key: str
    slug: str
    title: str
    sst: bool
    land: bool
    currents: bool


ALL_EXPORTS: list[ExportSpec] = [
    ExportSpec("stroem", "ozeanstroemungen", "Ozeanströmungen",             False, False, True),
    ExportSpec("sst",    "meerestemperatur", "Meeresoberflächentemperatur", True,  False, False),
    ExportSpec("klima",  "klimazonen",       "Klimazonen an Land",          False, True,  False),
    ExportSpec("ozean",  "ozeankarte",       "Ozeankarte",                  True,  False, True),
    ExportSpec("atlas",  "klimaatlas",       "Klimaatlas",                  True,  True,  False),
    ExportSpec("alles",  "weltklimakarte",   "Weltklimakarte",              True,  True,  True),
]
