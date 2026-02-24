# Klimakarten-Generator

Erzeugt SVG-Karten mit Meeresströmungen, Meeresoberflächentemperatur und Klimazonen an Land. Basiert auf einer Eingabe-Weltkarte (SVG).

## Voraussetzungen

Python 3.10+, siehe requirements.txt.

```bash
pip install -r requirements.txt
```

## Nutzung

```bash
python main.py <eingabe.svg> <ausgabe-praefix>
```

Beispiel: `python main.py karte.svg out/karte` erzeugt u.a. `out/karte_weltklimakarte.svg`, `out/karte_ozeanstroemungen.svg`, `out/karte_meerestemperatur.svg`.

### Optionen

| Option | Beschreibung |
|--------|--------------|
| `--raster-w` | Rasterbreite in Pixeln (Standard: 1800) |
| `--spacing` | Pfeilabstand in Pixeln (Standard: 24) |
| `--min-dist` | Mindestabstand zur Küste für Pfeile |
| `--exports` | Nur bestimmte Exporte (kommagetrennt: stroem, sst, klima, ozean, atlas, alles) |
| `--split` | Karte ohne Legende, Legende als separate PNG |
| `--legend-w` | Breite der Legenden-PNG bei --split |
| `--sst-debug` | A/B/C für SST-Diagnose |

## Jahresanimation

```bash
python animate_year.py karte.svg out/anim --exports atlas,ozean --fps 4
```

Erzeugt 12 Monate als MP4 (benötigt ffmpeg).

## Export-Varianten

- **stroem** – Ozeanströmungen (Pfeile)
- **sst** – Meeresoberflächentemperatur
- **klima** – Klimazonen an Land (Köppen-vereinfacht)
- **ozean** – SST + Strömungen
- **atlas** – SST + Landklima
- **alles** – Vollständige Weltklimakarte
