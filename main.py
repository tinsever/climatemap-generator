import argparse
from typing import Optional

from models import ALL_EXPORTS
from pipeline import generate_exports


def main() -> None:
    key_help = ", ".join(f"{s.key} ({s.title})" for s in ALL_EXPORTS)
    parser = argparse.ArgumentParser(
        prog="klimakarte",
        description=(
            "Klimakarten-Generator – erstellt SVG-Exporte mit Meeresströmungen, "
            "Meeresoberflächentemperatur und Klimazonen an Land."
        ),
    )
    parser.add_argument("input",  help="Eingabe-SVG (Weltkarte)")
    parser.add_argument(
        "output",
        help=(
            "Ausgabe-Präfix. Beispiel: 'out/karte' erzeugt "
            "'out/karte_weltklimakarte.svg', 'out/karte_ozeanstroemungen.svg' usw."
        ),
    )
    parser.add_argument(
        "--raster-w", type=int, default=1800, metavar="PX",
        help="Rasterbreite in Pixeln (Standard: 1800)",
    )
    parser.add_argument(
        "--spacing", type=int, default=24, metavar="PX",
        help="Pfeilabstand in Raster-Pixeln (Standard: 24)",
    )
    parser.add_argument(
        "--min-dist", type=float, default=0.8, metavar="PX",
        help="Min. Abstand zur Küste für Pfeil-Startpunkte in Pixeln (Standard: 0.8)",
    )
    parser.add_argument(
        "--exports", default=None, metavar="KEYS",
        help=(
            "Kommagetrennte Auswahl der Exporte. "
            f"Mögliche Werte: {key_help}. "
            "Standard: alle."
        ),
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help=(
            "Karte als SVG ohne Legende/Titel, Legende separat als PNG. "
            "Erzeugt z.B. out/karte_weltklimakarte.svg und out/karte_weltklimakarte_legende.png"
        ),
    )
    parser.add_argument(
        "--legend-w", type=int, default=1200, metavar="PX",
        help="Breite der Legende-PNG in Pixeln (nur bei --split, Standard: 1200)",
    )
    parser.add_argument(
        "--sst-debug",
        choices=["A", "B", "C"],
        metavar="TEST",
        help=(
            "SST-Diagnose: A=Strömung aus, B=Küsten-Term aus, C=Enclosure-Term aus. "
            "Zum Finden von Kältefleck-Ursachen."
        ),
    )
    parser.add_argument(
        "--month", type=int, default=4, metavar="N",
        help=(
            "Monat für SST und Landklima (1–12). Standard: 4 (April – nahe am "
            "Jahresmittel der SST, da der Ozean dem solaren Antrieb um ~1–2 Monate "
            "nacheilt; das Äquinoktium selbst liegt im März). "
            "1 = Januar (Nordhemisphäre-Winter), 7 = Juli (Nordhemisphäre-Sommer)."
        ),
    )
    args = parser.parse_args()

    if args.exports:
        selected: Optional[set] = {k.strip() for k in args.exports.split(",") if k.strip()}
        valid = {s.key for s in ALL_EXPORTS}
        unknown = selected - valid
        if unknown:
            parser.error(
                f"Unbekannte Export-Schlüssel: {', '.join(sorted(unknown))}. "
                f"Gültig: {', '.join(sorted(valid))}"
            )
    else:
        selected = None

    generate_exports(
        svg_path=args.input,
        out_prefix=args.output,
        raster_w=args.raster_w,
        spacing_px=args.spacing,
        min_dist_px=args.min_dist,
        selected_keys=selected,
        split_legend=args.split,
        legend_png_width=args.legend_w,
        sst_debug=args.sst_debug,
        month=args.month,
    )


if __name__ == "__main__":
    main()
