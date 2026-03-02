import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import cairosvg
from models import ALL_EXPORTS
from pipeline import build_all_layers_for_month, write_export
MONATSNAMEN = ['Januar', 'Februar', 'März', 'April', 'Mai', 'Juni', 'Juli', 'August', 'September', 'Oktober', 'November', 'Dezember']

def main() -> None:
    key_help = ', '.join((f'{s.key} ({s.title})' for s in ALL_EXPORTS))
    parser = argparse.ArgumentParser(prog='animate_year', description='Erzeugt Jahresanimationen (12 Monate) als MP4 für ausgewählte Exporte.')
    parser.add_argument('input', help='Eingabe-SVG (Weltkarte)')
    parser.add_argument('output', help='Ausgabe-Präfix. Erzeugt z.B. out/anim_klimaatlas_jahresanimation.mp4')
    parser.add_argument('--exports', default='atlas,ozean', metavar='KEYS', help=f'Kommagetrennte Exporte. Mögliche Werte: {key_help}. Standard: atlas,ozean')
    parser.add_argument('--no-legend', action='store_true', help='Legende nicht in die Animations-Frames einbetten (Standard: Legende sichtbar).')
    parser.add_argument('--fps', type=int, default=2, metavar='N', help='Frames pro Sekunde im MP4 (Standard: 2)')
    parser.add_argument('--raster-w', type=int, default=1800, metavar='PX', help='Rasterbreite (Standard: 1800)')
    args = parser.parse_args()
    selected = {k.strip() for k in args.exports.split(',') if k.strip()}
    valid = {s.key for s in ALL_EXPORTS}
    unknown = selected - valid
    if unknown:
        parser.error(f"Unbekannte Export-Schlüssel: {', '.join(sorted(unknown))}. Gültig: {', '.join(sorted(valid))}")
    with_legend = not args.no_legend
    if not shutil.which('ffmpeg'):
        print('Hinweis: ffmpeg nicht gefunden. Nur PNG-Frames werden erzeugt, kein MP4.')
    out_prefix = args.output.rstrip('/')
    os.makedirs(os.path.dirname(out_prefix) or '.', exist_ok=True)
    for spec in ALL_EXPORTS:
        if spec.key not in selected:
            continue
        print(f'\nAnimation: {spec.title} …')
        with tempfile.TemporaryDirectory(prefix='klima_anim_') as tmpdir:
            frame_paths = []
            for month in range(1, 13):
                print(f'  Monat {month}/12 …')
                data = build_all_layers_for_month(args.input, month=month, raster_w=args.raster_w)
                svg_path = os.path.join(tmpdir, f'frame_{month:02d}.svg')
                write_export(data, spec, svg_path, with_title=True, with_legend=with_legend, currents_style='arrows', month_label=MONATSNAMEN[month - 1])
                png_path = os.path.join(tmpdir, f'frame_{month:02d}.png')
                with open(svg_path, 'rb') as f:
                    svg_bytes = f.read()
                png_bytes = cairosvg.svg2png(bytestring=svg_bytes)
                with open(png_path, 'wb') as f:
                    f.write(png_bytes)
                frame_paths.append(png_path)
            mp4_path = f'{out_prefix}_{spec.slug}_jahresanimation.mp4'
            if shutil.which('ffmpeg'):
                print(f'  Erstelle MP4: {mp4_path}')
                subprocess.run(['ffmpeg', '-y', '-framerate', str(args.fps), '-i', os.path.join(tmpdir, 'frame_%02d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', mp4_path], check=True, capture_output=True)
                print(f'  Fertig: {mp4_path}')
            else:
                base = Path(out_prefix).parent or Path('.')
                frame_out = base / f'{spec.slug}_jahresanimation_frames'
                frame_out.mkdir(parents=True, exist_ok=True)
                for i, src in enumerate(frame_paths, start=1):
                    shutil.copy(src, frame_out / f'frame_{i:02d}.png')
                print(f'  Frames gespeichert in: {frame_out}')
    print('\nFertig.')
if __name__ == '__main__':
    main()
