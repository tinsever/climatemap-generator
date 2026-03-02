from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import maximum_filter, minimum_filter
from land_climate_physics import simulate_monthly_climate
from ocean import rasterize_svg, ocean_mask_from_rgba
MONTH_NAMES = ['Januar', 'Februar', 'März', 'April', 'Mai', 'Juni', 'Juli', 'August', 'September', 'Oktober', 'November', 'Dezember']

def _find_extrema(p: np.ndarray, size: int=40, land: np.ndarray | None=None) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Find local H and L centres in the pressure field."""
    p_smooth = p.copy()
    loc_max = maximum_filter(p_smooth, size=size, mode=('nearest', 'wrap'))
    loc_min = minimum_filter(p_smooth, size=size, mode=('nearest', 'wrap'))
    h_mask = (p_smooth == loc_max) & (p_smooth > 1.5)
    l_mask = (p_smooth == loc_min) & (p_smooth < -1.5)
    highs = _thin_points(h_mask, size)
    lows = _thin_points(l_mask, size)
    return (highs, lows)

def _thin_points(mask: np.ndarray, min_dist: int) -> list[tuple[int, int]]:
    ys, xs = np.where(mask)
    kept: list[tuple[int, int]] = []
    for y, x in zip(ys, xs):
        too_close = False
        for ky, kx in kept:
            if abs(y - ky) < min_dist and abs(x - kx) < min_dist:
                too_close = True
                break
        if not too_close:
            kept.append((int(y), int(x)))
    return kept

def generate_wind_pressure_maps(svg_path: str='welt.svg', out_prefix: str='out/wind_druck', raster_w: int=900, months: list[int] | None=None) -> None:
    if months is None:
        months = [1, 4, 7, 10]
    raster_h = raster_w // 2
    print('Rasterisiere SVG …')
    rgba = rasterize_svg(svg_path, out_w=raster_w, out_h=raster_h)
    ocean = ocean_mask_from_rgba(rgba)
    print('Simuliere Klima (12 Monate) …')
    sim = simulate_monthly_climate(ocean=ocean, raster_h=raster_h, coast_window_px=135)
    land = sim['land']
    lat_2d = sim['lat_2d']
    u_mon = sim['u_month']
    v_mon = sim['v_month']
    p_mon = sim['p_month']
    h, w = land.shape
    lons = np.linspace(-180, 180, w)
    lats = np.linspace(90, -90, h)
    step = max(1, raster_w // 45)
    for mi in months:
        idx = mi - 1
        u = u_mon[idx]
        v = v_mon[idx]
        p = p_mon[idx]
        fig, ax = plt.subplots(figsize=(14, 7), dpi=130)
        land_rgb = np.where(land[:, :, np.newaxis], np.array([0.88, 0.9, 0.85]), np.array([0.72, 0.82, 0.9]))
        ax.imshow(land_rgb, extent=[-180, 180, -90, 90], origin='upper', aspect='auto')
        norm = TwoSlopeNorm(vmin=-8, vcenter=0, vmax=8)
        cs = ax.contourf(lons, lats, p, levels=np.linspace(-8, 8, 17), cmap='RdBu_r', norm=norm, alpha=0.35, extend='both')
        ct = ax.contour(lons, lats, p, levels=np.linspace(-8, 8, 17), colors='k', linewidths=0.3, alpha=0.4)
        ys = np.arange(0, h, step)
        xs = np.arange(0, w, step)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        lon_q = lons[xx]
        lat_q = lats[yy]
        u_q = u[yy, xx]
        v_q = -v[yy, xx]
        speed = np.sqrt(u_q ** 2 + v_q ** 2)
        ax.quiver(lon_q, lat_q, u_q, v_q, speed, cmap='coolwarm', scale=25, width=0.003, headwidth=4, headlength=4, alpha=0.8, zorder=3)
        highs, lows = _find_extrema(p, size=max(20, raster_w // 30), land=land)
        for y, x in highs:
            ax.text(lons[x], lats[y], 'H', fontsize=16, fontweight='bold', color='#b00000', ha='center', va='center', zorder=5, bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))
        for y, x in lows:
            ax.text(lons[x], lats[y], 'T', fontsize=16, fontweight='bold', color='#0000b0', ha='center', va='center', zorder=5, bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))
        for lat_line in [23.5, -23.5, 66.5, -66.5, 0]:
            ax.axhline(lat_line, color='gray', lw=0.4, ls='--', alpha=0.5)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel('Längengrad', fontsize=10)
        ax.set_ylabel('Breitengrad', fontsize=10)
        ax.set_title(f'Oberflächenwind & Druck – {MONTH_NAMES[idx]}', fontsize=14, fontweight='bold')
        cbar = fig.colorbar(cs, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Druckanomalie [hPa]', fontsize=9)
        out_path = f'{out_prefix}_{mi:02d}_{MONTH_NAMES[idx].lower()}.png'
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  {out_path}')
    print('Fertig.')
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Wind/Druck-Diagnosekarten')
    parser.add_argument('--svg', default='karte.svg')
    parser.add_argument('--out', default='out/karte_wind_druck')
    parser.add_argument('--months', default='1,4,7,10', help='Kommagetrennte Monate (Standard: 1,4,7,10)')
    args = parser.parse_args()
    months = [int(m) for m in args.months.split(',')]
    generate_wind_pressure_maps(svg_path=args.svg, out_prefix=args.out, months=months)
