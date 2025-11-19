#!/usr/bin/env python3
"""Visualise netcoevolve simulation output (new schema).

Expected CSV header (after parameter comment line):
    time,col0,col1,e00,e01,e11,3cyc000,3cyc001,3cyc011,3cyc111
Triangle columns are (#type)/C(N,3); their sum is total triangle density (1 only for a complete graph).

Features:
    * Auto-pick newest simulation-*.csv if path omitted (searches CWD, then output/).
  * Reconstruct overall edge density exactly (using N from parameter line) and approx (ignoring 1/N terms).
  * Plot partition densities e00,e01,e11 (optional) and colour1 fraction.
  * Optionally plot triangle composition counts or normalised densities.
"""
from __future__ import annotations
import argparse, sys
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def darken_color(color: str, factor: float = 0.4):
    """Return a darker shade of the given color. factor in (0,1]; lower is darker."""
    try:
        r, g, b = mcolors.to_rgb(color)
        return (max(0.0, factor * r), max(0.0, factor * g), max(0.0, factor * b))
    except ValueError:
        # Fallback: return original if parsing fails
        return color


def latest_csv() -> Path | None:
    """Find newest simulation CSV in current directory or output/ subdir.

    Preference is simply newest lexicographically (timestamps in filename) across both.
    """
    cwd = Path.cwd()
    candidates = list(cwd.glob("simulation-*.csv"))
    out_dir = cwd / "output"
    if out_dir.is_dir():
        candidates.extend(out_dir.glob("simulation-*.csv"))
    if not candidates:
        return None
    # Filenames are simulation-YYYYMMDD-HHMMSS.csv so lexicographic sort works.
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def parse_param_line(path: Path) -> dict:
    try:
        with path.open('r') as f:
            first = f.readline().strip()
        if not first.startswith('#'):
            return {}
        params = {}
        for kv in first[1:].split():
            if '=' in kv:
                k, v = kv.split('=',1)
                params[k] = v
        return params
    except Exception:
        return {}


def compute_overall_density(df: pd.DataFrame, n: int) -> tuple[pd.Series, pd.Series]:
    # Exact: (e00*C(c0,2) + e01*c0*c1 + e11*C(c1,2)) / C(n,2)
    f0 = df['col0']
    f1 = df['col1']
    c0 = f0 * n
    c1 = f1 * n
    choose2 = lambda x: x*(x-1)/2.0
    num_exact = df['e00']*choose2(c0) + df['e01']*c0*c1 + df['e11']*choose2(c1)
    den_exact = choose2(float(n))
    overall_exact = num_exact / den_exact
    # Approx (ignore 1/n terms): e00 f0^2 + 2 e01 f0 f1 + e11 f1^2
    overall_approx = df['e00']*f0*f0 + df['e01']*2*f0*f1 + df['e11']*f1*f1
    return overall_exact, overall_approx


def add_triangle_normalised(df: pd.DataFrame, n: int) -> dict[str,pd.Series]:
    f0 = df['col0']; f1 = df['col1']
    c0 = (f0 * n).round()
    c1 = (f1 * n).round()
    choose3 = lambda x: x*(x-1)*(x-2)/6.0
    res = {}
    with pd.option_context('mode.use_inf_as_na', True):
        res['tri000_density'] = df['3cyc000'] / choose3(c0).replace(0, pd.NA)
        res['tri111_density'] = df['3cyc111'] / choose3(c1).replace(0, pd.NA)
    # Mixed triangles normalisation: cycles with exactly one 1 => choose(c0,2)*c1 ; two 1s => choose(c1,2)*c0
    mix_denom_001 = (c1 * (c0*(c0-1)/2.0)).replace(0, pd.NA)
    mix_denom_011 = (c0 * (c1*(c1-1)/2.0)).replace(0, pd.NA)
    res['tri001_density'] = df['3cyc001'] / mix_denom_001
    res['tri011_density'] = df['3cyc011'] / mix_denom_011
    return res


def main():
    ap = argparse.ArgumentParser(description="Plot simulation CSV")
    ap.add_argument('csv', nargs='?', type=Path, help='CSV file (optional)')
    ap.add_argument('--out', type=Path, help='Output filename (PNG by default)')
    ap.add_argument('--split-panels', action='store_true', help='Also write each panel as an individual image (suffix -1,-2,...) when --out is used')
    ap.add_argument('--dpi', type=int, default=300, help='DPI for saved figures (default 300)')
    ap.add_argument('--ratio', type=str, help='Aspect ratio W:H for each panel (e.g. 4:3, 16:10)')
    ap.add_argument('--show', action='store_true', help='Also display an interactive window (default is save-only)')
    ap.add_argument('--no-partitions', action='store_true', help='Hide e00/e01/e11 lines')
    ap.add_argument('--linewidth', type=float, default=0.5, help='Global line width for plots (default 0.5)')
    ap.add_argument('--triangles', action='store_true', help='Show triangles panel (default off)')
    # Renamed: --twopaths -> --2paths (keep old hidden for backward compatibility)
    ap.add_argument('--2paths', dest='two_paths', action='store_true', help='Show 2-paths panel from 2p*** columns (renamed from --twopaths)')
    ap.add_argument('--twopaths', dest='two_paths', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--3paths', dest='three_paths', action='store_true', help='Show 3-paths panel from 3p**** columns')
    ap.add_argument('--3stars', dest='three_stars', action='store_true', help='Show 3-star panel from 3s*_*** columns')
    ap.add_argument('--all', action='store_true', help='Enable all subgraph panels (triangles, 2-paths, 3-paths, 3-stars)')
    ap.add_argument('--projections', action='store_true', help='Overlay composition-based projections (dotted) on triangles and two-paths panels')
    # Triangles panel optional (raw counts)
    args = ap.parse_args()

    # If --all is set, turn on all motif panels
    if args.all:
        args.triangles = True
        args.two_paths = True
        args.three_paths = True
        args.three_stars = True

    path = args.csv or latest_csv()
    if path is None:
        print('No simulation-*.csv files found.', file=sys.stderr)
        sys.exit(1)
    if not path.exists():
        # If bare filename provided and not found, try output/ subdir
        if path.parent == Path('.'):
            alt = Path('output') / path
            if alt.exists():
                path = alt
        if not path.exists():
            print(f'File not found: {path}', file=sys.stderr); sys.exit(1)
    params = parse_param_line(path)
    try:
        # Support both legacy uppercase 'N' and new lowercase 'n'
        n = int(params.get('n', params.get('N', '0')))
    except ValueError:
        n = 0
    df = pd.read_csv(path, comment='#')
    required = {'time','col0','col1','e00','e01','e11'}
    if not required.issubset(df.columns):
        print('Missing required columns for new schema.', file=sys.stderr)
        sys.exit(1)

    overall_exact = overall_approx = None
    if n > 0:
        overall_exact, overall_approx = compute_overall_density(df, n)
        df['overall_density_exact'] = overall_exact
        df['overall_density_approx'] = overall_approx

    # --- Multi-panel layout ---
    three_paths_flag = args.three_paths
    three_stars_flag = args.three_stars
    n_rows = 2 
    if args.triangles:
        n_rows += 1
    if args.two_paths:
        n_rows += 1
    if three_paths_flag:
        n_rows += 1
    if three_stars_flag:
        n_rows += 1
    # Aspect ratio handling: width:height per panel
    panel_width = 9.0
    if args.ratio:
        try:
            w_str, h_str = args.ratio.split(':', 1)
            w_ratio = float(w_str)
            h_ratio = float(h_str)
            if w_ratio <= 0 or h_ratio <= 0:
                raise ValueError
            panel_height = panel_width * (h_ratio / w_ratio)
        except ValueError:
            print(f"Invalid --ratio '{args.ratio}'. Expected W:H with positive numbers, e.g. 4:3", file=sys.stderr)
            return
    else:
        # Previous default total height ~ 3.2 * n_rows, so per-panel ~3.2
        panel_height = 3.2
    total_height = panel_height * n_rows
    fig, axes = plt.subplots(n_rows, 1, figsize=(panel_width, total_height), sharex=True)
    # Normalise axes -> list of Axes
    if isinstance(axes, np.ndarray):
        axes = axes.ravel().tolist()
    elif isinstance(axes, (list, tuple)):
        axes = list(axes)
    else:
        axes = [axes]

    tvals = df['time']

    # Panel 1: colour fractions
    row_idx = 0
    axc = axes[row_idx]
    row_idx += 1
    axc.plot(tvals, df['col0'], label='Colour 0', color='#44c774', linewidth=args.linewidth)
    axc.plot(tvals, df['col1'], label='Colour 1', color='#532784', linewidth=args.linewidth)
    # Title
    axc.set_title('Colour Densities')
    axc.set_ylabel('Fraction of Vertices')
    axc.set_ylim(0,1)
    axc.grid(alpha=0.3)
    axc.legend(loc='upper right', fontsize='x-small')

    # Reconstruct edge counts for concordant / discordant using N if available
    if n > 0:
        concord_frac = df['e00'] + df['e11']
        discord_frac = df['e01']
        total_frac = concord_frac + discord_frac
    else:
        # stop script if N missing
        print("Missing parameter n; cannot reconstruct edge densities.", file=sys.stderr)
        sys.exit(1)


    # Panel 2: edge densities aggregated
    axe = axes[row_idx]
    row_idx += 1
    axe.plot(tvals, total_frac, label='Total Edge Density', color='black', zorder=3, linewidth=args.linewidth)
    axe.plot(tvals, concord_frac, label='Concordant Edge Density (0–0, 1–1)', color='#ffbe83', linewidth=args.linewidth)
    axe.plot(tvals, discord_frac, label='Discordant Edge Density (0–1)', color='#df4d70', linewidth=args.linewidth)
    axe.plot(tvals, df['ne00'] + df['ne11'], label='Concordant Non-Edge Density (0 0, 1 1)', color='#ffbe83', linestyle='dashed', linewidth=args.linewidth)
    axe.plot(tvals, df['ne01'], label='Discordant Non-Edge Density (0 1)', color='#df4d70', linestyle='dashed', linewidth=args.linewidth)
    # axe.plot(tvals, df['col0']**2+df['col1']**2, label='Sum of Colour Squares', color='#888888', zorder=2, linewidth=0.5)
    axe.set_title('Edge Densities')
    axe.set_ylabel('Fraction of Edges')
    axe.set_ylim(0,1)
    axe.grid(alpha=0.3)
    axe.legend(loc='upper right', fontsize='x-small')

    # Optional panel 3: triangles
    if args.triangles:
        # Panel 3: triangle counts + sum
        axt = axes[row_idx]
        row_idx += 1
        tri_cols = {'3cyc000','3cyc001','3cyc011','3cyc111'}
        if tri_cols.issubset(df.columns):
            tri_sum = df['3cyc000'] + df['3cyc001'] + df['3cyc011'] + df['3cyc111']
            axt.plot(tvals, tri_sum, label='Total Triangle Density', color='black', zorder=3, linewidth=args.linewidth)
            axt.plot(tvals, df['3cyc000'], label='Triangle Density 000', color='#1b9e77', linewidth=args.linewidth)
            axt.plot(tvals, df['3cyc001'], label='Triangle Density 001', color='#d95f02', linewidth=args.linewidth)
            axt.plot(tvals, df['3cyc011'], label='Triangle Density 011', color='#7570b3', linewidth=args.linewidth)
            axt.plot(tvals, df['3cyc111'], label='Triangle Density 111', color='#e7298a', linewidth=args.linewidth)
            # Optional projections: composition fractions times total edge density
            if args.projections:
                f0 = df['col0']; f1 = df['col1']
                proj_000 = (f0**3) * total_frac**3
                proj_001 = 3*(f0**2 * f1) * total_frac**3
                proj_011 = 3*(f0 * f1**2) * total_frac**3
                proj_111 = (f1**3) * total_frac**3
                axt.plot(tvals, proj_000, color=darken_color('#1b9e77'), linestyle='dotted', linewidth=args.linewidth, label='Triangle 000 (proj)')
                axt.plot(tvals, proj_001, color=darken_color('#d95f02'), linestyle='dotted', linewidth=args.linewidth, label='Triangle 001 (proj)')
                axt.plot(tvals, proj_011, color=darken_color('#7570b3'), linestyle='dotted', linewidth=args.linewidth, label='Triangle 011 (proj)')
                axt.plot(tvals, proj_111, color=darken_color('#e7298a'), linestyle='dotted', linewidth=args.linewidth, label='Triangle 111 (proj)')
            axt.set_title('Triangle Densities')
            axt.set_ylabel('Fraction of Triangles')
            try:
                ymax = float(tri_sum.max())
            except Exception:
                ymax = 0.0
            # Scale axis: 0 to 1.1 * max(total triangle density); fallback to 1 if zero
            if not np.isfinite(ymax) or ymax <= 0.0:
                axt.set_ylim(0,1)
            else:
                axt.set_ylim(0, 1.1 * ymax)
        else:
            axt.text(0.5,0.5,'Triangle columns missing', ha='center', va='center')
        axt.grid(alpha=0.3)
        axt.legend(loc='upper right', fontsize='x-small', ncol=1)

    # Optional panel 4: two-path densities
    if args.two_paths:
        axp = axes[row_idx]
        row_idx += 1
        path_cols = ['2p000','2p001','2p010','2p011','2p101','2p111']
        if all(c in df.columns for c in path_cols):
            p_sum = df[path_cols].sum(axis=1)
            axp.plot(tvals, p_sum, label='Total Two-Path Density', color='black', zorder=3, linewidth=args.linewidth)
            # Use a distinct palette for 6 series
            colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
            labels = ['2-path 000','2-path 001','2-path 010','2-path 011','2-path 101','2-path 111']
            for col, col_color, lab in zip(path_cols, colors, labels):
                axp.plot(tvals, df[col], label=lab, color=col_color, linewidth=args.linewidth)
            # Optional projections: composition fractions times total edge density
            if args.projections:
                f0 = df['col0']; f1 = df['col1']
                total_frac = (concord_frac + discord_frac)
                proj_vals = [
                    (f0**3) * total_frac**2,           # 000
                    2*(f0**2 * f1) * total_frac**2,      # 001
                    (f0**2 * f1) * total_frac**2,      # 010 same composition as 001
                    2*(f0 * f1**2) * total_frac**2,      # 011
                    (f0 * f1**2) * total_frac**2,      # 101 same composition as 011
                    (f1**3) * total_frac**2,           # 111
                ]
                proj_labels = ['2-path 000 (proj)','2-path 001 (proj)','2-path 010 (proj)','2-path 011 (proj)','2-path 101 (proj)','2-path 111 (proj)']
                for proj, col_color, lab in zip(proj_vals, colors, proj_labels):
                    axp.plot(tvals, proj, color=darken_color(col_color), linestyle='dotted', linewidth=args.linewidth, label=lab)
            axp.set_title('2-Path Densities')
            axp.set_ylabel('Fraction of 2-Paths')
            try:
                ymax_p = float(p_sum.max())
            except Exception:
                ymax_p = 0.0
            if not np.isfinite(ymax_p) or ymax_p <= 0.0:
                axp.set_ylim(0,1)
            else:
                axp.set_ylim(0, 1.1 * ymax_p)
        else:
            axp.text(0.5,0.5,'Two-path columns missing', ha='center', va='center')
        axp.grid(alpha=0.3)
        axp.legend(loc='upper right', fontsize='x-small', ncol=1)

    # Optional 3-path panel
    if three_paths_flag:
        ax3p = axes[row_idx]
        row_idx += 1
        p3_cols = ['3p0000','3p0001','3p0010','3p0011','3p0101','3p0110','3p0111','3p1001','3p1011','3p1111']
        existing = [c for c in p3_cols if c in df.columns]
        if len(existing) >= 2:
            p3_sum = df[existing].sum(axis=1)
            ax3p.plot(tvals, p3_sum, label='Total 3-Path Density', color='black', linewidth=args.linewidth)
            palette = ['#084081','#0868ac','#2b8cbe','#4eb3d3','#7bccc4','#a8ddb5','#ccebc5','#e0f3db','#f7fcf0','#fdd49e']
            for col, col_color in zip(existing, palette):
                ax3p.plot(tvals, df[col], label=col, linewidth=args.linewidth, color=col_color)
            ax3p.set_title('3-Path Densities')
            ax3p.set_ylabel('Fraction of 3-Paths')
            try:
                ymax3 = float(p3_sum.max())
            except Exception:
                ymax3 = 0.0
            if not np.isfinite(ymax3) or ymax3 <= 0.0:
                ax3p.set_ylim(0,1)
            else:
                ax3p.set_ylim(0, 1.1 * ymax3)
            ax3p.grid(alpha=0.3)
            # Optional projections for 3-paths if requested
            if args.projections:
                f0 = df['col0']; f1 = df['col1']
                # For a 3-path (4 vertices, 3 edges) naive random projection: product of colour fractions for sequence * total_frac^3
                # Interpret bit pattern as colours along path: v0-v1-v2-v3
                for col in existing:
                    pattern = col.replace('3p','')
                    if len(pattern) != 4 or not all(c in '01' for c in pattern):
                        continue
                    colour_prob = 1.0
                    for ch in pattern:
                        colour_prob = colour_prob * (f0 if ch == '0' else f1)
                    # Symmetry factor: if pattern != reversed(pattern), multiply by 2 to account for both orientations in homomorphism density sum
                    sym_factor = 2.0 if pattern != pattern[::-1] else 1.0
                    proj_series = colour_prob * (total_frac**3) * sym_factor
                    ax3p.plot(tvals, proj_series, linestyle='dotted', linewidth=args.linewidth, color=darken_color('#555555',0.7), label=f'{col} (proj)')
            ax3p.legend(loc='upper right', fontsize='x-small')
        else:
            ax3p.text(0.5,0.5,'3-path columns missing', ha='center', va='center')

    # Optional 3-star panel
    if three_stars_flag:
        axs3 = axes[row_idx]
        row_idx += 1
        star0 = ['3s0_000','3s0_001','3s0_011','3s0_111']
        star1 = ['3s1_000','3s1_001','3s1_011','3s1_111']
        have0 = all(c in df.columns for c in star0)
        have1 = all(c in df.columns for c in star1)
        if have0 or have1:
            total0 = df[star0].sum(axis=1) if have0 else 0.0
            total1 = df[star1].sum(axis=1) if have1 else 0.0
            combined_total = total0 + total1
            axs3.plot(tvals, combined_total, label='Total 3-Star Density', color='black', linewidth=args.linewidth)
            pal0 = ['#4d9221','#a1d76a','#e6f5d0','#fde0ef']
            pal1 = ['#c51b7d','#de77ae','#f1b6da','#fde0ef']
            if have0:
                for col, col_color in zip(star0, pal0):
                    axs3.plot(tvals, df[col], label=col, linewidth=args.linewidth, color=col_color)
            if have1:
                for col, col_color in zip(star1, pal1):
                    axs3.plot(tvals, df[col], label=col, linewidth=args.linewidth, color=col_color, linestyle='dashed')
            axs3.set_title('3-Star Densities')
            axs3.set_ylabel('Fraction of 3-Stars')
            try:
                ymax_candidates = []
                if have0 and isinstance(total0, pd.Series):
                    ymax_candidates.append(float(total0.max()))
                if have1 and isinstance(total1, pd.Series):
                    ymax_candidates.append(float(total1.max()))
                if isinstance(combined_total, pd.Series):
                    ymax_candidates.append(float(combined_total.max()))
                ymaxs = max(ymax_candidates) if ymax_candidates else 0.0
            except Exception:
                ymaxs = 0.0
            if not np.isfinite(ymaxs) or ymaxs <= 0.0:
                axs3.set_ylim(0,1)
            else:
                axs3.set_ylim(0, 1.1 * ymaxs)
            axs3.grid(alpha=0.3)
            # Optional projections for 3-stars
            if args.projections:
                f0 = df['col0']; f1 = df['col1']
                # A 3-star has 4 vertices (center + 3 leaves), 3 edges.
                # Column names: 3s{center}_XYZ where XYZ are leaf colours.
                def leaf_patterns(center_prefix, patterns, center_colour_series):
                    for col in patterns:
                        patt = col.split('_',1)[1]
                        if len(patt) != 3:
                            continue
                        leaf_prob = 1.0
                        for ch in patt:
                            leaf_prob *= (f0 if ch == '0' else f1)
                        # Multinomial symmetry factor: counts of leaves by colour
                        zeros = patt.count('0')
                        ones = 3 - zeros
                        # Factor is 1 for all same colour, 3 for two-one split (3 permutations)
                        if zeros in (0,3):
                            mult = 1.0
                        else:
                            mult = 3.0
                        proj = center_colour_series * leaf_prob * (total_frac**3) * mult
                        yield col, proj
                # Center 0
                if have0:
                    for col, proj in leaf_patterns('3s0', star0, f0):
                        axs3.plot(tvals, proj, linestyle='dotted', linewidth=args.linewidth, color=darken_color('#4d9221',0.6), label=f'{col} (proj)')
                if have1:
                    for col, proj in leaf_patterns('3s1', star1, f1):
                        axs3.plot(tvals, proj, linestyle='dotted', linewidth=args.linewidth, color=darken_color('#c51b7d',0.6), label=f'{col} (proj)')
            axs3.legend(loc='upper right', fontsize='x-small', ncol=2 if (have0 and have1) else 1)
        else:
            axs3.text(0.5,0.5,'3-star columns missing', ha='center', va='center')

    axes[-1].set_xlabel('time')
    fig.tight_layout()
    # Determine output path: explicit --out or derive from input CSV path
    if args.out:
        out_path = args.out if args.out.suffix else args.out.with_suffix('.png')
    else:
        # Derive default: same directory as input file with '-plot' appended to stem
        default_name = f"{path.stem}-plot.png"
        out_path = path.parent / default_name
    fig.savefig(out_path, dpi=args.dpi)
    print(f'Wrote {out_path}')
    if args.split_panels:
        stem = out_path.stem
        ext = out_path.suffix
        parent = out_path.parent
        # Map panel index to axis
        panel_files = []
        for idx, ax in enumerate(axes, start=1):
            # Derive individual panel size; keep width 6 for panel export
            ind_width = 6.0
            if args.ratio:
                try:
                    w_str2, h_str2 = args.ratio.split(':', 1)
                    w_ratio_sp = float(w_str2)
                    h_ratio_sp = float(h_str2)
                    ind_height = ind_width * (h_ratio_sp / w_ratio_sp) if w_ratio_sp > 0 else 4.0
                except Exception:
                    ind_height = 4.0
            else:
                ind_height = 4.0
            sub_fig = plt.figure(figsize=(ind_width, ind_height))
            # Copy artists by re-plotting data
            for line in ax.get_lines():
                sub_fig_ax = plt.gca()
                sub_fig_ax.plot(
                    line.get_xdata(),
                    line.get_ydata(),
                    label=line.get_label(),
                    color=line.get_color(),
                    linewidth=args.linewidth,
                    linestyle=line.get_linestyle(),
                    zorder=line.get_zorder(),
                )
            sub_fig_ax = plt.gca()
            # Titles / labels from original
            sub_fig_ax.set_title(ax.get_title())
            sub_fig_ax.set_xlabel(ax.get_xlabel())
            sub_fig_ax.set_ylabel(ax.get_ylabel())
            sub_fig_ax.grid(alpha=0.3)
            if ax.get_legend() is not None:
                sub_fig_ax.legend(loc='upper right', fontsize='x-small')
            # Enforce [0,1] y-limits for the first two panels (colour fractions, edge densities)
            if idx in (1,2):
                sub_fig_ax.set_ylim(0,1)
            panel_path = parent / f"{stem}-{idx}{ext}"
            sub_fig.tight_layout()
            sub_fig.savefig(panel_path, dpi=args.dpi)
            plt.close(sub_fig)
            panel_files.append(panel_path)
        print("Wrote panel images:")
        for p in panel_files:
            print(f"  {p}")
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()

