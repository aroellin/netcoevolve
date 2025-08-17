#!/usr/bin/env python3
"""Visualise netcoevolve simulation output (new schema).

Expected CSV header (after parameter comment line):
    time,frac0,frac1,e00,e01,e11,3cyc000,3cyc001,3cyc011,3cyc111
Triangle columns are (#type)/C(N,3); their sum is total triangle density (1 only for a complete graph).

Features:
    * Auto-pick newest simulation-*.csv if path omitted (searches CWD, then output/).
  * Reconstruct overall edge density exactly (using N from parameter line) and approx (ignoring 1/N terms).
  * Plot partition densities e00,e01,e11 (optional) and colour1 fraction.
  * Optionally plot triangle composition counts or normalised densities.
"""
from __future__ import annotations
import argparse, sys, re
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


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
    f0 = df['frac0']
    f1 = df['frac1']
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
    f0 = df['frac0']; f1 = df['frac1']
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
    ap.add_argument('--dpi', type=int, default=150, help='DPI for saved figures (default 150)')
    ap.add_argument('--ratio', type=str, help='Aspect ratio W:H for each panel (e.g. 4:3, 16:10)')
    ap.add_argument('--show', action='store_true', help='Show interactive window')
    ap.add_argument('--no-partitions', action='store_true', help='Hide e00/e01/e11 lines')
    # Triangles panel now always shown (raw counts)
    args = ap.parse_args()

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
    required = {'time','frac0','frac1','e00','e01','e11'}
    if not required.issubset(df.columns):
        print('Missing required columns for new schema.', file=sys.stderr)
        sys.exit(1)

    overall_exact = overall_approx = None
    if n > 0:
        overall_exact, overall_approx = compute_overall_density(df, n)
        df['overall_density_exact'] = overall_exact
        df['overall_density_approx'] = overall_approx

    # --- Multi-panel layout ---
    n_rows = 3
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
    axc = axes[0]
    axc.plot(tvals, df['frac0'], label='Colour 0', color='tab:blue')
    axc.plot(tvals, df['frac1'], label='Colour 1', color='tab:orange')
    # Title
    axc.set_title('Colour Densities')
    axc.set_ylabel('Fraction of Vertices')
    axc.set_ylim(0,1)
    axc.grid(alpha=0.3)
    axc.legend(loc='upper right', fontsize='small')

    # Reconstruct edge counts for concordant / discordant using N if available
    if n > 0:
        f0 = df['frac0']; f1 = df['frac1']
        c0 = f0 * n
        c1 = f1 * n
        m00 = df['e00'] * c0 * (c0 - 1) / 2.0
        m11 = df['e11'] * c1 * (c1 - 1) / 2.0
        m01 = df['e01'] * c0 * c1
        tot_pairs = n * (n - 1) / 2.0
        concord_frac = (m00 + m11) / tot_pairs
        discord_frac = m01 / tot_pairs
        total_frac = concord_frac + discord_frac
    else:
        # Fallback approximate using large-n assumption
        f0 = df['frac0']; f1 = df['frac1']
        concord_frac = df['e00'] * f0*f0 + df['e11'] * f1*f1
        discord_frac = df['e01'] * 2*f0*f1
        total_frac = concord_frac + discord_frac

    # Panel 2: edge densities aggregated
    axe = axes[1]
    axe.plot(tvals, total_frac, label='Total Edge Density', color='black', zorder=3)
    axe.plot(tvals, concord_frac, label='Concordant Edge Density (0–0, 1–1)', color='green')
    axe.plot(tvals, discord_frac, label='Discordant edge density (0–1)', color='red')
    axe.set_title('Edge Densities')
    axe.set_ylabel('Fraction of Edges')
    axe.set_ylim(0,1)
    axe.grid(alpha=0.3)
    axe.legend(loc='upper right', fontsize='small')

    # Optional panel 3: triangles
    # Panel 3: triangle counts + sum
    axt = axes[2]
    tri_cols = {'3cyc000','3cyc001','3cyc011','3cyc111'}
    if tri_cols.issubset(df.columns):
        tri_sum = df['3cyc000'] + df['3cyc001'] + df['3cyc011'] + df['3cyc111']
        axt.plot(tvals, tri_sum, label='Total Triangle Density', color='black', zorder=3)
        axt.plot(tvals, df['3cyc000'], label='Triangle Density 000', color='#1b9e77')
        axt.plot(tvals, df['3cyc001'], label='Triangle Density 001', color='#d95f02')
        axt.plot(tvals, df['3cyc011'], label='Triangle Density 011', color='#7570b3')
        axt.plot(tvals, df['3cyc111'], label='Triangle Density 111', color='#e7298a')
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
    axt.legend(loc='upper right', fontsize='small', ncol=1)

    axes[-1].set_xlabel('time')
    fig.tight_layout()
    if args.out:
        out_path = args.out if args.out.suffix else args.out.with_suffix('.png')
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
                    ind_height = ind_width * (h_ratio / w_ratio)
                else:
                    ind_height = 4.0
                sub_fig = plt.figure(figsize=(ind_width, ind_height))
                # Copy artists by re-plotting data
                for line in ax.get_lines():
                    sub_fig_ax = plt.gca()
                    sub_fig_ax.plot(line.get_xdata(), line.get_ydata(),
                                    label=line.get_label(),
                                    color=line.get_color(),
                                    linewidth=line.get_linewidth())
                sub_fig_ax = plt.gca()
                # Titles / labels from original
                sub_fig_ax.set_title(ax.get_title())
                sub_fig_ax.set_xlabel(ax.get_xlabel())
                sub_fig_ax.set_ylabel(ax.get_ylabel())
                sub_fig_ax.grid(alpha=0.3)
                if ax.get_legend() is not None:
                    sub_fig_ax.legend(loc='upper right', fontsize='small')
                panel_path = parent / f"{stem}-{idx}{ext}"
                sub_fig.tight_layout()
                sub_fig.savefig(panel_path, dpi=args.dpi)
                plt.close(sub_fig)
                panel_files.append(panel_path)
            print("Wrote panel images:")
            for p in panel_files:
                print(f"  {p}")
    if args.show or not args.out:
        plt.show()


if __name__ == '__main__':
    main()
