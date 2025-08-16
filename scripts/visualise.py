#!/usr/bin/env python3
"""Visualise netcoevolve simulation output (new schema).

Expected CSV header (after parameter comment line):
    time,frac0,frac1,e00,e01,e11,3cyc000,3cyc001,3cyc011,3cyc111
Triangle columns are (#type)/C(N,3); their sum is total triangle density (1 only for a complete graph).

Features:
  * Auto-pick newest simulation-*.csv if path omitted.
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
    files = sorted(Path.cwd().glob("simulation-*.csv"))
    return files[-1] if files else None


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
    ap.add_argument('--out', type=Path, help='Output PNG path')
    ap.add_argument('--show', action='store_true', help='Show interactive window')
    ap.add_argument('--no-partitions', action='store_true', help='Hide e00/e01/e11 lines')
    # Triangles panel now always shown (raw counts)
    args = ap.parse_args()

    path = args.csv or latest_csv()
    if path is None:
        print('No simulation-*.csv files found.', file=sys.stderr)
        sys.exit(1)
    if not path.exists():
        print(f'File not found: {path}', file=sys.stderr); sys.exit(1)
    params = parse_param_line(path)
    try:
        n = int(params.get('N','0'))
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
    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 3.2*n_rows), sharex=True)
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
    axc.plot(tvals, df['frac0'], label='frac0', color='tab:blue')
    axc.plot(tvals, df['frac1'], label='frac1', color='tab:orange')
    axc.set_ylabel('colour fraction')
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
    axe.plot(tvals, concord_frac, label='concordant edges', color='green')
    axe.plot(tvals, discord_frac, label='discordant edges', color='red')
    axe.plot(tvals, total_frac, label='total edges', color='black')
    axe.set_ylabel('edge density')
    axe.set_ylim(0,1)
    axe.grid(alpha=0.3)
    axe.legend(loc='upper right', fontsize='small')

    # Optional panel 3: triangles
    # Panel 3: triangle counts + sum
    axt = axes[2]
    tri_cols = {'3cyc000','3cyc001','3cyc011','3cyc111'}
    if tri_cols.issubset(df.columns):
        axt.plot(tvals, df['3cyc000'], label='000/C(N,3)', color='#1b9e77')
        axt.plot(tvals, df['3cyc001'], label='001/C(N,3)', color='#d95f02')
        axt.plot(tvals, df['3cyc011'], label='011/C(N,3)', color='#7570b3')
        axt.plot(tvals, df['3cyc111'], label='111/C(N,3)', color='#e7298a')
        tri_sum = df['3cyc000'] + df['3cyc001'] + df['3cyc011'] + df['3cyc111']
        axt.plot(tvals, tri_sum, label='total triangle density', color='black', linewidth=1.2)
        axt.set_ylabel('tri densities (per C(N,3))')
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
    axt.legend(loc='upper right', fontsize='small', ncol=3)

    axes[-1].set_xlabel('time')
    fig.tight_layout()
    if args.out:
        out_path = args.out if args.out.suffix else args.out.with_suffix('.png')
        fig.savefig(out_path, dpi=150)
        print(f'Wrote {out_path}')
    if args.show or not args.out:
        plt.show()


if __name__ == '__main__':
    main()
