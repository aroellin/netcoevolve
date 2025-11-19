#!/usr/bin/env python3
"""Unified analysis & visualization CLI for netcoevolve adjacency snapshots.

Subcommands:
  animate      Render (and optionally save) an adjacency matrix animation.
  diagnostics  Compute spectral / rank-1 metrics (optionally extended) without rendering.
  info         Summarize dataset metadata.
  (stubs) frames, permutation, verify - to be implemented.

Reordering modes (--order):
  none            Preserve on-disk order
  global-degree   Global total degree ascending (adds membership bar unless --membership-bar=off)
  degree          Per community total degree (group0 desc, group1 asc)
  in-degree       Per community internal degree (group0 desc, group1 asc)
  out-degree      Per community external degree (group0 desc, group1 asc)

Group direction override:
  --g0-dir asc|desc   (default desc for community modes)
  --g1-dir asc|desc   (default asc  for community modes)

Sampling selectors (diagnostics / future frames): choose one or combine logically:
  --sample K          Uniformly sample K frames (after other selection)
  --every N           Keep every Nth frame
  --frames LIST       Comma/range spec e.g. "0,5,10-20,100-200:10" (start-end:step)

Diagnostics tiers:
  --rank1             Global rank-1 closeness metrics (λ1..λ4, ratio, residual, tail, norm_second)
  --extended          Adds block eigen metrics, correlations, two-block baseline (implies --rank1)

Outputs:
  diagnostics: CSV (--csv) and/or JSON (--json)
  animate: MP4/GIF based on --out suffix.

NOTE: After feature parity this will supersede heatmap.py.
"""
from __future__ import annotations
import argparse, sys, json, re, time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class SnapshotSet:
    matrices: List[np.ndarray]
    times: List[float]
    group_sizes: List[int]
    colours: List[np.ndarray]  # NEW: per-frame colour bitvector in original vertex order


def parse_header(first_line: str) -> dict:
    if not first_line.startswith('#'):
        return {}
    parts = first_line[1:].strip().split()
    params = {}
    for kv in parts:
        if '=' in kv:
            k, v = kv.split('=', 1)
            params[k] = v
    return params


def load_snapshots(directory: Path, verbose: bool=True, progress: bool=True, *, lenient: bool=False, debug: bool=False) -> SnapshotSet:
    files = sorted(directory.glob('*.txt'))
    total_files = len(files)
    if verbose:
        print(f"Scanning directory: {directory} ({total_files} candidate .txt files)")
    mats: List[np.ndarray] = []
    times: List[float] = []
    gszs: List[int] = []
    colours: List[np.ndarray] = []
    if not files:
        return SnapshotSet(mats, times, gszs, colours)
    expected_n: Optional[int] = None
    last_len = 0

    def pb(cur: int):
        nonlocal last_len
        if not progress or not verbose:
            return
        width=40
        frac = cur/total_files if total_files else 1.0
        fill = int(round(width*frac))
        bar = '#' * fill + '-' * (width-fill)
        msg = f"[{bar}] {cur}/{total_files}\r"
        print(msg, end='', flush=True)
        last_len = len(msg)

    for idx,f in enumerate(files, start=1):
        try:
            with f.open() as fh:
                first = fh.readline().strip()
                if 'netcoevolve=' not in first or 'filetype=adjacency_matrix' not in first:
                    if verbose and not progress:
                        print(f"[skip] {f.name}: missing tokens")
                    continue
                params = parse_header(first)
                # NEW: optional second line with colours begins with '#'
                second_pos = fh.tell()
                second = fh.readline()
                colour_vec: Optional[np.ndarray] = None
                if second.startswith('#') and second.strip()[1:].isdigit():
                    bitstr = second.strip()[1:]
                    colour_vec = np.frombuffer(bitstr.encode('ascii'), dtype='S1') == b'1'
                else:
                    # rewind if not colour line
                    fh.seek(second_pos)
                n_str = params.get('n') or params.get('N')
                t_str = params.get('time')
                gsz = params.get('group_size')
                try:
                    n_val = int(float(n_str)) if n_str else None
                    t_val = float(t_str) if t_str else None
                except Exception:
                    n_val = None; t_val=None
                if n_val is None or t_val is None:
                    if verbose and not progress:
                        print(f"[skip] {f.name}: bad n/time")
                    continue
                # If colour_vec exists, validate length
                if colour_vec is not None and n_val is not None and colour_vec.size != n_val:
                    if debug:
                        print(f"[warn] {f.name}: colour line length {colour_vec.size} != n {n_val}; discarding colour line")
                    colour_vec = None
                # Read adjacency rows AFTER colour line handling
                rows = [line.strip() for line in fh if line.strip()]
                if len(rows) != n_val:
                    if len(rows) > n_val and lenient:
                        rows = rows[:n_val]
                    else:
                        if verbose and (debug or not progress):
                            print(f"[skip] {f.name}: row mismatch {len(rows)} != {n_val}")
                        continue
                if expected_n is None:
                    expected_n = n_val
                elif expected_n != n_val:
                    if verbose and (debug or not progress):
                        print(f"[skip] {f.name}: dimension mismatch")
                    continue
                mat = np.zeros((n_val,n_val), dtype=np.uint8)
                bad=False
                for i,r in enumerate(rows):
                    if len(r)!=n_val:
                        bad=True; break
                    mat[i,:] = np.frombuffer(r.encode('ascii'), dtype='S1') == b'1'
                if bad:
                    if verbose and not progress:
                        print(f"[skip] {f.name}: row length error")
                    continue
                mats.append(mat)
                times.append(t_val)
                try:
                    gszs.append(int(gsz) if gsz is not None else -1)
                except Exception:
                    gszs.append(-1)
                colours.append(colour_vec if colour_vec is not None else np.full(mat.shape[0], 0, dtype=bool))
        except Exception as e:
            print(f"Warning: {f} parse error: {e}", file=sys.stderr)
        pb(idx)
    if times:
        order = np.argsort(times)
        mats = [mats[i] for i in order]
        times = [times[i] for i in order]
        gszs  = [gszs[i] for i in order]
        colours = [colours[i] for i in order]
    if progress and verbose:
        print()
    if verbose:
        print(f"Loaded {len(mats)} snapshot(s).")
    return SnapshotSet(mats, times, gszs, colours)

# ---------------------------------------------------------------------------
# Reordering abstraction
# ---------------------------------------------------------------------------

@dataclass
class ReorderConfig:
    mode: str = 'none'  # none|global-degree|degree|in-degree|out-degree
    g0_dir: str = 'desc'
    g1_dir: str = 'asc'
    membership_bar: str = 'auto'  # auto|on|off

@dataclass
class ReorderResult:
    matrices: List[np.ndarray]
    perms: List[np.ndarray]
    membership: List[np.ndarray]  # 0/1 indicating original group0(0)/group1(1) BEFORE inversion for bar
    group_sizes: List[int]
    show_bar: bool


def apply_reorder(snap: SnapshotSet, cfg: ReorderConfig, *, verbose: bool=False) -> ReorderResult:
    """Apply vertex reordering for animation / diagnostics.

    IMPORTANT CHANGE: Snapshot matrices are now stored in ORIGINAL vertex order (not grouped).
    We reconstruct colour-group grouping on demand using the colour vectors so that community
    ordering modes still behave as before. This preserves consistent vertex indices for
    correlation calculations while keeping visual/community reordering semantics intact.
    """
    mats = snap.matrices
    gszs_file = snap.group_sizes  # from header (may assume grouped ordering historically)
    colours = getattr(snap, 'colours', [None]*len(mats))
    recomputed_group_sizes: List[int] = []
    perms: List[np.ndarray] = []
    membership: List[np.ndarray] = []
    out_mats: List[np.ndarray] = []
    show_bar = False
    warned_mismatch = False
    for frame,(mat, gsz_file, colour_vec) in enumerate(zip(mats, gszs_file, colours)):
        n = mat.shape[0]
        if colour_vec is not None:
            # ensure dtype int
            c = colour_vec.astype(np.uint8)
            idx0 = np.where(c==0)[0]
            idx1 = np.where(c==1)[0]
            gsz_col = idx0.size
            recomputed_group_sizes.append(gsz_col)
            if gsz_file not in (-1, gsz_col) and not warned_mismatch and verbose:
                print(f"[warn] Header group_size={gsz_file} differs from colour count={gsz_col} (frame {frame}); using colour count.")
                warned_mismatch = True
        else:
            idx0 = np.arange(gsz_file) if 0 < gsz_file < n else np.array([], dtype=int)
            idx1 = np.arange(gsz_file, n) if 0 < gsz_file < n else np.arange(n, dtype=int)
            recomputed_group_sizes.append(gsz_file)
        # Mode none: preserve original order precisely
        if cfg.mode == 'none' or n == 0:
            perm = np.arange(n)
            out_mats.append(mat)
            if colour_vec is not None:
                mem = colour_vec.astype(np.uint8)  # 0/1 in original order
            else:
                # fallback: assume header grouping
                mem = np.zeros(n, dtype=np.uint8)
                if 0 < recomputed_group_sizes[-1] < n:
                    mem[recomputed_group_sizes[-1]:] = 1
            membership.append(mem)
            perms.append(perm)
            continue
        # Global degree ordering: ignore colours for ordering, but keep membership for bar if available
        if cfg.mode == 'global-degree':
            deg = mat.sum(axis=1)
            perm = np.argsort(deg, kind='mergesort')  # ascending
            out_mats.append(mat[np.ix_(perm, perm)])
            if colour_vec is not None:
                membership.append(colour_vec.astype(np.uint8)[perm])
            else:
                mem = np.zeros(n, dtype=np.uint8)
                if 0 < recomputed_group_sizes[-1] < n:
                    mem[recomputed_group_sizes[-1]:] = 1
                membership.append(mem[perm])
            perms.append(perm)
            show_bar = True
            continue
        # Community modes: rebuild grouping by colours (idx0 then idx1), then sort within each group
        if idx0.size == 0 or idx1.size == 0:
            # Degenerate: fall back to global handling (no real split)
            deg = mat.sum(axis=1)
            perm = np.argsort(deg, kind='mergesort')
            out_mats.append(mat[np.ix_(perm, perm)])
            membership.append((colour_vec.astype(np.uint8) if colour_vec is not None else np.zeros(n, dtype=np.uint8))[perm])
            perms.append(perm)
            show_bar = show_bar or colour_vec is not None
            continue
        row_sum = mat.sum(axis=1)
        # Internal degree (within same colour)
        internal = np.zeros(n, dtype=int)
        # Build quick sets for membership check
        mask0 = np.zeros(n, dtype=bool); mask0[idx0] = True
        # For efficiency: slice submatrices instead of loops
        internal[idx0] = mat[np.ix_(idx0, idx0)].sum(axis=1)
        internal[idx1] = mat[np.ix_(idx1, idx1)].sum(axis=1)
        external = row_sum - internal
        if cfg.mode == 'degree':
            key = row_sum
        elif cfg.mode == 'in-degree':
            key = internal
        elif cfg.mode == 'out-degree':
            key = external
        else:
            key = row_sum
        def sort_dir(indices: np.ndarray, direction: str):
            arr = key[indices]
            local = np.argsort(-arr, kind='mergesort') if direction=='desc' else np.argsort(arr, kind='mergesort')
            return indices[local]
        order0 = sort_dir(idx0, cfg.g0_dir)
        order1 = sort_dir(idx1, cfg.g1_dir)
        perm = np.concatenate([order0, order1])
        out_mats.append(mat[np.ix_(perm, perm)])
        mem = np.zeros(n, dtype=np.uint8); mem[len(order0):] = 1  # after permutation first block zeros
        membership.append(mem)
        perms.append(perm)
        show_bar = show_bar or colour_vec is not None
    # Membership bar preference override
    if cfg.membership_bar == 'on':
        show_bar = True
    elif cfg.membership_bar == 'off':
        show_bar = False
    return ReorderResult(out_mats, perms, membership, recomputed_group_sizes, show_bar)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rank1_metrics(A: np.ndarray) -> dict:
    eigvals, eigvecs = np.linalg.eigh(A)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    lambda1 = eigvals[0]
    lambda2 = eigvals[1] if eigvals.size>1 else 0.0
    lambda3 = eigvals[2] if eigvals.size>2 else 0.0
    lambda4 = eigvals[3] if eigvals.size>3 else 0.0
    ratio = abs(lambda2)/lambda1 if lambda1!=0 else 0.0
    v1 = eigvecs[:,0]
    A1 = lambda1 * np.outer(v1,v1)
    fro_res = np.linalg.norm(A - A1, 'fro')
    fro_rel = fro_res / np.linalg.norm(A, 'fro')
    s2 = (eigvals[1:]**2).sum() if eigvals.size>1 else 0.0
    s_all = (eigvals**2).sum() if eigvals.size>0 else 1.0
    tail = s2/s_all if s_all>0 else 0.0
    d = A.sum(axis=1)
    with np.errstate(divide='ignore'): dinv = 1/np.sqrt(d)
    dinv[~np.isfinite(dinv)] = 0.0
    An = (dinv[:,None]*A)*dinv[None,:]
    neigs,_ = np.linalg.eigh(An); neigs.sort()
    norm_second = neigs[-2] if neigs.size>1 else 0.0
    return {
        'lambda1': float(lambda1), 'lambda2': float(lambda2), 'lambda3': float(lambda3), 'lambda4': float(lambda4),
        'lambda2_ratio': float(ratio), 'fro_residual_rel': float(fro_rel), 'tail_energy': float(tail), 'norm_second_eig': float(norm_second)
    }


def compute_extended_block_baseline_metrics(A: np.ndarray, gsz: int, *, eigvals: np.ndarray, eigvecs: np.ndarray, base_lambda2_ratio: float) -> dict:
    n = A.shape[0]
    nan_keys = [
        'block1_lambda1','block1_lambda2','block1_lambda3','block1_lambda4','block1_lambda2_ratio','block1_tail_energy',
        'block2_lambda1','block2_lambda2','block2_lambda3','block2_lambda4','block2_lambda2_ratio','block2_tail_energy',
        'corr_v1_deg','corr_v1_deg_block1','corr_v1_deg_block2',
        'baseline_lambda1','baseline_lambda2','baseline_lambda2_ratio','lambda2_ratio_minus_baseline'
    ]
    if not (1 < gsz < n-1):
        return {k: float('nan') for k in nan_keys}
    B1 = A[:gsz,:gsz]
    B2 = A[gsz:,gsz:]
    v1g = eigvecs[:,0]
    deg_global = A.sum(axis=1)
    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.size==0 or b.size==0: return float('nan')
        if np.allclose(a,a[0]) or np.allclose(b,b[0]): return float('nan')
        return float(np.corrcoef(a,b)[0,1])
    corr_global = corr(v1g, deg_global)
    def block_stats(B: np.ndarray):
        if B.shape[0] < 2:
            return [float('nan')]*4, float('nan'), float('nan'), np.array([])
        w,V = np.linalg.eigh(B); w.sort(); w=w[::-1]
        l1 = w[0]; l2 = w[1] if w.size>1 else 0.0; l3 = w[2] if w.size>2 else 0.0; l4 = w[3] if w.size>3 else 0.0
        ratio = abs(l2)/l1 if l1!=0 else float('nan')
        tail = (w[1:]**2).sum()/(w**2).sum() if w.size>1 else 0.0
        v1 = V[:, np.argsort(w)[::-1][0]]
        return [float(l1),float(l2),float(l3),float(l4)], float(ratio), float(tail), v1
    (b1_l1,b1_l2,b1_l3,b1_l4), b1_ratio, b1_tail, v1b1 = block_stats(B1)
    (b2_l1,b2_l2,b2_l3,b2_l4), b2_ratio, b2_tail, v1b2 = block_stats(B2)
    deg1 = A[:gsz,:].sum(axis=1)
    deg2 = A[gsz:,:].sum(axis=1)
    corr_b1 = corr(v1b1, deg1) if v1b1.size else float('nan')
    corr_b2 = corr(v1b2, deg2) if v1b2.size else float('nan')
    e1 = B1[np.triu_indices(B1.shape[0], k=1)].sum() if B1.shape[0]>1 else 0
    e2 = B2[np.triu_indices(B2.shape[0], k=1)].sum() if B2.shape[0]>1 else 0
    denom1 = gsz*(gsz-1)/2 if gsz>1 else 1
    denom2 = (n-gsz)*(n-gsz-1)/2 if (n-gsz)>1 else 1
    p11 = e1/denom1 if denom1>0 else float('nan')
    p22 = e2/denom2 if denom2>0 else float('nan')
    cross = A[:gsz, gsz:].sum()
    p12 = cross/(gsz*(n-gsz)) if gsz>0 and (n-gsz)>0 else float('nan')
    if gsz < 2 or (n-gsz) < 2:
        baseline_lambda1 = baseline_lambda2 = baseline_lambda2_ratio = float('nan')
    else:
        M = np.array([[p11*(gsz-1), p12*(n-gsz)], [p12*gsz, p22*(n-gsz-1)]], dtype=float)
        wb,_ = np.linalg.eigh(M); wb.sort(); baseline_lambda1 = wb[-1]; baseline_lambda2 = wb[-2] if wb.size>1 else float('nan')
        baseline_lambda2_ratio = abs(baseline_lambda2)/baseline_lambda1 if baseline_lambda1!=0 and not np.isnan(baseline_lambda2) else float('nan')
    if base_lambda2_ratio==base_lambda2_ratio and baseline_lambda2_ratio==baseline_lambda2_ratio:
        lambda2_ratio_minus_baseline = base_lambda2_ratio - baseline_lambda2_ratio
    else:
        lambda2_ratio_minus_baseline = float('nan')
    return {
        'block1_lambda1': b1_l1,
        'block1_lambda2': b1_l2,
        'block1_lambda3': b1_l3,
        'block1_lambda4': b1_l4,
        'block1_lambda2_ratio': b1_ratio,
        'block1_tail_energy': b1_tail,
        'block2_lambda1': b2_l1,
        'block2_lambda2': b2_l2,
        'block2_lambda3': b2_l3,
        'block2_lambda4': b2_l4,
        'block2_lambda2_ratio': b2_ratio,
        'block2_tail_energy': b2_tail,
        'corr_v1_deg': corr_global,
        'corr_v1_deg_block1': corr_b1,
        'corr_v1_deg_block2': corr_b2,
        'baseline_lambda1': float(baseline_lambda1),
        'baseline_lambda2': float(baseline_lambda2),
        'baseline_lambda2_ratio': float(baseline_lambda2_ratio),
        'lambda2_ratio_minus_baseline': float(lambda2_ratio_minus_baseline)
    }

# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------

def parse_frame_spec(spec: str) -> List[int]:
    result: List[int] = []
    for part in spec.split(','):
        part = part.strip()
        if not part: continue
        if '-' in part:
            range_part = part
            step = 1
            if ':' in part:
                range_part, step_s = part.split(':',1)
                step = int(step_s)
            a,b = range_part.split('-',1)
            start = int(a); end = int(b)
            if end < start: start,end = end,start
            result.extend(list(range(start,end+1,step)))
        else:
            result.append(int(part))
    return sorted(set(result))


def select_frames(total: int, *, every: Optional[int]=None, frames_spec: Optional[str]=None, sample: Optional[int]=None) -> np.ndarray:
    base = np.arange(total)
    if every and every>1:
        base = base[::every]
    if frames_spec:
        specified = np.array([i for i in parse_frame_spec(frames_spec) if 0<=i<total], dtype=int)
        base = np.intersect1d(base, specified, assume_unique=False)
    if sample and sample>0 and sample < base.size:
        idx = np.linspace(0, base.size-1, num=sample, dtype=int)
        base = base[idx]
    return base

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def build_animation(mats: List[np.ndarray],
                    times: List[float],
                    reorder: ReorderResult,
                    *,
                    interval: int,
                    show_time: bool,
                    loop: bool,
                    fig_size: float,
                    pixels: Optional[int],
                    dpi: int,
                    show_group_lines: bool=False,
                    avg_kernel: int = 1,
                    merge_kernel: int = 1) -> FuncAnimation:
    if not mats:
        raise ValueError("No matrices")
    if pixels and fig_size>0:
        dpi_eff = max(1, int(round(pixels/fig_size)))
    else:
        dpi_eff = dpi
    bar_px = 20 if reorder.show_bar and reorder.membership else 0
    h_frac = bar_px/(fig_size*dpi_eff) if (bar_px and dpi_eff>0) else 0.0
    fig = plt.figure(figsize=(fig_size,fig_size), dpi=dpi_eff)
    ax = fig.add_axes((0.0, h_frac, 1.0, 1.0-h_frac))
    bar_ax = None; bar_img=None
    if bar_px>0:
        bar_ax = fig.add_axes((0.0, 0.0, 1.0, h_frac))
        bar_ax.set_axis_off()
    # Helper transforms:
    def smooth_matrix(mat: np.ndarray) -> np.ndarray:
        k = avg_kernel
        if k > 1 and merge_kernel > 1:
            raise ValueError("avg and merge both active")
        if merge_kernel > 1:
            mk = merge_kernel
            h, w = mat.shape
            H = (h // mk) * mk
            W = (w // mk) * mk
            if H==0 or W==0: return mat.astype(float)
            trimmed = mat[:H,:W].astype(float)
            # reshape and average blocks
            return trimmed.reshape(H//mk, mk, W//mk, mk).mean(axis=(1,3))
        if k <= 1: return mat.astype(float)
        if k > mat.shape[0] or k > mat.shape[1]:
            return mat.astype(float)  # too large, skip
        A = mat.astype(float)
        I = A.cumsum(axis=0).cumsum(axis=1)
        res = (I[k-1:,k-1:] - np.pad(I[k-1:,:-k], ((0,0),(0,1)), constant_values=0) - np.pad(I[:-k,k-1:], ((0,1),(0,0)), constant_values=0) + np.pad(I[:-k,:-k], ((0,1),(0,1)), constant_values=0)) / (k*k)
        return res
    def smooth_bar(vec: np.ndarray) -> np.ndarray:
        if avg_kernel > 1 and merge_kernel > 1:
            raise ValueError("avg and merge both active")
        v = (1 - vec).astype(float)
        if merge_kernel > 1:
            mk = merge_kernel
            L = (v.size // mk) * mk
            if L==0: return v[None,:]
            trimmed = v[:L]
            return trimmed.reshape(L//mk, mk).mean(axis=1)[None,:]
        k = avg_kernel
        if k <= 1 or k > v.size:
            return v[None,:]
        kernel = np.ones(k)/k
        conv = np.convolve(v, kernel, mode='valid')
        return conv[None,:]
    smoothed0 = smooth_matrix(mats[0])
    img = ax.imshow(smoothed0, cmap='gray_r', interpolation='nearest', vmin=0, vmax=1, origin='lower')
    ax.set_axis_off()
    time_text = ax.text(0.01,0.99,'', transform=ax.transAxes, ha='left', va='top', color='black', fontsize=10) if show_time else None
    if bar_ax is not None:
        first_bar = smooth_bar(reorder.membership[0])
        bar_img = bar_ax.imshow(first_bar, cmap='gray_r', interpolation='nearest', vmin=0, vmax=1, aspect='auto')
    vline = hline = None
    if show_group_lines:
        # Adjust group boundary for reduced size.
        k = avg_kernel
        mk = merge_kernel
        original_g0 = reorder.group_sizes[0] if reorder.group_sizes and 0 < reorder.group_sizes[0] < mats[0].shape[0] else None
        if original_g0 is not None:
            if mk > 1:
                g0 = original_g0 // mk  # truncation
            elif k > 1:
                g0 = max(0, min(original_g0-1, mats[0].shape[0]-k))
            else:
                g0 = original_g0
        else:
            g0 = None
        if g0 is not None:
            pos = g0 - 0.5
            vline, = ax.plot([pos,pos], [-0.5, smoothed0.shape[0]-0.5], color='red', linewidth=1.0, alpha=0.8)
            hline, = ax.plot([-0.5, smoothed0.shape[1]-0.5], [pos,pos], color='red', linewidth=1.0, alpha=0.8)
        else:
            vline, = ax.plot([], [], color='red', linewidth=1.0, alpha=0.0)
            hline, = ax.plot([], [], color='red', linewidth=1.0, alpha=0.0)
    def update(i: int):
        smoothed = smooth_matrix(mats[i])
        img.set_data(smoothed)
        arts: List = [img]
        if time_text is not None:
            time_text.set_text(f"t={times[i]:.6f}")
            arts.append(time_text)
        if bar_img is not None:
            bar_img.set_data(smooth_bar(reorder.membership[i]))
            arts.append(bar_img)
        if show_group_lines and vline is not None and hline is not None:
            gsz = reorder.group_sizes[i] if i < len(reorder.group_sizes) else -1
            k = avg_kernel; mk = merge_kernel
            n_orig = mats[i].shape[0]
            sm = smoothed.shape[0]
            if 0 < gsz < n_orig:
                if mk > 1:
                    adj_g = gsz // mk
                elif k > 1:
                    adj_g = gsz if k<=1 else max(0, min(gsz-1, n_orig - k))
                else:
                    adj_g = gsz
                pos = adj_g - 0.5
                vline.set_data([pos,pos], [-0.5, sm-0.5])
                hline.set_data([-0.5, smoothed.shape[1]-0.5], [pos,pos])
                vline.set_alpha(0.8); hline.set_alpha(0.8)
            else:
                vline.set_alpha(0.0); hline.set_alpha(0.0)
            arts.extend([vline,hline])
        return tuple(arts)
    return FuncAnimation(fig, update, frames=len(mats), interval=interval, blit=True, repeat=loop)

# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_info(args):
    snap = load_snapshots(args.directory, verbose=not args.quiet, progress=not args.no_progress, lenient=args.lenient, debug=args.debug)
    n_frames = len(snap.matrices)
    if n_frames==0:
        print("No snapshots found."); return 0
    n = snap.matrices[0].shape[0]
    times = snap.times
    dt = np.diff(times) if len(times)>1 else np.array([])
    gs = [g for g in snap.group_sizes if g>0]
    summary = {
        'frames': n_frames,
        'n': n,
        'time_min': times[0],
        'time_max': times[-1],
        'avg_delta_t': float(dt.mean()) if dt.size else None,
        'group_size_mode': int(max(set(gs), key=gs.count)) if gs else None,
        'group_size_changes': int(sum(g1!=g2 for g1,g2 in zip(gs, gs[1:]))) if len(gs)>1 else 0,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2))
    if not args.quiet:
        for k,v in summary.items():
            print(f"{k}: {v}")
    return 0


def build_reorder_config(args) -> ReorderConfig:
    return ReorderConfig(mode=args.order, g0_dir=args.g0_dir, g1_dir=args.g1_dir, membership_bar=args.membership_bar)


def cmd_animate(args):
    snap = load_snapshots(args.directory, verbose=not args.quiet, progress=not args.no_progress, lenient=args.lenient, debug=args.debug)
    if not snap.matrices:
        print("No snapshots loaded", file=sys.stderr); return 1
    cfg = build_reorder_config(args)
    reorder = apply_reorder(snap, cfg, verbose=not args.quiet)
    show_lines = bool(getattr(args, 'group_lines', False) and cfg.mode != 'global-degree')
    if getattr(args, 'group_lines', False) and cfg.mode == 'global-degree' and not args.quiet:
        print("(Note) --group-lines ignored under global-degree ordering.")
    if args.avg < 1:
        print("--avg must be >=1", file=sys.stderr); return 2
    if args.merge < 1:
        print("--merge must be >=1", file=sys.stderr); return 2
    if args.avg>1 and args.merge>1:
        print("Cannot combine --avg and --merge (choose one)", file=sys.stderr); return 2
    anim = build_animation(reorder.matrices, snap.times, reorder, interval=args.interval, show_time=args.show_time, loop=not args.no_loop, fig_size=args.figure_size, pixels=args.pixels, dpi=args.dpi, show_group_lines=show_lines, avg_kernel=args.avg, merge_kernel=args.merge)
    out = args.out
    if out:
        suffix = out.suffix.lower()
        fps = args.fps or max(1, int(round(1000.0/args.interval)))
        savefig_kwargs={'pad_inches':0}
        start=time.time()
        if suffix=='.mp4':
            anim.save(out, writer='ffmpeg', fps=fps, savefig_kwargs=savefig_kwargs)
        elif suffix=='.gif':
            writer = PillowWriter(fps=fps)
            anim.save(out, writer=writer, savefig_kwargs=savefig_kwargs)
        else:
            print("Unsupported format (use .mp4 or .gif)", file=sys.stderr); return 2
        dur=time.time()-start
        if not args.quiet:
            print(f"Saved animation to {out} ({dur:.2f}s)")
    else:
        plt.show()
    return 0


def cmd_diagnostics(args):
    snap = load_snapshots(args.directory, verbose=not args.quiet, progress=not args.no_progress, lenient=args.lenient, debug=args.debug)
    if not snap.matrices:
        print("No snapshots loaded", file=sys.stderr); return 1
    cfg = build_reorder_config(args)
    reorder = apply_reorder(snap, cfg)
    sel = select_frames(len(reorder.matrices), every=args.every, frames_spec=args.frames, sample=args.sample)
    header=None; rows=[]; out_csv=None
    if args.csv:
        out_csv = open(args.csv,'w')
    try:
        for idx, frame in enumerate(sel, start=1):
            A = reorder.matrices[frame].astype(float)
            gsz = reorder.group_sizes[frame] if frame < len(reorder.group_sizes) else -1
            if args.extended or args.rank1:
                eigvals, eigvecs = np.linalg.eigh(A)
                order_desc = np.argsort(eigvals)[::-1]
                eigvals = eigvals[order_desc]
                eigvecs = eigvecs[:, order_desc]
                lambda1 = eigvals[0]
                lambda2 = eigvals[1] if eigvals.size>1 else 0.0
                lambda3 = eigvals[2] if eigvals.size>2 else 0.0
                lambda4 = eigvals[3] if eigvals.size>3 else 0.0
                base_lambda2_ratio = abs(lambda2)/lambda1 if lambda1!=0 else 0.0
                v1 = eigvecs[:,0]
                A1 = lambda1 * np.outer(v1,v1)
                fro_res = np.linalg.norm(A - A1, 'fro')
                fro_rel = fro_res / np.linalg.norm(A, 'fro')
                s2 = (eigvals[1:]**2).sum() if eigvals.size>1 else 0.0
                s_all = (eigvals**2).sum() if eigvals.size>0 else 1.0
                tail = s2/s_all if s_all>0 else 0.0
                d = A.sum(axis=1)
                with np.errstate(divide='ignore'): dinv = 1/np.sqrt(d)
                dinv[~np.isfinite(dinv)] = 0.0
                An = (dinv[:,None]*A)*dinv[None,:]
                neigs,_ = np.linalg.eigh(An); neigs.sort()
                norm_second = neigs[-2] if neigs.size>1 else 0.0
                base_metrics = {
                    'lambda1': float(lambda1), 'lambda2': float(lambda2), 'lambda3': float(lambda3), 'lambda4': float(lambda4),
                    'lambda2_ratio': float(base_lambda2_ratio), 'fro_residual_rel': float(fro_rel), 'tail_energy': float(tail), 'norm_second_eig': float(norm_second)
                }
                if args.extended:
                    ext = compute_extended_block_baseline_metrics(A, gsz, eigvals=eigvals, eigvecs=eigvecs, base_lambda2_ratio=base_metrics['lambda2_ratio'])
                    metrics_row = {'frame': int(frame), 'time': snap.times[frame]} | base_metrics | ext
                else:
                    metrics_row = {'frame': int(frame), 'time': snap.times[frame]} | base_metrics
            else:
                metrics_row = {'frame': int(frame), 'time': snap.times[frame]}
            if header is None:
                header = list(metrics_row.keys())
                if out_csv:
                    out_csv.write(','.join(header)+'\n')
                if not args.quiet:
                    print('# '+','.join(header))
            if out_csv:
                out_csv.write(','.join(str(metrics_row[k]) for k in header)+'\n')
                if idx % 25 == 0:
                    out_csv.flush()
            if not args.quiet:
                print(','.join(str(metrics_row[k]) for k in header))
            rows.append(metrics_row)
    finally:
        if out_csv:
            out_csv.close()
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
    return 0


def cmd_correlations(args):
    snap = load_snapshots(args.directory, verbose=not args.quiet, progress=not args.no_progress, lenient=args.lenient, debug=args.debug)
    if not snap.matrices:
        print("No snapshots loaded", file=sys.stderr); return 1
    times = np.array(snap.times)
    ref_time = args.reference if args.reference is not None else 0.0
    # Find nearest time index
    idx_ref = int(np.argmin(np.abs(times - ref_time)))
    ref_time_actual = times[idx_ref]
    if not args.quiet:
        print(f"Reference time requested={ref_time} -> using frame {idx_ref} at time {ref_time_actual}")
    # Vertex colours (bool arrays)
    colours = snap.colours
    if idx_ref >= len(colours):
        print("Reference index out of range", file=sys.stderr); return 2
    c_ref = colours[idx_ref].astype(int)
    # Edge vectors: flatten upper triangle (i<j)
    def edge_vector(mat: np.ndarray) -> np.ndarray:
        # Use upper triangle without diagonal; cast to int 0/1
        return mat[np.triu_indices(mat.shape[0], k=1)].astype(int)
    e_ref = edge_vector(snap.matrices[idx_ref])
    print(e_ref)
    n = snap.matrices[0].shape[0]
    # Prepare output
    rows = []
    if args.csv:
        out = open(args.csv, 'w')
        out.write('delta_t,vertex_correlation,edge_correlation\n')
    else:
        out = None
    try:
        for i,(t,mat,col) in enumerate(zip(snap.times, snap.matrices, colours)):
            delta_t = t - ref_time_actual
            col_i = col.astype(int)
            # Vertex correlation = Pearson corr of colour indicators (treat as 0/1 variables)
            def corr(a: np.ndarray, b: np.ndarray) -> float:
                """Pearson correlation with sensible fallback for constant vectors.

                If both vectors constant and equal -> 1.0.
                If both constant and different -> 0.0 (no variability but total disagreement on mean structure).
                If one constant only -> fraction agreement (equivalent to accuracy vs constant baseline),
                because Pearson would be undefined (division by zero variance).
                """
                if a.size != b.size or a.size==0:
                    return float('nan')
                a_const = np.all(a==a[0])
                b_const = np.all(b==b[0])
                if a_const and b_const:
                    return 1.0 if a[0]==b[0] else 0.0
                if a_const or b_const:
                    # fallback: agreement fraction
                    return float(np.mean(a==b))
                # standard Pearson
                return float(np.corrcoef(a,b)[0,1])
            v_corr = corr(c_ref, col_i)
            e_i = edge_vector(mat)
            e_corr = corr(e_ref, e_i)
            row = (delta_t, v_corr, e_corr)
            rows.append(row)
            if out:
                out.write(f"{delta_t:.9f},{v_corr:.9f},{e_corr:.9f}\n")
            if not args.quiet and not out:
                print(f"{delta_t:.9f},{v_corr:.9f},{e_corr:.9f}")
    finally:
        if out:
            out.close()
    return 0


def cmd_stub(_args, name: str):
    print(f"Command '{name}' not yet implemented.")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog='analyse.py', description='Analyse / animate netcoevolve adjacency snapshots (subcommand interface).')
    p.add_argument('--version', action='store_true', help='Show version and exit')
    sub = p.add_subparsers(dest='command', required=True)

    def add_common(sp):
        sp.add_argument('directory', type=Path, help='Snapshot directory')
        sp.add_argument('--quiet', action='store_true')
        sp.add_argument('--no-progress', action='store_true')
        sp.add_argument('--lenient', action='store_true')
        sp.add_argument('--debug', action='store_true')
        sp.add_argument('--order', choices=['none','global-degree','degree','in-degree','out-degree'], default='none')
        sp.add_argument('--g0-dir', choices=['asc','desc'], default='desc')
        sp.add_argument('--g1-dir', choices=['asc','desc'], default='asc')
        sp.add_argument('--membership-bar', choices=['auto','on','off'], default='auto')

    # animate
    spa = sub.add_parser('animate', help='Render animation of adjacency evolution.')
    add_common(spa)
    spa.add_argument('--interval', type=int, default=120)
    spa.add_argument('--no-loop', action='store_true')
    spa.add_argument('--show-time', action='store_true')
    spa.add_argument('--figure-size', type=float, default=5.0)
    spa.add_argument('--pixels', type=int)
    spa.add_argument('--dpi', type=int, default=150)
    spa.add_argument('--fps', type=int)
    spa.add_argument('--out', type=Path, help='Output animation file (.mp4 or .gif)')
    spa.add_argument('--group-lines', action='store_true', help='Draw red lines at community boundary (ignored for global-degree order)')
    spa.add_argument('--avg', type=int, default=1, help='Sliding kxk averaging (valid convolution) for display; overlapping windows; reduces size by k-1.')
    spa.add_argument('--merge', type=int, default=1, help='Non-overlapping kxk block merge (stride=k); truncates remainder; output size floor(n/k). Cannot combine with --avg>1.')

    # diagnostics
    spd = sub.add_parser('diagnostics', help='Compute spectral diagnostics (no rendering).')
    add_common(spd)
    spd.add_argument('--rank1', action='store_true', help='Compute rank-1 closeness metrics')
    spd.add_argument('--extended', action='store_true', help='Add block eigen metrics, correlations, and two-block baseline (implies --rank1)')
    spd.add_argument('--every', type=int)
    spd.add_argument('--frames', type=str, help='Frame list/ranges e.g. 0,10,20-30,100-200:10')
    spd.add_argument('--sample', type=int, help='Uniformly sample K frames after selection')
    spd.add_argument('--csv', type=Path, help='Write metrics CSV')
    spd.add_argument('--json', type=Path, help='Write metrics JSON array')

    # info
    spi = sub.add_parser('info', help='Summarize dataset metadata.')
    add_common(spi)
    spi.add_argument('--json', type=Path, help='Write summary JSON')

    # correlations
    spc = sub.add_parser('correlations', help='Compute vertex & edge correlations vs reference time.')
    add_common(spc)
    spc.add_argument('--reference', type=float, help='Reference time (default 0, nearest frame chosen).')
    spc.add_argument('--csv', type=Path, help='Write correlations CSV.')

    # stubs
    spf = sub.add_parser('frames', help='(stub) Export selected frames as images.')
    add_common(spf)
    spp = sub.add_parser('permutation', help='(stub) Export applied permutations.')
    add_common(spp)
    spv = sub.add_parser('verify', help='(stub) Verify integrity of snapshot set.')
    add_common(spv)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.version:
        print('analyse.py version 0.2.0')
        return 0
    if args.command == 'animate':
        return cmd_animate(args)
    if args.command == 'diagnostics':
        if args.extended:  # imply rank1
            args.rank1 = True
        return cmd_diagnostics(args)
    if args.command == 'info':
        return cmd_info(args)
    if args.command == 'correlations':
        return cmd_correlations(args)
    if args.command == 'frames':
        return cmd_stub(args, 'frames')
    if args.command == 'permutation':
        return cmd_stub(args, 'permutation')
    if args.command == 'verify':
        return cmd_stub(args, 'verify')
    print('Unknown command', file=sys.stderr)
    return 1

if __name__ == '__main__':
    raise SystemExit(main())
