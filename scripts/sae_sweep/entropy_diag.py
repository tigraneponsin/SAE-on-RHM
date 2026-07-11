"""Unified SAE entropy diagnostic (offline post-processing of *.sae_eval.pt).

One CLI, three subcommands (no SAE re-evaluation in any of them):

  aggregate   Normalized entropy H_bar_{fire,raw,dec}_norm vs lambda_1, one
              subplot per leaf position (the three weight schemes as three
              curves). Reads the stored entropy block directly.

  min         Min-conditional-entropy subtree alignment. Recomputes, per
              feature, the same-level and whole-tree argmin latent and draws the
              three constraint curves (parent / level / unconstrained) per
              position, plus the unconstrained argmin-target level distribution.

  splitcheck  min, plus the feature-splitting reassignment: features that pin
              their whole-tree child's structural PARENT beyond the RHM leak are
              reassigned up one level (absorption). Draws parent/level/
              unconstrained entropy and unconstrained vs reassigned level
              distributions. See feature_splitting_alpha_sweep.py to tune alpha.

Usage:
    python scripts/sae_sweep/entropy_diag.py aggregate  --artifacts_dir DIR ...
    python scripts/sae_sweep/entropy_diag.py min        --artifacts_dir DIR ...
    python scripts/sae_sweep/entropy_diag.py splitcheck --artifacts_dir DIR ...

All shared numeric helpers live in entropy_core.py; the per-feature labeling
core (shared with circuit_tracing) lives in absorption.py.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.common.notation import sae_label, report_level, add_report_flag
from scripts.sae_sweep.entropy_core import (
    SCHEMES, torch_nanmin, weighted_aggregate, weights_for_scheme,
    find_threshold_lambda, thr_label, parent_min_sanity)
from scripts.sae_sweep.absorption import per_feature_labels, resolve_leak_table

DEFAULT_ARTIFACTS_DIR = (
    '/work/pcsl/ponsin/Mean_Transformer/Small_SAE/latent_dim_4*512/'
    'v_16_L_3_m_16_wdecay_0.0001_dropout_0.1_nores/'
    'sweep_alltokens_layer1_lambda1_zoom/analysis_files'
)
DEFAULT_ALPHA = 0.5
# y-limits for every split-check entropy subplot (small margin so plateaus at
# 0/1 stay visible).
ENTROPY_YLIM = (-0.02, 1.02)
# Distinct per-level palette for the level-distribution plots (high-contrast,
# not the matplotlib default cycle, so those figures do not read like the
# entropy plot's C0/C1/C3).
LEVEL_PALETTE = ('#4daf4a', '#984ea3', '#a65628', '#f781bf', '#17becf',
                 '#bcbd22', '#000000')


# ===========================================================================
# aggregate mode  (former plot_entropy_lambda.py)
# ===========================================================================

ENTROPY_KEYS = ('H_bar_fire_norm', 'H_bar_raw_norm', 'H_bar_dec_norm')
ENTROPY_LABELS = ('H_fire', 'H_raw', 'H_dec')
ENTROPY_COLORS = ('C0', 'C1', 'C2')


def agg_load_entries(files):
    """Load artifacts, keep only those with the three normalized entropy
    tensors. Return list of dicts grouped later by layer."""
    entries = []
    skipped = []
    for f in files:
        art = torch.load(f, map_location='cpu', weights_only=False)
        if any(k not in art or art[k] is None for k in ENTROPY_KEYS):
            skipped.append(f.name)
            continue
        rhm = art.get('rhm')
        if rhm is None or 's' not in rhm or 'L' not in rhm:
            skipped.append(f.name)
            continue
        entries.append({
            'file': f,
            'layer_id': int(art['layer_id']),
            'lambda_l1': float(art.get('lambda_l1') or 0.0),
            'mode': art.get('mode', ''),
            's': int(rhm['s']),
            'L': int(rhm['L']),
            'token_positions': art['token_positions'].cpu().numpy().astype(np.int64),
            'H_fire': art['H_bar_fire_norm'].cpu().numpy().astype(np.float64),
            'H_raw': art['H_bar_raw_norm'].cpu().numpy().astype(np.float64),
            'H_dec': art['H_bar_dec_norm'].cpu().numpy().astype(np.float64),
        })
    if skipped:
        print(f'Skipped {len(skipped)} artifacts without entropy block:')
        for name in skipped[:10]:
            print(f'  {name}')
        if len(skipped) > 10:
            print(f'  ... and {len(skipped) - 10} more')
    return entries


def agg_build_matrix(entries_sorted, key, num_positions):
    """Stack entry[key] into [num_artifacts, num_positions], placing each
    value at its token_position index. Unfilled cells stay NaN."""
    mat = np.full((len(entries_sorted), num_positions), np.nan, dtype=np.float64)
    for i, e in enumerate(entries_sorted):
        for p_idx, p in enumerate(e['token_positions']):
            if 0 <= p < num_positions:
                mat[i, p] = e[key][p_idx]
    return mat


def agg_plot_meanpool_layer(layer, entries, outfile, xlim, threshold_lambda,
                            tolerance, report=False):
    """Single-panel plot for mean_pooled artifacts (P=1, root-class entropy)."""
    entries = sorted(entries, key=lambda e: e['lambda_l1'])
    lambdas = np.array([e['lambda_l1'] for e in entries], dtype=np.float64)
    s = entries[0]['s']
    L = entries[0]['L']

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    for key, label, color in zip(ENTROPY_KEYS, ENTROPY_LABELS, ENTROPY_COLORS):
        short = key.replace('H_bar_', '').replace('_norm', '')
        y = np.array([float(e[f'H_{short}'][0]) for e in entries], dtype=np.float64)
        ax.plot(lambdas, y, '-o', color=color, label=label,
                markersize=4, linewidth=1.2)
    ax.set_xscale('log')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('lambda_l1', fontsize=10)
    ax.set_ylabel('H / H_theoretical (root class)', fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.5)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if threshold_lambda is not None:
        ax.axvline(threshold_lambda, color='red', linestyle='--', linewidth=1.0,
                   alpha=0.7,
                   label=f'err thresh ({tolerance:.0%}): {threshold_lambda:.3g}')
    ax.legend(fontsize=10)
    fig.suptitle(
        f'Normalized entropy vs lambda_1 | {sae_label(layer, report)} | '
        f'mode=mean_pooled | s={s}, L={L}',
        fontsize=11, y=0.99,
    )
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  layer {layer} (mean_pooled): {len(entries)} artifacts, '
          f'lambda range [{lambdas.min():.3g}, {lambdas.max():.3g}] '
          f'-> {outfile}')


def agg_plot_layer(layer, entries, outfile, xlim, threshold_lambda, tolerance,
                   report=False, plot_avg=False):
    s_vals = {e['s'] for e in entries}
    L_vals = {e['L'] for e in entries}
    if len(s_vals) > 1 or len(L_vals) > 1:
        print(f'  layer {layer}: inconsistent rhm s/L across artifacts, '
              f'using first entry and dropping outliers')
        s0, L0 = entries[0]['s'], entries[0]['L']
        entries = [e for e in entries if e['s'] == s0 and e['L'] == L0]
    s = entries[0]['s']
    L = entries[0]['L']
    num_positions = s ** L

    entries = sorted(entries, key=lambda e: e['lambda_l1'])
    lambdas = np.array([e['lambda_l1'] for e in entries], dtype=np.float64)

    mats = {k: agg_build_matrix(entries, k, num_positions)
            for k in ('H_fire', 'H_raw', 'H_dec')}

    total_panels = num_positions + (1 if plot_avg else 0)
    ncols = min(4, total_panels)
    nrows = max(1, int(math.ceil(total_panels / ncols)))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.6 * ncols, 2.2 * nrows),
        sharex=True, sharey=True, squeeze=False,
    )
    axes_flat = axes.flatten()

    for p in range(num_positions):
        ax = axes_flat[p]
        for key, label, color in zip(ENTROPY_KEYS, ENTROPY_LABELS, ENTROPY_COLORS):
            short = key.replace('H_bar_', '').replace('_norm', '')
            y = mats[f'H_{short}'][:, p]
            ax.plot(lambdas, y, '-o', color=color, label=label,
                    markersize=3, linewidth=1.0)
        ax.set_title(f'pos {p}', fontsize=9)
        ax.set_xscale('log')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.5)
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if threshold_lambda is not None:
            ax.axvline(threshold_lambda, color='red', linestyle='--',
                       linewidth=0.8, alpha=0.7)

    if plot_avg:
        ax_agg = axes_flat[num_positions]
        for key, label, color in zip(ENTROPY_KEYS, ENTROPY_LABELS, ENTROPY_COLORS):
            short = key.replace('H_bar_', '').replace('_norm', '')
            y = np.nanmean(mats[f'H_{short}'], axis=1)
            ax_agg.plot(lambdas, y, '-o', color=color, label=label,
                        markersize=3, linewidth=1.2)
        ax_agg.set_title('mean over positions', fontsize=9, fontweight='bold')
        ax_agg.set_xscale('log')
        ax_agg.set_ylim(-0.05, 1.05)
        ax_agg.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.5)
        if xlim is not None:
            ax_agg.set_xlim(xlim[0], xlim[1])
        if threshold_lambda is not None:
            ax_agg.axvline(threshold_lambda, color='red', linestyle='--',
                           linewidth=1.0, alpha=0.8,
                           label=f'err thresh ({tolerance:.0%}): {threshold_lambda:.3g}')

    for j in range(total_panels, len(axes_flat)):
        axes_flat[j].axis('off')

    for ax in axes[-1, :]:
        ax.set_xlabel('lambda_l1', fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel('H / H_theoretical', fontsize=9)

    modes = sorted({e['mode'] for e in entries if e['mode']})
    mode_str = modes[0] if len(modes) == 1 else ','.join(modes)
    fig.suptitle(
        f'Normalized entropy vs lambda_1 | {sae_label(layer, report)} | '
        f'mode={mode_str} | s={s}, L={L}',
        fontsize=12, y=1.0,
    )

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if threshold_lambda is not None:
        handles.append(plt.Line2D([], [], color='red', linestyle='--',
                                  linewidth=1.0))
        labels.append(f'err thresh ({tolerance:.0%}): {threshold_lambda:.3g}')
    fig.legend(handles, labels, loc='lower center',
               ncol=len(labels), bbox_to_anchor=(0.5, -0.02), fontsize=10)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'  layer {layer}: {len(entries)} artifacts, '
          f'lambda range [{lambdas.min():.3g}, {lambdas.max():.3g}] '
          f'-> {outfile}')


def run_aggregate(args):
    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.is_dir():
        raise SystemExit(f'--artifacts_dir is not a directory: {artifacts_dir}')
    files = sorted(artifacts_dir.glob('*.sae_eval.pt'))
    if not files:
        raise SystemExit(f'No *.sae_eval.pt files in {artifacts_dir}')

    entries = agg_load_entries(files)
    if not entries:
        raise SystemExit(
            'No artifacts with H_bar_*_norm found. Re-run scripts/sae_eval/run.py '
            'with --with-entropy (or --with-all).'
        )

    modes = {e['mode'] for e in entries}
    if 'mean_pooled' in modes and len(modes) > 1:
        raise SystemExit(
            f'Mixed activation modes in {artifacts_dir}: {sorted(modes)}. '
            'mean_pooled artifacts must be plotted separately; re-run on a '
            'directory containing a single mode.'
        )
    is_meanpool = (modes == {'mean_pooled'})

    by_layer = defaultdict(list)
    for e in entries:
        by_layer[e['layer_id']].append(e)

    prefix = args.outfile_prefix or str(artifacts_dir / 'entropy_lambda')
    csv_path = args.csv_path or str(artifacts_dir / 'sweep_metrics.csv')

    print(f'Loaded {len(entries)} artifacts across {len(by_layer)} layer(s).')
    for layer in sorted(by_layer):
        layer_entries = by_layer[layer]
        modes_here = sorted({e['mode'] for e in layer_entries if e['mode']})
        mode_for_csv = modes_here[0] if len(modes_here) == 1 else ''
        threshold_lambda = find_threshold_lambda(
            csv_path, layer, mode_for_csv, args.err_tolerance,
        )
        if is_meanpool:
            outfile = f'{prefix}_layer{layer}_meanpool.png'
            agg_plot_meanpool_layer(layer, layer_entries, outfile, args.xlim,
                                    threshold_lambda, args.err_tolerance,
                                    args.report_notation)
        else:
            outfile = f'{prefix}_layer{layer}.png'
            agg_plot_layer(layer, layer_entries, outfile, args.xlim,
                           threshold_lambda, args.err_tolerance,
                           args.report_notation, args.plot_avg)


# ===========================================================================
# min mode  (former min_entropy_diag.py)
# ===========================================================================

def min_process_artifact(art, candidates='same_level'):
    """Per-position min-conditional-entropy diagnostics for one artifact.

    candidates: 'same_level' (matched-level latents only) or 'whole_tree' (every
    group). The min curves are always computed for BOTH candidate sets so the
    plot can draw parent / same-level / whole-tree together.
    """
    rhm = art['rhm']
    s = int(rhm['s'])
    L = int(rhm['L'])
    layer_id = int(art['layer_id'])

    pooled = (art.get('mode') == 'mean_pooled')
    matched_level = 0 if pooled else (L - 1 - layer_id)

    index_layout = art['index_layout']
    H_per_feature = art['H_per_feature']          # [num_groups, P, F]
    firing_rate = art['firing_rate']              # [P, F]
    baseline_mean = art['baseline_mean']          # [P, F]
    decoder_norms = art['decoder_norms']          # [F]
    token_positions = art['token_positions']      # [P]
    H_theoretical = art['H_theoretical']          # {(level,pos)->float}

    P = H_per_feature.shape[1]

    group_index = {(int(g['level']), int(g['position'])): idx
                   for idx, g in enumerate(index_layout)}

    same_level_groups = [(int(g['level']), int(g['position']))
                         for g in index_layout if int(g['level']) == matched_level]
    if not same_level_groups:
        raise ValueError('no index_layout groups at matched_level=%d' % matched_level)
    whole_tree_groups = [(int(g['level']), int(g['position'])) for g in index_layout]
    cand_sets = {'same_level': same_level_groups, 'whole_tree': whole_tree_groups}
    cand_idxs = {name: [group_index[k] for k in groups]
                 for name, groups in cand_sets.items()}

    def _href_tensor(groups):
        vals = []
        for g in groups:
            h = H_theoretical.get(g, float('nan'))
            vals.append(h if (isinstance(h, (int, float)) and h > 0
                              and math.isfinite(h)) else float('nan'))
        return torch.tensor(vals, dtype=torch.float64)
    cand_href = {name: _href_tensor(groups)
                 for name, groups in cand_sets.items()}

    if candidates not in cand_sets:
        raise ValueError('unknown candidates %r' % candidates)
    level_groups = cand_sets[candidates]
    level_g_idxs = cand_idxs[candidates]

    out = {
        'lambda_l1': float(art.get('lambda_l1') or 0.0),
        'layer_id': layer_id, 'P': P, 's': s, 'L': L,
        'mode': art.get('mode', ''), 'pooled': pooled,
        'matched_level': matched_level,
        'positions': [], 'parent_group': [],
        'stored_H_bar_fire': [],
    }
    for sc in SCHEMES:
        out['H_bar_%s' % sc] = []
        out['H_bar_%s_norm' % sc] = []
        out['gap_%s' % sc] = []
        for cand in cand_sets:
            out['H_bar_min_%s_%s' % (sc, cand)] = []
            out['H_bar_min_%s_%s_norm' % (sc, cand)] = []
        out['leak_frac_%s' % sc] = []
        out['leak_to_%s' % sc] = []
        out['target_dist_%s' % sc] = []
        out['target_dist_wt_%s' % sc] = []

    stored_fire = art.get('H_bar_fire')

    for p_idx in range(P):
        p_real = int(token_positions[p_idx].item())
        if pooled:
            parent_key = (0, 0)
        else:
            matched_j = p_real // (s ** (1 + layer_id))
            parent_key = (matched_level, matched_j)
        parent_gidx = group_index.get(parent_key)
        out['positions'].append(p_real)
        out['parent_group'].append(parent_key)
        out['stored_H_bar_fire'].append(
            float(stored_fire[p_idx]) if stored_fire is not None else float('nan'))

        H_min = {}
        argmins = {}
        for cand, gidxs in cand_idxs.items():
            H_cand = H_per_feature[gidxs, p_idx, :]              # [G, F]
            mn, am = torch_nanmin(H_cand, dim=0)                 # [F], [F]
            H_min[cand] = mn
            argmins[cand] = am
        argmin = argmins[candidates]
        H_parent = H_per_feature[parent_gidx, p_idx, :]          # [F]

        H_ref = H_theoretical.get(parent_key, float('nan'))
        ref_ok = (H_ref > 0) and math.isfinite(H_ref)

        parent_row = level_g_idxs.index(parent_gidx)

        for sc in SCHEMES:
            w = weights_for_scheme(sc, firing_rate[p_idx], baseline_mean[p_idx],
                                   decoder_norms)
            h_par = weighted_aggregate(H_parent, w)
            out['H_bar_%s' % sc].append(float(h_par))
            if ref_ok:
                out['H_bar_%s_norm' % sc].append(float(h_par) / H_ref)
            else:
                out['H_bar_%s_norm' % sc].append(float('nan'))
            for cand in cand_sets:
                h_min = weighted_aggregate(H_min[cand], w)
                out['H_bar_min_%s_%s' % (sc, cand)].append(float(h_min))
                href_feat = cand_href[cand][argmins[cand]]        # [F]
                h_min_norm = weighted_aggregate(
                    H_min[cand] / href_feat.clamp_min(1e-30), w)
                out['H_bar_min_%s_%s_norm' % (sc, cand)].append(float(h_min_norm))
            out['gap_%s' % sc].append(
                float(h_par) - out['H_bar_min_%s_%s' % (sc, candidates)][-1])

            live = torch.isfinite(H_min[candidates]) & (w > 0)
            w_live = torch.where(live, w.double(), torch.zeros_like(w.double()))
            tot = w_live.sum()
            leaks = live & (argmin != parent_row)
            leak_w = torch.where(leaks, w.double(), torch.zeros_like(w.double()))
            frac = float(leak_w.sum() / tot.clamp_min(1e-30)) if tot > 0 else float('nan')
            out['leak_frac_%s' % sc].append(frac)

            votes = torch.zeros(len(level_g_idxs), dtype=torch.float64)
            if live.any():
                votes.scatter_add_(0, argmin[live], w.double()[live])
            if tot > 0:
                votes = votes / tot
            dist = {level_groups[r]: float(votes[r])
                    for r in range(len(level_g_idxs)) if votes[r] > 0}
            out['target_dist_%s' % sc].append(dist)

            wt_groups = cand_sets['whole_tree']
            wt_argmin = argmins['whole_tree']
            wt_live = torch.isfinite(H_min['whole_tree']) & (w > 0)
            wt_w_live = torch.where(wt_live, w.double(),
                                    torch.zeros_like(w.double()))
            wt_tot = wt_w_live.sum()
            wt_votes = torch.zeros(len(wt_groups), dtype=torch.float64)
            if wt_live.any():
                wt_votes.scatter_add_(0, wt_argmin[wt_live],
                                      w.double()[wt_live])
            if wt_tot > 0:
                wt_votes = wt_votes / wt_tot
            wt_dist = {wt_groups[r]: float(wt_votes[r])
                       for r in range(len(wt_groups)) if wt_votes[r] > 0}
            out['target_dist_wt_%s' % sc].append(wt_dist)

            if leaks.any():
                votes_np = votes.clone()
                votes_np[parent_row] = 0.0
                dom_row = int(torch.argmax(votes_np).item())
                out['leak_to_%s' % sc].append(level_groups[dom_row])
            else:
                out['leak_to_%s' % sc].append(None)

    return out


def min_write_csv(path, diags):
    cols = ['file', 'lambda_l1', 'layer_id', 'matched_level', 'position',
            'parent_group']
    for sc in SCHEMES:
        cols += ['H_bar_%s' % sc,
                 'H_bar_min_%s_same_level' % sc,
                 'H_bar_min_%s_whole_tree' % sc,
                 'gap_%s' % sc, 'H_bar_%s_norm' % sc,
                 'H_bar_min_%s_same_level_norm' % sc,
                 'H_bar_min_%s_whole_tree_norm' % sc,
                 'leak_frac_%s' % sc, 'leak_to_%s' % sc]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for fname, d in diags:
            for p_idx in range(d['P']):
                row = [fname, d['lambda_l1'], d['layer_id'], d['matched_level'],
                       d['positions'][p_idx], '%d,%d' % d['parent_group'][p_idx]]
                for sc in SCHEMES:
                    lt = d['leak_to_%s' % sc][p_idx]
                    lt_str = '' if lt is None else '%d,%d' % lt
                    row += [d['H_bar_%s' % sc][p_idx],
                            d['H_bar_min_%s_same_level' % sc][p_idx],
                            d['H_bar_min_%s_whole_tree' % sc][p_idx],
                            d['gap_%s' % sc][p_idx],
                            d['H_bar_%s_norm' % sc][p_idx],
                            d['H_bar_min_%s_same_level_norm' % sc][p_idx],
                            d['H_bar_min_%s_whole_tree_norm' % sc][p_idx],
                            d['leak_frac_%s' % sc][p_idx], lt_str]
                w.writerow(row)


def min_write_target_dist_csv(path, diags):
    """Long-format distribution: one row per (file, position, scheme, target)."""
    cols = ['file', 'lambda_l1', 'layer_id', 'matched_level', 'position',
            'parent_group', 'scheme', 'target_level', 'target_position',
            'is_parent', 'share']
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for fname, d in diags:
            for p_idx in range(d['P']):
                pg = d['parent_group'][p_idx]
                for sc in SCHEMES:
                    dist = d['target_dist_%s' % sc][p_idx]
                    for (lvl, pos), share in sorted(dist.items()):
                        w.writerow([fname, d['lambda_l1'], d['layer_id'],
                                    d['matched_level'], d['positions'][p_idx],
                                    '%d,%d' % pg, sc, lvl, pos,
                                    int((lvl, pos) == pg), share])


def min_make_level_dist_plot(diags, out_prefix, scheme='fire', xlim=None,
                             report=False, threshold_lambda=None, tolerance=0.01):
    """Per-position share of the WHOLE-TREE (unconstrained) argmin-target
    distribution by RHM level vs lambda. Curves sum to ~1."""
    diags_sorted = sorted(diags, key=lambda fd: fd[1]['lambda_l1'])
    lambdas = [d['lambda_l1'] for _, d in diags_sorted]
    P = diags_sorted[0][1]['P']
    L = diags_sorted[0][1]['L']

    def _lvl(code_level):
        return report_level(code_level, L) if report else code_level

    dkey = 'target_dist_wt_%s' % scheme

    levels = sorted({gl for _, d in diags_sorted
                     for p in range(P)
                     for (gl, _gp) in d[dkey][p]})

    _PALETTE = ('#4daf4a', '#984ea3', '#a65628', '#f781bf', '#17becf',
                '#bcbd22', '#000000')
    level_color = {lvl: _PALETTE[i % len(_PALETTE)]
                   for i, lvl in enumerate(levels)}
    pooled = diags_sorted[0][1].get('pooled', False)

    ncol = min(4, P)
    nrow = math.ceil(P / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                             squeeze=False)
    axes = axes.flatten()

    for p_idx in range(P):
        ax = axes[p_idx]
        pg = diags_sorted[0][1]['parent_group'][p_idx]
        for lvl in levels:
            ys = []
            for _, d in diags_sorted:
                dist = d[dkey][p_idx]
                if not dist:
                    ys.append(float('nan'))
                else:
                    ys.append(sum(sh for (gl, _gp), sh in dist.items()
                                  if gl == lvl))
            ax.plot(lambdas, ys, 'o-', label='level %d' % _lvl(lvl),
                    color=level_color[lvl])
        ax.set_xscale('log')
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_ylim(-0.05, 1.05)
        if threshold_lambda is not None:
            ax.axvline(threshold_lambda, color='black', linestyle='--',
                       linewidth=0.8, alpha=0.7,
                       label=(thr_label(tolerance) if p_idx == 0 else None))
        ax.set_title('pooled readout (root %d,%d)' % (_lvl(pg[0]), pg[1])
                     if pooled
                     else 'pos %d (parent %d,%d)' % (p_idx, _lvl(pg[0]), pg[1]),
                     fontsize=9)
        ax.set_xlabel(r'$\lambda$', fontsize=12)
        ax.set_ylabel('weighted feature fraction', fontsize=12)
        if p_idx == 0:
            ax.legend(fontsize=7, loc='upper right')

    for k in range(P, len(axes)):
        axes[k].axis('off')

    weight_desc = 'unweighted' if scheme == 'unweighted' else '%s weights' % scheme
    fig.suptitle('Unconstrained argmin-target level distribution by position '
                 '(%s)' % weight_desc)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path('%s_leveldist_%s.png' % (out_prefix, scheme))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def min_make_plot(diags, out_prefix, scheme='fire', xlim=None, report=False,
                  plot_avg=False, threshold_lambda=None, tolerance=0.01):
    """Per-position conditional entropy vs lambda: parent / level / unconstrained."""
    diags_sorted = sorted(diags, key=lambda fd: fd[1]['lambda_l1'])
    lambdas = [d['lambda_l1'] for _, d in diags_sorted]
    P = diags_sorted[0][1]['P']
    L = diags_sorted[0][1]['L']

    def _lvl(code_level):
        return report_level(code_level, L) if report else code_level
    total_panels = P + (1 if plot_avg else 0)
    ncol = min(4, total_panels)
    nrow = math.ceil(total_panels / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                             squeeze=False)
    axes = axes.flatten()

    pooled = diags_sorted[0][1].get('pooled', False)
    par_key = 'H_bar_%s_norm' % scheme
    same_key = 'H_bar_min_%s_same_level_norm' % scheme
    whole_key = 'H_bar_min_%s_whole_tree_norm' % scheme
    if pooled:
        curves = [(par_key, 'o-', 'C0', 'root constrained'),
                  (whole_key, 'D-', 'C1', 'no constraint')]
    else:
        curves = [(par_key, 'o-', 'C0', 'parent constrained'),
                  (same_key, 's-', 'C3', 'level constrained'),
                  (whole_key, 'D-', 'C1', 'no constraint')]

    for p_idx in range(P):
        ax = axes[p_idx]
        pg = diags_sorted[0][1]['parent_group'][p_idx]
        for key, style, color, label in curves:
            ys = [d[key][p_idx] for _, d in diags_sorted]
            ax.plot(lambdas, ys, style, color=color, label=label)
        ax.set_xscale('log')
        if xlim:
            ax.set_xlim(*xlim)
        if threshold_lambda is not None:
            ax.axvline(threshold_lambda, color='black', linestyle='--',
                       linewidth=0.8, alpha=0.7,
                       label=(thr_label(tolerance) if p_idx == 0 else None))
        ax.set_title('pooled readout (root %d,%d)' % (_lvl(pg[0]), pg[1])
                     if pooled
                     else 'pos %d (parent %d,%d)' % (p_idx, _lvl(pg[0]), pg[1]),
                     fontsize=9)
        ax.set_xlabel(r'$\lambda$', fontsize=12)
        ax.set_ylabel('H_norm', fontsize=12)
        if p_idx == 0:
            ax.legend(fontsize=7, loc='upper left')

    if plot_avg:
        ax = axes[P]
        for key, style, color, label in curves:
            avg = [np.nanmean([d[key][p] for p in range(P)])
                   for _, d in diags_sorted]
            ax.plot(lambdas, avg, style, color=color, label=label)
        ax.set_xscale('log')
        if xlim:
            ax.set_xlim(*xlim)
        if threshold_lambda is not None:
            ax.axvline(threshold_lambda, color='black', linestyle='--',
                       linewidth=0.8, alpha=0.7, label=thr_label(tolerance))
        ax.set_title('avg over positions', fontsize=9)
        ax.set_xlabel(r'$\lambda$', fontsize=12)
        ax.set_ylabel('H_norm', fontsize=12)
        ax.legend(fontsize=7, loc='upper left')

    for k in range(total_panels, len(axes)):
        axes[k].axis('off')

    weight_desc = 'unweighted' if scheme == 'unweighted' else '%s weights' % scheme
    fig.suptitle('Parent / level / unconstrained conditional entropy '
                 '(%s)' % weight_desc)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path('%s_%s.png' % (out_prefix, scheme))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def run_min(args):
    adir = Path(args.artifacts_dir)
    files = sorted(adir.glob('*.sae_eval.pt'))
    if not files:
        raise SystemExit('no *.sae_eval.pt files in %s' % adir)

    tag = '' if args.candidates == 'same_level' else '_' + args.candidates
    out_csv = (Path(args.out_csv) if args.out_csv
               else adir / ('min_entropy_diag%s.csv' % tag))
    out_prefix = (args.out_plot_prefix
                  if args.out_plot_prefix
                  else str(adir.parent / 'analysis_plots'
                           / ('min_entropy_diag%s' % tag)))

    diags = []
    skipped = []
    for f in files:
        art = torch.load(f, map_location='cpu', weights_only=False)
        needed = ('H_per_feature', 'index_layout', 'firing_rate',
                  'baseline_mean', 'decoder_norms', 'token_positions',
                  'H_theoretical', 'rhm', 'layer_id')
        if any(k not in art or art[k] is None for k in needed):
            skipped.append(f.name)
            continue
        d = min_process_artifact(art, candidates=args.candidates)
        parent_min_sanity(art, d)
        diags.append((f.name, d))

    if not diags:
        raise SystemExit('no usable artifacts (skipped %d missing keys)'
                         % len(skipped))

    min_write_csv(out_csv, diags)
    print('wrote %s (%d artifacts x %d positions)'
          % (out_csv, len(diags), diags[0][1]['P']))
    dist_csv = out_csv.with_name(out_csv.stem + '_target_dist.csv')
    min_write_target_dist_csv(dist_csv, diags)
    print('wrote %s' % dist_csv)
    if skipped:
        print('skipped %d artifacts missing required keys' % len(skipped))

    csv_path = args.csv_path or str(adir / 'sweep_metrics.csv')
    layers = {d['layer_id'] for _, d in diags}
    modes = {d['mode'] for _, d in diags if d['mode']}
    if len(layers) > 1 or len(modes) > 1:
        print('threshold: mixed layers %s / modes %s in sweep; using first '
              'of each' % (sorted(layers), sorted(modes)))
    layer0 = diags[0][1]['layer_id']
    mode0 = diags[0][1]['mode']
    threshold_lambda = find_threshold_lambda(
        csv_path, layer0, mode0, args.err_tolerance)

    for sc in args.plot_schemes:
        p = min_make_plot(diags, out_prefix, scheme=sc, xlim=args.xlim,
                          report=args.report_notation, plot_avg=args.plot_avg,
                          threshold_lambda=threshold_lambda,
                          tolerance=args.err_tolerance)
        print('wrote %s' % p)
        p = min_make_level_dist_plot(diags, out_prefix, scheme=sc, xlim=args.xlim,
                                     report=args.report_notation,
                                     threshold_lambda=threshold_lambda,
                                     tolerance=args.err_tolerance)
        print('wrote %s' % p)

    target = min(diags, key=lambda fd: abs(fd[1]['lambda_l1'] - 0.01))
    fn, d = target
    print('\nsummary at lambda_1=%.4g (%s):' % (d['lambda_l1'], fn))
    print('%4s %10s %9s %9s %9s %7s %8s' %
          ('pos', 'parent', 'H_par', 'H_lvl', 'H_tree', 'gap', 'leak'))
    for p_idx in range(d['P']):
        pg = d['parent_group'][p_idx]
        lt = d['leak_to_fire'][p_idx]
        lt_s = '' if lt is None else ' ->%d,%d' % lt
        print('%4d %10s %9.4f %9.4f %9.4f %7.4f %8.3f%s' %
              (d['positions'][p_idx], '%d,%d' % pg,
               d['H_bar_fire'][p_idx],
               d['H_bar_min_fire_same_level'][p_idx],
               d['H_bar_min_fire_whole_tree'][p_idx],
               d['gap_fire'][p_idx], d['leak_frac_fire'][p_idx], lt_s))


# ===========================================================================
# splitcheck mode  (former feature_splitting_diag.py)
# ===========================================================================

def split_process_artifact(art, leak_norm, alpha, collect_ratios=False):
    """Per-position diagnostics for one artifact, with the split check.

    Superset of min_process_artifact (whole_tree mode). Adds the split-checked
    aggregate, the reassigned level distribution, and the reassigned fraction.
    collect_ratios stashes the raw per-feature ratios for the alpha sweep.
    """
    core = per_feature_labels(art, leak_norm, alpha)

    firing_rate = art['firing_rate']              # [P, F]
    baseline_mean = art['baseline_mean']          # [P, F]
    decoder_norms = art['decoder_norms']          # [F]

    P = core['P']
    cand_sets = core['cand_sets']
    cand_href = core['cand_href']

    out = {
        'lambda_l1': core['lambda_l1'],
        'layer_id': core['layer_id'], 'P': P, 's': core['s'], 'L': core['L'],
        'mode': core['mode'], 'pooled': core['pooled'],
        'matched_level': core['matched_level'],
        'positions': list(core['positions']),
        'parent_group': list(core['parent_group']),
        'stored_H_bar_fire': [],
    }
    for sc in SCHEMES:
        out['H_bar_%s' % sc] = []
        out['H_bar_%s_norm' % sc] = []
        for cand in cand_sets:
            out['H_bar_min_%s_%s' % (sc, cand)] = []
            out['H_bar_min_%s_%s_norm' % (sc, cand)] = []
        out['H_bar_splitcheck_%s_norm' % sc] = []
        out['reassign_frac_%s' % sc] = []
        out['target_dist_wt_%s' % sc] = []
        out['target_dist_reassigned_%s' % sc] = []
        if collect_ratios:
            out['ratio_%s' % sc] = []

    stored_fire = art.get('H_bar_fire')

    for p_idx in range(P):
        rec = core['per_position'][p_idx]
        out['stored_H_bar_fire'].append(
            float(stored_fire[p_idx]) if stored_fire is not None else float('nan'))

        H_min = rec['H_min']
        argmins = rec['argmin']
        H_parent = rec['H_parent']
        H_ref = rec['parent_href']
        ref_ok = math.isfinite(H_ref) and (H_ref > 0)

        child_level = rec['child_level']
        ratio = rec['ratio']
        reassign = rec['reassign']
        final_entropy = rec['final_entropy']
        final_href = rec['final_href']
        final_level = rec['final_level']
        can_reassign = child_level >= 1

        for sc in SCHEMES:
            w = weights_for_scheme(sc, firing_rate[p_idx], baseline_mean[p_idx],
                                   decoder_norms)
            h_par = weighted_aggregate(H_parent, w)
            out['H_bar_%s' % sc].append(float(h_par))
            out['H_bar_%s_norm' % sc].append(
                float(h_par) / H_ref if ref_ok else float('nan'))

            for cand in cand_sets:
                h_min = weighted_aggregate(H_min[cand], w)
                out['H_bar_min_%s_%s' % (sc, cand)].append(float(h_min))
                href_feat = cand_href[cand][argmins[cand]]
                h_min_norm = weighted_aggregate(
                    H_min[cand] / href_feat.clamp_min(1e-30), w)
                out['H_bar_min_%s_%s_norm' % (sc, cand)].append(float(h_min_norm))

            h_split = weighted_aggregate(
                final_entropy / final_href.clamp_min(1e-30), w)
            out['H_bar_splitcheck_%s_norm' % sc].append(float(h_split))

            live = torch.isfinite(H_min['whole_tree']) & (w > 0)
            w_live = torch.where(live, w.double(), torch.zeros_like(w.double()))
            tot = w_live.sum()
            re_w = torch.where(live & reassign, w.double(),
                               torch.zeros_like(w.double()))
            out['reassign_frac_%s' % sc].append(
                float(re_w.sum() / tot.clamp_min(1e-30)) if tot > 0 else float('nan'))

            if collect_ratios:
                eligible = (live & can_reassign & torch.isfinite(ratio))
                out['ratio_%s' % sc].append({
                    'ratio': ratio.cpu().numpy().copy(),
                    'w': w.double().cpu().numpy().copy(),
                    'live': live.cpu().numpy().copy(),
                    'eligible': eligible.cpu().numpy().copy()})

            out['target_dist_wt_%s' % sc].append(
                _level_votes(child_level, w, live, tot))
            out['target_dist_reassigned_%s' % sc].append(
                _level_votes(final_level, w, live, tot))

    return out


def _level_votes(level_per_feature, w, live, tot):
    """{level -> weighted share} over live features (sums to ~1 when tot>0)."""
    if tot <= 0 or not live.any():
        return {}
    lv = level_per_feature[live].long()
    ww = w.double()[live]
    votes = {}
    for level in torch.unique(lv).tolist():
        votes[int(level)] = float(ww[lv == level].sum() / tot)
    return votes


def split_write_csv(path, diags):
    cols = ['file', 'lambda_l1', 'layer_id', 'matched_level', 'position',
            'parent_group']
    for sc in SCHEMES:
        cols += ['H_bar_%s' % sc,
                 'H_bar_min_%s_same_level' % sc,
                 'H_bar_min_%s_whole_tree' % sc,
                 'H_bar_%s_norm' % sc,
                 'H_bar_min_%s_same_level_norm' % sc,
                 'H_bar_min_%s_whole_tree_norm' % sc,
                 'H_bar_splitcheck_%s_norm' % sc,
                 'reassign_frac_%s' % sc]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for fname, d in diags:
            for p_idx in range(d['P']):
                row = [fname, d['lambda_l1'], d['layer_id'], d['matched_level'],
                       d['positions'][p_idx], '%d,%d' % d['parent_group'][p_idx]]
                for sc in SCHEMES:
                    row += [d['H_bar_%s' % sc][p_idx],
                            d['H_bar_min_%s_same_level' % sc][p_idx],
                            d['H_bar_min_%s_whole_tree' % sc][p_idx],
                            d['H_bar_%s_norm' % sc][p_idx],
                            d['H_bar_min_%s_same_level_norm' % sc][p_idx],
                            d['H_bar_min_%s_whole_tree_norm' % sc][p_idx],
                            d['H_bar_splitcheck_%s_norm' % sc][p_idx],
                            d['reassign_frac_%s' % sc][p_idx]]
                w.writerow(row)


def write_leak_csv(path, leak_rows):
    cols = ['child_level', 'child_position', 'child_value', 'parent_level',
            'parent_position', 'P_child', 'leak_nats', 'parent_H_theoretical',
            'leak_norm']
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in leak_rows:
            w.writerow(r)


def split_write_target_dist_csv(path, diags):
    """Long format, one row per (file, position, scheme, variant, target_level)."""
    cols = ['file', 'lambda_l1', 'layer_id', 'matched_level', 'position',
            'parent_group', 'scheme', 'variant', 'target_level', 'share']
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for fname, d in diags:
            for p_idx in range(d['P']):
                pg = d['parent_group'][p_idx]
                for sc in SCHEMES:
                    for variant, key in (('unconstrained', 'target_dist_wt_%s'),
                                         ('reassigned', 'target_dist_reassigned_%s')):
                        dist = d[key % sc][p_idx]
                        for level, share in sorted(dist.items()):
                            w.writerow([fname, d['lambda_l1'], d['layer_id'],
                                        d['matched_level'], d['positions'][p_idx],
                                        '%d,%d' % pg, sc, variant, level, share])


def split_make_level_dist_plot(diags, out_prefix, variant, scheme='fire',
                               xlim=None, report=False, threshold_lambda=None,
                               tolerance=0.01):
    """Per-position level-share distribution vs lambda for one variant
    ('unconstrained' whole-tree argmin, or 'reassigned' split-checked final)."""
    dkey = {'unconstrained': 'target_dist_wt_%s' % scheme,
            'reassigned': 'target_dist_reassigned_%s' % scheme}[variant]
    diags_sorted = sorted(diags, key=lambda fd: fd[1]['lambda_l1'])
    lambdas = [d['lambda_l1'] for _, d in diags_sorted]
    P = diags_sorted[0][1]['P']
    L = diags_sorted[0][1]['L']

    def _lvl(code_level):
        return report_level(code_level, L) if report else code_level

    levels = sorted({lvl for _, d in diags_sorted for p in range(P)
                     for lvl in d[dkey][p]})
    level_color = {lvl: LEVEL_PALETTE[i % len(LEVEL_PALETTE)]
                   for i, lvl in enumerate(levels)}
    pooled = diags_sorted[0][1].get('pooled', False)

    ncol = min(4, P)
    nrow = math.ceil(P / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                             squeeze=False)
    axes = axes.flatten()

    for p_idx in range(P):
        ax = axes[p_idx]
        pg = diags_sorted[0][1]['parent_group'][p_idx]
        for lvl in levels:
            ys = []
            for _, d in diags_sorted:
                dist = d[dkey][p_idx]
                ys.append(dist.get(lvl, float('nan')) if dist else float('nan'))
            ax.plot(lambdas, ys, 'o-', label='level %d' % _lvl(lvl),
                    color=level_color[lvl])
        ax.set_xscale('log')
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_ylim(-0.05, 1.05)
        if threshold_lambda is not None:
            ax.axvline(threshold_lambda, color='black', linestyle='--',
                       linewidth=0.8, alpha=0.7,
                       label=(thr_label(tolerance) if p_idx == 0 else None))
        ax.set_title('pooled readout (root %d,%d)' % (_lvl(pg[0]), pg[1])
                     if pooled
                     else 'pos %d (parent %d,%d)' % (p_idx, _lvl(pg[0]), pg[1]),
                     fontsize=9)
        ax.set_xlabel(r'$\lambda$', fontsize=12)
        ax.set_ylabel('weighted feature fraction', fontsize=12)
        if p_idx == 0:
            ax.legend(fontsize=7, loc='upper left')

    for k in range(P, len(axes)):
        axes[k].axis('off')

    title = ('Unconstrained argmin' if variant == 'unconstrained'
             else 'Split-check reassigned')
    weight_desc = 'unweighted' if scheme == 'unweighted' else '%s weights' % scheme
    fig.suptitle('%s target level distribution by position (%s)'
                 % (title, weight_desc))
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path('%s_leveldist_%s_%s.png' % (out_prefix, variant, scheme))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def split_make_plot(diags, out_prefix, scheme='fire', xlim=None, report=False,
                    plot_avg=False, threshold_lambda=None, tolerance=0.01):
    """Per-position weighted entropy eta-bar vs lambda: parent / level /
    unconstrained whole-tree min BEFORE reassignment (orange)."""
    diags_sorted = sorted(diags, key=lambda fd: fd[1]['lambda_l1'])
    lambdas = [d['lambda_l1'] for _, d in diags_sorted]
    P = diags_sorted[0][1]['P']
    L = diags_sorted[0][1]['L']

    def _lvl(code_level):
        return report_level(code_level, L) if report else code_level
    total_panels = P + (1 if plot_avg else 0)
    ncol = min(4, total_panels)
    nrow = math.ceil(total_panels / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                             squeeze=False)
    axes = axes.flatten()

    pooled = diags_sorted[0][1].get('pooled', False)
    par_key = 'H_bar_%s_norm' % scheme
    same_key = 'H_bar_min_%s_same_level_norm' % scheme
    whole_key = 'H_bar_min_%s_whole_tree_norm' % scheme
    if pooled:
        curves = [(par_key, 'o-', 'C0', 'root constrained'),
                  (whole_key, '^-', 'orange', 'no constraint')]
    else:
        curves = [(par_key, 'o-', 'C0', 'parent constrained'),
                  (same_key, 's-', 'C3', 'level constrained'),
                  (whole_key, '^-', 'orange', 'no constraint')]

    for p_idx in range(P):
        ax = axes[p_idx]
        pg = diags_sorted[0][1]['parent_group'][p_idx]
        for key, style, color, label in curves:
            ys = [d[key][p_idx] for _, d in diags_sorted]
            ax.plot(lambdas, ys, style, color=color, label=label)
        ax.set_xscale('log')
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_ylim(*ENTROPY_YLIM)
        if threshold_lambda is not None:
            ax.axvline(threshold_lambda, color='black', linestyle='--',
                       linewidth=0.8, alpha=0.7,
                       label=(thr_label(tolerance) if p_idx == 0 else None))
        ax.set_title('pooled readout (root %d,%d)' % (_lvl(pg[0]), pg[1])
                     if pooled
                     else 'pos %d (parent %d,%d)' % (p_idx, _lvl(pg[0]), pg[1]),
                     fontsize=9)
        ax.set_xlabel(r'$\lambda$', fontsize=12)
        ax.set_ylabel(r'$\bar{\eta}$', fontsize=14)
        ax.tick_params(axis='y', labelsize=10)
        if p_idx == 0:
            ax.legend(fontsize=7, loc='upper left')

    if plot_avg:
        ax = axes[P]
        for key, style, color, label in curves:
            avg = [np.nanmean([d[key][p] for p in range(P)])
                   for _, d in diags_sorted]
            ax.plot(lambdas, avg, style, color=color, label=label)
        ax.set_xscale('log')
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_ylim(*ENTROPY_YLIM)
        if threshold_lambda is not None:
            ax.axvline(threshold_lambda, color='black', linestyle='--',
                       linewidth=0.8, alpha=0.7, label=thr_label(tolerance))
        ax.set_title('avg over positions', fontsize=9)
        ax.set_xlabel(r'$\lambda$', fontsize=12)
        ax.set_ylabel(r'$\bar{\eta}$', fontsize=14)
        ax.tick_params(axis='y', labelsize=10)
        ax.legend(fontsize=7, loc='upper left')

    for k in range(total_panels, len(axes)):
        axes[k].axis('off')

    fig.suptitle(r'Parent / level / unconstrained weighted entropy $\bar{\eta}$')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path('%s_%s.png' % (out_prefix, scheme))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def splitcheck_sanity(diag, scheme='fire', tol=1e-6):
    """Assert the split-checked aggregate is at least the unconstrained min, and
    each level distribution sums to ~1 when live."""
    for p_idx in range(diag['P']):
        whole = diag['H_bar_min_%s_whole_tree_norm' % scheme][p_idx]
        split = diag['H_bar_splitcheck_%s_norm' % scheme][p_idx]
        if math.isfinite(whole) and math.isfinite(split):
            assert whole <= split + tol, (
                'pos %d: split %.6f < whole_tree %.6f' % (p_idx, split, whole))
        for key in ('target_dist_wt_%s' % scheme,
                    'target_dist_reassigned_%s' % scheme):
            dist = diag[key][p_idx]
            if dist:
                total = sum(dist.values())
                assert abs(total - 1.0) < 1e-4, (
                    'pos %d %s: level shares sum to %.6f' % (p_idx, key, total))


def run_splitcheck(args):
    adir = Path(args.artifacts_dir)
    files = sorted(adir.glob('*.sae_eval.pt'))
    if not files:
        raise SystemExit('no *.sae_eval.pt files in %s' % adir)

    out_csv = (Path(args.out_csv) if args.out_csv
               else adir / 'feature_splitting_diag.csv')
    out_prefix = (args.out_plot_prefix
                  if args.out_plot_prefix
                  else str(adir.parent / 'analysis_plots'
                           / 'feature_splitting_diag'))

    needed = ('H_per_feature', 'index_layout', 'firing_rate',
              'baseline_mean', 'decoder_norms', 'token_positions',
              'H_theoretical', 'rhm', 'layer_id',
              'firing_count', 'joint_fire_count', 'targets')

    leak_norm = None
    leak_rows = None
    diags = []
    skipped = []
    for f in files:
        art = torch.load(f, map_location='cpu', weights_only=False)
        if any(k not in art or art[k] is None for k in needed):
            skipped.append(f.name)
            continue
        if leak_norm is None:
            leak_norm, leak_rows, src = resolve_leak_table(art, adir)
            print('resolved RHM rules (source: %s); value-specific leak table '
                  'over %d (child cell, value) entries' % (src, len(leak_norm)))
        d = split_process_artifact(art, leak_norm, args.alpha)
        parent_min_sanity(art, d)
        splitcheck_sanity(d)
        diags.append((f.name, d))

    if not diags:
        raise SystemExit('no usable artifacts (skipped %d missing keys)'
                         % len(skipped))

    split_write_csv(out_csv, diags)
    print('wrote %s (%d artifacts x %d positions)'
          % (out_csv, len(diags), diags[0][1]['P']))
    leak_csv = out_csv.with_name(out_csv.stem + '_leak_table.csv')
    write_leak_csv(leak_csv, leak_rows)
    print('wrote %s' % leak_csv)
    dist_csv = out_csv.with_name(out_csv.stem + '_target_dist.csv')
    split_write_target_dist_csv(dist_csv, diags)
    print('wrote %s' % dist_csv)
    if skipped:
        print('skipped %d artifacts missing required keys' % len(skipped))

    csv_path = args.csv_path or str(adir / 'sweep_metrics.csv')
    layer0 = diags[0][1]['layer_id']
    mode0 = diags[0][1]['mode']
    threshold_lambda = find_threshold_lambda(
        csv_path, layer0, mode0, args.err_tolerance)

    for sc in args.plot_schemes:
        p = split_make_plot(diags, out_prefix, scheme=sc, xlim=args.xlim,
                            report=args.report_notation, plot_avg=args.plot_avg,
                            threshold_lambda=threshold_lambda,
                            tolerance=args.err_tolerance)
        print('wrote %s' % p)
        for variant in ('unconstrained', 'reassigned'):
            p = split_make_level_dist_plot(diags, out_prefix, variant, scheme=sc,
                                           xlim=args.xlim,
                                           report=args.report_notation,
                                           threshold_lambda=threshold_lambda,
                                           tolerance=args.err_tolerance)
            print('wrote %s' % p)

    target = min(diags, key=lambda fd: abs(fd[1]['lambda_l1'] - 0.01))
    fn, d = target
    print('\nsummary at lambda_1=%.4g (%s), alpha=%g:'
          % (d['lambda_l1'], fn, args.alpha))
    print('%4s %10s %9s %9s %9s %9s %7s' %
          ('pos', 'parent', 'H_par', 'H_tree', 'H_split', 'reass', ''))
    for p_idx in range(d['P']):
        pg = d['parent_group'][p_idx]
        print('%4d %10s %9.4f %9.4f %9.4f %9.3f' %
              (d['positions'][p_idx], '%d,%d' % pg,
               d['H_bar_fire_norm'][p_idx],
               d['H_bar_min_fire_whole_tree_norm'][p_idx],
               d['H_bar_splitcheck_fire_norm'][p_idx],
               d['reassign_frac_fire'][p_idx]))


# ===========================================================================
# CLI
# ===========================================================================

def _add_common(sp, default_artifacts_dir):
    sp.add_argument('--artifacts_dir', default=default_artifacts_dir,
                    help='Directory of *.sae_eval.pt artifacts.')
    sp.add_argument('--xlim', nargs=2, type=float, default=None,
                    metavar=('XMIN', 'XMAX'))
    sp.add_argument('--csv_path', default=None,
                    help='sweep_metrics.csv for the error-onset threshold line. '
                         'Default: <artifacts_dir>/sweep_metrics.csv.')
    sp.add_argument('--err_tolerance', type=float, default=0.01,
                    help='Tolerance on (norm_err - baseline_err) for the '
                         'threshold lambda. Default: 0.01 (1%%).')
    sp.add_argument('--plot_avg', action='store_true',
                    help='Add an extra "avg over positions" panel.')
    add_report_flag(sp)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='mode', required=True)

    # aggregate
    ap = sub.add_parser('aggregate',
                        help='normalized entropy (fire/raw/dec) vs lambda')
    ap.add_argument('--artifacts_dir', required=True)
    ap.add_argument('--outfile_prefix', default=None)
    ap.add_argument('--xlim', nargs=2, type=float, default=None,
                    metavar=('XMIN', 'XMAX'))
    ap.add_argument('--csv_path', default=None)
    ap.add_argument('--err_tolerance', type=float, default=0.01)
    ap.add_argument('--plot_avg', action='store_true')
    add_report_flag(ap)

    # min
    mp = sub.add_parser('min', help='parent/level/unconstrained min-entropy')
    _add_common(mp, DEFAULT_ARTIFACTS_DIR)
    mp.add_argument('--candidates', choices=('same_level', 'whole_tree'),
                    default='same_level')
    mp.add_argument('--out_csv', default=None)
    mp.add_argument('--out_plot_prefix', default=None)
    mp.add_argument('--plot_schemes', nargs='+', choices=SCHEMES,
                    default=['dec'])

    # splitcheck
    spc = sub.add_parser('splitcheck',
                         help='min + feature-splitting reassignment')
    _add_common(spc, DEFAULT_ARTIFACTS_DIR)
    spc.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                     help='reassign when H_norm(parent|f)/leak_norm < alpha '
                          '(default %g)' % DEFAULT_ALPHA)
    spc.add_argument('--out_csv', default=None)
    spc.add_argument('--out_plot_prefix', default=None)
    spc.add_argument('--plot_schemes', nargs='+', choices=SCHEMES,
                     default=['raw'])

    args = parser.parse_args()
    if args.mode == 'aggregate':
        run_aggregate(args)
    elif args.mode == 'min':
        run_min(args)
    elif args.mode == 'splitcheck':
        run_splitcheck(args)


if __name__ == '__main__':
    main()
