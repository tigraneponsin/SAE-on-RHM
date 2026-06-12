"""Min-conditional-entropy diagnostic for SAE subtree alignment.

Offline post-processing of *.sae_eval.pt artifacts. No SAE re-evaluation.

The standard entropy metric (scripts/sae_eval/streaming.py) conditions each
SAE position's features ONLY on that position's own RHM parent latent at the
matched level (matched_level = L-1-layer_id, matched_j = p_real //
s^(1+layer_id)). If a feature firing on a left-subtree position actually
encodes the *sibling* (right) subtree's latent, it scores as high entropy
against its own parent and looks mis-aligned.

This script recomputes, for each feature, the SAME-LEVEL latent group (over
all positions at the matched level) that MINIMIZES that feature's conditional
entropy, then aggregates those minima with the same weight schemes as the
existing pipeline. A large gap (parent - min) and a high "leakage fraction"
(features whose argmin group is not the position's own parent) flag the
cross-subtree leakage.

All needed quantities are already in the artifact:
  H_per_feature [num_groups, P, F], index_layout, firing_rate [P, F],
  baseline_mean [P, F], decoder_norms [F], token_positions [P],
  H_theoretical {(level,pos) -> float}, rhm {v,n,m,s,L}, layer_id.

Usage:
    python scripts/sae_sweep/min_entropy_diag.py \\
        --artifacts_dir /path/to/sweep/analysis_files \\
        [--out_csv /path/to/min_entropy_diag.csv] \\
        [--out_plot_prefix /path/to/min_entropy_diag] \\
        [--xlim 1e-4 1]
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from notation import sae_label, report_level, add_report_flag
import numpy as np
import torch

DEFAULT_ARTIFACTS_DIR = (
    '/work/pcsl/ponsin/Mean_Transformer/Small_SAE/latent_dim_4*512/'
    'v_16_L_3_m_16_wdecay_0.0001_dropout_0.1_nores/'
    'sweep_alltokens_layer1_lambda1_zoom/analysis_files'
)

# weight schemes -> (csv suffix, label)
SCHEMES = ('fire', 'raw', 'dec')


def weighted_aggregate(H, weights):
    """Weighted mean of H ignoring NaN entries (dead features).

    Copied verbatim semantics from scripts/sae_eval/streaming.py:148 so the
    parent-conditioned recomputation reproduces the stored H_bar_* exactly.
    """
    if H.shape != weights.shape:
        raise ValueError('Shape mismatch: H %s vs weights %s.'
                         % (tuple(H.shape), tuple(weights.shape)))
    H64 = H.double()
    w64 = weights.double()
    mask = torch.isfinite(H64) & (w64 > 0)
    w_masked = torch.where(mask, w64, torch.zeros_like(w64))
    h_masked = torch.where(mask, H64, torch.zeros_like(H64))
    num = (w_masked * h_masked).sum(dim=-1)
    den = w_masked.sum(dim=-1)
    out = torch.where(den > 0, num / den.clamp_min(1e-30),
                      torch.full_like(den, float('nan')))
    return out.float()


def weights_for_scheme(scheme, firing_rate_p, baseline_mean_p, decoder_norms):
    """Per-position [F] weight vector, mirroring streaming.py:748-750."""
    if scheme == 'fire':
        return firing_rate_p
    if scheme == 'raw':
        return baseline_mean_p / decoder_norms.clamp_min(1e-30)
    if scheme == 'dec':
        return baseline_mean_p
    raise ValueError('unknown scheme %r' % scheme)


def process_artifact(art, candidates='same_level'):
    """Return a dict of per-position diagnostics for one artifact.

    candidates: 'same_level' restricts the argmin search to RHM latent groups
        at the matched level (the original behavior); 'whole_tree' searches
        every (level, position) group in index_layout, so a feature can be
        matched to an ancestor/descendant latent at a different level.

    Keys: lambda_l1, layer_id, P, s, L, positions [P],
          parent_group [P] (list of (level,pos)),
          H_bar_<s> [P], H_bar_min_<s> [P], gap_<s> [P],
          H_bar_<s>_norm [P], H_bar_min_<s>_norm [P],
          leak_frac_<s> [P], leak_to_<s> [P] (dominant non-parent group),
          stored_H_bar_fire [P] (for sanity check).
    """
    rhm = art['rhm']
    s = int(rhm['s'])
    L = int(rhm['L'])
    layer_id = int(art['layer_id'])
    matched_level = L - 1 - layer_id

    index_layout = art['index_layout']
    H_per_feature = art['H_per_feature']          # [num_groups, P, F]
    firing_rate = art['firing_rate']              # [P, F]
    baseline_mean = art['baseline_mean']          # [P, F]
    decoder_norms = art['decoder_norms']          # [F]
    token_positions = art['token_positions']      # [P]
    H_theoretical = art['H_theoretical']          # {(level,pos)->float}

    P = H_per_feature.shape[1]

    # group lookup: (level,pos) -> index into H_per_feature first dim
    group_index = {(int(g['level']), int(g['position'])): idx
                   for idx, g in enumerate(index_layout)}
    # candidate groups for the argmin search
    if candidates == 'same_level':
        level_groups = [(int(g['level']), int(g['position']))
                        for g in index_layout if int(g['level']) == matched_level]
        if not level_groups:
            raise ValueError('no index_layout groups at matched_level=%d' % matched_level)
    elif candidates == 'whole_tree':
        level_groups = [(int(g['level']), int(g['position'])) for g in index_layout]
    else:
        raise ValueError('unknown candidates %r' % candidates)
    level_g_idxs = [group_index[k] for k in level_groups]

    out = {
        'lambda_l1': float(art.get('lambda_l1') or 0.0),
        'layer_id': layer_id, 'P': P, 's': s, 'L': L,
        'matched_level': matched_level,
        'positions': [], 'parent_group': [],
        'stored_H_bar_fire': [],
    }
    for sc in SCHEMES:
        out['H_bar_%s' % sc] = []
        out['H_bar_min_%s' % sc] = []
        out['gap_%s' % sc] = []
        out['H_bar_%s_norm' % sc] = []
        out['H_bar_min_%s_norm' % sc] = []
        out['leak_frac_%s' % sc] = []
        out['leak_to_%s' % sc] = []
        out['target_dist_%s' % sc] = []

    stored_fire = art.get('H_bar_fire')

    for p_idx in range(P):
        p_real = int(token_positions[p_idx].item())
        matched_j = p_real // (s ** (1 + layer_id))
        parent_key = (matched_level, matched_j)
        parent_gidx = group_index.get(parent_key)
        out['positions'].append(p_real)
        out['parent_group'].append(parent_key)
        out['stored_H_bar_fire'].append(
            float(stored_fire[p_idx]) if stored_fire is not None else float('nan'))

        # candidate stack [num_level_groups, F]
        H_cand = H_per_feature[level_g_idxs, p_idx, :]            # [G, F]
        # min over candidate groups per feature; NaN-aware (dead features stay NaN)
        H_min, argmin = torch_nanmin(H_cand, dim=0)              # [F], [F]
        H_parent = H_per_feature[parent_gidx, p_idx, :]          # [F]

        H_ref = H_theoretical.get(parent_key, float('nan'))
        ref_ok = (H_ref > 0) and math.isfinite(H_ref)

        # which candidate-stack row is the parent
        parent_row = level_g_idxs.index(parent_gidx)

        for sc in SCHEMES:
            w = weights_for_scheme(sc, firing_rate[p_idx], baseline_mean[p_idx],
                                   decoder_norms)
            h_par = weighted_aggregate(H_parent, w)
            h_min = weighted_aggregate(H_min, w)
            out['H_bar_%s' % sc].append(float(h_par))
            out['H_bar_min_%s' % sc].append(float(h_min))
            out['gap_%s' % sc].append(float(h_par) - float(h_min))
            if ref_ok:
                out['H_bar_%s_norm' % sc].append(float(h_par) / H_ref)
                out['H_bar_min_%s_norm' % sc].append(float(h_min) / H_ref)
            else:
                out['H_bar_%s_norm' % sc].append(float('nan'))
                out['H_bar_min_%s_norm' % sc].append(float('nan'))

            # leakage: weighted fraction of live features whose argmin row
            # is not the parent row, and which non-parent group dominates.
            live = torch.isfinite(H_min) & (w > 0)
            w_live = torch.where(live, w.double(), torch.zeros_like(w.double()))
            tot = w_live.sum()
            leaks = live & (argmin != parent_row)
            leak_w = torch.where(leaks, w.double(), torch.zeros_like(w.double()))
            frac = float(leak_w.sum() / tot.clamp_min(1e-30)) if tot > 0 else float('nan')
            out['leak_frac_%s' % sc].append(frac)

            # full weighted distribution of argmin targets over ALL live
            # features (parent included), normalized to sum to 1. votes[r] is
            # the weighted fraction of live features whose argmin is row r.
            votes = torch.zeros(len(level_g_idxs), dtype=torch.float64)
            if live.any():
                votes.scatter_add_(0, argmin[live], w.double()[live])
            if tot > 0:
                votes = votes / tot
            dist = {level_groups[r]: float(votes[r])
                    for r in range(len(level_g_idxs)) if votes[r] > 0}
            out['target_dist_%s' % sc].append(dist)

            # dominant leak target group (largest non-parent share)
            if leaks.any():
                votes_np = votes.clone()
                votes_np[parent_row] = 0.0
                dom_row = int(torch.argmax(votes_np).item())
                out['leak_to_%s' % sc].append(level_groups[dom_row])
            else:
                out['leak_to_%s' % sc].append(None)

    return out


def torch_nanmin(x, dim):
    """NaN-aware min along dim. Returns (values, argmin_index).

    NaNs are treated as +inf for the min; if all entries along dim are NaN the
    value stays NaN and the index is 0 (the aggregate ignores NaN values).
    """
    big = torch.full_like(x, float('inf'))
    filled = torch.where(torch.isnan(x), big, x)
    vals, idx = filled.min(dim=dim)
    all_nan = torch.isnan(x).all(dim=dim)
    vals = torch.where(all_nan, torch.full_like(vals, float('nan')), vals)
    return vals, idx


def sanity_check(art, diag, tol=1e-4):
    """Assert min<=parent everywhere and stored H_bar_fire matches recompute."""
    for p_idx in range(diag['P']):
        par = diag['H_bar_fire'][p_idx]
        mn = diag['H_bar_min_fire'][p_idx]
        if math.isfinite(par) and math.isfinite(mn):
            assert mn <= par + 1e-6, (
                'pos %d: min %.6f > parent %.6f' % (p_idx, mn, par))
        stored = diag['stored_H_bar_fire'][p_idx]
        if math.isfinite(stored) and math.isfinite(par):
            assert abs(stored - par) < tol, (
                'pos %d: recomputed parent H_bar_fire %.6f != stored %.6f'
                % (p_idx, par, stored))


def write_csv(path, diags):
    cols = ['file', 'lambda_l1', 'layer_id', 'matched_level', 'position',
            'parent_group']
    for sc in SCHEMES:
        cols += ['H_bar_%s' % sc, 'H_bar_min_%s' % sc, 'gap_%s' % sc,
                 'H_bar_%s_norm' % sc, 'H_bar_min_%s_norm' % sc,
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
                            d['H_bar_min_%s' % sc][p_idx],
                            d['gap_%s' % sc][p_idx],
                            d['H_bar_%s_norm' % sc][p_idx],
                            d['H_bar_min_%s_norm' % sc][p_idx],
                            d['leak_frac_%s' % sc][p_idx], lt_str]
                w.writerow(row)


def write_target_dist_csv(path, diags):
    """Long-format distribution: one row per (file, position, scheme, target).

    'share' is the weighted fraction of live features at that SAE position
    whose min-entropy argmin lands on (target_level, target_position).
    is_parent marks the position's own matched-level parent latent.
    """
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


def make_level_dist_plot(diags, out_prefix, scheme='fire', xlim=None,
                         report=False):
    """Per-position: share of the argmin-target distribution by RHM level
    vs lambda. For each position one subplot; one curve per level (0..L-1)
    giving the summed share of all targets at that level. Curves sum to ~1.
    """
    diags_sorted = sorted(diags, key=lambda fd: fd[1]['lambda_l1'])
    lambdas = [d['lambda_l1'] for _, d in diags_sorted]
    P = diags_sorted[0][1]['P']
    L = diags_sorted[0][1]['L']

    def _lvl(code_level):
        return report_level(code_level, L) if report else code_level

    dkey = 'target_dist_%s' % scheme

    # levels actually present as argmin targets, derived from the data. The
    # candidate groups span every RHM level including the leaf level L (8 leaf
    # positions for L=3,s=2), so this must NOT be hardcoded to range(L) or the
    # leaf-level mass is silently dropped and the per-level curves fail to
    # sum to 1.
    levels = sorted({gl for _, d in diags_sorted
                     for p in range(P)
                     for (gl, _gp) in d[dkey][p]})

    ncol = 4
    nrow = math.ceil(P / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                             squeeze=False)
    axes = axes.flatten()

    for p_idx in range(P):
        ax = axes[p_idx]
        pg = diags_sorted[0][1]['parent_group'][p_idx]
        # per level: list over lambda of summed share. An empty dist means all
        # features are dead at this (position, lambda) (high lambda); the
        # distribution is undefined -> NaN (gap in curve, not a drop to 0).
        # When live, the per-level shares sum to 1.
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
                    color='C%d' % lvl)
        ax.set_xscale('log')
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_ylim(0, 1)
        ax.set_title('pos %d (parent %d,%d)' % (p_idx, _lvl(pg[0]), pg[1]),
                     fontsize=9)
        ax.set_xlabel('lambda_1', fontsize=8)
        ax.set_ylabel('share', fontsize=8)
        if p_idx == 0:
            ax.legend(fontsize=7, loc='upper right')

    for k in range(P, len(axes)):
        axes[k].axis('off')

    fig.suptitle('Argmin-target level distribution by position (%s weights)'
                 % scheme)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path('%s_leveldist_%s.png' % (out_prefix, scheme))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def make_plot(diags, out_prefix, scheme='fire', xlim=None,
              candidates='same_level', report=False):
    """Per-position parent vs min entropy curves + leakage panel vs lambda."""
    min_label = ('min-same-level' if candidates == 'same_level'
                 else 'min-whole-tree')
    # group by lambda, all diags share P/s/L
    diags_sorted = sorted(diags, key=lambda fd: fd[1]['lambda_l1'])
    lambdas = [d['lambda_l1'] for _, d in diags_sorted]
    P = diags_sorted[0][1]['P']
    L = diags_sorted[0][1]['L']

    def _lvl(code_level):
        return report_level(code_level, L) if report else code_level
    ncol = 4
    nrow = math.ceil((P + 1) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                             squeeze=False)
    axes = axes.flatten()

    par_key = 'H_bar_%s_norm' % scheme
    min_key = 'H_bar_min_%s_norm' % scheme
    leak_key = 'leak_frac_%s' % scheme

    for p_idx in range(P):
        ax = axes[p_idx]
        par = [d[par_key][p_idx] for _, d in diags_sorted]
        mn = [d[min_key][p_idx] for _, d in diags_sorted]
        leak = [d[leak_key][p_idx] for _, d in diags_sorted]
        pg = diags_sorted[0][1]['parent_group'][p_idx]
        ax.plot(lambdas, par, 'o-', color='C0', label='parent')
        ax.plot(lambdas, mn, 's-', color='C3', label=min_label)
        ax2 = ax.twinx()
        ax2.plot(lambdas, leak, '^--', color='C2', alpha=0.6)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('leak frac', color='C2', fontsize=8)
        ax.set_xscale('log')
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_title('pos %d (parent %d,%d)' % (p_idx, _lvl(pg[0]), pg[1]), fontsize=9)
        ax.set_xlabel('lambda_1', fontsize=8)
        ax.set_ylabel('H_norm', fontsize=8)
        if p_idx == 0:
            ax.legend(fontsize=7, loc='upper left')

    # averaged panel
    ax = axes[P]
    par_avg = [np.nanmean([d[par_key][p] for p in range(P)]) for _, d in diags_sorted]
    min_avg = [np.nanmean([d[min_key][p] for p in range(P)]) for _, d in diags_sorted]
    leak_avg = [np.nanmean([d[leak_key][p] for p in range(P)]) for _, d in diags_sorted]
    ax.plot(lambdas, par_avg, 'o-', color='C0', label='parent')
    ax.plot(lambdas, min_avg, 's-', color='C3', label=min_label)
    ax2 = ax.twinx()
    ax2.plot(lambdas, leak_avg, '^--', color='C2', alpha=0.6)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('leak frac', color='C2', fontsize=8)
    ax.set_xscale('log')
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_title('avg over positions', fontsize=9)
    ax.set_xlabel('lambda_1', fontsize=8)
    ax.set_ylabel('H_norm', fontsize=8)
    ax.legend(fontsize=7, loc='upper left')

    for k in range(P + 1, len(axes)):
        axes[k].axis('off')

    fig.suptitle('Parent vs %s conditional entropy (%s weights)'
                 % (min_label, scheme))
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path('%s_%s.png' % (out_prefix, scheme))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--artifacts_dir', default=DEFAULT_ARTIFACTS_DIR)
    ap.add_argument('--candidates', choices=('same_level', 'whole_tree'),
                    default='same_level',
                    help='argmin search space: matched-level latents only '
                         '(default) or every (level,position) group in the tree')
    ap.add_argument('--out_csv', default=None)
    ap.add_argument('--out_plot_prefix', default=None)
    ap.add_argument('--xlim', nargs=2, type=float, default=None)
    ap.add_argument('--plot_schemes', nargs='+', choices=SCHEMES, default=None,
                    help='subset of weight schemes to plot (default: all). '
                         'CSVs always contain all schemes.')
    add_report_flag(ap)
    args = ap.parse_args()

    adir = Path(args.artifacts_dir)
    files = sorted(adir.glob('*.sae_eval.pt'))
    if not files:
        raise SystemExit('no *.sae_eval.pt files in %s' % adir)

    # tag default output names by mode so whole_tree does not clobber same_level
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
        d = process_artifact(art, candidates=args.candidates)
        sanity_check(art, d)
        diags.append((f.name, d))

    if not diags:
        raise SystemExit('no usable artifacts (skipped %d)' % len(skipped))

    write_csv(out_csv, diags)
    print('wrote %s (%d artifacts x %d positions)'
          % (out_csv, len(diags), diags[0][1]['P']))
    dist_csv = out_csv.with_name(out_csv.stem + '_target_dist.csv')
    write_target_dist_csv(dist_csv, diags)
    print('wrote %s' % dist_csv)
    if skipped:
        print('skipped %d artifacts missing required keys' % len(skipped))

    plot_schemes = args.plot_schemes if args.plot_schemes else SCHEMES
    for sc in plot_schemes:
        p = make_plot(diags, out_prefix, scheme=sc, xlim=args.xlim,
                      candidates=args.candidates, report=args.report_notation)
        print('wrote %s' % p)
        p = make_level_dist_plot(diags, out_prefix, scheme=sc, xlim=args.xlim,
                                 report=args.report_notation)
        print('wrote %s' % p)

    # quick console summary at the mid lambda closest to 0.01
    target = min(diags, key=lambda fd: abs(fd[1]['lambda_l1'] - 0.01))
    fn, d = target
    print('\nsummary at lambda_1=%.4g (%s):' % (d['lambda_l1'], fn))
    print('%4s %10s %9s %9s %7s %8s' %
          ('pos', 'parent', 'H_par', 'H_min', 'gap', 'leak'))
    for p_idx in range(d['P']):
        pg = d['parent_group'][p_idx]
        lt = d['leak_to_fire'][p_idx]
        lt_s = '' if lt is None else ' ->%d,%d' % lt
        print('%4d %10s %9.4f %9.4f %7.4f %8.3f%s' %
              (d['positions'][p_idx], '%d,%d' % pg,
               d['H_bar_fire'][p_idx], d['H_bar_min_fire'][p_idx],
               d['gap_fire'][p_idx], d['leak_frac_fire'][p_idx], lt_s))


if __name__ == '__main__':
    main()
