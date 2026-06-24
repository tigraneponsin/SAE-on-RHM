"""Feature-splitting diagnostic for SAE subtree alignment.

Offline post-processing of *.sae_eval.pt artifacts. No SAE re-evaluation.

Builds on min_entropy_diag.py. That script finds, per feature, the unconstrained
whole-tree argmin latent (the "child" (l*, j*)). This script adds the missing
test: does the feature ALSO pin the child's structural PARENT (l*-1, j*//s)
beyond what the RHM tree's leak already forces? If so, the feature is really a
parent feature carved toward one child value (absorption / feature splitting),
and we REASSIGN its label up to the parent while keeping the child as a carve
tag.

Procedure (per SAE position p, per weight scheme, per feature f):
  child  = whole-tree argmin cell (l*, j*).
  parent = (l*-1, j*//s)            (root=level 0, leaf=level L; parent toward
                                     root. l*=0 has no parent -> never reassign.)
  c_i    = the child value the feature specialized to, argmax over the child
           cell's values of P(child = c | f fires) = joint_fire_count /
           firing_count (the value the feature fires most on).
  leak(child, c_i) = H(Z_parent | Z_child = c_i), the VALUE-SPECIFIC structural
                leak from the RHM rules. Same for all lambdas in a sweep ->
                computed ONCE. Indexed by (CHILD cell, value).
  leak_norm[child, c_i] = leak(child, c_i) / H_theoretical[parent].
  ratio_f = (H_per_feature[parent_gidx, p, f] / H_theoretical[parent])
            / leak_norm[child, c_i].
  reassign when ratio_f < alpha (default 0.25): the feature's selectivity on the
  parent is small relative to the value-specific leak, i.e. it pins the parent
  far beyond what the child value alone forces -> it is really a parent feature.

Plots (report-notation aware):
  - entropy vs lambda, parent / level / unconstrained-split-checked (the
    aggregate AFTER reassignment, drawn orange). The raw unconstrained-argmin
    curve is no longer drawn; the split-checked curve replaces it.
  - level-distribution vs lambda, TWO variants: unconstrained (simple argmin)
    and reassigned (after the split check), so the level evolution can be
    compared directly.

Usage:
    python scripts/sae_sweep/feature_splitting_diag.py \\
        --artifacts_dir /path/to/sweep/analysis_files \\
        [--alpha 0.25] [--out_csv ...] [--out_plot_prefix ...] [--xlim 1e-4 1]
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))                       # sibling sweep modules
sys.path.insert(0, str(_here.parent / 'common'))     # sae_loading
sys.path.insert(0, str(_here.parent.parent))         # repo root (datasets)

from notation import report_level, add_report_flag
from plot_entropy_lambda import _find_threshold_lambda
from min_entropy_diag import (weighted_aggregate, weights_for_scheme,
                              SCHEMES, sanity_check)
# Shared per-feature labeling core. resolve_leak_table is re-exported so
# existing importers (feature_splitting_alpha_sweep) keep working.
from absorption import per_feature_labels, resolve_leak_table  # noqa: F401
import numpy as np
import torch

DEFAULT_ARTIFACTS_DIR = (
    '/work/pcsl/ponsin/Mean_Transformer/Small_SAE/latent_dim_4*512/'
    'v_16_L_3_m_16_wdecay_0.0001_dropout_0_nores/'
    'sweep_alltokens_layer1_lambda1_zoom/analysis_files'
)

DEFAULT_ALPHA = 0.5
# y-limits for every entropy subplot (small margin so plateaus at 0/1 stay
# visible). Shared by make_plot and the averaged panel.
ENTROPY_YLIM = (-0.02, 1.02)


def process_artifact(art, leak_norm, alpha, collect_ratios=False):
    """Per-position diagnostics for one artifact, with the split check.

    Superset of min_entropy_diag.process_artifact (whole_tree mode). Keeps the
    parent / same-level / whole-tree min curves and the unconstrained level
    distribution, and ADDS the split-checked aggregate, the reassigned level
    distribution, and the reassigned fraction.

    collect_ratios: also stash, per position and scheme, the raw per-feature
        split-check ratio with its weight and reassign-eligibility mask
        (out['ratio_<sc>'][p] = {'ratio','w','eligible'} numpy arrays). Used by
        the alpha-sweep diagnostic to threshold the SAME ratios at many alphas
        without recomputing entropies. Off by default (heavier output).
    """
    # Per-feature labeling core (no weighting). Provides, per position, the
    # parent / level / whole-tree mins + argmins, the split-check ratio, and the
    # reassign mask + final cell -- everything the weighted aggregates below
    # need. Weighting stays here (aggregation-only); the core is shared with
    # circuit_tracing/labels.py.
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

            # split-checked aggregate: normalize EACH feature by its FINAL
            # cell's H_theoretical, then aggregate (mirrors the whole_tree_norm
            # block; a single H_ref would be wrong when levels differ).
            h_split = weighted_aggregate(
                final_entropy / final_href.clamp_min(1e-30), w)
            out['H_bar_splitcheck_%s_norm' % sc].append(float(h_split))

            # weighted fraction of live features reassigned.
            live = torch.isfinite(H_min['whole_tree']) & (w > 0)
            w_live = torch.where(live, w.double(), torch.zeros_like(w.double()))
            tot = w_live.sum()
            re_w = torch.where(live & reassign, w.double(),
                               torch.zeros_like(w.double()))
            out['reassign_frac_%s' % sc].append(
                float(re_w.sum() / tot.clamp_min(1e-30)) if tot > 0 else float('nan'))

            if collect_ratios:
                # raw ratios for the alpha sweep: weighted denominator is all
                # live features (tot), numerator at a given alpha is the live &
                # reassign-eligible features with ratio < alpha. Store ratio, w,
                # and the live/eligible masks so the sweep just thresholds.
                eligible = (live & can_reassign & torch.isfinite(ratio))
                out['ratio_%s' % sc].append({
                    'ratio': ratio.cpu().numpy().copy(),
                    'w': w.double().cpu().numpy().copy(),
                    'live': live.cpu().numpy().copy(),
                    'eligible': eligible.cpu().numpy().copy()})

            # unconstrained child-level distribution (per level).
            out['target_dist_wt_%s' % sc].append(
                _level_votes(child_level, w, live, tot))
            # reassigned distribution: same features, voting at their FINAL
            # level (only difference: reassigned features moved up one level).
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


def write_csv(path, diags):
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


def write_target_dist_csv(path, diags):
    """Long format, one row per (file, position, scheme, variant, target_level).

    variant in {unconstrained, reassigned}. 'share' is the weighted fraction of
    live features whose argmin (unconstrained) or final-assigned (reassigned)
    cell lands on that level.
    """
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


def _thr_label(tolerance):
    return '%g%% err onset' % (tolerance * 100)


LEVEL_PALETTE = ('#4daf4a', '#984ea3', '#a65628', '#f781bf', '#17becf',
                 '#bcbd22', '#000000')


def make_level_dist_plot(diags, out_prefix, variant, scheme='fire', xlim=None,
                         report=False, threshold_lambda=None, tolerance=0.01):
    """Per-position level-share distribution vs lambda for one variant.

    variant 'unconstrained' uses the simple whole-tree argmin levels;
    'reassigned' uses the split-checked final levels. Curves sum to ~1.
    """
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
                       label=(_thr_label(tolerance) if p_idx == 0 else None))
        ax.set_title('pooled readout (root %d,%d)' % (_lvl(pg[0]), pg[1])
                     if pooled
                     else 'pos %d (parent %d,%d)' % (p_idx, _lvl(pg[0]), pg[1]),
                     fontsize=9)
        ax.set_xlabel(r'$\lambda$', fontsize=12)
        ax.set_ylabel('share', fontsize=12)
        if p_idx == 0:
            ax.legend(fontsize=7, loc='upper right')

    for k in range(P, len(axes)):
        axes[k].axis('off')

    title = ('Unconstrained argmin' if variant == 'unconstrained'
             else 'Split-check reassigned')
    fig.suptitle('%s target level distribution by position' % title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path('%s_leveldist_%s_%s.png' % (out_prefix, variant, scheme))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def make_plot(diags, out_prefix, scheme='fire', xlim=None, report=False,
              plot_avg=False, threshold_lambda=None, tolerance=0.01):
    """Per-position weighted entropy eta-bar vs lambda. Three curves per
    subplot: parent-constrained, level-constrained, and the unconstrained
    whole-tree min BEFORE any reassignment (drawn orange)."""
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
    # orange curve: unconstrained whole-tree min, BEFORE reassignment.
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
                       label=(_thr_label(tolerance) if p_idx == 0 else None))
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
                       linewidth=0.8, alpha=0.7, label=_thr_label(tolerance))
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
    each reassigned/unconstrained level distribution sums to ~1 when live.

    Only whole_tree <= split is a true invariant: reassignment moves a feature
    from its child to that child's tree-parent (entropy can only rise or stay).
    Split is NOT bounded above by the position's NOMINAL parent curve -- a
    reassigned feature lands on its own child's parent cell, which may differ
    from (matched_level, matched_j), so split can exceed the nominal-parent
    aggregate. We therefore do not assert split <= parent.
    """
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--artifacts_dir', default=DEFAULT_ARTIFACTS_DIR)
    ap.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                    help='reassign a feature from child to parent when '
                         'ratio = H_norm(parent|f)/leak_norm < alpha '
                         '(default %g)' % DEFAULT_ALPHA)
    ap.add_argument('--out_csv', default=None)
    ap.add_argument('--out_plot_prefix', default=None)
    ap.add_argument('--xlim', nargs=2, type=float, default=None)
    ap.add_argument('--plot_schemes', nargs='+', choices=SCHEMES,
                    default=['raw'],
                    help='subset of weight schemes to plot (default: raw only). '
                         'CSVs always contain all schemes.')
    ap.add_argument('--plot_avg', action='store_true',
                    help='add an extra "avg over positions" panel to the '
                         'entropy plot.')
    ap.add_argument('--csv_path', default=None,
                    help='Path to sweep_metrics.csv for the error-onset '
                         'threshold lambda. Default: <artifacts_dir>/sweep_metrics.csv.')
    ap.add_argument('--err_tolerance', type=float, default=0.01,
                    help='Tolerance on (norm_err - baseline_err) for the '
                         'threshold lambda. Default: 0.01 (1%%).')
    add_report_flag(ap)
    args = ap.parse_args()

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

    # resolve the structural leak ONCE from the first usable artifact.
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
        d = process_artifact(art, leak_norm, args.alpha)
        sanity_check(art, d)
        splitcheck_sanity(d)
        diags.append((f.name, d))

    if not diags:
        raise SystemExit('no usable artifacts (skipped %d missing keys)'
                         % len(skipped))

    write_csv(out_csv, diags)
    print('wrote %s (%d artifacts x %d positions)'
          % (out_csv, len(diags), diags[0][1]['P']))
    leak_csv = out_csv.with_name(out_csv.stem + '_leak_table.csv')
    write_leak_csv(leak_csv, leak_rows)
    print('wrote %s' % leak_csv)
    dist_csv = out_csv.with_name(out_csv.stem + '_target_dist.csv')
    write_target_dist_csv(dist_csv, diags)
    print('wrote %s' % dist_csv)
    if skipped:
        print('skipped %d artifacts missing required keys' % len(skipped))

    csv_path = args.csv_path or str(adir / 'sweep_metrics.csv')
    layer0 = diags[0][1]['layer_id']
    mode0 = diags[0][1]['mode']
    threshold_lambda = _find_threshold_lambda(
        csv_path, layer0, mode0, args.err_tolerance)

    for sc in args.plot_schemes:
        p = make_plot(diags, out_prefix, scheme=sc, xlim=args.xlim,
                      report=args.report_notation, plot_avg=args.plot_avg,
                      threshold_lambda=threshold_lambda,
                      tolerance=args.err_tolerance)
        print('wrote %s' % p)
        for variant in ('unconstrained', 'reassigned'):
            p = make_level_dist_plot(diags, out_prefix, variant, scheme=sc,
                                     xlim=args.xlim, report=args.report_notation,
                                     threshold_lambda=threshold_lambda,
                                     tolerance=args.err_tolerance)
            print('wrote %s' % p)

    # console summary at the lambda closest to 0.01.
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


if __name__ == '__main__':
    main()
