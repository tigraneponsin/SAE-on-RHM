"""Alpha-sweep diagnostic for the feature-splitting reassignment.

Companion to feature_splitting_diag.py. Computes the per-feature split-check
ratio ONCE (ratio = H(parent|f) / H(parent|child=c_i), value-specific), then:
  - plots reassign_frac vs alpha (weighted fraction of live features with
    ratio < alpha), per SAE position and averaged, at a chosen lambda; and
  - plots the histogram of per-feature ratios (weighted), to look for a valley
    that would justify a natural alpha.

A good alpha sits in a valley of the ratio histogram (bimodal -> genuine
splitting) and/or on a plateau of reassign_frac vs alpha (insensitive). Judge
both in the low/mid-lambda regime, before the error-onset line: at high lambda
most features die and the fraction is noisy.

Usage:
    python scripts/sae_sweep/feature_splitting_alpha_sweep.py \\
        --artifacts_dir /path/to/sweep/analysis_files \\
        [--lambda_target 0.01] [--alphas 0.05 ... 0.6] \\
        [--out_plot_prefix ...] [--report-notation]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent / 'common'))
sys.path.insert(0, str(_here.parent.parent))

from notation import report_level, add_report_flag
from plot_entropy_lambda import _find_threshold_lambda
from min_entropy_diag import SCHEMES
from feature_splitting_diag import (resolve_leak_table, process_artifact,
                                    DEFAULT_ARTIFACTS_DIR)

DEFAULT_ALPHAS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]


def reassign_frac_at(ratio_rec, alpha):
    """Weighted fraction of LIVE features reassigned at this alpha.

    Denominator = total weight of live features (same as the main script).
    Numerator   = weight of live & reassign-eligible features with ratio<alpha.
    """
    w = ratio_rec['w']
    live = ratio_rec['live']
    elig = ratio_rec['eligible']
    r = ratio_rec['ratio']
    tot = w[live].sum()
    if tot <= 0:
        return float('nan')
    hit = elig & np.isfinite(r) & (r < alpha)
    return float(w[hit].sum() / tot)


def _thr_label(tol):
    return '%g%% err onset' % (tol * 100)


def make_alpha_plot(diags_at_lambda, alphas, out_prefix, scheme, report,
                    lambda_val, threshold_lambda=None, tolerance=0.01):
    """reassign_frac vs alpha, one curve per SAE position + an avg curve."""
    fn, d = diags_at_lambda
    P = d['P']
    L = d['L']
    pooled = d.get('pooled', False)

    def _lvl(cl):
        return report_level(cl, L) if report else cl

    # frac[p][a] = reassign fraction at position p, alpha a.
    recs = d['ratio_%s' % scheme]
    frac = np.array([[reassign_frac_at(recs[p], a) for a in alphas]
                     for p in range(P)])                      # (P, A)
    avg = np.nanmean(frac, axis=0)                            # (A,)

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap('viridis')
    for p in range(P):
        pg = d['parent_group'][p]
        lab = ('root %d,%d' % (_lvl(pg[0]), pg[1]) if pooled
               else 'pos %d (par %d,%d)' % (p, _lvl(pg[0]), pg[1]))
        ax.plot(alphas, frac[p], 'o-', color=cmap(p / max(P - 1, 1)),
                alpha=0.7, linewidth=1, markersize=3, label=lab)
    ax.plot(alphas, avg, 'k^-', linewidth=2.2, markersize=6,
            label='avg over positions', zorder=5)
    ax.set_xlabel(r'$\alpha$', fontsize=14)
    ax.set_ylabel('reassigned fraction', fontsize=12)
    ax.set_ylim(-0.02, 1.02)
    ax.axvline(0.5, color='red', linestyle=':', linewidth=1,
               alpha=0.6, label=r'$\alpha=0.5$')
    ax.set_title(r'Reassigned fraction vs $\alpha$  ($\lambda$=%.4g)'
                 % lambda_val, fontsize=11)
    ax.legend(fontsize=7, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path('%s_alphasweep_%s.png' % (out_prefix, scheme))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def make_ratio_hist(diags_at_lambda, out_prefix, scheme, report, lambda_val):
    """Weighted histogram of per-feature ratios (eligible features only), per
    position. A valley between a near-0 mode and a near-1 mode justifies an
    alpha in that valley."""
    fn, d = diags_at_lambda
    P = d['P']
    L = d['L']
    pooled = d.get('pooled', False)

    def _lvl(cl):
        return report_level(cl, L) if report else cl

    recs = d['ratio_%s' % scheme]
    ncol = min(4, P)
    nrow = int(np.ceil(P / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                             squeeze=False)
    axes = axes.flatten()
    bins = np.linspace(0, 2, 41)        # ratio in [0,2]; >1 = leak-explained
    for p in range(P):
        ax = axes[p]
        rec = recs[p]
        m = rec['eligible'] & np.isfinite(rec['ratio'])
        r = np.clip(rec['ratio'][m], 0, 2)
        w = rec['w'][m]
        pg = d['parent_group'][p]
        if w.sum() > 0:
            ax.hist(r, bins=bins, weights=w / w.sum(), color='#4daf4a',
                    edgecolor='k', linewidth=0.3)
        ax.axvline(0.5, color='red', linestyle=':', linewidth=1.2,
                   label=r'$\alpha=0.5$')
        ax.axvline(1.0, color='gray', linestyle='--', linewidth=0.8,
                   label='ratio=1 (=leak)')
        ax.set_title('pooled (root %d,%d)' % (_lvl(pg[0]), pg[1]) if pooled
                     else 'pos %d (par %d,%d)' % (p, _lvl(pg[0]), pg[1]),
                     fontsize=9)
        ax.set_xlabel('ratio = H(par|f)/H(par|child=c_i)', fontsize=8)
        ax.set_ylabel('weighted share', fontsize=8)
        if p == 0:
            ax.legend(fontsize=7)
    for k in range(P, len(axes)):
        axes[k].axis('off')
    fig.suptitle(r'Split-check ratio histogram  ($\lambda$=%.4g)'
                 % lambda_val)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path('%s_ratiohist_%s.png' % (out_prefix, scheme))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--artifacts_dir', default=DEFAULT_ARTIFACTS_DIR)
    ap.add_argument('--alphas', nargs='+', type=float, default=DEFAULT_ALPHAS)
    ap.add_argument('--lambda_target', type=float, default=0.01,
                    help='examine the sweep file whose lambda_l1 is closest to '
                         'this (default 0.01).')
    ap.add_argument('--out_plot_prefix', default=None)
    ap.add_argument('--plot_schemes', nargs='+', choices=SCHEMES,
                    default=['raw'])
    ap.add_argument('--csv_path', default=None)
    ap.add_argument('--err_tolerance', type=float, default=0.01)
    add_report_flag(ap)
    args = ap.parse_args()

    adir = Path(args.artifacts_dir)
    files = sorted(adir.glob('*.sae_eval.pt'))
    if not files:
        raise SystemExit('no *.sae_eval.pt files in %s' % adir)

    out_prefix = (args.out_plot_prefix or
                  str(adir.parent / 'analysis_plots'
                      / 'feature_splitting_alpha_sweep'))

    needed = ('H_per_feature', 'index_layout', 'firing_rate', 'baseline_mean',
              'decoder_norms', 'token_positions', 'H_theoretical', 'rhm',
              'layer_id', 'firing_count', 'joint_fire_count', 'targets')

    # only need the lambda closest to the target; load all, pick the nearest,
    # process THAT one with collect_ratios.
    arts = []
    leak_norm = None
    for f in files:
        art = torch.load(f, map_location='cpu', weights_only=False)
        if any(k not in art or art[k] is None for k in needed):
            continue
        if leak_norm is None:
            leak_norm, _rows, src = resolve_leak_table(art, adir)
            print('resolved RHM rules (source: %s)' % src)
        arts.append((f.name, art))
    if not arts:
        raise SystemExit('no usable artifacts')

    fn, art = min(arts, key=lambda fa: abs(
        float(fa[1].get('lambda_l1') or 0.0) - args.lambda_target))
    lambda_val = float(art.get('lambda_l1') or 0.0)
    print('examining lambda_l1=%.4g (%s)' % (lambda_val, fn))
    d = process_artifact(art, leak_norm, alpha=0.25, collect_ratios=True)
    diags_at = (fn, d)

    csv_path = args.csv_path or str(adir / 'sweep_metrics.csv')
    threshold_lambda = _find_threshold_lambda(
        csv_path, d['layer_id'], d['mode'], args.err_tolerance)

    for sc in args.plot_schemes:
        p1 = make_alpha_plot(diags_at, args.alphas, out_prefix, sc,
                             args.report_notation, lambda_val,
                             threshold_lambda, args.err_tolerance)
        print('wrote %s' % p1)
        p2 = make_ratio_hist(diags_at, out_prefix, sc, args.report_notation,
                             lambda_val)
        print('wrote %s' % p2)

    # console table: avg reassign_frac at each alpha (dec scheme).
    sc = args.plot_schemes[0]
    recs = d['ratio_%s' % sc]
    print('\navg reassigned fraction vs alpha (%s, lambda=%.4g):' % (sc, lambda_val))
    print('%8s %10s' % ('alpha', 'avg_frac'))
    for a in args.alphas:
        fr = np.nanmean([reassign_frac_at(recs[p], a) for p in range(d['P'])])
        print('%8.3f %10.3f' % (a, fr))


if __name__ == '__main__':
    main()
