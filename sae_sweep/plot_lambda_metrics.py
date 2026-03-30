"""Plot SAE quality metrics as a function of lambda_1.

Reads the CSV produced by eval_sweep.py and plots four panels:
  1. Ever-active features: binary (>0), >1% of max mean, >10% of max mean
  2. Mean active features per token: binary (>0), >1% of token max, >10% of token max
  3. Classification error (with baseline reference)
  4. Effective features (IPR)

Within panels 1 and 2, the three thresholds are shown with different
linestyles (solid / dashed / dotted) and the same color per layer.

Usage:
    python sae_sweep/plot_lambda_metrics.py \
        --csv /work/pcsl/ponsin/Mean_Transformer/SAE/sweep_round2_lambda/eval_results.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _load_csv(path):
    """Load CSV rows, grouped by layer and sorted by lambda_l1."""
    by_layer = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            by_layer[int(row['layer'])].append(row)
    # Sort each layer's rows by lambda_l1
    for layer in by_layer:
        by_layer[layer].sort(key=lambda r: float(r['lambda_l1']))
    return dict(by_layer)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--csv', required=True, help='Path to eval_results.csv from eval_sweep.py')
    parser.add_argument('--outfile', default=None,
                        help='Output figure path (default: <csv_dir>/lambda_metrics.png)')
    parser.add_argument('--xlim', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'),
                        help='Lambda axis limits, e.g. --xlim 1e-2 1')
    args = parser.parse_args()

    by_layer = _load_csv(args.csv)
    outfile = args.outfile or str(Path(args.csv).parent / 'lambda_metrics.png')

    layers = sorted(by_layer.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(layers), 1)))
    layer_color = {l: colors[i] for i, l in enumerate(layers)}

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax_ever  = axes[0][0]
    ax_mean  = axes[0][1]
    ax_class = axes[1][0]
    ax_ipr   = axes[1][1]

    # Linestyle legend entries (drawn once, outside the layer loop)
    thresh_styles = [
        ('solid',  'o', '>0 (all active)'),
        ('dashed', 's', '>1% of max'),
        ('dotted', '^', '>10% of max'),
    ]

    for layer in layers:
        rows = by_layer[layer]
        lam = np.array([float(r['lambda_l1']) for r in rows])
        latent_dim = np.array([int(r['latent_dim']) for r in rows])
        dead = np.array([int(r['dead_features']) for r in rows])
        mean_active = np.array([float(r['mean_active']) for r in rows])
        sae_err = np.array([float(r['sae_err']) for r in rows])
        ipr = np.array([float(r['ipr']) for r in rows])
        above_1pct = np.array([int(r['active_above_1pct']) for r in rows])
        above_10pct = np.array([int(r['active_above_10pct']) for r in rows])
        mean_above_1pct = np.array([float(r['mean_active_above_1pct']) for r in rows])
        mean_above_10pct = np.array([float(r['mean_active_above_10pct']) for r in rows])

        # Filter to xlim range if specified
        if args.xlim:
            mask = (lam >= args.xlim[0]) & (lam <= args.xlim[1])
            lam, latent_dim, dead = lam[mask], latent_dim[mask], dead[mask]
            mean_active, sae_err, ipr = mean_active[mask], sae_err[mask], ipr[mask]
            above_1pct, above_10pct = above_1pct[mask], above_10pct[mask]
            mean_above_1pct, mean_above_10pct = mean_above_1pct[mask], mean_above_10pct[mask]

        c = layer_color[layer]

        # Panel 1: ever-active — three thresholds, same color, different linestyles
        for vals, (ls, mk, _) in zip(
            [latent_dim - dead, above_1pct, above_10pct], thresh_styles
        ):
            ax_ever.plot(lam, vals, color=c, linestyle=ls, marker=mk,
                         markersize=4, linewidth=1.5,
                         label=f'L{layer}' if ls == 'solid' else '_nolegend_')

        # Panel 2: mean active — three thresholds
        for vals, (ls, mk, _) in zip(
            [mean_active, mean_above_1pct, mean_above_10pct], thresh_styles
        ):
            ax_mean.plot(lam, vals, color=c, linestyle=ls, marker=mk,
                         markersize=4, linewidth=1.5,
                         label=f'L{layer}' if ls == 'solid' else '_nolegend_')

        # Panel 3: classification error
        ax_class.plot(lam, sae_err, '-o', color=c, label=f'Layer {layer}', markersize=4)

        # Panel 4: effective features (IPR)
        ax_ipr.plot(lam, ipr, '-o', color=c, label=f'Layer {layer}', markersize=4)

    # Baseline error reference line
    first_row = next(iter(by_layer.values()))[0]
    baseline_err = float(first_row['baseline_err'])
    ax_class.axhline(baseline_err, color='grey', linestyle='--', linewidth=1,
                      label=f'Baseline ({baseline_err:.4f})')

    # Add linestyle legend to panels 1 and 2
    for ax in (ax_ever, ax_mean):
        thresh_handles = [
            plt.Line2D([0], [0], color='grey', linestyle=ls, marker=mk,
                       markersize=4, label=lbl)
            for ls, mk, lbl in thresh_styles
        ]
        layer_handles, layer_labels = ax.get_legend_handles_labels()
        ax.legend(handles=layer_handles + thresh_handles, fontsize=8)

    panel_info = [
        (ax_ever,  'Ever-active features',   'Feature count'),
        (ax_mean,  'Mean active features',   'Mean active per token'),
        (ax_class, 'Classification error',   'Error rate'),
        (ax_ipr,   'Effective features (IPR)', 'IPR'),
    ]

    for ax, title, ylabel in panel_info:
        ax.set_xscale('log')
        if args.xlim:
            ax.set_xlim(args.xlim)
        ax.set_xlabel('λ₁', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)

    ax_class.legend(fontsize=8)
    ax_ipr.legend(fontsize=8)

    fig.suptitle('SAE metrics vs λ₁', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Figure saved to {outfile}')
    plt.close(fig)


if __name__ == '__main__':
    main()
