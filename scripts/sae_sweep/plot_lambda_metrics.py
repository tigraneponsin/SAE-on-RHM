"""Plot SAE quality metrics as a function of lambda.

Reads the CSV produced by scripts/sae_eval/run.py and plots three panels:
  1. Ever-active features: binary (>0)
  2. Mean active features per token: binary (>0)
  3. Classification error (with baseline reference)

Usage:
    python sae_sweep/plot_lambda_metrics.py \
        --csv /work/pcsl/ponsin/Mean_Transformer/SAE/sweep_round2_lambda/eval_results.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common.notation import sae_label, add_report_flag


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


def _build_suptitle(by_layer, layers, report=False):
    """Build a contextual figure title from mode/token_idx in the CSV rows."""
    if len(layers) == 1:
        layer_label = sae_label(layers[0], report)
    else:
        layer_label = ', '.join(sae_label(l, report) for l in layers)

    if report:
        return rf'SAE metrics vs $\lambda$  |  {layer_label}'

    all_rows = [r for rows in by_layer.values() for r in rows]
    modes = sorted(set(r.get('mode', 'all_tokens') for r in all_rows))
    token_idxs = sorted(set(
        r.get('token_idx', '')
        for r in all_rows
        if r.get('token_idx') not in ('', None, 'None')
    ))

    if len(modes) == 1:
        mode_str = modes[0]
        if mode_str == 'one_token' and len(token_idxs) == 1:
            mode_label = f'one_token (tok={token_idxs[0]})'
        elif mode_str == 'one_token':
            mode_label = f'one_token (tok={",".join(token_idxs)})'
        else:
            mode_label = mode_str
    else:
        mode_label = ','.join(modes)

    return rf'SAE metrics vs $\lambda$  |  {layer_label}  |  {mode_label}'


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--csv', required=True,
                        help='Path to the sweep CSV emitted by scripts/sae_eval/run.py --outcsv')
    parser.add_argument('--outfile', default=None,
                        help='Output figure path (default: <csv_dir>/lambda_metrics.png)')
    parser.add_argument('--xlim', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'),
                        help='Lambda axis limits, e.g. --xlim 1e-2 1')
    parser.add_argument('--log-y', dest='log_y', action='store_true', default=True,
                        help='Use log scale on the y-axis for ever-active and mean-active '
                             'panels (default: on)')
    parser.add_argument('--no-log-y', dest='log_y', action='store_false',
                        help='Use linear scale on the y-axis instead')
    add_report_flag(parser)
    args = parser.parse_args()

    by_layer = _load_csv(args.csv)
    outfile = args.outfile or str(Path(args.csv).parent / 'lambda_metrics.png')

    layers = sorted(by_layer.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(layers), 1)))
    layer_color = {l: colors[i] for i, l in enumerate(layers)}

    suptitle = _build_suptitle(by_layer, layers, args.report_notation)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.25))
    ax_ever  = axes[0]
    ax_mean  = axes[1]
    ax_class = axes[2]

    for layer in layers:
        rows = by_layer[layer]
        lam = np.array([float(r['lambda_l1']) for r in rows])
        latent_dim = np.array([int(r['latent_dim']) for r in rows])
        dead = np.array([int(r['dead_features']) for r in rows])
        mean_active = np.array([float(r['mean_active']) for r in rows])
        norm_err = np.array([float(r['norm_err']) for r in rows])

        # Filter to xlim range if specified
        if args.xlim:
            mask = (lam >= args.xlim[0]) & (lam <= args.xlim[1])
            lam, latent_dim, dead = lam[mask], latent_dim[mask], dead[mask]
            mean_active, norm_err = mean_active[mask], norm_err[mask]

        c = layer_color[layer]

        # Panel 1: ever-active features (>0)
        ax_ever.plot(lam, latent_dim - dead, color=c, linestyle='solid',
                     marker='o', markersize=4, linewidth=1.5,
                     label=sae_label(layer, args.report_notation, short=True))

        # Panel 2: mean active features (>0)
        ax_mean.plot(lam, mean_active, color=c, linestyle='solid',
                     marker='o', markersize=4, linewidth=1.5,
                     label=sae_label(layer, args.report_notation, short=True))

        # Panel 3: normalized classification error
        ax_class.plot(lam, norm_err, '-o', color=c,
                      label=sae_label(layer, args.report_notation), markersize=4)

    # Normalized baseline error reference line
    first_row = next(iter(by_layer.values()))[0]
    baseline_err = float(first_row['baseline_err'])
    random_err = 1.0 - 1.0 / len(by_layer)  # fallback; use actual ratio from CSV
    # norm_baseline = baseline_err / random_err, but we can read it directly
    # from the CSV: baseline_err is already in the row, random_err = baseline's
    # denominator. Since norm_err = sae_err / random_err, the normalized baseline
    # is baseline_err / random_err. We recover random_err from the first row.
    first_norm = float(first_row['norm_err'])
    first_sae = float(first_row['sae_err'])
    random_err = first_sae / first_norm if first_norm != 0 else 1.0
    norm_baseline = baseline_err / random_err
    ax_class.axhline(norm_baseline, color='grey', linestyle='--', linewidth=1,
                      label=f'Baseline ({norm_baseline:.2f})')

    # Layer legend on panels 1 and 2
    for ax in (ax_ever, ax_mean):
        ax.legend(fontsize=8)

    mean_ylabel = r'Mean $L_0$' if args.report_notation else 'Mean active per token'
    panel_info = [
        (ax_ever,  'Ever-active features',   'Feature count'),
        (ax_mean,  'Mean active features',   mean_ylabel),
        (ax_class, 'Normalized classification error',   'Normalized error'),
    ]

    for ax, title, ylabel in panel_info:
        ax.set_xscale('log')
        if args.log_y and ax in (ax_ever, ax_mean):
            ax.set_yscale('log')
        if args.xlim:
            ax.set_xlim(args.xlim)
        ax.set_xlabel(r'$\lambda$', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)

    ax_class.legend(fontsize=8)

    fig.suptitle(suptitle, fontsize=13, y=1.03)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Figure saved to {outfile}')
    plt.close(fig)


if __name__ == '__main__':
    main()
