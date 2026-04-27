"""Plot SAE sparsity metrics + probe reconstruction error as a function of lambda_1.

Combines the CSV outputs of scripts/sae_eval/run.py and eval_probe.py to plot three panels:
  1. Ever-active features: binary (>0), >1% of max mean, >10% of max mean
  2. Mean active features per token: binary (>0), >1% of token max, >10% of token max
  3. Probe reconstruction error (recon_id_error_norm) with clean baseline reference

Usage:
    python sae_sweep/plot_probe_metrics.py \\
        --sweep_csv /path/to/eval_results.csv \\
        --probe_csv /path/to/probe_results.csv \\
        [--outfile /path/to/probe_metrics.png] \\
        [--xlim 1e-4 1] \\
        [--log-y]
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _load_sweep_csv(path):
    """Load eval_results.csv rows, grouped by layer and sorted by lambda_l1."""
    by_layer = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            by_layer[int(row['layer'])].append(row)
    for layer in by_layer:
        by_layer[layer].sort(key=lambda r: float(r['lambda_l1']))
    return dict(by_layer)


def _load_probe_csv(path):
    """Load probe_results.csv rows, grouped by (layer, token_idx) and sorted by lambda_l1."""
    by_pair = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get('lambda_l1') in ('', None, 'None'):
                continue
            key = (int(row['layer']), int(row['token_idx']))
            by_pair[key].append(row)
    for key in by_pair:
        by_pair[key].sort(key=lambda r: float(r['lambda_l1']))
    return dict(by_pair)


def _build_suptitle(sweep_data, probe_data, layers):
    """Build a contextual figure title from mode/token_idx in the sweep CSV rows."""
    all_sweep_rows = [r for rows in sweep_data.values() for r in rows]
    modes = sorted(set(r.get('mode', 'all_tokens') for r in all_sweep_rows))
    token_idxs = sorted(set(
        r.get('token_idx', '')
        for r in all_sweep_rows
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

    if len(layers) == 1:
        layer_label = f'Layer {layers[0]}'
    else:
        layer_label = f'Layers {",".join(str(l) for l in layers)}'

    return f'SAE metrics vs lambda_1  |  {layer_label}  |  {mode_label}'


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--sweep_csv', required=True,
                        help='Path to the sweep CSV emitted by scripts/sae_eval/run.py --outcsv')
    parser.add_argument('--probe_csv', required=True,
                        help='Path to probe_results.csv from eval_probe.py')
    parser.add_argument('--outfile', default=None,
                        help='Output figure path (default: <probe_csv_dir>/probe_metrics.png)')
    parser.add_argument('--xlim', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'),
                        help='Lambda axis limits, e.g. --xlim 1e-4 1')
    parser.add_argument('--log-y', action='store_true',
                        help='Use log scale on y-axis for ever-active and mean-active panels')
    args = parser.parse_args()

    sweep_data = _load_sweep_csv(args.sweep_csv)
    probe_data = _load_probe_csv(args.probe_csv)
    outfile = args.outfile or str(Path(args.probe_csv).parent / 'probe_metrics.png')

    layers = sorted(sweep_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(layers), 1)))
    layer_color = {l: colors[i] for i, l in enumerate(layers)}

    suptitle = _build_suptitle(sweep_data, probe_data, layers)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_ever  = axes[0]
    ax_mean  = axes[1]
    ax_probe = axes[2]

    thresh_styles = [
        ('solid',  'o', '>0 (all active)'),
        ('dashed', 's', '>1% of max'),
        ('dotted', '^', '>10% of max'),
    ]

    # Panels 1 and 2: sparsity metrics from eval_results.csv
    for layer in layers:
        rows = sweep_data[layer]
        lam = np.array([float(r['lambda_l1']) for r in rows])
        latent_dim = np.array([int(r['latent_dim']) for r in rows])
        dead = np.array([int(r['dead_features']) for r in rows])
        mean_active = np.array([float(r['mean_active']) for r in rows])
        above_1pct = np.array([int(r['active_above_1pct']) for r in rows])
        above_10pct = np.array([int(r['active_above_10pct']) for r in rows])
        mean_above_1pct = np.array([float(r['mean_active_above_1pct']) for r in rows])
        mean_above_10pct = np.array([float(r['mean_active_above_10pct']) for r in rows])

        if args.xlim:
            mask = (lam >= args.xlim[0]) & (lam <= args.xlim[1])
            lam = lam[mask]
            latent_dim, dead = latent_dim[mask], dead[mask]
            mean_active = mean_active[mask]
            above_1pct, above_10pct = above_1pct[mask], above_10pct[mask]
            mean_above_1pct, mean_above_10pct = mean_above_1pct[mask], mean_above_10pct[mask]

        c = layer_color[layer]

        for vals, (ls, mk, _) in zip(
            [latent_dim - dead, above_1pct, above_10pct], thresh_styles
        ):
            ax_ever.plot(lam, vals, color=c, linestyle=ls, marker=mk,
                         markersize=4, linewidth=1.5,
                         label=f'L{layer}' if ls == 'solid' else '_nolegend_')

        for vals, (ls, mk, _) in zip(
            [mean_active, mean_above_1pct, mean_above_10pct], thresh_styles
        ):
            ax_mean.plot(lam, vals, color=c, linestyle=ls, marker=mk,
                         markersize=4, linewidth=1.5,
                         label=f'L{layer}' if ls == 'solid' else '_nolegend_')

    # Panel 3: probe reconstruction error from probe_results.csv
    pairs = sorted(probe_data.keys())
    # Use linestyle to distinguish token_idx within the same layer
    token_idxs = sorted(set(tok for _, tok in pairs))
    linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    tok_style = {tok: linestyles[i % len(linestyles)] for i, tok in enumerate(token_idxs)}

    for (layer, token_idx) in pairs:
        rows = probe_data[(layer, token_idx)]
        lam = np.array([float(r['lambda_l1']) for r in rows])
        recon_err = np.array([float(r['recon_id_error_norm']) for r in rows])
        clean_err = float(rows[0]['clean_id_error_norm'])

        if args.xlim:
            mask = (lam >= args.xlim[0]) & (lam <= args.xlim[1])
            lam, recon_err = lam[mask], recon_err[mask]

        c = layer_color.get(layer, 'black')
        ls = tok_style[token_idx]
        label = f'L{layer} tok{token_idx}' if len(token_idxs) > 1 else f'Layer {layer}'
        ax_probe.plot(lam, recon_err, color=c, linestyle=ls, marker='o',
                      markersize=4, linewidth=1.5, label=label)

        # Baseline reference line (draw once per layer/token)
        ax_probe.axhline(clean_err, color=c, linestyle=':', linewidth=0.8, alpha=0.5)

    # Threshold legend for panels 1 and 2
    for ax in (ax_ever, ax_mean):
        thresh_handles = [
            plt.Line2D([0], [0], color='grey', linestyle=ls, marker=mk,
                       markersize=4, label=lbl)
            for ls, mk, lbl in thresh_styles
        ]
        layer_handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=layer_handles + thresh_handles, fontsize=8)

    panel_info = [
        (ax_ever,  'Ever-active features',        'Feature count'),
        (ax_mean,  'Mean active features',         'Mean active per token'),
        (ax_probe, 'Probe reconstruction error',   'Normalized ID error'),
    ]

    for ax, title, ylabel in panel_info:
        ax.set_xscale('log')
        if args.log_y and ax in (ax_ever, ax_mean):
            ax.set_yscale('log')
        if args.xlim:
            ax.set_xlim(args.xlim)
        ax.set_xlabel('lambda_1', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)

    ax_probe.legend(fontsize=8)

    fig.suptitle(suptitle, fontsize=13, y=1.03)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Figure saved to {outfile}')
    plt.close(fig)


if __name__ == '__main__':
    main()
