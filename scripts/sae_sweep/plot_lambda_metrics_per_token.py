"""Plot per-token SAE quality metrics as a function of lambda_1.

Reads the per-position CSV produced by scripts/sae_eval/run.py
(--per_position_csv) and plots two panels:
  1. Ever-active features: binary (>0), >1% of max mean, >10% of max mean
  2. Mean active features per token: binary (>0), >1% of token max, >10% of token max

Unlike plot_lambda_metrics.py (which reads the aggregated-over-tokens CSV),
this script keeps the leaf_position axis, so a subset of tokens can be
selected and either averaged together or drawn as one curve per token.

The classification-error panel is omitted: classification impact is a global
per-checkpoint metric with no per-token decomposition.

Usage:
    # average over tokens 0,3,7 (one curve per layer)
    python sae_sweep/plot_lambda_metrics_per_token.py \
        --csv /work/.../per_position_metrics.csv \
        --tokens 0 3 7 --agg mean

    # one curve per selected token, per layer
    python sae_sweep/plot_lambda_metrics_per_token.py \
        --csv /work/.../per_position_metrics.csv \
        --tokens 0 3 7 --agg per_token
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


# Per-position metric columns averaged / drawn per token.
_METRIC_COLS = (
    'mean_active', 'ever_active', 'dead_features',
    'active_above_1pct', 'active_above_10pct',
    'mean_active_above_1pct', 'mean_active_above_10pct',
)


def _load_csv(path, tokens=None):
    """Load per-position rows grouped by (layer, leaf_position).

    Returns dict[layer] -> dict[leaf_position] -> list of rows sorted by
    lambda_l1. If `tokens` is given, only those leaf positions are kept.
    """
    by_layer = defaultdict(lambda: defaultdict(list))
    token_set = set(tokens) if tokens is not None else None
    with open(path) as f:
        for row in csv.DictReader(f):
            pos = int(row['leaf_position'])
            if token_set is not None and pos not in token_set:
                continue
            by_layer[int(row['layer'])][pos].append(row)
    for layer in by_layer:
        for pos in by_layer[layer]:
            by_layer[layer][pos].sort(key=lambda r: float(r['lambda_l1']))
    return {l: dict(p) for l, p in by_layer.items()}


def _series_for_position(rows, xlim):
    """Return (lam, {metric: array}) for one layer/position's sorted rows."""
    lam = np.array([float(r['lambda_l1']) for r in rows])
    data = {col: np.array([float(r[col]) for r in rows]) for col in _METRIC_COLS}
    if xlim:
        mask = (lam >= xlim[0]) & (lam <= xlim[1])
        lam = lam[mask]
        data = {col: arr[mask] for col, arr in data.items()}
    return lam, data


def _average_positions(positions_data):
    """Average a list of (lam, data) over positions sharing the same lam grid.

    Assumes every position has the same lambda grid (true within a sweep).
    Returns (lam, {metric: mean array}).
    """
    lam = positions_data[0][0]
    stacked = {
        col: np.mean([d[col] for _, d in positions_data], axis=0)
        for col in _METRIC_COLS
    }
    return lam, stacked


def _build_suptitle(by_layer, layers, tokens, agg, report=False):
    """Build a contextual figure title from mode and the token selection."""
    all_rows = [
        r for pos_map in by_layer.values()
        for rows in pos_map.values() for r in rows
    ]
    modes = sorted(set(r.get('mode', 'all_tokens') for r in all_rows))
    mode_label = ','.join(modes)

    if len(layers) == 1:
        layer_label = sae_label(layers[0], report)
    else:
        layer_label = ', '.join(sae_label(l, report) for l in layers)

    if tokens is None:
        tok_label = 'all tokens'
    else:
        tok_label = f'tokens {",".join(str(t) for t in tokens)}'
    tok_label += f' ({agg})'

    return f'SAE per-token metrics vs lambda_1  |  {layer_label}  |  {mode_label}  |  {tok_label}'


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--csv', required=True,
                        help='Path to the per-position CSV emitted by '
                             'scripts/sae_eval/run.py --per_position_csv')
    parser.add_argument('--tokens', type=int, nargs='+', default=None,
                        help='Leaf positions (0-based) to include. Default: all.')
    parser.add_argument('--agg', choices=('mean', 'per_token'), default='mean',
                        help="'mean' averages the selected tokens into one curve "
                             "per layer; 'per_token' draws one curve per token.")
    parser.add_argument('--outfile', default=None,
                        help='Output figure path (default: '
                             '<csv_dir>/lambda_metrics_per_token.png)')
    parser.add_argument('--xlim', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'),
                        help='Lambda axis limits, e.g. --xlim 1e-2 1')
    parser.add_argument('--log-y', action='store_true',
                        help='Use log scale on the y-axis')
    add_report_flag(parser)
    args = parser.parse_args()

    by_layer = _load_csv(args.csv, tokens=args.tokens)
    if not by_layer:
        raise SystemExit(
            'No rows matched. Check --tokens against the leaf_position column '
            '(an all_tokens SAE is required for per-token plots).'
        )
    outfile = args.outfile or str(
        Path(args.csv).parent / 'lambda_metrics_per_token.png'
    )

    layers = sorted(by_layer.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(layers), 1)))
    layer_color = {l: colors[i] for i, l in enumerate(layers)}

    suptitle = _build_suptitle(by_layer, layers, args.tokens, args.agg,
                               args.report_notation)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax_ever = axes[0]
    ax_mean = axes[1]

    # Linestyle legend entries (drawn once, outside the layer loop).
    thresh_styles = [
        ('solid',  'o', '>0 (all active)'),
        ('dashed', 's', '>1% of max'),
        ('dotted', '^', '>10% of max'),
    ]

    def _plot_curves(lam, data, color, label):
        ever_series = [
            data['ever_active'], data['active_above_1pct'], data['active_above_10pct'],
        ]
        mean_series = [
            data['mean_active'], data['mean_active_above_1pct'],
            data['mean_active_above_10pct'],
        ]
        for vals, (ls, mk, _) in zip(ever_series, thresh_styles):
            ax_ever.plot(lam, vals, color=color, linestyle=ls, marker=mk,
                         markersize=4, linewidth=1.5,
                         label=label if ls == 'solid' else '_nolegend_')
        for vals, (ls, mk, _) in zip(mean_series, thresh_styles):
            ax_mean.plot(lam, vals, color=color, linestyle=ls, marker=mk,
                         markersize=4, linewidth=1.5,
                         label=label if ls == 'solid' else '_nolegend_')

    for layer in layers:
        pos_map = by_layer[layer]
        positions = sorted(pos_map.keys())
        c = layer_color[layer]

        if args.agg == 'mean':
            positions_data = [
                _series_for_position(pos_map[p], args.xlim) for p in positions
            ]
            lam, data = _average_positions(positions_data)
            _plot_curves(lam, data, c,
                         sae_label(layer, args.report_notation, short=True))
        else:  # per_token
            shades = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(positions), 1)))
            for shade, p in zip(shades, positions):
                lam, data = _series_for_position(pos_map[p], args.xlim)
                _plot_curves(lam, data, shade,
                             f'{sae_label(layer, args.report_notation, short=True)} tok{p}')

    # Threshold (linestyle) legend appended to both panels.
    for ax in (ax_ever, ax_mean):
        thresh_handles = [
            plt.Line2D([0], [0], color='grey', linestyle=ls, marker=mk,
                       markersize=4, label=lbl)
            for ls, mk, lbl in thresh_styles
        ]
        series_handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=series_handles + thresh_handles, fontsize=8)

    panel_info = [
        (ax_ever, 'Ever-active features', 'Feature count'),
        (ax_mean, 'Mean active features', 'Mean active per token'),
    ]
    for ax, title, ylabel in panel_info:
        ax.set_xscale('log')
        if args.log_y:
            ax.set_yscale('log')
        if args.xlim:
            ax.set_xlim(args.xlim)
        ax.set_xlabel('lambda_1', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)

    fig.suptitle(suptitle, fontsize=13, y=1.03)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Figure saved to {outfile}')
    plt.close(fig)


if __name__ == '__main__':
    main()
