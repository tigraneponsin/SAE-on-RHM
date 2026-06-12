"""Plot normalized entropy aggregates vs lambda_1 from a sweep of
.sae_eval.pt artifacts.

For each transformer layer found in the artifacts directory, produces a
figure with s^L + 1 subplots:
  - The first s^L subplots show, for each leaf position p, the three
    normalized entropy aggregates H_bar_{fire,raw,dec}_norm[p] as a
    function of lambda_l1.
  - The last subplot shows the same three curves averaged over all leaf
    positions (nanmean).

Input: a directory of *.sae_eval.pt artifacts produced by
scripts/sae_eval/run.py with --with-entropy (or --with-all).

Usage:
    python sae_sweep/plot_entropy_lambda.py \\
        --artifacts_dir /path/to/sweep/sae_eval_artifacts/ \\
        [--outfile_prefix /path/to/entropy_lambda] \\
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
from notation import sae_label, add_report_flag
import numpy as np
import torch


ENTROPY_KEYS = ('H_bar_fire_norm', 'H_bar_raw_norm', 'H_bar_dec_norm')
ENTROPY_LABELS = ('H_fire', 'H_raw', 'H_dec')
ENTROPY_COLORS = ('C0', 'C1', 'C2')


def _load_entries(files):
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


def _build_matrix(entries_sorted, key, num_positions):
    """Stack entry[key] into [num_artifacts, num_positions], placing each
    value at its token_position index. Unfilled cells stay NaN."""
    mat = np.full((len(entries_sorted), num_positions), np.nan, dtype=np.float64)
    for i, e in enumerate(entries_sorted):
        for p_idx, p in enumerate(e['token_positions']):
            if 0 <= p < num_positions:
                mat[i, p] = e[key][p_idx]
    return mat


def _find_threshold_lambda(csv_path, layer, mode, tolerance):
    """Smallest lambda_l1 for which norm_err - baseline_err exceeds tolerance.

    Reads sweep_metrics.csv, filters rows matching (layer, mode), and returns
    the threshold lambda or None if no row crosses it (or the csv is missing /
    malformed).
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        print(f'  threshold: csv not found at {csv_path}, skipping line')
        return None
    rows = []
    try:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row['layer']) != int(layer):
                    continue
                if mode and row.get('mode', '') and row['mode'] != mode:
                    continue
                try:
                    lam = float(row['lambda_l1'])
                    ne = float(row['norm_err'])
                    be = float(row['baseline_err'])
                except (KeyError, ValueError):
                    continue
                rows.append((lam, ne, be))
    except Exception as ex:
        print(f'  threshold: failed to read {csv_path}: {ex}')
        return None
    if not rows:
        print(f'  threshold: no rows for layer={layer} mode={mode} in {csv_path.name}')
        return None
    rows.sort(key=lambda r: r[0])
    for lam, ne, be in rows:
        if (ne - be) > tolerance:
            print(f'  threshold (layer={layer} mode={mode}): lambda_l1={lam:g} '
                  f'(norm_err={ne:.4g}, baseline_err={be:.4g}, '
                  f'tolerance={tolerance})')
            return lam
    print(f'  threshold: norm_err never exceeds baseline + {tolerance} for '
          f'layer={layer} mode={mode}')
    return None


def _plot_meanpool_layer(layer, entries, outfile, xlim, threshold_lambda, tolerance,
                         report=False):
    """Single-panel plot for mean_pooled artifacts.

    Mean_pooled SAEs have one entropy value per artifact (P=1), already
    conditioned on the root class. No per-position grid is meaningful.
    """
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


def _plot_layer(layer, entries, outfile, xlim, threshold_lambda, tolerance,
                report=False):
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

    mats = {k: _build_matrix(entries, k, num_positions)
            for k in ('H_fire', 'H_raw', 'H_dec')}

    total_panels = num_positions + 1
    nrows = max(1, int(math.ceil(math.sqrt(total_panels))))
    ncols = int(math.ceil(total_panels / nrows))

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

    for j in range(num_positions + 1, len(axes_flat)):
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

    handles, labels = ax_agg.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=len(labels), bbox_to_anchor=(0.5, -0.02), fontsize=10)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'  layer {layer}: {len(entries)} artifacts, '
          f'lambda range [{lambdas.min():.3g}, {lambdas.max():.3g}] '
          f'-> {outfile}')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--artifacts_dir', required=True,
                        help='Directory of *.sae_eval.pt artifacts '
                             '(produced with --with-entropy).')
    parser.add_argument('--outfile_prefix', default=None,
                        help='Output prefix; figures land at '
                             '<prefix>_layer{k}.png. Default: '
                             '<artifacts_dir>/entropy_lambda.')
    parser.add_argument('--xlim', type=float, nargs=2, default=None,
                        metavar=('XMIN', 'XMAX'),
                        help='Optional log-x axis limits shared across subplots.')
    parser.add_argument('--csv_path', default=None,
                        help='Path to sweep_metrics.csv used to locate the '
                             'error-onset threshold lambda. Default: '
                             '<artifacts_dir>/sweep_metrics.csv.')
    parser.add_argument('--err_tolerance', type=float, default=0.01,
                        help='Additive tolerance on (norm_err - baseline_err) '
                             'used to define the threshold lambda. Default: 0.01.')
    add_report_flag(parser)
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.is_dir():
        raise SystemExit(f'--artifacts_dir is not a directory: {artifacts_dir}')
    files = sorted(artifacts_dir.glob('*.sae_eval.pt'))
    if not files:
        raise SystemExit(f'No *.sae_eval.pt files in {artifacts_dir}')

    entries = _load_entries(files)
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
        threshold_lambda = _find_threshold_lambda(
            csv_path, layer, mode_for_csv, args.err_tolerance,
        )
        if is_meanpool:
            outfile = f'{prefix}_layer{layer}_meanpool.png'
            _plot_meanpool_layer(layer, layer_entries, outfile, args.xlim,
                                 threshold_lambda, args.err_tolerance,
                                 args.report_notation)
        else:
            outfile = f'{prefix}_layer{layer}.png'
            _plot_layer(layer, layer_entries, outfile, args.xlim,
                        threshold_lambda, args.err_tolerance,
                        args.report_notation)


if __name__ == '__main__':
    main()
