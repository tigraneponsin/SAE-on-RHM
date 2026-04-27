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
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


def _plot_layer(layer, entries, outfile, xlim):
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
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.5)
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])

    ax_agg = axes_flat[num_positions]
    for key, label, color in zip(ENTROPY_KEYS, ENTROPY_LABELS, ENTROPY_COLORS):
        short = key.replace('H_bar_', '').replace('_norm', '')
        y = np.nanmean(mats[f'H_{short}'], axis=1)
        ax_agg.plot(lambdas, y, '-o', color=color, label=label,
                    markersize=3, linewidth=1.2)
    ax_agg.set_title('mean over positions', fontsize=9, fontweight='bold')
    ax_agg.set_xscale('log')
    ax_agg.set_ylim(0.0, 1.05)
    ax_agg.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.5)
    if xlim is not None:
        ax_agg.set_xlim(xlim[0], xlim[1])

    for j in range(num_positions + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    for ax in axes[-1, :]:
        ax.set_xlabel('lambda_l1', fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel('H / H_theoretical', fontsize=9)

    modes = sorted({e['mode'] for e in entries if e['mode']})
    mode_str = modes[0] if len(modes) == 1 else ','.join(modes)
    fig.suptitle(
        f'Normalized entropy vs lambda_1 | layer {layer} | '
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

    by_layer = defaultdict(list)
    for e in entries:
        by_layer[e['layer_id']].append(e)

    prefix = args.outfile_prefix or str(artifacts_dir / 'entropy_lambda')

    print(f'Loaded {len(entries)} artifacts across {len(by_layer)} layer(s).')
    for layer in sorted(by_layer):
        outfile = f'{prefix}_layer{layer}.png'
        _plot_layer(layer, by_layer[layer], outfile, args.xlim)


if __name__ == '__main__':
    main()
