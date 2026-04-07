"""Plot SAE loss curves from multiple sweep directories on the same figure.

Usage:

    python sae_sweep/plot_multi_sweep.py \\
        --sweep_dirs /work/.../sweep_round1a_bs /work/.../sweep_round1b_steps \\
        --labels "best bs" "best steps" \\
        --outfile /work/.../comparison.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the curve loader from plot_loss_curves
from plot_loss_curves import _load_curves, LOSS_LABELS


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--sweep_dirs', nargs='+', required=True,
                   help='Paths to sweep directories containing .pt checkpoints')
    p.add_argument('--labels', nargs='+', default=None,
                   help='Legend labels for each sweep directory (default: directory names)')
    p.add_argument('--outfile', type=str, required=True,
                   help='Output figure path')
    p.add_argument('--loss', type=str, default='all',
                   choices=['total', 'recon', 'sparse', 'all'])
    p.add_argument('--max_steps', type=int, default=None)
    p.add_argument('--layer', type=str, default=None,
                   help='Comma-separated layers to plot (default: all found)')
    return p.parse_args()


def main():
    args = _parse_args()

    sweep_dirs = [Path(d) for d in args.sweep_dirs]
    labels = args.labels or [d.name for d in sweep_dirs]
    if len(labels) != len(sweep_dirs):
        print('ERROR: number of --labels must match number of --sweep_dirs', file=sys.stderr)
        sys.exit(1)

    if args.loss == 'all':
        loss_types = ['total', 'recon', 'sparse']
    else:
        loss_types = [args.loss]

    layer_filter = None
    if args.layer is not None:
        layer_filter = set(int(x.strip()) for x in args.layer.split(','))

    # Load records from all directories, tagging each with its source
    all_records = []
    for sweep_dir, label in zip(sweep_dirs, labels):
        ckpt_files = sorted(sweep_dir.glob('*.pt'))
        if not ckpt_files:
            print(f'WARNING: No .pt files in {sweep_dir}')
            continue
        for f in ckpt_files:
            r = _load_curves(str(f))
            if r is not None:
                r['source_label'] = label
                all_records.append(r)

    if not all_records:
        print('No valid checkpoints found across all directories.')
        sys.exit(0)

    layers = sorted({r['layer'] for r in all_records})
    if layer_filter is not None:
        layers = [l for l in layers if l in layer_filter]
    source_labels = list(dict.fromkeys(r['source_label'] for r in all_records))

    n_rows = len(layers)
    n_cols = len(loss_types)

    cmap = cm.get_cmap('tab10', max(len(source_labels), 1))
    label_color = {lab: cmap(i) for i, lab in enumerate(source_labels)}

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    for row, layer in enumerate(layers):
        layer_recs = [r for r in all_records if r['layer'] == layer]

        for col, loss_key in enumerate(loss_types):
            ax = axes[row][col]

            for r in layer_recs:
                steps = r['step']
                if args.max_steps is not None:
                    mask = steps <= args.max_steps
                    steps = steps[mask]
                    vals = r[loss_key][mask]
                else:
                    vals = r[loss_key]

                pos = steps > 0
                ax.plot(
                    steps[pos], vals[pos],
                    color=label_color[r['source_label']],
                    linewidth=1.5, alpha=0.8,
                    label=r['source_label'],
                )

            ax.set_xscale('log')
            ax.set_xlabel('Steps', fontsize=10)
            ax.set_ylabel(LOSS_LABELS[loss_key], fontsize=10)
            ax.set_title(f'Layer {layer}', fontsize=11)
            ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)

    # Deduplicated legend
    handles_seen = {}
    for ax_row in axes:
        for ax in ax_row:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in handles_seen:
                    handles_seen[l] = h

    fig.legend(
        list(handles_seen.values()), list(handles_seen.keys()),
        title='Sweep',
        loc='lower center',
        ncol=min(len(source_labels), 6),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
    )

    fig.suptitle('Cross-sweep comparison', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(args.outfile, dpi=150, bbox_inches='tight')
    print(f'Figure saved to {args.outfile}')
    plt.close(fig)


if __name__ == '__main__':
    main()
