"""Plot tree reconstruction accuracy from .tree_recon.csv or .pt outputs.

Produces one figure :
  1. Per-level accuracy (activation vs uniform weighting)

Usage:
    python scripts/sae_tree_reconstruction/plot_results.py \\
        --csv /path/to/<stem>.tree_recon.csv \\
        [--outfile /path/to/plot.png]

    # Or load from .pt files directly:
    python scripts/sae_tree_reconstruction/plot_results.py \\
        --pt /path/to/<stem>.tree_recon.activation.pt \\
            /path/to/<stem>.tree_recon.uniform.pt \\
        [--outfile /path/to/plot.png]
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _load_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                'weighting': row['weighting'],
                'level': int(row['level']),
                'per_level_acc': float(row['per_level_acc']),
                'whole_tree_acc': float(row['whole_tree_acc']),
            })
    return rows


def _load_pt(paths):
    import torch
    rows = []
    for p in paths:
        blob = torch.load(str(p), map_location='cpu', weights_only=False)
        m = blob['metrics']
        w = blob['weighting']
        whole = m['whole_tree_acc']
        for l, acc in m['per_level_acc'].items():
            rows.append({
                'weighting': w,
                'level': int(l),
                'per_level_acc': acc,
                'whole_tree_acc': whole,
            })
    return rows


def _plot(rows, outfile):
    weightings = sorted(set(r['weighting'] for r in rows))
    # levels in the CSV are RHM levels (0=root, L-1=just above leaves).
    # We plot against transformer layer k = L-1-level so x=0 is first-above-leaves,
    # x=L-1 is root.
    levels = sorted(set(r['level'] for r in rows))
    L = len(levels)

    def _layer_label(k):
        if k == 0:
            return f"layer {k}\n(above leaves)"
        if k == L - 1:
            return f"layer {k}\n(root)"
        return f"layer {k}"

    colors = {'activation': '#1f77b4', 'uniform': '#ff7f0e'}
    markers = {'activation': 'o', 'uniform': 's'}

    fig, ax = plt.subplots(figsize=(5, 4))

    for w in weightings:
        wrows = sorted([r for r in rows if r['weighting'] == w],
                       key=lambda r: r['level'])
        # transformer layer k = L-1-level
        xs = [L - 1 - r['level'] for r in wrows]
        accs = [r['per_level_acc'] for r in wrows]
        whole = wrows[0]['whole_tree_acc'] if wrows else None
        label = f"{w} (whole={whole:.3f})" if whole is not None else w
        ax.plot(xs, accs, marker=markers.get(w, 'o'),
                color=colors.get(w), label=label, linewidth=1.5)

    layer_ticks = list(range(L))
    ax.set_xticks(layer_ticks)
    ax.set_xticklabels([_layer_label(k) for k in layer_ticks], rotation=20, ha='right')
    ax.set_xlabel('Transformer layer')
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Accuracy')
    ax.set_title('Tree reconstruction accuracy per layer')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Saved: {outfile}')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--csv', type=str, help='.tree_recon.csv file')
    src.add_argument('--pt', type=str, nargs='+',
                     help='One or two .tree_recon.*.pt files')
    p.add_argument('--outfile', type=str, default=None,
                   help='Output PNG path (default: next to input file)')
    args = p.parse_args()

    if args.csv:
        rows = _load_csv(args.csv)
        default_out = Path(args.csv).with_suffix('.png')
    else:
        rows = _load_pt([Path(x) for x in args.pt])
        default_out = Path(args.pt[0]).with_suffix('.png')

    outfile = Path(args.outfile) if args.outfile else default_out
    _plot(rows, outfile)


if __name__ == '__main__':
    main()
