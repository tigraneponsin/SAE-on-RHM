"""Plot histograms of per-feature mean weighted activations.

Shows, for each layer, the distribution of mean activation magnitudes across
features, with one histogram per lambda_1 value. Log-scale bins reveal both
the dominant features and the long tail of near-zero activations.

Input: a directory of *.sae_eval.pt artifacts produced by scripts/sae_eval/run.py
(with at least --with-per-feature so feature_mean_activations is populated).

Usage:
    python sae_sweep/plot_rank_magnitude.py \
        --artifacts_dir /path/to/sweep/sae_eval_artifacts/
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common.notation import sae_label, add_report_flag


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--artifacts_dir', required=True,
                        help='Directory of *.sae_eval.pt artifacts from scripts/sae_eval/run.py')
    parser.add_argument('--outfile', default=None,
                        help='Output figure path (default: <artifacts_dir>/activation_histograms.png)')
    parser.add_argument('--bins', type=int, default=50,
                        help='Number of histogram bins (default: 50)')
    add_report_flag(parser)
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.is_dir():
        raise SystemExit(f'--artifacts_dir is not a directory: {artifacts_dir}')
    files = sorted(artifacts_dir.glob('*.sae_eval.pt'))
    if not files:
        raise SystemExit(f'No *.sae_eval.pt files in {artifacts_dir}')
    outfile = args.outfile or str(artifacts_dir / 'activation_histograms.png')

    by_layer = defaultdict(list)
    for f in files:
        art = torch.load(f, map_location='cpu', weights_only=False)
        feat = art.get('feature_mean_activations')
        if feat is None:
            print(f'  skipping {f.name}: no feature_mean_activations '
                  '(re-run with --with-per-feature)')
            continue
        by_layer[int(art['layer_id'])].append({
            'layer': int(art['layer_id']),
            'lambda_l1': float(art.get('lambda_l1') or 0.0),
            'feature_mean_activations': feat,
        })

    layers = sorted(by_layer.keys())
    n_layers = len(layers)

    # Collect all lambda values for consistent coloring
    all_lambdas = sorted({float(e['lambda_l1']) for entries in by_layer.values() for e in entries})
    cmap = cm.get_cmap('viridis', max(len(all_lambdas), 1))
    lam_color = {lam: cmap(i) for i, lam in enumerate(all_lambdas)}

    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5), squeeze=False)

    for col, layer in enumerate(layers):
        ax = axes[0][col]
        entries = sorted(by_layer[layer], key=lambda e: float(e['lambda_l1']))

        # Compute shared log-scale bins from all entries in this layer
        all_positive = []
        for entry in entries:
            feat = entry['feature_mean_activations'].numpy()
            positive = feat[feat > 0]
            if len(positive) > 0:
                all_positive.append(positive)

        if all_positive:
            all_pos = np.concatenate(all_positive)
            bin_edges = np.logspace(np.log10(all_pos.min()), np.log10(all_pos.max()), args.bins + 1)
        else:
            bin_edges = np.logspace(-10, 0, args.bins + 1)

        for entry in entries:
            lam = float(entry['lambda_l1'])
            feat = entry['feature_mean_activations'].numpy()
            positive = feat[feat > 0]

            if len(positive) == 0:
                continue

            ax.hist(positive, bins=bin_edges, color=lam_color[lam],
                    alpha=0.5, label=f'λ₁={lam:.3g}', edgecolor='none')

        ax.set_xscale('log')
        ax.set_xlabel('Mean weighted activation', fontsize=11)
        ax.set_ylabel('Number of features', fontsize=11)
        ax.set_title(sae_label(layer, args.report_notation), fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)

    # Shared legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, title='λ₁', loc='lower center',
               ncol=min(len(all_lambdas), 9), bbox_to_anchor=(0.5, -0.04), fontsize=9)

    fig.suptitle('Distribution of per-feature mean weighted activations', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Figure saved to {outfile}')
    plt.close(fig)


if __name__ == '__main__':
    main()
