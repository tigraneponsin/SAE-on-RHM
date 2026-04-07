"""Plot histograms of per-feature mean weighted activations.

Shows, for each layer, the distribution of mean activation magnitudes across
features, with one histogram per lambda_1 value. Log-scale bins reveal both
the dominant features and the long tail of near-zero activations.

Input: the .features.pt file produced by eval_sweep.py alongside the CSV.

Usage:
    python sae_sweep/plot_rank_magnitude.py \
        --features /work/pcsl/ponsin/Mean_Transformer/SAE/sweep_round2_lambda/eval_results.features.pt
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--features', required=True,
                        help='Path to .features.pt file from eval_sweep.py')
    parser.add_argument('--outfile', default=None,
                        help='Output figure path (default: <features_dir>/activation_histograms.png)')
    parser.add_argument('--bins', type=int, default=50,
                        help='Number of histogram bins (default: 50)')
    args = parser.parse_args()

    data = torch.load(args.features, map_location='cpu')
    outfile = args.outfile or str(Path(args.features).parent / 'activation_histograms.png')

    # Group by layer
    by_layer = defaultdict(list)
    for ckpt_name, entry in data.items():
        by_layer[int(entry['layer'])].append(entry)

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
        ax.set_title(f'Layer {layer}', fontsize=12)
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
