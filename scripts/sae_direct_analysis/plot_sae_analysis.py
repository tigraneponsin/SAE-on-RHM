"""Plot the per-feature latent-selectivity artifacts from analyze_sae.py.

For a single .feature_latent.pt artifact, produces one figure type:

  1. expected_<ckpt>_pos<p>.png -- for each analyzed real-token position p,
      the top-K features ranked by selectivity to the target that the
      hierarchy hypothesis predicts this layer should track:
          level  = L - 1 - layer_id
          pos_j  = p // s^(1 + layer_id)
      Grouped bar chart of E[f | target=v] for each value v of that target,
      for the top K features.

Usage:
    python scripts/sae_direct_analysis/plot_sae_analysis.py \\
        --artifact /path/to/<ckpt>.feature_latent.pt \\
        [--out_dir /path/to/figures/]          \\
        [--top_k 12]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


def _load_artifact(path: Path) -> dict:
    return torch.load(str(path), map_location='cpu', weights_only=False)


def _expected_target(layer_id: int, token_pos: int, rhm: dict):
    """Given an SAE at layer k and real-token index p, return (level, j) of
    the latent the hierarchy hypothesis says it should track.
    """
    L = int(rhm['L'])
    s = int(rhm['s'])
    level = L - 1 - int(layer_id)
    if level < 0 or level > L:
        return None
    if token_pos < 0:
        return None  # CLS: no expected latent position
    j = int(token_pos) // (s ** (1 + int(layer_id)))
    return level, j


def _find_target_indices(targets: list, level: int, position: int):
    """Return sorted list of target indices matching (level, position)."""
    return [i for i, t in enumerate(targets)
            if int(t['level']) == level and int(t['position']) == position]


def _plot_expected(artifact: dict, pos_idx: int, top_k: int, out_path: Path):
    targets = artifact['targets']
    cond_mean = artifact['conditional_mean']            # [T, P, F]
    token_pos = int(artifact['token_positions'][pos_idx].item())
    rhm = artifact['rhm']
    layer_id = int(artifact['layer_id'])

    expected = _expected_target(layer_id, token_pos, rhm)
    if expected is None:
        return False
    level, j = expected
    idxs = _find_target_indices(targets, level, j)
    if not idxs:
        return False

    # Sub-matrix: [V, F] of conditional means at this (level, j, pos_idx).
    sub = cond_mean[idxs, pos_idx, :].clone()           # [V, F]
    sub[torch.isnan(sub)] = 0.0

    # Rank features by max_v |cond - baseline| at the expected target.
    baseline = artifact['baseline_mean'][pos_idx, :]    # [F]
    score = (sub - baseline.unsqueeze(0)).abs().amax(dim=0)  # [F]
    top = torch.argsort(score, descending=True)[:top_k].numpy()

    values = [int(targets[i]['value']) for i in idxs]
    heights = sub[:, top].numpy()                       # [V, K]
    K = heights.shape[1]
    V = heights.shape[0]

    fig_w = max(6.0, 0.8 * K + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 4.0))
    bar_w = 0.8 / V
    x = np.arange(K)
    for vi, v in enumerate(values):
        ax.bar(x + (vi - (V - 1) / 2.0) * bar_w, heights[vi], width=bar_w,
               label=f'v={v}')
    ax.axhline(0.0, color='black', linewidth=0.5, alpha=0.5)
    # Baseline reference line (average baseline over the top-K features).
    ax.axhline(float(baseline[top].mean().item()), color='gray',
               linewidth=0.8, linestyle='--', alpha=0.7, label='baseline (mean)')

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(f)) for f in top], rotation=0, fontsize=8)
    ax.set_xlabel(f'SAE feature id  (top-{K} by selectivity)')
    ax.set_ylabel('E[ f_i | target ]  (decoder-weighted)')
    title = (f'{Path(artifact["ckpt_path"]).name}  layer={layer_id}  '
             f'token={token_pos}  expected target = trees[{level}][:, {j}]')
    ax.set_title(title, fontsize=9)
    ax.legend(loc='best', fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--artifact', required=True,
                        help='Path to a .feature_latent.pt from analyze_sae.py')
    parser.add_argument('--out_dir', default=None,
                        help='Directory to write figures to (default: next to the artifact)')
    parser.add_argument('--top_k', type=int, default=12,
                        help='Number of top features to show in the expected-target bar chart')
    args = parser.parse_args()

    artifact_path = Path(args.artifact)
    artifact = _load_artifact(artifact_path)

    out_dir = Path(args.out_dir) if args.out_dir else artifact_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = artifact_path.stem.replace('.feature_latent', '')

    num_positions = int(artifact['token_positions'].numel())
    print(f'Plotting {artifact_path.name}: '
          f'T={len(artifact["targets"])}, '
          f'P={num_positions}, '
          f'F={artifact["latent_dim"]}')

    for pos_idx in range(num_positions):
        expected_out = out_dir / f'expected_{stem}_pos{pos_idx:02d}.png'
        ok = _plot_expected(artifact, pos_idx, args.top_k, expected_out)
        if ok:
            print(f'  wrote {expected_out}')
        else:
            print(f'  skipped expected-target plot for pos_idx={pos_idx} '
                  '(CLS token or out-of-range layer)')


if __name__ == '__main__':
    main()
