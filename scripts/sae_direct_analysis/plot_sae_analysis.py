"""Plot the per-feature latent-selectivity artifacts from analyze_sae.py.

For a single .feature_latent.pt artifact, produces two figure types per
analyzed real-token position p. Both use the same top-K features, ranked by
selectivity to the target the hierarchy hypothesis predicts this layer
should track:
    level  = L - 1 - layer_id
    pos_j  = p // s^(1 + layer_id)
with selectivity score = max_v |E[f|v] - baseline|.

  1. expected_<ckpt>_pos<p>.png -- grouped bar chart with x = top-K feature
      ids and V colored bars per feature (one per value v of the expected
      latent). Good for comparing features at a glance.

  2. per_feature_<ckpt>_pos<p>.png -- grid of K subplots, one per top-K
      feature. In each subplot, x = value v of the expected latent and the
      bars are E[f_i | v]. Gray dashed line = baseline for that feature.
      Good for checking whether a single feature responds to multiple
      values of the same latent (several tall bars in one subplot) or is
      exclusively selective to one value (a single peak).

Disable the per-feature grid with --skip_per_feature.

Usage:
    python scripts/sae_direct_analysis/plot_sae_analysis.py \\
        --artifact /path/to/<ckpt>.feature_latent.pt \\
        [--out_dir /path/to/figures/]          \\
        [--top_k 12]                           \\
        [--skip_per_feature]
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

    baseline = artifact['baseline_mean'][pos_idx, :]    # [F]
    F_total = int(artifact['latent_dim'])

    # Select top-K features by max_v |cond - baseline|, but keep their real IDs.
    score = (sub - baseline.unsqueeze(0)).abs().amax(dim=0)  # [F]
    top = torch.argsort(score, descending=True)[:top_k].numpy()

    values = [int(targets[i]['value']) for i in idxs]
    heights = sub.numpy()                               # [V, F]
    V = heights.shape[0]

    # Visual style: use tab20 colors first; only add hatch patterns when V > 20.
    _HATCHES = ['', '//', '\\\\', 'xx', '..', 'oo', '++', '**', 'OO', '--']
    cmap = plt.get_cmap('tab20')
    colors = [cmap(vi % 20) for vi in range(V)]
    hatches = [_HATCHES[vi // 20] if V > 20 else '' for vi in range(V)]

    # Sort top by feature id so x-axis is monotonically increasing.
    top = top[np.argsort(top)]

    K = len(top)
    fig_w = max(6.0, 0.8 * K + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 4.0))
    bar_w = 0.8 / V
    # Pack bars at positions 0..K-1; label ticks with real feature ids.
    x = np.arange(K)
    for vi, v in enumerate(values):
        y = heights[vi, top]
        ax.bar(x + (vi - (V - 1) / 2.0) * bar_w, y, width=bar_w,
               color=colors[vi], hatch=hatches[vi], edgecolor='black',
               linewidth=0.4, label=f'v={v}')
    ax.axhline(0.0, color='black', linewidth=0.5, alpha=0.5)
    ax.axhline(float(baseline[top].mean().item()), color='gray',
               linewidth=0.8, linestyle='--', alpha=0.7, label='baseline (mean)')

    ax.set_xticks(x)
    ax.set_xticklabels([str(f) for f in top], rotation=45, ha='right', fontsize=8)
    ax.set_xlim(-0.5, K - 0.5)
    ax.set_xlabel('SAE feature id (top-K by selectivity, ordered by id)')
    ax.set_ylabel('E[ f_i | target ]  (decoder-weighted)')
    title = (f'{Path(artifact["ckpt_path"]).name}  layer={layer_id}  '
             f'token={token_pos}  expected target = trees[{level}][:, {j}]')
    ax.set_title(title, fontsize=9)
    ax.legend(loc='upper right', fontsize=7, ncol=max(1, V // 10))

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def _plot_per_feature(artifact: dict, pos_idx: int, top_k: int, out_path: Path):
    """For each top-K feature at this (layer, token position), draw one
    subplot with x-axis = value v of the expected latent and bar heights =
    E[f_i | latent = v]. Top-K selection matches `_plot_expected` exactly so
    the two figures show the same features in the same order.
    """
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

    sub = cond_mean[idxs, pos_idx, :].clone()           # [V, F]
    sub[torch.isnan(sub)] = 0.0

    baseline = artifact['baseline_mean'][pos_idx, :]    # [F]

    # Same top-K selection as _plot_expected.
    score = (sub - baseline.unsqueeze(0)).abs().amax(dim=0)  # [F]
    top = torch.argsort(score, descending=True)[:top_k].numpy()
    top = top[np.argsort(top)]
    K = len(top)
    if K == 0:
        return False

    values = [int(targets[i]['value']) for i in idxs]
    V = len(values)
    cmap = plt.get_cmap('tab20')
    bar_colors = [cmap(vi % 20) for vi in range(V)]

    n_cols = min(4, K)
    n_rows = int(np.ceil(K / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)

    sub_np = sub.numpy()                                # [V, F]
    baseline_np = baseline.numpy()                      # [F]
    x = np.arange(V)

    for k, feat_id in enumerate(top):
        r = k // n_cols
        c = k % n_cols
        ax = axes[r][c]
        heights = sub_np[:, feat_id]
        ax.bar(x, heights, color=bar_colors, edgecolor='black', linewidth=0.4)
        ax.axhline(0.0, color='black', linewidth=0.5, alpha=0.5)
        ax.axhline(float(baseline_np[feat_id]), color='gray',
                   linewidth=0.8, linestyle='--', alpha=0.7, label='baseline')
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in values], rotation=45, ha='right', fontsize=8)
        ax.set_xlabel(f'latent value v  (trees[{level}][:, {j}])', fontsize=9)
        ax.set_ylabel('E[ f_i | v ]', fontsize=9)
        ax.set_title(f'f={int(feat_id)}', fontsize=10)

    # Hide any unused axes in the last row.
    for k in range(K, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].axis('off')

    suptitle = (f'{Path(artifact["ckpt_path"]).name}  layer={layer_id}  '
                f'token={token_pos}  expected target = trees[{level}][:, {j}]')
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
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
                        help='Number of top features to highlight (plotted at their real feature id)')
    parser.add_argument('--skip_per_feature', action='store_true',
                        help='Skip the per-feature subplot grid figures (on by default)')
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

        if not args.skip_per_feature:
            per_feat_out = out_dir / f'per_feature_{stem}_pos{pos_idx:02d}.png'
            ok2 = _plot_per_feature(artifact, pos_idx, args.top_k, per_feat_out)
            if ok2:
                print(f'  wrote {per_feat_out}')
            else:
                print(f'  skipped per-feature plot for pos_idx={pos_idx} '
                      '(CLS token or out-of-range layer)')


if __name__ == '__main__':
    main()
