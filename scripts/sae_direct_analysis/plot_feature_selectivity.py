"""Population-level feature selectivity and co-firing plots.

For each analyzed real-token position p in a .feature_latent.pt artifact,
produces a single PNG with:

  Main panel (scatter):
    x  = selectivity count per feature, where "selectivity" at a given (layer,
         position, feature) is the number of latent values l such that
         delta_mean[t(l), p, f] > alpha * max_l' delta_mean[t(l'), p, f].
         Features whose max score is <= 0 are assigned selectivity = 0.
    y  = mean_cofire[p, f] = E[ L0_p | f_i > 0 ].
    c  = log10(firing_rate[p, f])  (viridis, colorbar).
    Dead features (firing_count[p, f] == 0) are excluded.

  Top marginal:
    Histogram of selectivity over non-dead features at this position.

The artifact must be produced by an analyze_sae.py run that had
--no_cofire UNSET (the default). If firing_rate / firing_count /
mean_cofire / L0_mean are missing the script exits with a clear message.

Usage:
    python scripts/sae_direct_analysis/plot_feature_selectivity.py \\
        --artifact /path/to/<ckpt>.feature_latent.pt \\
        [--out_dir /path/to/figures/] \\
        [--alpha 0.3] \\
        [--positions 0 3 5] \\
        [--suffix ""]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sae_direct_analysis.plot_sae_analysis import (
    _expected_target,
    _find_target_indices,
    _load_artifact,
    _num_values_for_level,
)


REQUIRED_COFIRE_KEYS = ('firing_rate', 'firing_count', 'mean_cofire', 'L0_mean')


def _compute_selectivity(artifact: dict, pos_idx: int, alpha: float):
    """Build the per-feature selectivity vector for this (artifact, pos_idx).

    Returns (selectivity [F], level, j, num_values) or None if the SAE at this
    position has no expected latent target (e.g. CLS token, out-of-range layer).
    """
    rhm = artifact['rhm']
    layer_id = int(artifact['layer_id'])
    token_pos = int(artifact['token_positions'][pos_idx].item())
    expected = _expected_target(layer_id, token_pos, rhm)
    if expected is None:
        return None
    level, j = expected

    num_values = _num_values_for_level(level, rhm)
    if num_values <= 0:
        return None

    F_total = int(artifact['latent_dim'])
    targets = artifact['targets']
    delta_mean = artifact['delta_mean']           # [T, P, F]

    idxs = _find_target_indices(targets, level, j)
    # S[l, f] for l in 0..num_values-1; unobserved values -> NaN.
    S = torch.full((num_values, F_total), float('nan'), dtype=torch.float32)
    for t_idx in idxs:
        v = int(targets[t_idx]['value'])
        if 0 <= v < num_values:
            S[v] = delta_mean[t_idx, pos_idx, :]

    S_for_max = S.clone()
    S_for_max[torch.isnan(S_for_max)] = -float('inf')
    peak = S_for_max.max(dim=0).values                       # [F]

    # selectivity is 0 where the feature never has a positive response.
    has_positive = peak > 0
    threshold = (alpha * peak).clamp_min(0.0)                # [F]
    S_valid = torch.nan_to_num(S, nan=-float('inf'))         # [V, F]
    above = (S_valid > threshold.unsqueeze(0)).sum(dim=0)    # [F]
    selectivity = torch.where(
        has_positive, above, torch.zeros_like(above)
    ).long()
    # peak_score is the feature's score for its winning latent value.
    # Features with no positive response are clamped to 0 so they don't
    # contribute negatively to importance-weighted histograms.
    peak_score = peak.clamp_min(0.0)
    return selectivity, peak_score, level, j, num_values


def _plot_selectivity_figure(artifact: dict, pos_idx: int, alpha: float,
                             out_path: Path) -> bool:
    missing = [k for k in REQUIRED_COFIRE_KEYS if k not in artifact]
    if missing:
        raise RuntimeError(
            f'Artifact is missing co-firing keys {missing}. '
            f'Re-run analyze_sae.py without --no_cofire.'
        )

    info = _compute_selectivity(artifact, pos_idx, alpha)
    if info is None:
        return False
    selectivity, peak_score, level, j, num_values = info

    firing_rate = artifact['firing_rate'][pos_idx, :]        # [F]
    firing_count = artifact['firing_count'][pos_idx, :]      # [F]
    mean_cofire = artifact['mean_cofire'][pos_idx, :]        # [F]
    L0_mean = float(artifact['L0_mean'][pos_idx].item())

    alive = firing_count > 0
    F_total = firing_rate.numel()
    n_dead = int((~alive).sum().item())

    token_pos = int(artifact['token_positions'][pos_idx].item())
    layer_id = int(artifact['layer_id'])

    x_all = selectivity.numpy()
    fr_all = firing_rate.numpy()
    cof_all = mean_cofire.numpy()

    # Histogram edges span 0..num_values inclusive; each integer gets its own bin.
    edges = np.arange(num_values + 2) - 0.5
    centers = np.arange(num_values + 1)
    hist_unweighted, _ = np.histogram(x_all[alive.numpy()], bins=edges)

    fig = plt.figure(figsize=(7.5, 6.0))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[20, 1],
        height_ratios=[1, 3],
        hspace=0.05, wspace=0.04,
    )
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[1, 0], sharex=ax_hist)
    ax_cbar = fig.add_subplot(gs[1, 1])

    # --- Top marginal histogram ---
    ax_hist.bar(
        centers, hist_unweighted, width=1.0,
        color='tab:gray', edgecolor='black', linewidth=0.4,
    )
    ax_hist.set_ylabel('feature count', fontsize=9)
    ax_hist.tick_params(axis='y', labelsize=8)
    ax_hist.tick_params(axis='x', labelbottom=False)

    # --- Main scatter ---
    alive_np = alive.numpy()
    x = x_all[alive_np]
    y = cof_all[alive_np]
    fr_alive = fr_all[alive_np]
    c = np.log10(np.clip(fr_alive, 1e-12, None))

    # Small x-jitter so points at the same integer don't fully overlap.
    rng = np.random.default_rng(0)
    x_jitter = x.astype(np.float32) + rng.uniform(-0.25, 0.25, size=x.shape)

    sc = ax_scatter.scatter(
        x_jitter, y, c=c, cmap='viridis', s=14, alpha=0.75,
        edgecolors='none',
    )

    ax_scatter.axhline(
        L0_mean, color='tab:red', linestyle='--', linewidth=1.0,
        label=f'L0_mean = {L0_mean:.2f}',
    )
    ax_scatter.axvline(
        1.0, color='black', linestyle=':', linewidth=0.8,
        label='selectivity = 1',
    )

    ax_scatter.set_xlabel('selectivity (# latent values above alpha * peak)', fontsize=9)
    ax_scatter.set_ylabel('mean cofire = E[L0_p | f_i > 0]', fontsize=9)
    ax_scatter.set_xticks(centers)
    ax_scatter.set_xticklabels([str(i) for i in centers], fontsize=8)
    ax_scatter.tick_params(axis='y', labelsize=8)
    ax_scatter.set_xlim(edges[0], edges[-1])
    ax_scatter.legend(fontsize=7, loc='upper left', frameon=False)

    cb = fig.colorbar(sc, cax=ax_cbar)
    cb.set_label('log10(firing_rate)', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ckpt_name = Path(artifact['ckpt_path']).stem
    line1 = ckpt_name
    line2 = (
        f'layer={layer_id}  token={token_pos}  '
        f'target=trees[{level}][:, {j}]  '
        f'alpha={alpha}  dead={n_dead}/{F_total}'
    )
    fig.suptitle(f'{line1}\n{line2}', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1.0, 0.90))
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
                        help='Directory to write figures to (default: next to artifact)')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Fraction-of-max threshold for selectivity counting (default: 0.3)')
    parser.add_argument('--positions', type=int, nargs='*', default=None,
                        help='Subset of pos_idx values to plot (default: all)')
    parser.add_argument('--suffix', type=str, default='',
                        help='Optional filename suffix (e.g. "_a05" for alpha variants)')
    args = parser.parse_args()

    artifact_path = Path(args.artifact)
    artifact = _load_artifact(artifact_path)

    missing = [k for k in REQUIRED_COFIRE_KEYS if k not in artifact]
    if missing:
        raise SystemExit(
            f'Artifact {artifact_path} is missing co-firing keys {missing}. '
            f'Re-run analyze_sae.py without --no_cofire to regenerate it.'
        )

    out_dir = Path(args.out_dir) if args.out_dir else artifact_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = artifact_path.stem.replace('.feature_latent', '')

    num_positions = int(artifact['token_positions'].numel())
    if args.positions is None:
        pos_list = list(range(num_positions))
    else:
        pos_list = [p for p in args.positions if 0 <= p < num_positions]

    print(f'Plotting {artifact_path.name}: '
          f'T={len(artifact["targets"])}, P={num_positions}, '
          f'F={artifact["latent_dim"]}, alpha={args.alpha}')

    for pos_idx in pos_list:
        out_path = (
            out_dir
            / f'selectivity_scatter_{stem}_pos{pos_idx:02d}{args.suffix}.png'
        )
        ok = _plot_selectivity_figure(artifact, pos_idx, args.alpha, out_path)
        if ok:
            print(f'  wrote {out_path}')
        else:
            print(f'  skipped pos_idx={pos_idx} (no expected latent target)')


if __name__ == '__main__':
    main()
