"""Plot top-K feature selectivity for a given token position and artifact.

For an SAE trained at layer k and a user-supplied real-token index p, the
target latent this position should track is
    level = L - 1 - k
    j     = p // s^(1 + k),
i.e. trees[level][:, j].

For each value v of that target latent, we pick the top-K features by
    score_i(v) = E[f_i | trees[level][:, j] == v] - baseline_i
(same scoring as plot_sae_analysis.py). Then, for each of those K features,
we plot a bar chart of that feature's score across ALL values v' of the
target latent. The "home" value v (the one that elected the feature into
the top-K) is highlighted in orange.

Output: one PNG per value v, each containing K subplots.
    topk_selectivity_<stem>_pos<p:02d>_val<v:02d>.png

Usage:
    python scripts/sae_direct_analysis/plot_topk_selectivity.py \\
        --artifact /path/to/<ckpt>.feature_latent.pt \\
        --token_pos <p>              \\
        [--top_k 12]                 \\
        [--out_dir /path/to/figures/]
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
sys.path.insert(0, str(REPO_ROOT))

from scripts.sae_direct_analysis.plot_sae_analysis import (
    _expected_target,
    _find_target_indices,
    _load_artifact,
    _num_values_for_level,
)


def _resolve_pos_idx(token_positions: torch.Tensor, token_pos: int) -> int:
    """Return the array index in token_positions where the value equals
    token_pos. Raise ValueError if not present or ambiguous."""
    tp = token_positions.long().tolist()
    matches = [i for i, v in enumerate(tp) if int(v) == int(token_pos)]
    if not matches:
        raise ValueError(
            f'token_pos={token_pos} is not present in artifact token_positions={tp}. '
            'For one_token artifacts, pass the token index the SAE was trained on; '
            'for all_tokens artifacts, pass a real-token index within range.'
        )
    if len(matches) > 1:
        raise ValueError(
            f'token_pos={token_pos} appears multiple times in token_positions={tp}'
        )
    return matches[0]


def _plot_topk_selectivity_for_value(score_matrix: torch.Tensor,
                                     valid_mask: torch.Tensor,
                                     observed_value_to_row: dict,
                                     count_by_value: dict,
                                     all_values: list,
                                     home_value: int,
                                     top_k: int,
                                     title_suffix: str,
                                     out_path: Path) -> bool:
    """Save a figure with K subplots, one per top-K feature for home_value.

    score_matrix: [V_obs, F], float tensor with NaNs where missing.
    valid_mask:   [V_obs, F], bool tensor.
    observed_value_to_row: dict {value -> row index in score_matrix}.
    count_by_value:        dict {value -> sample count}.
    all_values: list of every value v' in range(num_values) for the target level.
    home_value: the value v whose top-K features we are inspecting.
    top_k: requested number of top features; actual K may be smaller.
    """
    if home_value not in observed_value_to_row:
        print(f'  skipping value={home_value}: no samples observed')
        return False

    v_row = observed_value_to_row[home_value]
    row_valid = valid_mask[v_row]
    if not bool(row_valid.any()):
        print(f'  skipping value={home_value}: no valid features')
        return False

    score_for_argmax = score_matrix[v_row].clone()
    score_for_argmax[~row_valid] = -float('inf')

    k_this = min(int(top_k), int(row_valid.sum().item()))
    if k_this <= 0:
        print(f'  skipping value={home_value}: k_this == 0')
        return False

    top_feats = torch.argsort(score_for_argmax, descending=True)[:k_this].tolist()

    num_values = len(all_values)
    n_cols = min(4, k_this)
    n_rows = int(np.ceil(k_this / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )

    for rank, f in enumerate(top_feats):
        r = rank // n_cols
        c = rank % n_cols
        ax = axes[r][c]
        ax.axhline(0.0, color='black', linewidth=0.5, alpha=0.5)

        bars = np.full(num_values, np.nan, dtype=np.float64)
        for v_prime in all_values:
            if v_prime not in observed_value_to_row:
                continue
            o_row = observed_value_to_row[v_prime]
            if not bool(valid_mask[o_row, f]):
                continue
            bars[v_prime] = float(score_matrix[o_row, f].item())

        x = np.arange(num_values)
        plot_mask = ~np.isnan(bars)
        colors = ['tab:orange' if i == home_value else 'tab:blue'
                  for i in range(num_values)]
        ax.bar(
            x[plot_mask],
            bars[plot_mask],
            color=[colors[i] for i in range(num_values) if plot_mask[i]],
            edgecolor='black',
            linewidth=0.4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in all_values], fontsize=7)
        ax.set_xlabel('latent value', fontsize=8)
        ax.set_ylabel('score = E[f|value] - baseline', fontsize=8)
        ax.set_title(
            f'feature {int(f)}  (rank {rank + 1}/{k_this} for value={home_value})',
            fontsize=9,
        )

    for panel_idx in range(k_this, n_rows * n_cols):
        axes[panel_idx // n_cols][panel_idx % n_cols].axis('off')

    home_count = count_by_value.get(home_value, 0)
    suptitle = (
        f'top-{k_this} selectivity: value={home_value}  '
        f'(count={home_count})  {title_suffix}'
    )
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1.0, 0.95))
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
    parser.add_argument('--token_pos', type=int, required=True,
                        help='Real-token index p (0-based). Must be present in the '
                             "artifact's token_positions tensor.")
    parser.add_argument('--top_k', type=int, default=12,
                        help='Number of top features per value to plot (default: 12).')
    parser.add_argument('--out_dir', default=None,
                        help='Directory to write figures (default: next to the artifact)')
    args = parser.parse_args()

    artifact_path = Path(args.artifact)
    artifact = _load_artifact(artifact_path)

    out_dir = Path(args.out_dir) if args.out_dir else artifact_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = artifact_path.stem.replace('.feature_latent', '')

    token_positions = artifact['token_positions']
    pos_idx = _resolve_pos_idx(token_positions, args.token_pos)

    targets = artifact['targets']
    rhm = artifact['rhm']
    layer_id = int(artifact['layer_id'])

    expected = _expected_target(layer_id, args.token_pos, rhm)
    if expected is None:
        raise SystemExit(
            f'Cannot resolve expected target for layer={layer_id}, '
            f'token_pos={args.token_pos} (CLS token or out-of-range layer).'
        )
    level, j = expected
    idxs = _find_target_indices(targets, level, j)
    if not idxs:
        raise SystemExit(
            f'No targets matching (level={level}, position={j}) in artifact.'
        )

    cond_mean = artifact['conditional_mean']            # [T, P, F]
    baseline = artifact['baseline_mean'][pos_idx, :]    # [F]
    sub = cond_mean[idxs, pos_idx, :].clone()           # [V_obs, F]
    score_matrix = sub - baseline.unsqueeze(0)          # [V_obs, F]
    valid_mask = ~torch.isnan(score_matrix)

    observed_value_to_row = {
        int(targets[idx]['value']): row_idx for row_idx, idx in enumerate(idxs)
    }
    count_by_value = {
        int(targets[idx]['value']): int(targets[idx]['count']) for idx in idxs
    }

    num_values = _num_values_for_level(level, rhm)
    if num_values <= 0:
        raise SystemExit(f'_num_values_for_level returned {num_values}')
    all_values = list(range(num_values))

    title_suffix = (
        f'{Path(artifact["ckpt_path"]).name}  layer={layer_id}  '
        f'token={args.token_pos}  target = trees[{level}][:, {j}]'
    )

    print(f'Plotting {artifact_path.name}: '
          f'token_pos={args.token_pos} -> pos_idx={pos_idx}, '
          f'target=(level={level}, j={j}), '
          f'num_values={num_values}, F={artifact["latent_dim"]}')

    n_written = 0
    for v in all_values:
        out_path = out_dir / (
            f'topk_selectivity_{stem}_pos{int(args.token_pos):02d}_val{v:02d}.png'
        )
        ok = _plot_topk_selectivity_for_value(
            score_matrix=score_matrix,
            valid_mask=valid_mask,
            observed_value_to_row=observed_value_to_row,
            count_by_value=count_by_value,
            all_values=all_values,
            home_value=v,
            top_k=args.top_k,
            title_suffix=title_suffix,
            out_path=out_path,
        )
        if ok:
            print(f'  wrote {out_path}')
            n_written += 1

    print(f'done. {n_written}/{num_values} figures written.')


if __name__ == '__main__':
    main()
