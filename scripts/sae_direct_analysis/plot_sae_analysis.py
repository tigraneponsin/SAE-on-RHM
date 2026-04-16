"""Plot per-value top-K SAE feature selectivity from analyze_sae.py artifacts.

For a single .feature_latent.pt artifact, produces one figure per analyzed
real-token position p:

  expected_<ckpt>_pos<p>.png

For the expected target this layer should track,
    level  = L - 1 - layer_id
    pos_j  = p // s^(1 + layer_id),
the figure is a grid of subplots, one subplot per latent value. In each
subplot we compute top-K features specifically for that value using
    score_i(v) = E[f_i | v] - baseline_i,
and plot:
    - bars:     score_i(v) for those selected features

Usage:
    python scripts/sae_direct_analysis/plot_sae_analysis.py \\
        --artifact /path/to/<ckpt>.feature_latent.pt \\
        [--out_dir /path/to/figures/] \\
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


def _load_decoder_weights(artifact: dict) -> torch.Tensor:
    """Read the SAE checkpoint referenced by the artifact and return decoder
    weights W_dec of shape [input_dim, latent_dim]."""
    ckpt_path = artifact['ckpt_path']
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    layer_id = int(artifact['layer_id'])
    sae_state = ckpt.get('sae_state', {})
    state = sae_state.get(layer_id) or sae_state.get(str(layer_id))
    if state is None:
        raise RuntimeError(
            f'No SAE state for layer {layer_id} in {ckpt_path}'
        )
    return state['decoder.weight'].detach().cpu()


def _normalize_columns(W: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return W / (W.norm(dim=0, keepdim=True) + eps)


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
    return sorted(
        [
            i for i, t in enumerate(targets)
            if int(t['level']) == level and int(t['position']) == position
        ],
        key=lambda i: int(targets[i]['value']),
    )


def _num_values_for_level(level: int, rhm: dict) -> int:
    """Return value cardinality for this latent level.

    Level 0 is the class/root level and uses n when available. Other levels
    use v.
    """
    if level == 0 and 'n' in rhm:
        return int(rhm['n'])
    return int(rhm['v'])


def _plot_expected_per_value(artifact: dict, pos_idx: int, top_k: int, out_path: Path,
                             W_norm: torch.Tensor):
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
    F_total = int(artifact['latent_dim'])
    if F_total <= 0:
        return False

    # Sub-matrix: [V_obs, F] of conditional means at this (level, j, pos_idx).
    sub = cond_mean[idxs, pos_idx, :].clone() if idxs else torch.empty(0, F_total)
    baseline = artifact['baseline_mean'][pos_idx, :]    # [F]
    top_k_use = max(1, min(int(top_k), F_total))

    # Map observed values to row index in `sub` for fast lookup.
    observed_value_to_row = {
        int(targets[idx]['value']): row_idx for row_idx, idx in enumerate(idxs)
    }
    count_by_value = {int(targets[idx]['value']): int(targets[idx]['count']) for idx in idxs}

    num_values = _num_values_for_level(level, rhm)
    if num_values <= 0:
        return False
    all_values = list(range(num_values))

    n_cols = min(4, num_values)
    n_rows = int(np.ceil(num_values / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )
    last_im = None

    for panel_idx, value in enumerate(all_values):
        r = panel_idx // n_cols
        c = panel_idx % n_cols
        ax = axes[r][c]
        ax.axhline(0.0, color='black', linewidth=0.5, alpha=0.5)

        if value not in observed_value_to_row:
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=9, color='gray')
            ax.set_xticks([])
            ax.set_title(f'value={value}  (count=0)', fontsize=9)
            continue

        row = sub[observed_value_to_row[value], :]      # [F]
        valid = ~torch.isnan(row)
        if not valid.any():
            ax.text(0.5, 0.5, 'no valid features', transform=ax.transAxes,
                    ha='center', va='center', fontsize=9, color='gray')
            ax.set_xticks([])
            ax.set_title(f'value={value}  (count={count_by_value[value]})', fontsize=9)
            continue

        score = (row - baseline)                        # [F]
        score[~valid] = -float('inf')
        k_this = min(top_k_use, int(valid.sum().item()))
        if k_this == 0:
            ax.text(0.5, 0.5, 'no valid features', transform=ax.transAxes,
                    ha='center', va='center', fontsize=9, color='gray')
            ax.set_xticks([])
            ax.set_title(f'value={value}  (count={count_by_value[value]})', fontsize=9)
            continue

        top = torch.argsort(score, descending=True)[:k_this].numpy()
        heights = score[top].numpy()
        x = np.arange(k_this)

        ax.bar(x, heights, color='tab:blue', edgecolor='black', linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(f)) for f in top], rotation=45, ha='right', fontsize=7)
        ax.set_xlabel('feature id (top-k for this value)', fontsize=8)
        ax.set_ylabel('score = E[f_i | value] - baseline_i', fontsize=8)
        ax.set_title(f'value={value}  (count={count_by_value[value]})', fontsize=9)

        # k*k decoder cosine similarity inset (upper-right corner of subplot).
        top_idx = torch.as_tensor(top, dtype=torch.long)
        W_sub = W_norm.index_select(1, top_idx)              # [D, k_this]
        C = (W_sub.T @ W_sub).detach().cpu().numpy()         # [k_this, k_this]
        ax_inset = ax.inset_axes([0.62, 0.62, 0.36, 0.36])
        im = ax_inset.imshow(
            C, vmin=-1, vmax=1, cmap='RdBu_r',
            aspect='equal', interpolation='nearest',
        )
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        for spine in ax_inset.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)
        last_im = im

    # Hide any unused axes in the last row.
    for panel_idx in range(num_values, n_rows * n_cols):
        axes[panel_idx // n_cols][panel_idx % n_cols].axis('off')

    suptitle = (
        f'{Path(artifact["ckpt_path"]).name}  layer={layer_id}  '
        f'token={token_pos}  expected target = trees[{level}][:, {j}]'
    )
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.93, 0.96))
    if last_im is not None:
        cax = fig.add_axes([0.945, 0.30, 0.012, 0.40])
        cb = fig.colorbar(last_im, cax=cax)
        cb.set_label('decoder cos sim', fontsize=8)
        cb.ax.tick_params(labelsize=7)
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
                        help='Per-value top-K features to plot in each value subplot')
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

    W_dec = _load_decoder_weights(artifact)      # [input_dim, latent_dim]
    W_norm = _normalize_columns(W_dec)

    for pos_idx in range(num_positions):
        expected_out = out_dir / f'expected_{stem}_pos{pos_idx:02d}.png'
        ok = _plot_expected_per_value(artifact, pos_idx, args.top_k, expected_out, W_norm)
        if ok:
            print(f'  wrote {expected_out}')
        else:
            print(f'  skipped expected-target plot for pos_idx={pos_idx} '
                  '(CLS token or out-of-range layer)')


if __name__ == '__main__':
    main()
