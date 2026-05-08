"""Per-(layer, position, feature) latent labels from .sae_eval.pt artifacts.

For each (layer k, position p, feature f), the label is the most likely
parent latent value given that the feature fires:

    label_value = argmax_v  P(latent_value = v | feature_f fires)

P(value | fire) is computed as joint_fire_count / firing_count via
scripts.sae_tree_reconstruction.run._build_cond_prob.

The relevant (level, position) at which to evaluate this is
    level = L - 1 - k
    parent_position = p // s ** (1 + k)
where (s, L) are the RHM branching factor and depth.
"""

from __future__ import annotations

import math

import torch

# Reuse the existing implementation to avoid drift.
from scripts.sae_tree_reconstruction.run import _build_cond_prob


def parent_position(p: int, k: int, s: int) -> int:
    """Ancestor position at level L-1-k for leaf position p.

    From CLAUDE.md and the handoff doc: a_{L-1-k}(p) = p // s^(1+k).
    """
    return int(p) // int(s) ** (1 + int(k))


def parent_level(k: int, L: int) -> int:
    return int(L) - 1 - int(k)


def build_labels(eval_artifact: dict, layer_id: int, s: int, L: int) -> dict:
    """Argmax-label table for one SAE eval artifact.

    Returns:
        dict[(p, f)] -> {'value': int, 'p_value_given_fire': float,
                          'level': int, 'parent_position': int,
                          'firing_count': int}
        for every (p, f) where firing_count[p, f] > 0. Features that never
        fired in the eval set are labeled with value=None.

    Uses the eval artifact's joint_fire_count, firing_count, and
    index_layout. Requires that the artifact was produced with both
    per_feature and joint_fire_and_entropy flags set.
    """
    if 'joint_fire_count' not in eval_artifact or 'firing_count' not in eval_artifact:
        raise ValueError(
            f'eval artifact for layer {layer_id} is missing joint_fire_count '
            f'or firing_count. Re-run the eval with per_feature and '
            f'joint_fire_and_entropy flags enabled.'
        )

    cond = _build_cond_prob(eval_artifact)  # (lvl, pos) -> {values, cond_prob [V_g, P, F]}
    firing = eval_artifact['firing_count'].long()  # [P, F]

    P, F = firing.shape
    target_level = parent_level(layer_id, L)

    out: dict = {}
    for p in range(P):
        target_pos = parent_position(p, layer_id, s)
        key = (target_level, target_pos)
        if key not in cond:
            # Either the (level, position) was empty in the eval set, or the
            # artifact came from trees missing this level. Skip silently.
            continue
        group = cond[key]
        values = group['values']  # [V_g] long
        cp = group['cond_prob']  # [V_g, P, F] (NaN for dead features)
        if values.numel() == 0:
            continue
        cp_pf = cp[:, p, :]  # [V_g, F]
        # NaN columns at this (p, f) mean firing_count[p, f] == 0 -> dead.
        for f in range(F):
            fc = int(firing[p, f].item())
            if fc == 0:
                out[(p, f)] = {
                    'value': None,
                    'p_value_given_fire': float('nan'),
                    'level': target_level,
                    'parent_position': target_pos,
                    'firing_count': 0,
                }
                continue
            col = cp_pf[:, f]  # [V_g] float64, sums to <=1 (1 if all values were observed)
            if torch.isnan(col).any():
                out[(p, f)] = {
                    'value': None,
                    'p_value_given_fire': float('nan'),
                    'level': target_level,
                    'parent_position': target_pos,
                    'firing_count': fc,
                }
                continue
            v_idx = int(col.argmax().item())
            out[(p, f)] = {
                'value': int(values[v_idx].item()),
                'p_value_given_fire': float(col[v_idx].item()),
                'level': target_level,
                'parent_position': target_pos,
                'firing_count': fc,
            }
    return out


def build_labels_per_layer(eval_artifacts: list[dict], s: int, L: int) -> list[dict]:
    """Vectorize build_labels over a list of artifacts (one per layer)."""
    return [build_labels(art, k, s, L) for k, art in enumerate(eval_artifacts)]
