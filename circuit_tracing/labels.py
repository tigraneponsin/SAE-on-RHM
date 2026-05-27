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
    """Per-(p, f) label table for one SAE eval artifact.

    Returns:
        dict[(p, f)] -> {
            'value': int | None,
            'p_value_given_fire': float,
            'level': int,
            'parent_position': int,
            'firing_count': int,
            'value_distribution': FloatTensor[V_g] | None,
            'values': LongTensor[V_g] | None,
            'normalized_entropy': float | None,
        }

    `value_distribution` is the full conditional P(value | feature fires)
    vector (length V_g, the number of observed values in the (level,
    parent_position) group), and `values` is the matching index tensor so
    callers can lift to the full parent-level vocab. Both are None for
    dead features.

    `normalized_entropy` is `H_per_feature[(level, parent_position), p, f]
    / H_theoretical[(level, parent_position)]`, clipped to [0, 1], or None
    if the reference is non-positive / non-finite, or the feature is dead.

    Uses the eval artifact's joint_fire_count, firing_count, index_layout,
    H_per_feature, and H_theoretical. Requires `--with-per-feature` and
    `--with-entropy` (joint_fire_and_entropy flag) when the artifact was
    produced.
    """
    if 'joint_fire_count' not in eval_artifact or 'firing_count' not in eval_artifact:
        raise ValueError(
            f'eval artifact for layer {layer_id} is missing joint_fire_count '
            f'or firing_count. Re-run the eval with per_feature and '
            f'joint_fire_and_entropy flags enabled.'
        )

    cond = _build_cond_prob(eval_artifact)  # (lvl, pos) -> {values, cond_prob [V_g, P, F]}
    firing = eval_artifact['firing_count'].long()  # [P, F]
    # H_per_feature and H_theoretical are optional: when the artifact was
    # produced without --with-entropy, fall back to normalized_entropy=None
    # for every feature (visualizers render the gray fallback). All other
    # fields still populate.
    H_per_feature = eval_artifact.get('H_per_feature')   # [num_groups, P, F] or None
    H_theoretical = eval_artifact.get('H_theoretical')   # dict or None
    index_layout = eval_artifact['index_layout']
    group_index = {
        (int(g['level']), int(g['position'])): idx
        for idx, g in enumerate(index_layout)
    }
    has_entropy = (H_per_feature is not None) and (H_theoretical is not None)

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
        values_t = group['values'].clone().long()  # [V_g]
        cp = group['cond_prob']  # [V_g, P, F] (NaN for dead features)
        if values_t.numel() == 0:
            continue
        cp_pf = cp[:, p, :]  # [V_g, F]

        g_idx = group_index.get(key)
        if has_entropy:
            H_ref = float(H_theoretical.get(key, float('nan')))
            H_ref_ok = math.isfinite(H_ref) and H_ref > 0
        else:
            H_ref = float('nan')
            H_ref_ok = False

        def _norm_entropy(f_idx: int) -> float | None:
            if not has_entropy or g_idx is None or not H_ref_ok:
                return None
            h = float(H_per_feature[g_idx, p, f_idx].item())
            if not math.isfinite(h):
                return None
            v = h / H_ref
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            return v

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
                    'value_distribution': None,
                    'values': None,
                    'normalized_entropy': None,
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
                    'value_distribution': None,
                    'values': None,
                    'normalized_entropy': None,
                }
                continue
            v_idx = int(col.argmax().item())
            out[(p, f)] = {
                'value': int(values_t[v_idx].item()),
                'p_value_given_fire': float(col[v_idx].item()),
                'level': target_level,
                'parent_position': target_pos,
                'firing_count': fc,
                'value_distribution': col.float().clone(),
                'values': values_t.clone(),
                'normalized_entropy': _norm_entropy(f),
            }
    return out


def build_labels_per_layer(eval_artifacts: list[dict], s: int, L: int) -> list[dict]:
    """Vectorize build_labels over a list of artifacts (one per layer)."""
    return [build_labels(art, k, s, L) for k, art in enumerate(eval_artifacts)]
