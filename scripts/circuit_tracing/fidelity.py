"""Fidelity diagnostics for circuit_trace.

Two entry points:

    compute_pre_prune_fidelity(...) -> dict
        Per-layer error mass, feature-mediated logit fraction, and
        subtree-alignment over the FULL attribution edge set. Moved verbatim
        from circuit_trace.py.

    compute_postprune_alignment(...) -> dict
        Subtree-alignment over the PRUNED edge set. Iterates pruned_edges,
        classifies each by (src_kind, dst_kind), and applies the same
        non-degenerate denominator rule used by the pre-prune metric.

Both are pure functions of their inputs; no torch state mutation.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Pre-prune fidelity (moved verbatim from circuit_trace.py)
# ---------------------------------------------------------------------------

def compute_pre_prune_fidelity(
    layer_pairs_feat: list,
    layer_pairs_err: list,
    embed_to_l0: list,
    feat_to_logit: list,
    err_to_logit: list,
    logits: torch.Tensor,
    y_true: int,
    num_classes: int,
    sink_mode: str,
    s: int,
    L: int,
    N: int,
    pooled_last_layer: bool = False,
) -> dict:
    """Pre-prune fidelity diagnostics over the full attribution edge set.

    Returns dict with:
      per_layer_error_fraction         : list[float], one per (k -> k+1) pair
      feature_mediated_logit_fraction  : float
      subtree_alignment_fraction       : float
      per_layer_subtree_alignment      : list[float], [embed, k=0, ..., k=K-2]

    When pooled_last_layer is True, the final feat->feat pair (k=K-2 -> pooled
    K-1) has all destination positions == 0, so its subtree-alignment is
    degenerate. Its per-layer slot is reported as None and it is excluded from
    the global non-degenerate fraction (it already would be, since its group
    = s**K = N). The per_layer_error_fraction slot stays well-defined.
    """
    K = L  # transformer has L blocks; matches num_layers in cfg.

    # Per-layer error mass (using all attribution edges, not the pruned graph).
    per_layer_err_fraction = []
    for k in range(K - 1):
        feat_abs = sum(abs(w) for (_, _, _, _, w) in layer_pairs_feat[k])
        err_abs = sum(abs(w) for (_, _, _, w) in layer_pairs_err[k])
        denom = feat_abs + err_abs
        per_layer_err_fraction.append((err_abs / denom) if denom > 0 else 0.0)

    # Feature-mediated logit fraction: per class, then averaged with the
    # softmax weights (for softmax_logits mode) or evaluated at y_true (for
    # true_class). Reported as a single scalar.
    if sink_mode == 'true_class':
        c_target = y_true
        feat_abs_l = sum(abs(w) for (_, _, c, w) in feat_to_logit if c == c_target)
        err_abs_l = sum(abs(w) for (_, c, w) in err_to_logit if c == c_target)
        denom = feat_abs_l + err_abs_l
        feat_mediated_frac = (feat_abs_l / denom) if denom > 0 else 0.0
    else:
        probs = torch.softmax(logits, dim=0).tolist()
        weighted_num = 0.0
        weighted_den = 0.0
        for c in range(num_classes):
            feat_abs_l = sum(abs(w) for (_, _, cc, w) in feat_to_logit if cc == c)
            err_abs_l = sum(abs(w) for (_, cc, w) in err_to_logit if cc == c)
            denom = feat_abs_l + err_abs_l
            if denom > 0:
                weighted_num += probs[c] * feat_abs_l
                weighted_den += probs[c] * denom
        feat_mediated_frac = (weighted_num / weighted_den) if weighted_den > 0 else 0.0

    # Subtree alignment: feature->feature edges (existing) PLUS embed->layer0
    # edges as a new "k=-1" slot. Per-layer plus a global non-degenerate
    # average.
    per_layer_subtree_alignment = []
    nondeg_aligned = 0.0
    nondeg_total = 0.0

    # Embed -> layer 0: group = s (per design decision; matches s ** (2 + (-1))).
    embed_aligned = 0.0
    embed_total = 0.0
    embed_group = s
    for (p, j, q, w) in embed_to_l0:
        embed_total += abs(w)
        if (p // embed_group) == (q // embed_group):
            embed_aligned += abs(w)
    embed_frac = (embed_aligned / embed_total) if embed_total > 0 else 0.0
    per_layer_subtree_alignment.append(embed_frac)
    if embed_group < N:
        nondeg_aligned += embed_aligned
        nondeg_total += embed_total

    for k, lst in enumerate(layer_pairs_feat):
        # The pooled last layer collapses dst position to 0, making subtree
        # alignment for the final pair (k == K-2) meaningless. Report None and
        # skip the (already-degenerate, group == N) global accumulation.
        if pooled_last_layer and k == K - 2:
            per_layer_subtree_alignment.append(None)
            continue
        group = s ** (2 + k)
        aligned_k = 0.0
        total_k = 0.0
        for (i, p, j, q, w) in lst:
            total_k += abs(w)
            if (p // group) == (q // group):
                aligned_k += abs(w)
        frac_k = (aligned_k / total_k) if total_k > 0 else 0.0
        per_layer_subtree_alignment.append(frac_k)
        if group < N:
            nondeg_aligned += aligned_k
            nondeg_total += total_k
    subtree_alignment_fraction = (
        nondeg_aligned / nondeg_total if nondeg_total > 0 else 0.0
    )

    return {
        'per_layer_error_fraction': per_layer_err_fraction,
        'feature_mediated_logit_fraction': feat_mediated_frac,
        'subtree_alignment_fraction': subtree_alignment_fraction,
        'per_layer_subtree_alignment': per_layer_subtree_alignment,
    }


# ---------------------------------------------------------------------------
# Post-prune alignment
# ---------------------------------------------------------------------------

def compute_postprune_alignment(
    pruned_edges: list,
    s: int,
    L: int,
    N: int,
    pooled_last_layer: bool = False,
) -> dict:
    """Subtree alignment over the PRUNED edge set.

    Classifies each pruned edge by (src_kind, dst_kind):
      - ('embed', 'feat'): group = s. Compare src.position // s vs
        dst.position // s.
      - ('feat',  'feat'): group = s ** (2 + k_src). Compare ancestor
        indices.
      - Skip ('err', *), (*, 'logit'), and any unexpected combinations
        (alignment is undefined for those, matching the pre-prune
        convention).

    Returns dict with:
      subtree_alignment_fraction_postprune       : float
      per_layer_subtree_alignment_postprune      : list[float]
        [embed -> k=0, k=0 -> k=1, ..., k=(K-2) -> k=(K-1)].
    """
    K = L

    # Buckets: index 0 = embed -> k=0, then k=0->1, k=1->2, ..., k=(K-2)->(K-1).
    n_buckets = K  # 1 (embed) + (K-1) feat->feat pairs
    aligned = [0.0] * n_buckets
    total = [0.0] * n_buckets
    bucket_group = [s] + [s ** (2 + k) for k in range(K - 1)]

    for (src, dst, w) in pruned_edges:
        src_kind = src[0]
        dst_kind = dst[0]
        aw = abs(float(w))
        if src_kind == 'embed' and dst_kind == 'feat':
            # dst layer must be 0.
            _, k_dst, p_dst, _ = dst
            if int(k_dst) != 0:
                continue
            _, p_src = src
            group = s
            total[0] += aw
            if (int(p_src) // group) == (int(p_dst) // group):
                aligned[0] += aw
        elif src_kind == 'feat' and dst_kind == 'feat':
            _, k_src, p_src, _ = src
            _, k_dst, p_dst, _ = dst
            if int(k_dst) != int(k_src) + 1:
                # Unexpected: edges should only span adjacent layers.
                continue
            bucket = 1 + int(k_src)
            # Pooled last layer: the final feat->feat bucket (k_src == K-2,
            # dst at pooled K-1) has all dst positions == 0, so alignment is
            # degenerate. Leave its totals at 0 -> reported as None below.
            if pooled_last_layer and int(k_src) == K - 2:
                continue
            group = bucket_group[bucket]
            total[bucket] += aw
            if (int(p_src) // group) == (int(p_dst) // group):
                aligned[bucket] += aw
        # err -> *, * -> logit: skipped.

    per_layer = [
        (aligned[b] / total[b]) if total[b] > 0
        else (None if (pooled_last_layer and b == K - 1) else 0.0)
        for b in range(n_buckets)
    ]

    nondeg_aligned = 0.0
    nondeg_total = 0.0
    for b in range(n_buckets):
        if bucket_group[b] < N:
            nondeg_aligned += aligned[b]
            nondeg_total += total[b]
    global_frac = (nondeg_aligned / nondeg_total) if nondeg_total > 0 else 0.0

    return {
        'subtree_alignment_fraction_postprune': global_frac,
        'per_layer_subtree_alignment_postprune': per_layer,
    }
