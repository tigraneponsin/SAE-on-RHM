"""Indirect-influence pruning for the circuit-tracing DAG.

Two-stage algorithm:
  1. Build the signed adjacency W [N_nodes, N_nodes] indexed as
     W[target, source] = signed edge weight.
  2. Build A = |W|, row-normalized so each row sums to 1 (rows of pure-input
     nodes -- embeddings, errors, logits -- naturally have row sum 0 and
     stay zero after a clamped-min eps guard).
  3. Compute the indirect influence
        B = A + A^2 + ... + A^D     where D = K + 2
     using the polynomial form. With a strict DAG (every edge advances the
     virtual layer index by at least one), A is nilpotent and the sum is
     exact in <= D iterations.
  4. logit_weights vector w_L of shape [N_nodes]:
        - sink_mode='softmax_logits': softmax(logits)[c] at each logit node.
        - sink_mode='true_class':     1.0 at the y_true logit, 0 elsewhere.
  5. node_score = B @ w_L. Sort prunable (kind=='feature') nodes by node_score
     desc, keep prefix until cumsum / total >= node_threshold.
  6. Restrict edges to kept_features UNION embedding UNION error UNION logit.
     Recompute A_sub, B_sub. Override logit-node node_score_sub entries
     with logit_weights (otherwise they collapse to 0). Edge score
        edge_score[t, s] = A_sub[t, s] * node_score_sub[t]
     (only over actual edges). Sort desc, keep prefix until
     cumsum / total >= edge_threshold.

Inputs:
- nodes: dict node_key -> attrs (must contain 'kind').
- edges: list of (src_key, dst_key, signed_weight: float).
- num_classes, K, N, s, L: RHM/transformer parameters.
- logits: [num_classes] float tensor (anchors.logits).
- y_true: int.
- sink_mode: 'softmax_logits' | 'true_class'.
- node_threshold, edge_threshold: floats in (0, 1].

Returns:
- kept_edges: list of (src, dst, signed_weight: float) -- final pruned edges.
- kept_node_keys: set of node keys present in the pruned subgraph.
- diagnostics: dict with all the fidelity-relevant counts and scores.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def _virtual_layer(key, K: int) -> int:
    """Strict-DAG virtual layer index for ordering A.

    embedding -> -1
    feature/error at layer k -> k
    logit -> K
    """
    kind = key[0]
    if kind == 'embed':
        return -1
    if kind == 'feat':
        return int(key[1])
    if kind == 'err':
        return int(key[1])
    if kind == 'logit':
        return K
    raise ValueError(f'unrecognized node key {key!r}')


def _index_nodes(nodes: dict, K: int) -> tuple[dict, list]:
    """Build a stable node ordering compatible with strict-upper-triangular A.

    Returns (idx_of, ordered_keys) where idx_of[key] = i in [0, N_nodes).
    Order: by virtual_layer first, then arbitrary tiebreak (we use key tuple).
    Within a layer, edges between nodes of the same virtual layer must not
    exist (verified during W construction).
    """
    keys = sorted(nodes.keys(), key=lambda k: (_virtual_layer(k, K), k))
    idx_of = {k: i for i, k in enumerate(keys)}
    return idx_of, keys


# ---------------------------------------------------------------------------
# Adjacency
# ---------------------------------------------------------------------------

def _build_adjacency(
    nodes: dict, edges: list, K: int, device, dtype,
) -> tuple[torch.Tensor, dict, list]:
    """Build dense W of shape [N_nodes, N_nodes] with W[t, s] = signed weight,
    plus the (idx_of, ordered_keys) used to index it.

    Verifies that every edge respects the strict-DAG ordering
    virtual_layer(src) < virtual_layer(dst). Hard-fails otherwise (this
    indicates a bug in edge assembly).
    """
    idx_of, ordered_keys = _index_nodes(nodes, K)
    N_nodes = len(ordered_keys)
    W = torch.zeros(N_nodes, N_nodes, device=device, dtype=dtype)
    for (src, dst, w) in edges:
        if src not in idx_of:
            raise RuntimeError(f'edge source {src!r} not in node table')
        if dst not in idx_of:
            raise RuntimeError(f'edge target {dst!r} not in node table')
        ls = _virtual_layer(src, K)
        ld = _virtual_layer(dst, K)
        if not (ls < ld):
            raise RuntimeError(
                f'edge {src!r} -> {dst!r} violates DAG ordering: '
                f'virtual_layer src={ls}, dst={ld}'
            )
        si = idx_of[src]
        ti = idx_of[dst]
        # Multiple edges from same src to same dst are not expected, but
        # accumulate just in case.
        W[ti, si] = W[ti, si] + float(w)
    return W, idx_of, ordered_keys


def _row_normalize_abs(W: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """A = |W| with each row L1-normalized to 1 (or zero if the row sum was 0).

    Rows for pure-input nodes (embed/err/logit) have row_sum = 0 -> stay zero.
    """
    A = W.abs()
    row_sums = A.sum(dim=1, keepdim=True)
    safe = row_sums.clamp_min(eps)
    A = torch.where(row_sums > 0, A / safe, torch.zeros_like(A))
    return A


def _indirect_influence(A: torch.Tensor, max_depth: int) -> torch.Tensor:
    """B = A + A^2 + ... + A^max_depth using the polynomial form.

    Asserts A^max_depth is essentially zero (cycle detection); raises with a
    descriptive message otherwise.
    """
    B = torch.zeros_like(A)
    Ak = A.clone()
    last_max = float('inf')
    for _ in range(max_depth):
        B = B + Ak
        Ak = Ak @ A
        last_max = float(Ak.abs().max().item())
        if last_max < 1e-12:
            return B
    if last_max >= 1e-10:
        raise RuntimeError(
            f'indirect_influence: A^{max_depth} not nilpotent '
            f'(max abs = {last_max:.3e}). The graph likely has a cycle.'
        )
    return B


# ---------------------------------------------------------------------------
# Logit weights
# ---------------------------------------------------------------------------

def _logit_weights(
    ordered_keys: list, logits: torch.Tensor, y_true: int, sink_mode: str,
    device, dtype,
) -> torch.Tensor:
    N_nodes = len(ordered_keys)
    w = torch.zeros(N_nodes, device=device, dtype=dtype)
    if sink_mode == 'softmax_logits':
        probs = torch.softmax(logits.to(device=device, dtype=dtype), dim=0)
        for i, k in enumerate(ordered_keys):
            if k[0] == 'logit':
                w[i] = probs[int(k[1])]
    elif sink_mode == 'true_class':
        for i, k in enumerate(ordered_keys):
            if k[0] == 'logit' and int(k[1]) == int(y_true):
                w[i] = 1.0
    else:
        raise ValueError(
            f"sink_mode must be 'softmax_logits' or 'true_class', got {sink_mode!r}"
        )
    return w


# ---------------------------------------------------------------------------
# Threshold-prefix helper
# ---------------------------------------------------------------------------

def _keep_prefix(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """Given scores >= 0, return a boolean mask of the same shape selecting
    the smallest prefix (after sorting desc) whose cumsum / total >= threshold.
    The threshold-crossing element is included.

    Returns the keep mask aligned to the input order (not sorted).
    """
    n = scores.numel()
    if n == 0:
        return torch.zeros(0, dtype=torch.bool, device=scores.device)
    total = float(scores.sum().item())
    if total <= 0:
        # Caller decides what to do; return a None-like signal by raising.
        # We surface a special case: keep nothing (caller may override).
        return torch.zeros(n, dtype=torch.bool, device=scores.device)
    sorted_scores, sort_idx = torch.sort(scores, descending=True)
    cumsum = torch.cumsum(sorted_scores, dim=0)
    target = threshold * total
    # First index where cumsum >= target.
    crossed = (cumsum >= target).nonzero(as_tuple=False)
    if crossed.numel() == 0:
        first_cross = n - 1  # keep everything
    else:
        first_cross = int(crossed[0].item())
    keep_sorted = torch.zeros(n, dtype=torch.bool, device=scores.device)
    keep_sorted[: first_cross + 1] = True
    keep = torch.zeros_like(keep_sorted)
    keep[sort_idx] = keep_sorted

    # Off-by-one assertions (cheap).
    sum_kept = float(scores[keep].sum().item())
    assert sum_kept / total >= threshold - 1e-9, (
        f'prefix kept = {sum_kept / total:.6f} < threshold = {threshold}'
    )
    if int(keep.sum().item()) < n:
        # Removing the last kept element should drop us below threshold.
        last_kept_score = float(sorted_scores[first_cross].item())
        assert (sum_kept - last_kept_score) / total < threshold + 1e-9, (
            f'tighter prefix would also clear threshold'
        )
    return keep


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def prune_indirect_influence(
    nodes: dict, edges: list,
    K: int,
    logits: torch.Tensor, y_true: int,
    sink_mode: str,
    node_threshold: float, edge_threshold: float,
    device=None, dtype=torch.float32,
):
    """Two-stage indirect-influence pruning.

    Returns (kept_edges, kept_node_keys, diagnostics).
    """
    if device is None:
        device = logits.device

    if not (0.0 < node_threshold <= 1.0):
        raise ValueError(f'node_threshold must be in (0, 1], got {node_threshold}')
    if not (0.0 < edge_threshold <= 1.0):
        raise ValueError(f'edge_threshold must be in (0, 1], got {edge_threshold}')

    # ---- Build adjacency and influence ----
    W, idx_of, ordered_keys = _build_adjacency(nodes, edges, K, device, dtype)
    A = _row_normalize_abs(W)
    max_depth = K + 2
    B = _indirect_influence(A, max_depth)

    w_logit = _logit_weights(ordered_keys, logits, y_true, sink_mode, device, dtype)
    # B is indexed [target, source]: B[i, j] is the polynomial sum of path
    # mass from source j to target i. We want, for each source j, the total
    # mass flowing from j to ANY logit weighted by logit_weights -- that is
    # `sum_i B[i, j] * w_i = (B.T @ w_logit)[j]`. The spec writes
    # `node_score = B @ logit_weights` but uses the opposite W convention
    # (W indexed [source, target]). With our [target, source] indexing the
    # transpose is required.
    node_score = B.t() @ w_logit  # [N_nodes]

    # ---- Pre-prune counts ----
    counts_pre = {'feature': 0, 'error': 0, 'embedding': 0, 'logit': 0}
    for k in ordered_keys:
        kind = nodes[k]['kind']
        counts_pre[kind] = counts_pre.get(kind, 0) + 1
    n_edges_pre = len(edges)

    # ---- Node pruning: prunable = features ----
    feature_idx = [i for i, k in enumerate(ordered_keys) if nodes[k]['kind'] == 'feature']
    feat_scores = node_score[feature_idx]  # [num_features]
    total_feat = float(feat_scores.sum().item())
    n_features_pre = len(feature_idx)

    if total_feat <= 0:
        print('  WARNING: total prunable feature score is 0; keeping all features.')
        keep_feat_local = torch.ones(len(feature_idx), dtype=torch.bool, device=device)
    else:
        keep_feat_local = _keep_prefix(feat_scores, node_threshold)
    kept_feature_indices = {feature_idx[i] for i in range(len(feature_idx)) if bool(keep_feat_local[i].item())}

    kept_node_indices = set()
    for i, k in enumerate(ordered_keys):
        kind = nodes[k]['kind']
        if kind == 'feature':
            if i in kept_feature_indices:
                kept_node_indices.add(i)
        else:
            kept_node_indices.add(i)
    kept_node_keys = {ordered_keys[i] for i in kept_node_indices}

    # ---- Restrict edges to kept-node subgraph ----
    sub_edges = [
        (src, dst, w) for (src, dst, w) in edges
        if (src in kept_node_keys and dst in kept_node_keys)
    ]

    # ---- Build sub-adjacency, recompute B_sub ----
    # Reuse the dense matrix with rows/cols of pruned nodes zeroed -- this
    # preserves indexing into ordered_keys.
    keep_mask = torch.zeros(len(ordered_keys), dtype=torch.bool, device=device)
    for i in kept_node_indices:
        keep_mask[i] = True
    W_sub = W.clone()
    W_sub[~keep_mask, :] = 0
    W_sub[:, ~keep_mask] = 0
    A_sub = _row_normalize_abs(W_sub)
    B_sub = _indirect_influence(A_sub, max_depth)
    node_score_sub = B_sub.t() @ w_logit  # [N_nodes]

    # Override logit-node entries: logit nodes have no outgoing edges so
    # B_sub.t() @ w_logit gives them 0 (no path leaves a logit). Set them
    # directly to logit_weights (Anthropic's rule:
    # `node_score[logit_weights > 0] = logit_weights[logit_weights > 0]`).
    # Without this every edge into a logit gets edge_score = 0.
    for i, k in enumerate(ordered_keys):
        if k[0] == 'logit':
            node_score_sub[i] = w_logit[i]

    # ---- Edge pruning ----
    if len(sub_edges) == 0:
        kept_edges = []
        edge_threshold_total = 0.0
    else:
        edge_scores = []
        for (src, dst, w) in sub_edges:
            ti = idx_of[dst]
            si = idx_of[src]
            score = float(A_sub[ti, si].item()) * float(node_score_sub[ti].item())
            edge_scores.append(abs(score))
        edge_scores_t = torch.tensor(edge_scores, device=device, dtype=dtype)
        total_edge = float(edge_scores_t.sum().item())
        edge_threshold_total = total_edge
        if total_edge <= 0:
            print('  WARNING: total edge score is 0; keeping all subgraph edges.')
            keep_edge_local = torch.ones(len(sub_edges), dtype=torch.bool, device=device)
        else:
            keep_edge_local = _keep_prefix(edge_scores_t, edge_threshold)
        kept_edges = [sub_edges[i] for i in range(len(sub_edges)) if bool(keep_edge_local[i].item())]

    # ---- Post-prune counts ----
    counts_post = {'feature': 0, 'error': 0, 'embedding': 0, 'logit': 0}
    for k in kept_node_keys:
        kind = nodes[k]['kind']
        counts_post[kind] = counts_post.get(kind, 0) + 1
    n_edges_post = len(kept_edges)

    # ---- completeness_score ----
    # Fraction of |signed_weight| on incoming edges to kept feature/logit
    # nodes that comes from feature or embedding sources (not error).
    target_kinds = {'feature', 'logit'}
    non_error_src_kinds = {'feature', 'embedding'}
    num_w = 0.0
    den_w = 0.0
    for (src, dst, w) in kept_edges:
        if nodes[dst]['kind'] in target_kinds:
            den_w += abs(float(w))
            if nodes[src]['kind'] in non_error_src_kinds:
                num_w += abs(float(w))
    completeness_score = (num_w / den_w) if den_w > 0 else 0.0

    # ---- replacement_score ----
    # On the pruned subgraph, zero out the COLUMNS of A_sub corresponding to
    # error nodes (block any flow leaving an error node), recompute B, and
    # compare embedding -> logit path mass against the original B_sub.
    # For sink_mode=softmax_logits, average per-class with softmax weights.
    err_idx = [i for i, k in enumerate(ordered_keys) if k[0] == 'err']
    embed_idx = [i for i, k in enumerate(ordered_keys) if k[0] == 'embed']
    logit_idx = [i for i, k in enumerate(ordered_keys) if k[0] == 'logit']

    A_no_err = A_sub.clone()
    if err_idx:
        A_no_err[:, err_idx] = 0.0
    B_no_err = _indirect_influence(A_no_err, max_depth)

    def _emb_to_logit_mass(B_mat: torch.Tensor) -> dict[int, float]:
        """Per-logit-class total embedding-to-logit mass under B."""
        out = {}
        for li in logit_idx:
            c = int(ordered_keys[li][1])
            mass = 0.0
            for ei in embed_idx:
                mass += float(B_mat[li, ei].item())
            out[c] = mass
        return out

    mass_full = _emb_to_logit_mass(B_sub)
    mass_no_err = _emb_to_logit_mass(B_no_err)

    if sink_mode == 'true_class':
        full_v = mass_full.get(int(y_true), 0.0)
        rep_v = mass_no_err.get(int(y_true), 0.0)
        replacement_score = (rep_v / full_v) if full_v > 0 else 0.0
    else:
        probs = torch.softmax(logits.to(device=device, dtype=dtype), dim=0).tolist()
        weighted_full = 0.0
        weighted_rep = 0.0
        for c, p in enumerate(probs):
            weighted_full += p * mass_full.get(c, 0.0)
            weighted_rep += p * mass_no_err.get(c, 0.0)
        replacement_score = (weighted_rep / weighted_full) if weighted_full > 0 else 0.0

    diagnostics = {
        'sink_mode': sink_mode,
        'node_threshold': float(node_threshold),
        'edge_threshold': float(edge_threshold),
        'n_nodes_pre_by_kind': dict(counts_pre),
        'n_nodes_post_by_kind': dict(counts_post),
        'n_edges_pre': int(n_edges_pre),
        'n_edges_post': int(n_edges_post),
        'n_features_pre': int(n_features_pre),
        'n_features_post': int(counts_post.get('feature', 0)),
        'completeness_score': float(completeness_score),
        'replacement_score': float(replacement_score),
        'completeness_weight_convention': 'abs(signed_weight) on pruned subgraph',
    }

    return kept_edges, kept_node_keys, diagnostics
