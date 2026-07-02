"""Indirect-influence pruning for the circuit-tracing DAG.

Two-stage algorithm:
  1. Build the signed adjacency W [N_nodes, N_nodes] indexed as
     W[target, source] = signed edge weight.
  2. Build A = |W|, row-normalized so each row sums to 1 (rows of pure-input
     nodes -- embeddings, errors, logits -- naturally have row sum 0 and
     stay zero after a clamped-min eps guard).
  3. Compute the indirect influence
        B = A + A^2 + ... + A^D     where D = K + 1
     using the polynomial form. K + 1 is the longest path length (in edges):
     embedding (virtual layer -1) -> feat 0 -> ... -> feat K-1 -> logit
     (virtual layer K) is K + 1 hops. With a strict DAG (every edge advances
     the virtual layer index by at least one), A is nilpotent and the sum is
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
  7. Reachability trim: keep only edges on some path
     (embedding|error) -> ... -> logit within the kept-edge set, dropping
     any newly orphaned nodes from kept_node_keys.

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
# Reachability trim
# ---------------------------------------------------------------------------

def _reachability_trim(
    kept_edges: list, kept_node_keys: set, nodes: dict,
) -> tuple[list, set]:
    """Keep only edges on some path input -> ... -> logit in kept_edges.

    Inputs are embedding/error nodes (no incoming edges in this DAG).
    Sinks are logit nodes. A node survives iff it is forward-reachable from
    some input AND backward-reachable from some logit, using only kept_edges.
    An edge survives iff both endpoints survive.
    """
    if not kept_edges:
        # No edges -> no node can sit on a source->logit path. Keep only
        # nodes that are themselves both source and sink, which is empty.
        return [], set()

    fwd: dict = {}
    bwd: dict = {}
    for (src, dst, _) in kept_edges:
        fwd.setdefault(src, []).append(dst)
        bwd.setdefault(dst, []).append(src)

    sources = {k for k in kept_node_keys if k[0] in ('embed', 'err')}
    sinks = {k for k in kept_node_keys if k[0] == 'logit'}

    def _bfs(seeds: set, adj: dict) -> set:
        seen = set(seeds)
        frontier = list(seeds)
        while frontier:
            node = frontier.pop()
            for nxt in adj.get(node, ()):  # neighbors via adj
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        return seen

    forward_reachable = _bfs(sources, fwd)
    backward_reachable = _bfs(sinks, bwd)
    survivors = forward_reachable & backward_reachable

    trimmed_edges = [
        (s, d, w) for (s, d, w) in kept_edges
        if s in survivors and d in survivors
    ]
    return trimmed_edges, survivors


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
    # Longest path embedding -> feat 0 -> ... -> feat K-1 -> logit is K + 1
    # edges, so the polynomial B = A + ... + A^{K+1} captures every path.
    max_depth = K + 1
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

    # ---- Node pruning: prunable = features AND errors ----
    # Error nodes carry their own indirect influence (node_score gives the
    # error->logit path mass, same scale as features), so they compete with
    # features for the kept budget under one node_threshold. Embeddings and
    # logits are pure inputs/sinks and are always kept.
    prunable_idx = [i for i, k in enumerate(ordered_keys)
                    if nodes[k]['kind'] in ('feature', 'error')]
    prunable_scores = node_score[prunable_idx]
    total_prunable = float(prunable_scores.sum().item())
    n_features_pre = sum(1 for i in prunable_idx
                         if nodes[ordered_keys[i]]['kind'] == 'feature')

    if total_prunable <= 0:
        print('  WARNING: total prunable (feature+error) score is 0; '
              'keeping all prunable nodes.')
        keep_local = torch.ones(len(prunable_idx), dtype=torch.bool, device=device)
    else:
        keep_local = _keep_prefix(prunable_scores, node_threshold)
    kept_prunable_indices = {prunable_idx[i] for i in range(len(prunable_idx))
                             if bool(keep_local[i].item())}

    kept_node_indices = set()
    for i, k in enumerate(ordered_keys):
        kind = nodes[k]['kind']
        if kind in ('feature', 'error'):
            if i in kept_prunable_indices:
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

    # ---- Reachability trim ----
    # Drop edges (and their now-orphan endpoints) that do not lie on any
    # path embedding/error -> ... -> logit within the current kept-edge set.
    # Without this pass we can keep edges into a feature whose own outgoing
    # edges were all pruned, leaving visually noisy dead-end branches.
    kept_edges, kept_node_keys = _reachability_trim(
        kept_edges, kept_node_keys, nodes,
    )

    # ---- Post-prune counts ----
    counts_post = {'feature': 0, 'error': 0, 'embedding': 0, 'logit': 0}
    for k in kept_node_keys:
        kind = nodes[k]['kind']
        counts_post[kind] = counts_post.get(kind, 0) + 1
    n_edges_post = len(kept_edges)

    # ---- completeness_score ----
    # Fraction of |signed_weight| on incoming edges to kept feature/logit
    # nodes that comes from feature or embedding sources (not error). Note:
    # error-node pruning can raise this, since pruned error->target edges drop
    # out of the denominator.
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
    # Computed on the final reachability-trimmed kept-edge graph.
    # Build W_trim from kept_edges, row-normalize to A_trim, sum the
    # polynomial B_trim = A_trim + A_trim^2 + ... For each logit c the
    # total polynomial path mass arriving from embedding sources is
    # emb_mass[c], from error sources err_mass[c]. Then
    #     replacement_score[c] = emb_mass[c] / (emb_mass[c] + err_mass[c]).
    # 1.0 means the trimmed circuit explains the logit entirely through
    # feature/embedding pathways; 0.0 means error terms dominate. Errors
    # plugged in directly at the logit count more than errors deep in the
    # feature hierarchy, since the latter get diluted by feature mixing.
    embed_idx = [i for i, k in enumerate(ordered_keys) if k[0] == 'embed']
    err_idx = [i for i, k in enumerate(ordered_keys) if k[0] == 'err']
    logit_idx = [i for i, k in enumerate(ordered_keys) if k[0] == 'logit']

    if len(kept_edges) == 0:
        replacement_score = 0.0
    else:
        N_nodes = len(ordered_keys)
        W_trim = torch.zeros(N_nodes, N_nodes, device=device, dtype=dtype)
        for (src, dst, w) in kept_edges:
            W_trim[idx_of[dst], idx_of[src]] += float(w)
        A_trim = _row_normalize_abs(W_trim)
        B_trim = _indirect_influence(A_trim, max_depth)

        def _src_mass(src_indices: list) -> dict[int, float]:
            out = {}
            for li in logit_idx:
                c = int(ordered_keys[li][1])
                mass = 0.0
                for si in src_indices:
                    mass += float(B_trim[li, si].item())
                out[c] = mass
            return out

        emb_mass = _src_mass(embed_idx)
        err_mass = _src_mass(err_idx)

        def _per_class_score(c: int) -> float:
            e = emb_mass.get(c, 0.0)
            r = err_mass.get(c, 0.0)
            denom = e + r
            return (e / denom) if denom > 0 else 0.0

        if sink_mode == 'true_class':
            replacement_score = _per_class_score(int(y_true))
        else:
            probs = torch.softmax(logits.to(device=device, dtype=dtype), dim=0).tolist()
            replacement_score = sum(
                p * _per_class_score(c) for c, p in enumerate(probs)
            )

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
        'n_errors_pre': int(counts_pre.get('error', 0)),
        'n_errors_post': int(counts_post.get('error', 0)),
        'completeness_score': float(completeness_score),
        'replacement_score': float(replacement_score),
        'completeness_weight_convention': 'abs(signed_weight) on pruned subgraph',
    }

    return kept_edges, kept_node_keys, diagnostics
