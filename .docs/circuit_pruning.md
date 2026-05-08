# Circuit Tracing — Pruning Modification Spec

## Context

This spec modifies the pruning step of the circuit tracing pipeline described
in `circuit_tracing_implementation_plan.md`. The current pipeline (Steps A–F)
already produces a DAG with feature nodes, error nodes, and a sink. This spec
replaces Step G (the per-node top-fraction-by-|weight| rule) with the
indirect-influence pruning algorithm from Anthropic's circuit tracing paper
(https://transformer-circuits.pub/2025/attribution-graphs/methods.html,
appendix "Graph Pruning"), adapted for the RHM setting.

Steps A–F (anchor capture, linearization, edge weight computation,
bit-identity check, fidelity diagnostics) are unchanged. The output artifacts
listed in Step G of the original plan are largely the same (`nodes.pt`,
`edges.pt`, `fidelity.pt`, `graph.gpickle`), with additional fields described
below.

## Decisions (locked)

- **Sink choice (CLI flag)**: two modes, switched by `--sink_mode`:
  - `softmax_logits` (default): one logit node per class. Each logit node is
    weighted by `softmax(logits)[c]` from the original forward pass. This
    matches Anthropic's recipe (they cap to top-K covering 95%; we keep all
    `v` classes since `v` is small).
  - `true_class`: a single logit node `logit[y_true]` with weight 1. This
    answers "why did the model produce the correct answer."
  - **Drop `logit_diff` entirely.** No more `logit[y_true] - logit[c_runner_up]`
    sink. The asymmetry between promoting truth and suppressing runner-up is
    handled correctly by `softmax_logits` mode.
- **Embedding nodes**: added as new graph inputs. One node per leaf position
  `p`, representing the perturbation `tok_emb[x[p]] + pos_emb[p]` fed into
  block 0. Edge weights to layer-0 features computed via `M_0` (the
  linearization of block 0), same machinery as error nodes.
- **Pruning algorithm**: two-stage indirect-influence pruning. Nodes first,
  then edges. Embedding, error, and logit nodes are never pruned.
- **Thresholds (CLI flags)**: `--node_threshold` (default 0.8),
  `--edge_threshold` (default 0.98). Anthropic's defaults.
- **Indirect influence computation**: polynomial form
  `B = A + A^2 + ... + A^D` where `D = K + 2` is the longest path length
  (embedding -> K feature layers -> logit). Exact in finite steps because the
  DAG has no cycles. Don't use `(I - A)^-1`; the polynomial is faster and
  numerically safer for nilpotent `A`.


## Implementation steps

### Step 1 — Embedding nodes (`embeddings.py`, `circuit_trace.py`)

For the single input `x` of length `N` (leaves), build embedding nodes:

- One node per position `p in range(N)`: key `('embed', p)`.
- The "activation" stored on the node is the perturbation vector
  `e_emb[p] = tok_emb[x[p]] + pos_emb[p]` (shape `[d]`). Saved on the node
  for diagnostics; not used as a scalar.

For each embedding node `('embed', p)` and each active layer-0 feature
`(0, q, j)` (i.e. `z_0[q, j] > 0`), compute the edge weight:

```
contribution[(0, q, j) <- ('embed', p)]
  = act_scale_0 * W_enc_0[j, :] @ M_0(delta_p ⊗ e_emb[p])[q, :]
```

This is identical in shape to the existing error->feature contribution
formula (`circuit_tracing_implementation_plan.md` Step F), with `e_emb[p]`
in place of `e_k[p]`. Reuse the same code path with a different source
vector.

`M_0` is the closure for block 0 with all anchors frozen (ln1/ln2 scales,
attention pattern, MLP ReLU mask captured during the original forward
pass). The plan's `linearize.py` needs to expose this; currently it only
documents `M_{k+1}` for `k >= 0`. Add `M_0` as the same construction with
anchors taken from block 0.

Important: embedding nodes are PURE INPUTS. They have no incoming edges in
the DAG, exactly like error nodes.

Important : Embedding nodes will count in the subtree alignement verification. They correspond to the leaf tokens, so will group by groups of s.

### Step 2 — Per-class logit attribution (`attribution.py`)

Drop `logit_diff`. For each class `c in range(v)`:

- Feature -> logit_c attribution at position `p`:
  ```
  attr[c <- (K-1, p, i)]
    = (1/N) * W_cls[c, :] @ ln_f_jvp_at_p(W_dec_{K-1}[:, i] / act_scale_{K-1})
      * z_{K-1}[p, i]
  ```
- Error -> logit_c attribution at position `p`:
  ```
  attr[c <- (K-1, p, 'error')]
    = (1/N) * W_cls[c, :] @ ln_f_jvp_at_p(e_{K-1}[p])
  ```

Save logit nodes with key `('logit', c)` for `c in range(v)`. Each logit
node stores the raw logit value and `softmax(logits)[c]`.

### Step 3 — Node kinds and adjacency (`dag.py`, `prune.py`)

Add a `kind` field to nodes. Four kinds:

- `'feature'`: keys `(k, p, i)` with `z_k[p, i] > 0`.
- `'error'`: keys `(k, p, 'error')` for every (layer, position).
- `'embedding'`: keys `('embed', p)` for every position.
- `'logit'`: keys `('logit', c)` for every class.

Order nodes for the adjacency matrix by `(layer_index, kind_priority,
position, feature_id)`. Define a virtual layer index for each kind:

```
embedding         -> -1
feature at layer k -> k
error at layer k   -> k
logit              -> K
```

This ordering makes `A` strictly upper-triangular block (no self-loops,
no backward edges), which is required for the polynomial form to terminate.

Build the signed adjacency matrix `W` of shape `[N_nodes, N_nodes]` indexed
as `W[target, source] = signed edge weight`. Then build the unsigned,
row-normalized `A`:

```
A = |W|
row_sums = A.sum(axis=1)        # incoming sums per target
row_sums = max(row_sums, eps)   # eps = 1e-8
A = A / row_sums[:, None]       # broadcast, normalize each row to 1
```

Rows for embedding, error, and logit nodes that have no incoming edges
will have `row_sum = 0` and produce a zero row after the eps guard. That
is correct: pure-input nodes contribute no path mass via incoming edges.

### Step 4 — Indirect influence (polynomial form) (`prune.py`)

Compute `B = sum_{k=1}^{D} A^k` where `D = K + 2` (longest path length:
embedding -> layer 0 -> ... -> layer K-1 -> logit).

```python
def indirect_influence(A, max_depth):
    B = torch.zeros_like(A)
    Ak = A.clone()
    for _ in range(max_depth):
        B = B + Ak
        Ak = Ak @ A
        if Ak.abs().max() < 1e-12:
            break
    return B
```

Assert that `Ak.abs().max() < 1e-10` after `D` iterations; if not, the DAG
has a cycle and something is wrong upstream. Hard-fail with a clear error.

### Step 5 — Logit weights (`prune.py`)

Build `logit_weights` of shape `[N_nodes]`:

- For `--sink_mode softmax_logits`: `logit_weights[('logit', c)] =
  softmax(logits)[c]` for every class. Zero elsewhere.
- For `--sink_mode true_class`: `logit_weights[('logit', y_true)] = 1`.
  Zero elsewhere.

### Step 6 — Node pruning (`prune.py`)

```
node_score = B @ logit_weights        # shape [N_nodes]
```

Identify the prunable set: every node whose kind is `'feature'`. Embedding,
error, and logit nodes are never pruned (skip them in the sort).

Sort prunable nodes by `node_score` descending. Compute cumulative sum and
keep the smallest prefix such that
`cumulative[i] / total_prunable_score >= node_threshold`. Include the node
that crosses the threshold (off-by-one matters: keep up to and including
the first index where cumsum exceeds threshold).

Construct the kept-node set: kept-feature-nodes UNION embedding UNION error
UNION logit. Build a subgraph DAG restricted to these nodes; drop edges
whose source or target is not in the kept set.

### Step 7 — Edge pruning (`prune.py`)

On the subgraph from Step 6:

- Recompute `A_sub` (re-normalize rows; the row sums change after pruning).
- Recompute `B_sub` and `node_score_sub = B_sub @ logit_weights_sub`.
- **Fix logit node scores**: logit nodes have no outgoing edges so
  `B_sub @ logit_weights_sub` gives them zero. Override:
  ```
  node_score_sub[i] = logit_weights[i] for i in logit_nodes
  ```
  (This is the line in Anthropic's pseudocode:
  `node_score[logit_weights > 0] = logit_weights[logit_weights > 0]`.
  Without it every edge into a logit gets edge_score = 0 and survives only
  by accident.)
- Edge score:
  ```
  edge_score[target, source] = A_sub[target, source] * node_score_sub[target]
  ```
  Only nonzero where there is an actual edge.
- Flatten edge_score over the edge set (not the full matrix), sort
  descending. Keep prefix to `edge_threshold * total_edge_score`. Same
  off-by-one rule as Step 6.
- Construct final pruned edge set.

### Step 8 — Save and diagnose (`circuit_trace.py`)

Save artifacts (extending the original Step G):

- `nodes.pt`: dict `key -> {kind, ...}`. Per kind:
  - `'feature'`: `{kind: 'feature', layer: k, position: p, feature_id: i,
    z: float, label: dict, level: int, parent_position: int}`
  - `'error'`: `{kind: 'error', layer: k, position: p, level: int,
    parent_position: int, e_norm: float}`
  - `'embedding'`: `{kind: 'embedding', position: p, token_id: int,
    e_emb_norm: float}`
  - `'logit'`: `{kind: 'logit', class: c, logit: float, prob: float,
    is_true_class: bool}`
- `edges.pt`: `{'all': [...], 'pruned': [...]}`. Each entry:
  `{src: key, dst: key, weight: float, signed_weight: float}`. Save both
  the unsigned/normalized weight used for pruning and the original signed
  weight for downstream analysis.
- `fidelity.pt`: existing fields PLUS:
  - `node_threshold`, `edge_threshold`, `sink_mode`.
  - `n_nodes_pre_prune`, `n_nodes_post_prune` per kind.
  - `n_edges_pre_prune`, `n_edges_post_prune`.
  - `completeness_score`: fraction of incoming-edge mass on kept feature
    and logit nodes that comes from feature or embedding nodes (not error).
  - `replacement_score`: fraction of total path mass from embedding to
    logit nodes that flows entirely through feature nodes (no error nodes
    on the path). Compute this by running the influence computation a
    second time with all error nodes' rows of `A` zeroed out, and
    comparing the embedding-to-logit path mass before and after.
  - `subtree_alignment_fraction`: fraction of feature->feature edge mass
    in the pruned graph where the edge respects RHM structure
    (`q == p // s` for an edge from layer-k position `p` to layer-(k+1)
    position `q`). This was already in the original plan as a stdout
    diagnostic; promote it to a saved field.
- `graph.gpickle`: networkx DiGraph of the pruned graph.

Print to stdout:

- All previous diagnostics (bit-identity, per-layer error fraction,
  feature-mediated logit fraction).
- `node_threshold`, `edge_threshold`, `sink_mode`.
- Node counts pre/post by kind.
- Edge counts pre/post.
- `completeness_score`, `replacement_score`,
  `subtree_alignment_fraction`.

## CLI changes

Add to `circuit_tracing.circuit_trace`:

```
--sink_mode {softmax_logits, true_class}   default: softmax_logits
--node_threshold FLOAT                     default: 0.8
--edge_threshold FLOAT                     default: 0.98
```

Remove (or deprecate with a clear error message): `--prune_fraction`. The
old single-threshold rule is gone.

## Verification

End-to-end test (manual; extend the existing one in
`circuit_tracing_implementation_plan.md` Verification) (it will most likely by hard for you to do so on your own, I will work on it):

1. Run on a known-good RHM config with three SAEs. Use
   `transformer_meanclass_nores` for cleaner math.
2. Run with `--sink_mode true_class` and `--input_idx 0`. Confirm:
   - Bit-identity max error < 1e-4 (unchanged from before).
   - Embedding nodes appear: there are exactly `s^L` of them.
   - Logit nodes: exactly one (the true class) under `true_class` mode.
   - `node_threshold = 0.8` keeps roughly 10x fewer feature nodes than
     pre-pruning. Print the ratio; flag if it is outside [3x, 30x].
   - `completeness_score >= 0.7` if SAEs are good; `replacement_score`
     usually lower, around 0.5-0.7.
   - `subtree_alignment_fraction >= 0.5`. If significantly lower the
     SAEs are leaking levels; cross-reference with the
     with-residual / without-residual gap from the progress report.
3. Re-run with `--sink_mode softmax_logits`. Confirm `v` logit nodes
   appear and the graph differs from `true_class` (more diverse paths
   when the model is uncertain).
4. Re-run with `--node_threshold 0.95 --edge_threshold 0.99`. Confirm
   more nodes and edges kept; completeness_score should be higher.
5. Re-run with `--node_threshold 0.5`. Confirm aggressive pruning;
   completeness_score should drop.
6. Sanity: `--sink_mode true_class` on a misclassified input should
   show small total path mass on the logit, and the graph should
   look structurally degenerate (low subtree_alignment).

## Out of scope (deferred)

- Visualization (still Step G of the original plan; unchanged).
- Multi-input aggregation.
- Replacement-score computation for prompts where the model is
  uncertain across many classes (the path-mass reweighting may need
  more care; for now compute it per logit node and average weighted
  by `softmax(logits)[c]`).
- Attention QK attribution.
- Pruning embedding or error nodes (we never do this).

## Things to NOT change

- Step A (CLI loading), Step B (anchor capture except adding `M_0`
  exposure), Step C (SAE pass), Step D (bit-identity), Step E (`M_{k+1}`
  closures), Step F (edge weight computation) of the original plan.
- The rules-consistency invariant (still required, still hard-fails on
  mismatch).
- The label format on feature nodes
  (`{level, parent_position, value, p_value_given_fire}`).
- The act_scale convention.
- Saving both pre- and post-pruning edge lists.