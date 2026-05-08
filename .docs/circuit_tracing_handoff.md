# Circuit Tracing with SAE Insertion — Implementation Handoff

## Context

This handoff assumes familiarity with the existing codebase. The transformer and SAE
training pipelines are working. The direct analysis pipeline (conditional entropy,
selectivity, tree reconstruction) is working. This task is the next step: **circuit
tracing across SAE layers for a single input**.

The goal is a deterministic, per-input attribution DAG over SAE features, analogous
to what Anthropic produces with CLTs but using the existing SAEs inserted into the
forward pass. The output is a weighted, pruned graph that can be inspected against
the RHM tree structure.

---

## RHM and architecture parameters

The codebase is general over the RHM parameters (s, L, v, m, n_c) and transformer
depth. All formulas below are written in terms of these parameters. Do not hardcode
any specific configuration.

Key quantities derived from (s, L):
- Number of leaf positions (sequence length): `N = s^L`
- Number of transformer blocks: `K = L` (one block per RHM level)
- Block index: `k ∈ {0, ..., K-1}`, bottom-to-top
- Hypothesized RHM level resolved by block k: `level = L - 1 - k`
- Ancestor position at level l for leaf position p: `a_l(p) = p // s^(L - l)`

Model variant in use: `transformer_meanclass` (or `_nores`).
- One SAE per layer, weight-shared across all `N` token positions, hooked post-block
  on the residual stream.
- After block K-1: mean pool over all `N` positions → linear classifier → `n_c` logits.

Layer indexing: 0-based, bottom-to-top. SAE at layer k is stored as
`model.blocks[k]`'s hook in the existing `train_sae.py` infrastructure.

---

## Step 0 — SAE checkpoint selection (manual for now)

SAE checkpoints are provided manually. For each layer k in `{0, ..., K-1}`, the
caller specifies the path to the SAE checkpoint to use.

The intended checkpoint at each layer is the most sparse one (highest λ) for which
the single-layer normalized classification error is still 0, as determined by the
sweep infrastructure. But the selection itself is done outside this script for now —
the script just loads what it is given.

Interface: the script takes a list of K checkpoint paths, one per layer, in
bottom-to-top order. Example:

```
python circuit_trace.py \
  --model_ckpt path/to/transformer.pt \
  --sae_ckpts path/to/sae_layer0.pt path/to/sae_layer1.pt path/to/sae_layer2.pt \
  --input_idx 42
```

Each SAE checkpoint must contain: `W_enc`, `b_enc`, `W_dec`, `b_dec`, `act_scale`.
Check `train_sae.py` for the exact checkpoint format.

---

## Method: linearized forward pass with SAE insertion and error nodes

### Step 1 — Forward pass on the original model, cache linearization anchors

Run the original transformer forward pass on a single input x (shape: `[N, d_model]`).
Cache, at every layer k:

- `x_k`: post-block residual stream, shape `[N, d]`. This is the SAE's input.
- `attn_pattern_k`: attention weights, shape `[n_heads, N, N]`, **after softmax**.
  Frozen here; treated as a fixed matrix for the rest of the procedure.
- `ln_scale_k`: LayerNorm scale factor applied before the attention and MLP sublayers
  at layer k+1, shape `[N, 1]` or `[N, d]` depending on implementation. Frozen here.
- `mlp_relu_mask_k`: binary mask from the sign of the MLP hidden pre-activations at
  layer k+1, shape `[N, d_mlp]`. Freezing this makes the MLP a linear operator.

**Why the MLP ReLU mask is needed here but not in Anthropic's setup:** Anthropic's
CLT replaces the MLP entirely, so there is no MLP ReLU to worry about. Here the
transformer's MLP blocks are still present inside each block. Freezing the ReLU mask
(on/off per hidden unit, from the first forward pass) linearizes the MLP so that the
full block becomes an affine map, enabling exact linear attribution.

**Note:** `transformer_meanclass_nores` has no residual skip connections, which makes
the linearization cleaner and the graph sparser. Prefer this variant for initial
experiments if available.

### Step 2 — Run SAEs, compute error nodes

For each layer k, run the SAE on `x_k` using the `act_scale` from the checkpoint:

```
x_scaled_k  = act_scale_k * x_k                                    # shape [N, d]
z_k         = ReLU(W_enc_k @ x_scaled_k.T + b_enc_k[:, None]).T   # shape [N, d_latent]
x_hat_k     = (W_dec_k @ z_k.T).T + b_dec_k                        # shape [N, d], scaled space
x_hat_k     = x_hat_k / act_scale_k                                 # back to original space
e_k         = x_k - x_hat_k                                         # error node, shape [N, d]
```

`z_k` is the **raw post-ReLU encoder activation** (before any decoder-norm
reweighting). This is the quantity used for attribution edge weights — see Step 4.

The reconstruction identity `x_hat_k + e_k = x_k` holds by construction. The spliced
residual stream fed into block k+1 is `x_hat_k + e_k = x_k` exactly. **No behavior
is lost.** The error nodes make the spliced model bit-identical to the original.

Store `z_k` (raw sparse activations), `x_hat_k`, and `e_k` for each k.

### Step 3 — Linearize each block

With attention patterns, LayerNorm scales, and ReLU masks all frozen from Step 1,
block k+1 is an affine map from `R^{N×d}` to `R^{N×d}`. Represent it implicitly as
a function `M_{k+1}(v)` that applies the frozen linearized block to an input `v`.

Do **not** materialize the full `[N*d, N*d]` matrix explicitly. For large N or d
this is wasteful and does not generalize. Instead, implement `M_{k+1}` as a function
and compute attributions via VJPs using `torch.func.jacrev` or explicit `.backward()`
calls on the frozen-graph forward pass.

The linearized block consists of:
- Attention sublayer: `sum_h attn_pattern_k[h, p, p'] * W_V_h * W_O_h`, scaled by
  the frozen LayerNorm denominator. Cross-position in general (all p' can contribute to p).
- MLP sublayer: `W_out @ diag(relu_mask_k[p]) @ W_in`, scaled by frozen LayerNorm.
  Position-local (no cross-position mixing in the MLP).
- Residual connections (if present): add identity to both sublayers.

### Step 4 — Compute feature-to-feature attribution edges

**Why raw activations z_k, not decoder-weighted f_k:**

The attribution edge is the linear effect of feature i's firing on the pre-activation
of feature j at the next layer. Feature i's contribution to the residual stream is
`W_dec_k[:, i] * z_k[p, i]` (in scaled space). The decoder direction `W_dec_k[:, i]`
already encodes the feature's direction and norm. Using the decoder-weighted activation
`f_k[p, i] = z_k[p, i] * ||W_dec_k[:, i]||` would double-count the decoder norm.
Always use raw `z_k`.

**Edge weight formula:**

Attribution from feature i at (layer k, position p) to feature j at (layer k+1,
position q):

```
A[j, q, i, p] = W_enc_{k+1}[j, :] @ M_{k+1}[q, p] @ (W_dec_k[:, i] / act_scale_k)
```

where `M_{k+1}[q, p]` is the `[d, d]` block of the linearized operator mapping
position p input to position q output, and dividing `W_dec_k[:, i]` by `act_scale_k`
converts it from scaled space to residual-stream space before the block reads it.

The actual edge contribution on this input:

```
contribution[j, q ← i, p] = A[j, q, i, p] * z_k[p, i]
```

For the error node at (layer k, position p) to feature j at (layer k+1, position q):

```
contribution[j, q ← err, k, p] = W_enc_{k+1}[j, :] @ M_{k+1}[q, p] @ e_k[p]
```

where `e_k[p]` is already in original (unscaled) residual-stream space.

**act_scale consistency check:** Before proceeding, assert that
`(W_dec_k @ z_k.T).T / act_scale_k + e_k` reconstructs `x_k` to float32 precision.

In practice: for each active feature j at (k+1, q) — active means `z_{k+1}[q, j] > 0`
— compute its total pre-activation as the sum of all upstream contributions. This
linearly decomposes the pre-activation into per-source terms, which become the edge
weights.

### Step 5 — Attribution to logits

The mean pool + linear classifier is globally linear:

```
logit_c = W_cls[c, :] @ (1/N) * sum_p x_hat_{K-1}[p]
```

Attribution from feature i at (layer K-1, position p) to logit c:

```
A_logit[c, i, p] = (1/N) * W_cls[c, :] @ (W_dec_{K-1}[:, i] / act_scale_{K-1})
```

This is **input-independent** — compute once for all inputs. The actual contribution
on a given input is `A_logit[c, i, p] * z_{K-1}[p, i]`.

Attribution from error node at (layer K-1, position p) to logit c:

```
contribution[c ← err, K-1, p] = (1/N) * W_cls[c, :] @ e_{K-1}[p]
```

Use the **logit difference** between the correct class and the runner-up as the
primary scalar to trace back. This is cleaner than tracing a single logit.

### Step 6 — Build the DAG and prune

**Nodes:**
- One node per active (feature i, layer k, position p) triple.
  Active means `z_k[p, i] > 0`.
- One error node per (layer k, position p): `K × N` total.
- One logit-difference node (correct class minus runner-up).

**Edges:**
- Feature → feature (adjacent layers): weight = `A[j, q, i, p] * z_k[p, i]`
- Error → feature: weight = `W_enc_{k+1}[j, :] @ M_{k+1}[q, p] @ e_k[p]`
- Feature → logit: weight = `A_logit[c, i, p] * z_{K-1}[p, i]`
- Error → logit: weight = `(1/N) * W_cls[c, :] @ e_{K-1}[p]`

**Pruning:** per-node, keep top incoming edges by |weight| that together account for
80% of absolute incoming flow. Adjustable threshold as needed — the graph is small for
typical RHM configurations but pruning is still needed for readability.

---

## Fidelity diagnostics

Compute and log these for every circuit before inspecting the graph.

**1. Bit-identity check (sanity):**
Assert that the fully spliced model (SAE at every layer with error nodes added back)
produces logits matching the original forward pass to float32 precision. Guaranteed
by construction but assert it anyway.

**2. Per-layer error mass:**
```
error_fraction_k = sum_{j,q} |err contribution to j at q from layer k| /
                   sum_{j,q} |total pre-activation of j at q|
```
If > 20% at any layer, the SAE at that layer is not capturing causally relevant
directions. This is the quantitative signal to consider switching to transcoders.

**3. Feature-mediated logit fraction:**
Sum of |feature → logit| edge weights divided by |logit difference|. Should be close
to 1.0 if the SAE basis is causally sufficient at the final layer.

---

## RHM-specific checks on the graph

**Subtree alignment:** For an edge from (layer k, position p) to (layer k+1, position
q), the RHM predicts `q = p // s` (ancestor one level up). The fraction of total edge
mass respecting this constraint is the primary alignment score. Flag off-subtree edges.

General formula: a feature at (layer k, position p) should only have strong edges to
features at (layer k+1, position q) where `q = a_{L-1-k}(p) = p // s^(1 + ...)`.
Check `entropy_specificity_handoff.md` for the exact level-to-position mapping.

**Feature latent labels:** Each active feature at (layer k, position p) has stored a conditionnal distribution on parent latent value when running `run.py`. Annotate graph node with argmax of this conditionnal distribution.  A clean circuit should show: level-0 leaf-value features at layer 0 →
level-1 latent features at layer 1 → ... → level-(L-1) root features at layer K-1
→ correct logit.



---

## Output artifacts

Per input, save:

- `nodes.pt`: dict `(layer, position, feature_idx)` → `{z_activation, latent_label,
  layer, position}`. Error nodes keyed as `(layer, position, 'error')`.
- `edges.pt`: list of `{src, dst, weight}` dicts, before and after pruning.
- `fidelity.pt`: per-layer error mass, feature-mediated logit fraction.
- A networkx `DiGraph` for visualization.

Visualization: layered layout, x-axis = position, y-axis = layer. Annotate nodes
with latent labels (argmax of the conditionnal distribution). Color edges by sign (positive = orange, negative = blue).

---

## What is out of scope for this handoff

- Automatic SAE selection by sweep. Checkpoints are provided manually for now.
- Training transcoders or CLTs. Fallback if per-layer error mass is large.
- Multi-input aggregation of graphs. Per-input only for now.
- Attention attribution (QK circuits). Attention patterns are frozen infrastructure here.

---

## Files to read before starting

- `train_sae.py` — SAE checkpoint format, `act_scale` convention.
- `scripts/sae_eval/run.py` — how `x_k` is hooked from
  `model.blocks[layer_id]`, and the `.feature_latent.pt` artifact format.
- `scripts/sae_eval/streaming.py` - the library that we use to analyze SAEs
- `entropy_specificity_handoff.md` — latent label conventions, level/position
  mapping formula (`level = L-1-k`, `ancestor = p // s^(1+k)`).
