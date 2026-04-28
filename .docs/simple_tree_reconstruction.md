# Tree reconstruction from SAE features

## Goal

Given a trained transformer + a set of SAEs (one per layer), reconstruct the
full RHM latent tree for each input by aggregating per-feature evidence about
each latent `Z_{l, j}`.

This relies on the per-feature conditionals `P(Z = z | f_i > 0)` already
computed by the entropy/specificity pipeline. No new statistics need to be
estimated — only an inference-time aggregation.

## Inputs

Per `(SAE, layer k, position p, feature i)`:
- `cond_prob[z]`: `P(Z_{l, j} = z | f_i > 0)` for `z = 0, ..., v_l - 1`,
  with `l = L - 1 - k` and `j = p // s^(1+k)`.
- `firing_rate`: `P(f_i > 0)` (used only to filter dead features).

These already exist as `.feature_latent.pt` artifacts. Reuse them; do not
recompute.

## Per-input reconstruction

For one input `x`:

1. Run `x` through the transformer + every SAE; collect, per `(k, p)`, the
   decoder-weighted activations `f_i(x)` for all features at that tap point.
2. For every latent `(l, j)` in the tree:
   - Identify the canonical layer `k = L - 1 - l` and the set of canonical
     positions `P(l, j) = { p : p // s^(1+k) == j }`. Size is `s^(1+k)`.
   - For each `p in P(l, j)`:
     - Let `A_p = { i : f_i(x) > 0 and feature i is alive }` be the firing
       features at `(k, p)`.
     - If `A_p` is empty, skip this position (do not contribute).
     - Else compute the position score (a length-`v_l` vector):
       ```
       score_p[z] = sum_{i in A_p} f_i(x) * cond_prob[i][z]
                    / sum_{i in A_p} f_i(x)
       ```
   - Aggregate across positions by uniform mean over the non-empty positions:
     ```
     score[l, j][z] = mean_{p with non-empty A_p} score_p[z]
     ```
   - Predicted latent: `Z_hat[l, j] = argmax_z score[l, j][z]`.
3. The collection `{ Z_hat[l, j] }` over all `(l, j)` is the reconstructed
   tree.

Notes:
- Only features that fire contribute. Silent features are ignored.
- Activation weights are decoder-weighted `f_i = z_i * ||W_dec[:, i]||`,
  consistent with the rest of the pipeline.
- Dead features (those with `firing_rate == 0` on the eval set used to fit
  the conditionals) are excluded.
- If every position at `(l, j)` has empty `A_p`, mark `Z_hat[l, j]` as
  undefined (NaN / sentinel) — do not silently fall back to a prior.

## Configuration toggles

Both must be implemented and runnable from the same script via flags:

- `--weighting {activation, uniform}`:
  - `activation` (default): use `f_i(x)` as the weight inside `score_p`.
  - `uniform`: replace `f_i(x)` with `1` (mean of `cond_prob` over firing
    features).
- `--token_mode {one_token, all_tokens}`: matches the SAE token mode. Reuse
  the existing `(layer, position) -> (level, position_in_level)` mapping
  from the entropy pipeline. `cls_token` is out of scope.

## Outputs

For each eval input, store:
- `Z_hat[l, j]` for every `(l, j)` (integer or sentinel).
- `score[l, j][z]` for every `(l, j)` (full vector, float64).
- A per-(l, j) flag indicating whether any position contributed.

Aggregate over the eval set and report:
- Per-level accuracy: `mean over (l, j, x) of 1[Z_hat[l, j](x) == Z[l, j](x)]`,
  grouped by `l`.
- Whole-tree accuracy: fraction of inputs where all `(l, j)` are correct.
- Coverage: fraction of `(l, j, x)` where at least one position contributed.


## Sanity checks (run before trusting numbers)

1. **Root-level agreement.** At `l = 0` (root class), the reconstructor's
   accuracy should be close to (and not exceed by much) the transformer's
   own classifier head accuracy on the same eval set. Large discrepancies
   either way signal a bug.
2. **Oracle conditionals.** Replace `cond_prob[i]` with a one-hot at the
   true `Z` for a synthetic feature that fires iff `Z == z_0`; the
   reconstructor must recover `Z` exactly at that `(l, j)`.
3. **Position consistency.** For a single `(l, j)` with multiple canonical
   positions, the per-position `score_p` distributions should be
   qualitatively similar (same argmax most of the time). If they
   systematically disagree, the `(layer, position) -> (level,
   position_in_level)` mapping is likely off by one — fix before reporting.


## Out of scope (do not implement here)

- Logistic probe baseline on SAE activations (separate script, separate PR).
- Cross-layer aggregation (using non-canonical layers as additional
  evidence).
- Silent-feature contributions / Bayes-factor formulation.
- Activation-bin conditionals `P(Z | f_i ∈ bin)`.
- Calibration / confidence reporting (argmax only for now).

## Where this fits

Tell me if you think this would work or if you have a better option : A new script in  that recovers the inputs from the analysis artifacts that were created by scripts/sae_eval/run.py