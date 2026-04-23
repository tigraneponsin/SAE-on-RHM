# Entropy-based feature specificity: handoff

## Goal

A new metric for SAE features trained on transformers trained on the Random
Hierarchy Model (RHM): per-feature conditional entropy of the RHM latent
given the feature's firing, plus three weighted aggregations per
(SAE, layer, position).

## The math

Fix an SAE, a layer `k`, a token position `p`. The expected RHM latent is
`Z = Z_{l, j}` with `l = L - 1 - k` and `j = p // s^(1+k)`, taking values in
`{0, ..., v_l - 1}`.

For each feature `i`:
- `z_i(x)`: raw post-ReLU encoder activation.
- `f_i(x) = z_i(x) * ||W_dec[:, i]||`: decoder-weighted activation.
- "Fires" means `f_i(x) > 0` (equivalently `z_i(x) > 0`).

### Per-feature entropy (one definition, used for every feature)

Estimated directly from joint counts on the eval set:

```
p_i(z) = P_emp(Z = z | F_i > 0)
       = #{x : F_i(x) > 0 and Z(x) = z} / #{x : F_i(x) > 0}

H_i = -sum_z p_i(z) * log(p_i(z))
```

### Three aggregations per (layer, position)

```
H_bar_fire = sum_i P(F_i > 0) * H_i  /  sum_i P(F_i > 0)

H_bar_raw  = sum_i E[z_i] * H_i      /  sum_i E[z_i]

H_bar_dec  = sum_i E[f_i] * H_i      /  sum_i E[f_i]
```

Sums over alive features at this (layer, position). Each aggregate should
also be available normalized by `H(Z)` so the score lies in `[0, 1]`.

## The latent prior `P(Z)` and its entropy

Two versions are computed and stored everywhere `H(Z)` would be referenced:

**Theoretical (rule-based):** `P(Z_{l, j} = z)` computed exactly from the
RHM composition rules by descending the tree — start from a uniform class
prior, propagate down via `P(child slot value | parent value) = (1/m) *
(count of that value in the parent's m allowed s-tuples at that slot)`,
marginalize over parents at each step. This is exact, free of sampling
noise, and is the canonical normalizer for the entropy aggregates.

**Empirical:** `P_emp(Z = z)` from histogramming the eval set, with the
corresponding empirical `H_emp(Z)`.

Both should be reported alongside each other. The comparison is itself
informative: large discrepancies signal eval-set sampling issues
(insufficient size, biased sampling, dedup artifacts) that would also
contaminate the per-feature `H_i` estimates.

## What ultimately needs to exist

- For every (SAE, layer, position, feature): the per-feature entropy `H_i`
  and the three weights (`P(F_i > 0)`, `E[z_i]`, `E[f_i]`) that go into the
  aggregations. Keep these queryable per feature, not only collapsed into
  aggregates — downstream work (clustering, sensitivity, sample-split nulls)
  will consume them.
- For every (SAE, layer, position): the three aggregates above, in nats and
  normalized by the theoretical `H(Z)`. These need to land somewhere
  accessible to the sweep evaluation alongside L0, dead-feature fraction,
  etc., so SAE configs can be compared on specificity.
- For every (RHM instance, level, position): both the theoretical `P(Z)` /
  `H(Z)` and the empirical `P_emp(Z)` / `H_emp(Z)`, side by side.

## Things worth being careful about

**Numerical precision.** Entropies should be computed in float64. The
`p log p` term loses meaningful precision in float32 when probabilities span
orders of magnitude. Use `xlogy` or an explicit zero mask for the
`0 log 0 = 0` convention.

**Aggregate, then normalize.** Aggregate `H_i` in nats with the chosen
weights, then divide the aggregate by `H(Z)` once. `H(Z)` is constant
across features at a fixed (layer, position) so it factors out — but
`H(Z)` varies across positions, which makes consistent end-of-pipeline
normalization the sane choice for cross-position comparisons.

**Token modes.** The `(layer, position) -> (level, position_in_level)`
mapping is well-defined for the leaf-position token modes (`one_token`,
`all_tokens`). For `cls_token`, the latent target is not very well defined right now. 

**Dead features.** Features with `P(F_i > 0) == 0` have no defined `H_i`
and should be excluded from both per-feature computation and aggregation.

## Sanity / acceptance checks

These are cheap and catch the likely failure modes. Run them before relying
on any number.

1. **Theoretical vs empirical marginal.** On `>= 1e5` eval samples,
   `P_emp(Z_{l, j})` should agree with the theoretical `P(Z_{l, j})` to
   within `1/sqrt(N)`-scale fluctuations, for every `(l, j)`. Same for
   `H_emp(Z)` vs `H(Z)`. Catches off-by-one bugs in the position-to-parent
   mapping in the propagator and signals eval-set quality issues.

2. **Oracle feature.** A synthetic feature with `F(x) = 1[Z(x) == z_0]`
   should yield `H_i = 0` exactly (within float64 precision), for any `z_0`.

3. **Uniform feature.** A synthetic feature with `F(x) = 1` always should
   yield `p_i = P_emp(Z)` and therefore `H_i = H_emp(Z)`.

4. **Aggregation degeneracy.** If all alive features have identical `H_i`,
   All three aggregates should equal that value regardless of the weights.

If any of these fail, the implementation is wrong; fix before proceeding.

## Out of scope (deferred, but design with it in mind)

- Significance testing against shuffle nulls.
- Clustering features and computing cluster-level entropies.
- Sensitivity analysis (`I(Z; F)`, per-cluster recall).
- Causal interventions.

These all consume the per-feature `H_i` and weight tensors. Whatever is
stored should be rich enough that they don't require re-running the
streaming activation pass.