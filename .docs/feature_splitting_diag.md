# Feature-splitting diagnostic

Script: `scripts/sae_sweep/feature_splitting_diag.py`. Offline post-processing of
`*.sae_eval.pt` artifacts (no SAE re-evaluation). Builds on
`min_entropy_diag.py`.

## Why

Some SAE features look selective to BOTH a parent latent and a child latent.
That is the signature of feature splitting / absorption: a feature that is
really a parent's, carved toward one child value. The simple unconstrained
argmin (min_entropy_diag) labels such a feature by its child and never asks
whether it also pins the parent beyond what the RHM tree's structural leak
already forces. This script adds that test and a reassignment rule.

## Conventions

Code level convention: root = level 0, leaf = level L. For a cell `(l, j)`:
- tree-parent (toward root) = `(l-1, j//s)`, defined only for `l >= 1`.
- `H_theoretical[(l, j)]` = entropy of latent `Z_{l,j}` (nats), from the artifact.
- `H_per_feature[group, p, f]` = conditional entropy `H(Z_group | f fires)`
  (nats), feature binarized to {0, active}.

## Procedure (per SAE position p, weight scheme, feature f)

1. **Child** = the unconstrained whole-tree argmin cell `(l*, j*)` (the latent
   feature f is most selective for).
2. **Parent** = the child's tree-parent `(l*-1, j*//s)`. Root children
   (`l* = 0`) have no parent and are never reassigned.
3. **Specialized value** `c_i` = the child value the feature fires most on,
   `argmax_c P(child = c | f fires)` over the child cell's values, with
   `P(child = c | f fires) = joint_fire_count[child_row(c), p, f] /
   firing_count[p, f]` straight from the artifact.
4. **Value-specific structural leak**
   `leak(child, c_i) = H(Z_parent | Z_child = c_i)`, in nats. This is a property
   of the RHM rules only, identical across the whole sweep, so it is computed
   ONCE. Index it by `(CHILD cell, value)`.
   `leak_norm[child, c_i] = leak(child, c_i) / H_theoretical[parent]`.
5. **Ratio**
   `ratio_f = (H_per_feature[parent, p, f] / H_theoretical[parent]) / leak_norm[child, c_i]`.
   The normalizer `H_theoretical[parent]` cancels, so `ratio_f = H(parent | f
   fires) / H(parent | child = c_i)`: how much the feature pins the parent
   against how much the SPECIFIC child value it specialized to already forces
   on the parent (the value-specific leak).
6. **Reassign** when `ratio_f < alpha` (CLI `--alpha`, default 0.5): the
   feature moves from child to parent, keeping the child as a carve tag. A low
   ratio means the feature determines the parent far beyond leak, i.e. it is
   really a parent feature. Dead features, root children, and undefined-leak
   cases (unreachable child value) are not reassigned.

The per-value leak comes from the joint `P(parent=u, child=a)`:
`P(parent=u) * (1/m) * #{u's m rules with value a at child-slot c = j_child mod s}`,
the same `trans` tensor built inside
`datasets/random_hierarchy_model.latent_prior`. Conditioning on `child = c_i`
normalizes the column `P(parent=u, child=c_i)` to `P(parent | child=c_i)` and
takes its entropy. Averaging the per-value leaks over `P(child)` recovers the
cell-level `H(parent|child)` (a cross-check). Rules are recovered offline via
the SAE checkpoint -> transformer checkpoint -> `resolve_rules` chain (with a
deterministic `seed_rules` fallback).

## Outputs (per run, tagged with the alpha in the filename)

Plots in `analysis_plots/`:
- `..._dec.png`: per-position weighted entropy `eta-bar` vs lambda, THREE
  curves: parent / level / unconstrained-split-checked (orange). The
  split-checked curve is the aggregate AFTER reassignment; the raw
  unconstrained-argmin curve is not drawn. It is always >= the unconstrained
  min; it usually sits below the parent curve but can exceed it when a
  reassigned feature lands on a parent cell other than the position's nominal
  one. The y-axis is the normalized weighted entropy `eta-bar` (LaTeX
  `\bar{\eta}`).
- `..._leveldist_unconstrained_dec.png` and `..._leveldist_reassigned_dec.png`:
  per-position share of the argmin target by RHM level vs lambda, before and
  after the split check (BOTH kept). Reassignment moves mass from level `l*` up
  to `l*-1`; curves sum to ~1.

CSVs in `analysis_files/`:
- `....csv`: per-position metrics, including `H_bar_splitcheck_<scheme>_norm`
  and `reassign_frac_<scheme>`.
- `..._leak_table.csv`: the value-specific structural leak per
  `(child cell, child value)`, with `P_child` and the raw nats.
- `..._target_dist.csv`: long-format level distributions, `variant` column
  `{unconstrained, reassigned}`.

## Running

```
python scripts/sae_sweep/feature_splitting_diag.py \
    --artifacts_dir /path/to/sweep/analysis_files \
    --alpha 0.5 --plot_schemes dec --report-notation \
    --out_csv  /path/to/sweep/analysis_files/feature_splitting_diag_alpha0p5.csv \
    --out_plot_prefix /path/to/sweep/analysis_plots/feature_splitting_diag_alpha0p5
```

Notes:
- Works for any L, for `all_tokens` and for `mean_pooled` sweeps (pooled: parent
  is the root class, no level-constrained curve).
- If eval artifacts sit in a nested subfolder (e.g.
  `analysis_files/new_analysis/`), point `--artifacts_dir` at that subfolder;
  the checkpoint lookup searches `sae_checkpoints/` up the directory tree, and
  `--out_csv` / `--out_plot_prefix` let you keep outputs at the sweep root.
- Per default the CSVs always carry all three weight schemes; `--plot_schemes`
  only selects which schemes get plotted.
