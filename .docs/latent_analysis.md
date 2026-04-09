# Plan: Direct SAE Feature-Latent Analysis Script

## Context

The user wants to **directly** inspect what a trained SAE has learned by asking,
for each SAE feature `f` and each ground truth RHM latent `(level, position, value)`:

> How much more (or less) does feature `f` fire when the latent takes this value,
> compared to its baseline firing rate?

Concretely, we want two quantities per SAE feature `f` (at a given transformer
layer `k` and input token position `p`):

1. **Baseline**: `E[z_f]` over a large sample of RHM data.
2. **Conditional**: `E[z_f | trees[level][sample, j] = value]` for every
   `(level, j, value)` triple we care about.

If the SAE is clean and the hierarchy hypothesis holds, a feature at layer `k`,
token `p` should light up selectively for the level-`(L-1-k)` ancestor of `p`
taking a particular value, i.e., `trees[L-1-k][:, p // s^(1+k)]`.

The existing [latent_analysis.py](latent_analysis.py) already implements this
idea (it has `collect_weighted_sae_activations`, `LatentRowTable`,
`build_full_grid_target_masks`, `compute_observational_metrics`). But:

- It lives at the repo root with a library-only API.
- It is only used from one notebook
  ([notebooks/test_train_sae_from_meantransformer.ipynb](notebooks/test_train_sae_from_meantransformer.ipynb)),
  not from any CLI / sweep script.
- It does not use the `act_scale` normalization that `train_sae.py` and
  `eval_sweep.py` apply (so conditional means would be off by a factor).
- It stores full [num_rows, latent_dim] activation tables in memory, which is
  fine for small sweeps but scales poorly if we want to evaluate on every
  reachable RHM tree.
- There is no CLI entry point, no artifact layout, and no visualization.

The user calls it "obsolete" — the right move is to **replace** it with a
targeted CLI script that follows the existing `scripts/sae_sweep/eval_sweep.py`
pattern, computes baseline and conditional means in a single streaming pass,
saves a compact artifact, and provides a companion plotting script.

## Design decisions (resolved)

1. **Target scope**: **full (level, position, value) grid**. For each SAE
   feature, compute conditionals across every level, every position at that
   level, and every observed value. Cheap in practice (see memory sanity
   check below) and exposes cross-level / cross-position selectivity.

2. **Eval data**: large Monte Carlo split (default `--eval_size 32768`,
   `--eval_seed 99999`) plus an **optional `--dedupe` flag**. With `--dedupe`,
   the script drops duplicate trees in memory before running the forward pass
   so the reported expectations are exact over the observed unique trees.
   Dedupe is implemented as `torch.unique(trees[L], dim=0, return_inverse=True)`
   followed by rebuilding the other `trees[l]` rows from the kept indices.

3. **Token positions**: always respect the SAE's training mode. For
   `one_token` analyze only `sae_token_idx`; for `cls_token` analyze only
   position 0; for `all_tokens` analyze every real token and store
   per-position tables so position-specific selectivity is visible.

4. **Feature definition**: a "feature activation" throughout this script is
   the **decoder-weighted** value `f = z * ||W_dec[:, f]||`, matching the
   convention used by [scripts/sae_sweep/eval_sweep.py](scripts/sae_sweep/eval_sweep.py)
   (`_activity_stats`) and the old `latent_analysis.py` (`collect_weighted_sae_activations`).
   All accumulators (`sum_z`, `sum_z2`, `sum_z_cond`) are populated from the
   weighted value. The artifact's `baseline_mean`, `baseline_std`,
   `conditional_mean`, `delta_mean`, and `z_score` are all decoder-weighted.
   `decoder_norms` is still saved in the artifact for reference, but plots do
   NOT re-weight the data.

5. **RHM rule consistency (critical)**: the analysis data MUST come from the
   exact same RHM rules the transformer and the SAE were trained on.
   [scripts/sae_sweep/eval_sweep.py](scripts/sae_sweep/eval_sweep.py) already
   implements this invariant in `_resolve_rules()` (lines 47-91) and the
   inline `sae_rules_source` check in `main` (lines 423-434). The new script
   inherits both mechanisms by importing them from the shared
   [scripts/common/sae_loading.py](scripts/common/sae_loading.py) module:

   - `_resolve_rules(blob)` prefers `blob['output']['rules']` (the exact
     rules object saved alongside the trained transformer). If that field
     is missing, it regenerates deterministically from
     `cfg.seed_rules` and the RHM structural parameters via
     `sample_rules(v, n, m, s, L, seed=seed_rules)`. Because
     [train_sae.py](train_sae.py) uses the same priority when picking the
     rules to train the SAE on, both paths produce identical rules as long
     as the saved `cfg` has not changed.
   - The SAE checkpoint stores `sae_dataset_split.rules_source` (either
     `'artifact'` or `'seed_rules_resampled'`). Before running any forward
     pass, the script compares this string to the resolved rules source
     and **skips the checkpoint with an error** on mismatch (same behavior
     as `eval_sweep.py`).
   - The plan adds one extra safeguard: when `_resolve_rules` falls back to
     `'seed_rules_resampled'`, the script logs a warning naming the
     checkpoint and asks the user to rerun training with `--save_models`
     (which writes `output.rules` directly). This turns a silent assumption
     into a visible one.

6. **Loader helpers**: `_resolve_rules`, `_load_transformer`, `_load_sae` are
   moved out of [scripts/sae_sweep/eval_sweep.py](scripts/sae_sweep/eval_sweep.py)
   into a new shared module [scripts/common/sae_loading.py](scripts/common/sae_loading.py).
   Both `eval_sweep.py` and the new `analyze_sae.py` import from there.
   This is a zero-behavior-change refactor — I will run `eval_sweep.py` on an
   existing sweep directory before and after to confirm identical output.

7. **Disposition of obsolete code**: delete
   [latent_analysis.py](latent_analysis.py). The notebook
   [notebooks/test_train_sae_from_meantransformer.ipynb](notebooks/test_train_sae_from_meantransformer.ipynb)
   is the only caller and will break on its import cell; it stays on disk
   untouched and can be migrated later if still useful.

8. **Visualization format**:
   - **Main heatmap**: per SAE checkpoint and per analyzed token position,
     a matrix where rows = SAE features (sorted by max |z_score| across
     targets), columns = `(level, position, value)` targets, cell =
     `z_score[target, pos, f] = (E[z_f|T] - E[z_f]) / (std(z_f) + eps)`.
     Diverging colormap centered at 0, vertical lines separating levels.
   - **Expected-latent bar chart**: for each analyzed token position `p`,
     compute the "expected" target `level = L-1-k`, `j = p // s^(1+k)`;
     show the top-K features ranked by their `max_v |E[z_f|target=v]|` at
     that target, as a grouped bar chart (one bar per value).
   - **Selectivity summary scatter**: per feature, scatter of
     `baseline_mean` vs `max |z_score|` colored by the level of the
     argmax target. Lets the user see at a glance whether active features
     are also selective.

## Proposed implementation

### New files

- [scripts/sae_direct_analysis/analyze_sae.py](scripts/sae_direct_analysis/analyze_sae.py) --
  CLI entry point. Loads an SAE checkpoint (or a sweep directory), runs the
  frozen transformer + SAE over RHM eval data, computes baseline and
  conditional SAE feature statistics per `(level, position, value)` target,
  writes an artifact file.
- [scripts/sae_direct_analysis/plot_sae_analysis.py](scripts/sae_direct_analysis/plot_sae_analysis.py) --
  Loads the artifact and produces heatmaps + top-feature bar charts.
- [scripts/sae_direct_analysis/__init__.py](scripts/sae_direct_analysis/__init__.py) --
  empty package init.

I chose a new subdirectory `sae_direct_analysis/` rather than dropping into
`sae_sweep/` because this is a different kind of evaluation (feature-level
interpretability, not sweep-level metrics) and separating them keeps
`sae_sweep/` focused on hyperparameter comparison.

### Files to reuse (do not copy, import)

- `sample_rules`, `sample_trees` from
  [datasets/random_hierarchy_model.py](datasets/random_hierarchy_model.py) --
  eval data generation, seed-consistent with training.
- `_resolve_rules`, `_load_transformer`, `_load_sae` from
  [scripts/sae_sweep/eval_sweep.py](scripts/sae_sweep/eval_sweep.py) --
  I will **refactor these three helpers into a shared module**
  (`scripts/common/sae_loading.py`) and import from both `eval_sweep.py` and
  `analyze_sae.py`. This is the simplest way to avoid drift and matches the
  user's preference for editing over duplicating.
- `init.init_model`, `init.init_data` for consistent data pipeline.
- `models.SparseAutoencoder` (used by the loader helpers).

### Files to delete

- [latent_analysis.py](latent_analysis.py) -- obsolete, per user. The only
  caller is [notebooks/test_train_sae_from_meantransformer.ipynb](notebooks/test_train_sae_from_meantransformer.ipynb),
  which is left untouched (its import cell will break; the user accepted
  that trade-off).

### Core algorithm (single streaming pass)

Given a loaded `(model, sae, act_scale)` for layer `layer_id` and eval trees
`trees` of `num_samples` RHM samples — where the rules used to build `trees`
are resolved via `_resolve_rules()` and verified against
`sae_dataset_split.rules_source` **before** any forward pass:

```
# Accumulators (on device for speed, cast to float64 to avoid precision loss)
# F = sae.latent_dim, P = number of token positions analyzed
# Feature values below are ALWAYS decoder-weighted: f = z * ||W_dec[:, f]||
sum_f  [P, F]                         # running sum of weighted features
sum_f2 [P, F]                         # running sum of weighted features squared
count_total [P]                       # sample count per token position

# For each (level, j, value) target we need sum_f and count conditioned on
# trees[level][:, j] == value. To keep memory bounded, we pre-enumerate the
# targets: for level in 0..L, for j in range(s^level), for value in
# unique(trees[level][:, j]). For each target store:
sum_f_cond  [T, P, F]                 # conditional sum of weighted features
count_cond  [T, P]                    # conditional count
```

Where `T` is the total number of `(level, j, value)` triples. Naively this
is up to `sum_l s^l * v` but we only keep triples whose `value` is actually
observed, so in practice it's exactly
`sum_l (num_unique_values_at_level_l * s^l)`.

**Memory sanity check**: for `v=8, s=2, L=3, F=256, P=8` and dtype float64:
`T ~ 8 * (1 + 2 + 4 + 8) = 120`, so `sum_z_cond` is
`120 * 8 * 256 * 8 bytes = ~2 MB`. Even `L=4, F=1024, P=16` is ~40 MB. Fine.

Streaming loop:

```
dec_norms = sae.decoder_feature_norms()         # [F], computed once

for batch in eval_loader:           # batch inputs have shape [B, s^L]
    with hook on model.blocks[layer_id]:
        model(batch_inputs)
    act = captured_buffer.pop()     # [B, seq_len, emb_dim]
    act = select_positions(act, mode, has_cls, sae_token_idx)   # [B, P, emb_dim]
    act = act * act_scale           # same scaling train_sae.py uses
    flat = act.reshape(B * P, emb_dim)
    _, z = sae(flat)                # [B*P, F]    raw hidden activation
    f = z * dec_norms.unsqueeze(0)  # [B*P, F]    DECODER-WEIGHTED feature
    f = f.view(B, P, F)

    sum_f  += f.sum(dim=0)          # [P, F]
    sum_f2 += (f * f).sum(dim=0)
    count_total += B

    # Conditional accumulation: for each target, mask the batch and add
    batch_trees = {l: trees_l[batch_indices] for l in ...}   # [B, s^l]
    for target_id, (level, j, value) in enumerate(target_list):
        mask = (batch_trees[level][:, j] == value)           # [B]
        if not mask.any(): continue
        sel = f[mask]                                        # [M, P, F]
        sum_f_cond[target_id]  += sel.sum(dim=0)             # [P, F]
        count_cond[target_id]  += int(mask.sum())
```

**Important detail**: the transformer `eval_loader` is built via
`init.init_data(trees[L], trees[0], cfg)` which shuffles at batch time unless
we override. I will build a non-shuffled loader so that a running batch
index maps cleanly back to `trees[level]` rows -- OR, simpler, carry the
batch row indices through the loader by wrapping it with `enumerate` on a
fixed-order `TensorDataset` (`shuffle=False`). `eval_sweep.py`'s loader is
already built with defaults; I will build a dedicated one with `shuffle=False`
for this script.

At the end (all quantities refer to the decoder-weighted feature
`f = z * ||W_dec[:, f]||`):

```
mean_f       = sum_f / count_total.unsqueeze(-1)
var_f        = sum_f2 / count_total.unsqueeze(-1) - mean_f ** 2
std_f        = var_f.clamp_min(0).sqrt()
mean_f_cond  = sum_f_cond / count_cond.unsqueeze(-1).unsqueeze(-1)
delta        = mean_f_cond - mean_f                       # [T, P, F]
zscore       = delta / (std_f + eps)                      # [T, P, F]
```

No second pass is needed: weighting is baked into the accumulators.

### Artifact format

`<output_dir>/<ckpt_basename>.feature_latent.pt`:

```
{
  'ckpt_path': str,
  'layer_id': int,
  'mode': str,                                    # all_tokens / one_token / cls_token
  'sae_token_idx': int,
  'act_scale': float,
  'latent_dim': int,
  'embedding_dim': int,
  'eval_size': int,
  'eval_seed': int,
  'num_samples_used': int,
  'token_positions': LongTensor [P],              # 0-based real-token indices analyzed
  'rhm': {'v': int, 'n': int, 'm': int, 's': int, 'L': int},
  'targets': list of {'level': int, 'position': int, 'value': int, 'count': int},
  'rules_source':     str,                        # 'artifact' or 'seed_rules_resampled'
  'sae_rules_source': str,                        # from SAE ckpt; must match rules_source
  # All feature-level quantities below are DECODER-WEIGHTED:
  # feature f at position p has value z_f(p) * ||W_dec[:, f]||
  'baseline_mean':    FloatTensor [P, F],
  'baseline_std':     FloatTensor [P, F],
  'conditional_mean': FloatTensor [T, P, F],
  'conditional_count':LongTensor  [T, P],         # per-target, per-pos sample count
  'delta_mean':       FloatTensor [T, P, F],      # = conditional_mean - baseline_mean
  'z_score':          FloatTensor [T, P, F],      # = delta_mean / (baseline_std + eps)
  'decoder_norms':    FloatTensor [F],            # stored for reference; already baked in
}
```

### CLI shape

```
python scripts/sae_direct_analysis/analyze_sae.py \
    --sweep_dir /path/to/sweep/    # or --ckpt /path/to/one.pt
    --eval_size 32768 \
    --eval_seed 99999 \
    --batch_size 512 \
    --device cuda \
    --out_dir /path/to/analysis/out/
```

Same grouping logic as `eval_sweep.py`: group checkpoints by transformer
source so each transformer is loaded once. Skip checkpoints whose
`sae_rules_source` disagrees with the resolved rules (same safety check
`eval_sweep.py` already enforces).

### Visualization (`plot_sae_analysis.py`)

Loads one `.feature_latent.pt` artifact and produces:

1. **Main heatmap** (`heatmap_<ckpt>_pos<p>.png`): one per token position `p`
   analyzed. X axis = target index grouped by `(level, position)` then sorted
   by `value`, Y axis = SAE features sorted by their single max
   `|z_score|` across targets. Cell = `z_score[target, p, f]`. Diverging
   colormap centered at 0. Vertical lines separating levels.

2. **Expected-latent bar chart** (`expected_<ckpt>.png`): for the SAE's
   layer `k` and each position `p`, compute the "expected" target
   `level = L - 1 - k`, `j = p // s^(1+k)`. For each feature `f`, bar chart of
   `mean_z_cond[all values of that target]` at that feature, for top-K
   features ranked by selectivity to that target.

3. **Activity/selectivity summary** (`summary_<ckpt>.png`): scatter of
   `baseline_mean` vs `max |z_score|` per feature, colored by
   which level the argmax target lives at. Lets the user see at a glance
   whether "active" features are "selective".

Plotting uses matplotlib with Agg backend (same convention as
`scripts/sae_sweep/plot_*.py`).

## Critical files and their roles

| File | What I will do |
|------|----------------|
| [scripts/sae_direct_analysis/analyze_sae.py](scripts/sae_direct_analysis/analyze_sae.py) | NEW. Main CLI: streams eval data, accumulates baseline + conditional means, saves artifact |
| [scripts/sae_direct_analysis/plot_sae_analysis.py](scripts/sae_direct_analysis/plot_sae_analysis.py) | NEW. Loads artifact, renders heatmap + bar charts + summary |
| [scripts/sae_direct_analysis/__init__.py](scripts/sae_direct_analysis/__init__.py) | NEW. Empty |
| [scripts/common/__init__.py](scripts/common/__init__.py) | NEW. Empty package init |
| [scripts/common/sae_loading.py](scripts/common/sae_loading.py) | NEW. Houses `resolve_rules`, `load_transformer`, `load_sae` pulled out of `eval_sweep.py` (renamed to drop the underscore since they are now public) |
| [scripts/sae_sweep/eval_sweep.py](scripts/sae_sweep/eval_sweep.py) | EDIT. Replace local helpers with imports from `scripts/common/sae_loading.py`. No behavior change; verified by running before / after on an existing sweep |
| [latent_analysis.py](latent_analysis.py) | DELETE. The sole caller is a notebook; notebook stays untouched |

## Verification plan

1. **Unit-level sanity test** (can be a small `__main__` block or a pytest):
   generate a tiny `v=4, n=4, m=2, s=2, L=2` RHM, pick 1000 random samples,
   build a deterministic "fake SAE" whose decoder-weighted feature `f` **is**
   a known indicator of `trees[1][:, 0]`. Run the script, assert that for
   the matching target the `z_score` and `delta_mean` are large while all
   other targets are near zero. This catches indexing bugs in the accumulator
   and a sign / weighting bug if decoder norms were applied at the wrong step.

2. **End-to-end on an existing SAE** from a sweep directory on disk. Pick
   one SAE with known good `norm_err` from `eval_sweep.py` output and one
   with bad `norm_err`:
   - Good SAE should show strong selectivity (big `max |z_score|`) for
     targets at the hypothesized level.
   - Bad / random SAE should show weak / diffuse selectivity.
   - Baseline accumulator should match
     `feature_mean_activations` saved by `eval_sweep.py` to within 1e-4
     when decoder-weighted (cross-check: we are computing the same
     thing in two scripts, they must agree).

3. **Visualization spot-check**: render the heatmap for the good SAE and
   eyeball that the block-diagonal structure (one strongly lit feature
   per value) is visible.

4. **Numerical stability**: run with `--eval_size 256` (small) and
   `--eval_size 65536` (large) and confirm baseline means are stable;
   conditional means should converge but may wiggle on small sizes.

5. **act_scale round-trip**: run on a checkpoint where `act_scale != 1.0`
   and one where `act_scale == 1.0`; confirm that the feature activity
   magnitudes match `eval_sweep.py`'s `feature_mean_activations` for the
   same checkpoint.
