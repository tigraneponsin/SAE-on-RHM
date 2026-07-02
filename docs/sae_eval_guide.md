# SAE streaming eval: library + CLI guide

The `scripts/sae_eval/` package handles SAE post-hoc evaluation on the Random
Hierarchy Model. It replaces the old
`scripts/sae_direct_analysis/analyze_sae.py` and
`scripts/sae_sweep/eval_sweep.py`.

## Package layout

- `scripts/sae_eval/streaming.py` -- library. No `__main__`. Import from it.
- `scripts/sae_eval/run.py` -- the only CLI driver. Scanning, orchestration,
  saving. All math lives in the library.
- `scripts/sae_eval/test_sanity.py` -- four acceptance checks. Run once
  before trusting any number:

  ```bash
  python scripts/sae_eval/test_sanity.py
  ```

  Exit code is non-zero on failure.

## Library surface (streaming.py)

### `StreamingFlags` -- what gets computed

A dataclass with six booleans, all default `False`. Each gates one block.

| Flag | What it produces | Cost |
|------|------------------|------|
| `scalar_aggregates` | `dead_features`, `dead_ratio`, `mean_active`, `ipr`, `active_above_{1,10}pct`, `mean_active_above_{1,10}pct` | cheap |
| `per_position` | `per_position_*` versions of the above, `[P]`-shaped | cheap |
| `per_feature` | `baseline_mean/std [P,F]`, `firing_count/rate [P,F]`, `mean_cofire [P,F]`, `L0_mean [P]`, `decoder_norms [F]` | medium |
| `conditional` | `conditional_mean [T,P,F]`, `delta_mean`, `z_score`, `conditional_count`, `targets`, `index_layout` | heavy (`[T,P,F]` tensor) |
| `joint_fire_and_entropy` | `joint_fire_count [T,P,F]`, `H_per_feature [num_groups,P,F]`, theoretical + empirical `P(Z)` / `H(Z)`, six `H_bar_*[P]` aggregates | heavy; requires `per_feature` |
| `classification_impact` | `baseline_err`, `sae_err`, `norm_err`, `baseline_ce`, `sae_ce` -- separate second streaming pass | roughly 1x the cost of the main pass |

**Dependency rules** (enforced by `.validate()`):
`scalar_aggregates`, `conditional`, and `joint_fire_and_entropy` all require
`per_feature`.

### `stream_sae_eval(...)` -- the main entry point

```python
from scripts.sae_eval.streaming import stream_sae_eval, StreamingFlags

artifact = stream_sae_eval(
    model=frozen_transformer,
    sae=trained_sae,
    trees=rhm_trees,                    # dict[level] -> LongTensor
    layer_id=0,
    mode='all_tokens',                  # or 'one_token', 'cls_token'
    has_cls=hasattr(model, 'cls_token'),
    token_idx=0,                        # used only for one_token
    act_scale=1.0,                      # from the SAE checkpoint
    batch_size=256,
    device='cuda',
    flags=StreamingFlags(per_feature=True, joint_fire_and_entropy=True),
    rhm={'n': 8, 'v': 8, 'm': 4, 's': 2, 'L': 4},  # required with entropy
    rules=rules_dict,                                # required with entropy
)
```

One forward pass over all `trees[L]` batches. Returns a dict; absent keys
mean "you didn't ask for that block."

### `stream_classification_impact(...)`

Separate from the main pass because it uses a different hook (replaces
residuals with the SAE reconstruction, reads classifier logits). Same
signature minus the flags; always returns the five `*_err` / `*_ce`
scalars.

### Pure helpers (useful for custom analyses)

- `shannon_entropy(p, dim=-1)` -- nats, `xlogy` convention so `0 log 0 = 0`.
- `per_feature_entropy(joint_fire[..., V, F], marginal_fire[..., F])` --
  returns `H_i` per feature, NaN on dead features.
- `weighted_aggregate(H[..., F], weights[..., F])` -- NaN-safe weighted mean.
- `enumerate_targets(trees)` -- build `(level, position, value)` target list.
- `dedupe_trees(trees)` -- drop duplicate leaf sequences, keeping levels
  internally consistent.
- `select_activation_tokens(act, mode, has_cls, token_idx)` -- port of the
  token-mode slicer used in training.

## The CLI (run.py)

### Anatomy of a command

```bash
python scripts/sae_eval/run.py \
    --sweep_dir /path/to/sweep/ \       # OR --ckpt /path/to/one.pt
    --out_dir   /path/to/artifacts/ \   # where .sae_eval.pt files land
    [--eval_size 32768] [--eval_seed 99999] [--batch_size 512] \
    [--dedupe] [--device cuda] \
    [--with-<block> ...] [--with-all] \
    [--outcsv /sweep/results.csv] \
    [--per_position_csv /sweep/per_position.csv]
```

### Nothing is computed unless you ask

Every block defaults to OFF. Either enable individual blocks or pass
`--with-all`:

- `--with-scalar-aggregates`  / `--no-scalar-aggregates`
- `--with-per-position`       / `--no-per-position`
- `--with-per-feature`        / `--no-per-feature`
- `--with-conditional`        / `--no-conditional`
- `--with-entropy`            / `--no-entropy`
- `--with-classification-impact` / `--no-classification-impact`
- `--with-all`                -- shortcut: everything on

Guardrails in the CLI:

- `--outcsv` requires `--with-scalar-aggregates` (otherwise there is
  nothing to summarize).
- `--per_position_csv` requires `--with-per-position`.
- Entropy columns in the CSV appear only when `--with-entropy` is on.

### Common recipes

**Replacement for the old `analyze_sae.py`** (rich per-feature artifacts,
no CSV):

```bash
python scripts/sae_eval/run.py \
    --sweep_dir /path/to/sweep/ \
    --out_dir   /path/to/sweep/sae_eval_artifacts/ \
    --with-per-feature --with-conditional --with-entropy
```

**Replacement for the old `eval_sweep.py`** (sweep CSV for comparison
plots):

```bash
python scripts/sae_eval/run.py \
    --sweep_dir /path/to/sweep/ \
    --out_dir   /path/to/sweep/sae_eval_artifacts/ \
    --outcsv    /path/to/sweep/eval_results.csv \
    --per_position_csv /path/to/sweep/per_position.csv \
    --with-all
```

**Light run** (basic activity stats, no entropy):

```bash
python scripts/sae_eval/run.py \
    --ckpt /one.pt --out_dir /out/ \
    --with-per-feature --with-scalar-aggregates --with-per-position
```

**Classification impact only** (fastest way to get reconstruction error):

```bash
python scripts/sae_eval/run.py \
    --ckpt /one.pt --out_dir /out/ \
    --with-classification-impact
```

The slurm wrapper for eval + sweep plots:

- `slurm/sae/run_analysis_and_plots.sh` runs the streaming eval (`--with-all`)
  and then the sweep plots (`plot_lambda_metrics.py`, `entropy_diag.py`). It is
  the dependent step submitted by `run_full_sweep.sh`. (Eval-only variants
  `run_analysis.sh` / `run_eval.sh` are kept locally but untracked.)

## What lands on disk

### `<ckpt_stem>.sae_eval.pt`

One file per input SAE checkpoint. Load with
`torch.load(path, weights_only=False)`. Expect a dict with these groups
(keys present only if their flag was enabled):

```python
art = torch.load('.../sae_layer0.sae_eval.pt', weights_only=False)

# Provenance (always present):
art['ckpt_path'], art['layer_id'], art['mode'], art['sae_token_idx']
art['act_scale'], art['rhm']              # {v, n, m, s, L}
art['rules_source']                       # 'artifact' or 'seed_rules_resampled'
art['token_positions']                    # LongTensor [P]
art['num_samples_used']
art['lambda_l1'], art['lr'], art['steps']  # from the training checkpoint

# Per-feature (per_feature flag):
art['baseline_mean']        # [P, F]  E[f_i]
art['firing_count']         # [P, F]  long
art['firing_rate']          # [P, F]
art['decoder_norms']        # [F]     ||W_dec[:, i]||
art['mean_cofire']          # [P, F]  E[L0_p | f_i > 0]

# Conditional (conditional flag):
art['targets']              # list of {level, position, value, count}
art['index_layout']         # groups, for indexing conditional_mean
art['conditional_mean']     # [T, P, F]
art['delta_mean']           # [T, P, F]  conditional_mean - baseline_mean

# Entropy (joint_fire_and_entropy flag):
art['joint_fire_count']     # [T, P, F]  #{x: Z=v AND f_i > 0}
art['H_per_feature']        # [num_groups, P, F]  H(Z_group | f_i > 0 at p)
art['prior_theoretical']    # dict (level, pos) -> [V] float
art['prior_empirical']      # dict (level, pos) -> [V] float
art['H_theoretical']        # dict (level, pos) -> float (nats)
art['H_empirical']          # dict (level, pos) -> float
art['H_bar_fire'], art['H_bar_raw'], art['H_bar_dec']                 # [P] nats
art['H_bar_fire_norm'], art['H_bar_raw_norm'], art['H_bar_dec_norm']  # [P] in [0, 1]

# Classification (classification_impact flag):
art['baseline_err'], art['sae_err'], art['norm_err']
art['baseline_ce'], art['sae_ce']
```

### `--outcsv` main CSV

One row per SAE checkpoint. Columns follow the flags you enabled:

- **Always:** `ckpt`, `layer`, `mode`, `token_idx`, `latent_dim`,
  `lambda_l1`, `lr`, `steps`, `batch_size`,
  `train_{total,recon,sparse}_loss`
- **With `scalar_aggregates`:** `dead_features`, `dead_ratio`,
  `mean_active`, `ipr`, `active_above_{1,10}pct`,
  `mean_active_above_{1,10}pct`
- **With `classification_impact`:** `baseline_err`, `sae_err`, `norm_err`,
  `baseline_ce`, `sae_ce`
- **With `entropy`:** `H_bar_{fire,raw,dec}_mean` and
  `H_bar_{fire,raw,dec}_norm_mean` (means over the `[P]` per-position
  aggregates)

### `--per_position_csv` long-format CSV

One row per `(ckpt, leaf_position)`. Same scalar columns as above but
per-position, plus per-position entropy aggregates when entropy is on.
Feed straight into pandas for subset analyses (the old
`--subset_positions` trick becomes a `groupby('leaf_position')` in a
notebook).

## Typical workflow from zero

1. Sanity-check once: `python scripts/sae_eval/test_sanity.py`. Must print
   `All 4 checks passed.`
2. Run the CLI on a sweep with `--with-all` (or enable just the blocks you
   need).
3. For deep dives on one checkpoint, open `*.sae_eval.pt` in a notebook,
   pick a `(layer, slot)` group from `index_layout`, slice
   `H_per_feature[group_idx, :, :]` for per-feature entropy.
4. For sweep comparison, join the main CSV on `ckpt`. The plot scripts in
   `scripts/sae_direct_analysis/` and `scripts/sae_sweep/` still work;
   they just consume `.sae_eval.pt` now.

## Entropy-based specificity: what the numbers mean

For a feature `f_i` hooked to layer `k` at SAE position `p`, the default
("matched") RHM latent to compare against is `Z_{l,j}` with `l = L - 1 - k` and
`j = p // s^(1+k)`. This follows the layer-to-level working hypothesis (see
`docs/project_pipelines.md` section 1.3); it is a reference, not an assumption.
The diagnostics in `scripts/sae_sweep/entropy_diag.py` also score each feature
against other same-level cells and against every cell in the tree, so a feature
that is more selective for a different latent than its matched one is caught
rather than hidden. Per-feature conditional entropy against a chosen latent:

```
H_i = H(Z_{l,j} | F_i > 0) = -sum_z P(Z = z | F_i > 0) log P(Z = z | F_i > 0)
```

Estimated from joint counts on the eval set:

```
p_i(z) = joint_fire_count[t(z), p, i] / firing_count[p, i]
H_i    = -sum_z p_i(z) log p_i(z)
```

Three aggregates per `(layer, SAE position)` differ only in how features
are weighted:

```
H_bar_fire = sum_i P(F_i > 0) * H_i / sum_i P(F_i > 0)
H_bar_raw  = sum_i E[z_i]     * H_i / sum_i E[z_i]
H_bar_dec  = sum_i E[f_i]     * H_i / sum_i E[f_i]
```

Normalized versions divide by the theoretical `H(Z_{l,j})` computed from
the RHM rules (via `latent_prior` / `latent_entropy` in
`datasets/random_hierarchy_model.py`), so the score falls in `[0, 1]`: closer to
0 indicates a feature more selective for `Z`, closer to 1 a feature whose firing
carries little information about `Z`.
