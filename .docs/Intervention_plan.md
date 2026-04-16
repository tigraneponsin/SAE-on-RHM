# Plan: Dead-token interventions and token-subset SAE metrics

## Context

For one trained RHM transformer (L=3, s=2, so 8 leaf tokens), the user observed
a "dead token" phenomenon visible through SAEs: for each adjacent leaf pair,
only one token's post-layer-0 residual stream is load-bearing. Evidence so far:
a `one_token` SAE on position 1 at layer 0 shows a clear lambda transition in
the whole-transformer reconstruction error curve, while the SAE on position 0
does not (since the downstream network already ignores that position). The
pattern appears to cascade: at layer 1 only positions {0,4} matter, at layer 2
only position 0 matters. This matches the RHM bottom-up composition
(layer k resolves level L-1-k).

Goal: build two tools to pin down this phenomenon without relying on SAEs as
the measurement instrument, plus a small set of corroborating probes.

## Tool 1: Direct token-residual intervention

Replace the post-block residual at chosen (layer, positions) with a
non-informative surrogate, and measure downstream classification. No SAE
involved.

Reuses the existing forward-hook pattern from
[scripts/sae_sweep/eval_sweep.py:167-190](scripts/sae_sweep/eval_sweep.py#L167-L190)
(`_make_sae_hook`), but the hook body becomes an ablation function.

### New file: `scripts/intervention/ablate_tokens.py`

CLI:
```
python scripts/intervention/ablate_tokens.py \
    --train_output /path/to/transformer.pt \
    --experiments /path/to/experiments.json \
    --eval_size 32768 --eval_seed 99999 --batch_size 256 \
    --outcsv /path/to/ablate_results.csv
```

`experiments.json` is a list of experiments. Each experiment has a name and a
list of interventions applied simultaneously in a single forward pass:
```json
[
  {"name": "baseline", "interventions": []},
  {"name": "L0_kill_{1,3,5,7}_mean",
   "interventions": [{"layer": 0, "positions": [1,3,5,7], "mode": "mean"}]},
  {"name": "L0_kill_{0,2,4,6}_mean",
   "interventions": [{"layer": 0, "positions": [0,2,4,6], "mode": "mean"}]},
  {"name": "L0_{1,3,5,7}+L1_{2,3,6,7}_mean",
   "interventions": [{"layer": 0, "positions": [1,3,5,7], "mode": "mean"},
                     {"layer": 1, "positions": [2,3,6,7], "mode": "mean"}]}
]
```

Supported `mode` values (all three requested):
- `mean`: replace `x[:, p, :]` with `E[x[:, p, :]]` where the expectation is
  taken over the eval set at that layer and position. Means are precomputed in
  a single pass (one capture hook per target layer) before running experiments
  and cached in a `{layer: Tensor[T, D]}` dict.
- `zero`: replace `x[:, p, :]` with 0.
- `resample`: replace `x[:, p, :]` with `x[perm, p, :]` where `perm` is a
  random permutation of the batch, applied independently per forward. Keeps
  activations in-distribution at that position while breaking input-specific
  content.

### Implementation sketch

```python
def make_ablation_hook(interventions_at_layer, pos_offset, mean_cache, layer_id):
    """interventions_at_layer: list of (positions: List[int], mode: str)."""
    def hook(_m, _i, output):
        out = output.clone()
        for positions, mode in interventions_at_layer:
            seq_positions = [pos_offset + p for p in positions]
            if mode == 'zero':
                out[:, seq_positions, :] = 0.0
            elif mode == 'mean':
                out[:, seq_positions, :] = mean_cache[layer_id][seq_positions].to(out)
            elif mode == 'resample':
                perm = torch.randperm(out.size(0), device=out.device)
                out[:, seq_positions, :] = output[perm][:, seq_positions, :]
        return out
    return hook
```

Reuse helpers:
- [scripts/common/sae_loading.py:load_transformer](scripts/common/sae_loading.py) for
  model + eval_loader + resolved cfg (it already builds a deterministic eval
  split matching `eval_sweep.py`).
- [scripts/sae_sweep/eval_sweep.py:144-164](scripts/sae_sweep/eval_sweep.py#L144-L164)
  `_eval_classification` for the per-experiment accuracy/CE computation.
- `has_cls = hasattr(model, 'cls_token')` to set `pos_offset = 1 if has_cls
  else 0` (the same check used in `_make_sae_hook`).

### Output

One CSV row per experiment:
`name, interventions_json, acc, cross_entropy, err, err_over_random, delta_err_vs_baseline`.

### On the "zero vs mean vs resample" question

Run all three in the same sweep. Expected reading:
- If a position is truly dead, all three ablations yield accuracy near the
  baseline (the residual was ignored anyway).
- If a position is load-bearing, mean and resample usually degrade less than
  zero (zero is off-distribution after LayerNorm), and resample measures
  content-specific info while mean measures position-mean-centered info.
  The gap between zero and mean tells you how much of the degradation is from
  LayerNorm weirdness vs genuine information loss.

## Tool 2: Token-subset lambda metrics (+ per-position breakdown)

Extend the existing `_activity_stats` pipeline so that, for `all_tokens` SAEs,
metrics can be restricted to a user-specified subset of leaf positions, and a
per-position breakdown is also emitted.

### Changes in `scripts/sae_sweep/eval_sweep.py`

1. Modify `_activity_stats` at
   [scripts/sae_sweep/eval_sweep.py:65-141](scripts/sae_sweep/eval_sweep.py#L65-L141):
   - Add parameters `token_subset: Optional[Sequence[int]] = None` and
     `per_position: bool = True`.
   - For `mode == 'all_tokens'`: after slicing away CLS (same as current
     line 96), reshape activations as `[B, T_real, D]`, not
     `[B*T_real, D]`, until after per-position bookkeeping.
   - Maintain per-position accumulators:
     - `active_per_pos: Tensor[T_real]` (sum of active features)
     - `ever_active_per_pos: Tensor[T_real, latent_dim]` (bool)
     - `feature_sum_per_pos: Tensor[T_real, latent_dim]` (float64)
     - `count_per_pos: Tensor[T_real]`
   - At the end, build:
     - Subset-aggregated versions of the existing metrics
       (`mean_active`, `mean_active_ratio`, `ipr`, `dead_features`,
       `ever_active`, `mean_active_above_{1,10}pct`) computed over only the
       selected positions (or all positions if `token_subset is None`).
     - Per-position versions of the same metrics returned as 1-D / 2-D
       tensors of length `T_real`.

2. Return the enriched dict (back-compatible: existing keys unchanged when
   `token_subset is None`; new keys prefixed `per_position_` and `subset_`).

3. Plumbing in `main`
   ([scripts/sae_sweep/eval_sweep.py:201-374](scripts/sae_sweep/eval_sweep.py#L201-L374)):
   - New flags: `--subset_positions 0,2,4,6` and `--per_position_csv
     /path/to/...`.
   - When `--subset_positions` is given, pass through to
     `_activity_stats`; emit both the existing aggregate row and a
     `subset_*` row into the main CSV.
   - When `--per_position_csv` is given, write a long-format CSV with
     columns `ckpt, layer, lambda_l1, position, mean_active, ever_active,
     ipr, active_above_1pct, active_above_10pct,
     mean_active_above_1pct, mean_active_above_10pct`.

4. `one_token` and `cls_token` modes: `token_subset` is ignored (no effect),
   since those SAEs already live at a single position. Per-position breakdown
   collapses to a single row in that case.

### Why decoder-weighted activations stay

Keep the current `features = z * dec_norms.unsqueeze(0)` weighting from
[line 102](scripts/sae_sweep/eval_sweep.py#L102) so the per-position metrics
are directly comparable to the existing aggregate metrics.

## Recommended corroborating probes (optional, small)

These are small additions that triangulate the "dead token" story from other
angles. I recommend doing at least the first two before drawing conclusions
from the intervention curves alone.

### A. Residual-norm trajectories per position

In `ablate_tokens.py`, during the mean-precomputation pass we already see
`x[layer_id]` for each layer. At no extra forward-pass cost, accumulate
`mean_norm[layer_id, position] = E[ ||x[:, position, :]|| ]`. Emit as a
second CSV (`*_norms.csv`). If dead positions have vanishing norm at later
layers, that is visible at a glance and independent of SAE / probe choices.

### B. Probe accuracy per (layer, position, level)

Use the existing
[linear_probe.py:ancestor_labels](linear_probe.py#L106-L133) and
[linear_probe.py:collect_probe_data](linear_probe.py#L140) to train a
per-position probe at each layer against each RHM level. Expected outcome if
the hypothesis holds:
- Layer 0, position 1: chance accuracy at level 2 (the relevant ancestor for
  that pair), since the token is dead.
- Layer 0, position 0: high accuracy at level 2.
- Layer 1, positions {1,2,3}: chance accuracy at level 1.
This either confirms or falsifies the hypothesis without any ablation or SAE.

### C. Progressive dead-set curve (uses Tool 1)

For each layer, run a sweep of ablations that zero/mean out increasingly
large candidate-dead sets, plus the dual sweep on candidate-live sets.
The smallest set whose ablation breaks classification == the minimal live
set. This is just a pre-written `experiments.json` invoked through Tool 1.

## Files to create / modify

- **Create** [scripts/intervention/__init__.py](scripts/intervention/__init__.py) (empty).
- **Create** [scripts/intervention/ablate_tokens.py](scripts/intervention/ablate_tokens.py)
  (Tool 1 + residual-norm trajectory as side product).
- **Modify** [scripts/sae_sweep/eval_sweep.py](scripts/sae_sweep/eval_sweep.py)
  (Tool 2, changes at lines 65-141 and 201-374 as described above).
- **Optional**: a small script
  `scripts/intervention/probe_per_position.py` that wraps the existing
  `collect_probe_data` / probe-training code for corroboration B.
- **No changes** to `models/transformer.py` or `train_sae.py`. The existing
  hook points are sufficient.

## Verification

1. Sanity checks for Tool 1:
   - `experiments.json` with an empty `interventions` list must reproduce the
     baseline accuracy reported by `eval_sweep.py`.
   - Mean-ablating ALL leaf positions at layer `L-1` (the last block) should
     destroy accuracy down to ~chance.
   - Zero-ablating an empty position list must match baseline exactly.
2. Sanity checks for Tool 2:
   - `_activity_stats` with `token_subset=None` and `per_position=False` must
     produce numerically identical values to the current implementation on a
     held-out checkpoint (regression test; run on one existing sweep CSV and
     diff).
   - `token_subset=list(range(T_real))` must equal the unrestricted aggregate.
   - Sum of per-position `active_counts` divided by `T_real * N` must equal
     the aggregate `mean_active_ratio`.
3. End-to-end on the identified trained instance:
   - Reproduce the reported pattern: for layer 0, the ablation curve for
     positions `{1,3,5,7}` (candidate dead) should be flat; for `{0,2,4,6}`
     (candidate live) it should collapse.
   - For layer 1, ablating `{2,3,6,7}` should leave accuracy near baseline;
     ablating `{0,4}` should break it.
   - Cumulative `L0_{1,3,5,7}+L1_{2,3,6,7}` should still preserve accuracy.
4. Per-position Tool 2 output on the `all_tokens` SAE sweep should show the
   candidate-dead positions having much lower `mean_active` at each lambda
   compared to candidate-live positions, matching the `one_token` SAE story.
