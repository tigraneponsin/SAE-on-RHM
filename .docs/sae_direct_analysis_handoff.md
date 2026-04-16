# SAE Direct Analysis Plotting Handoff

## Scope
Refactor `scripts/sae_direct_analysis/plot_sae_analysis.py` to replace global top-k plotting with per-value top-k plotting for the expected latent target at each token position.

## What was changed
- Kept one plotting mode only (removed second per-feature option and related CLI flow).
- For each value panel, top-k selection is now value-specific.
- Selectivity ranking criterion is signed:
  - `score_i(v) = E[f_i | v] - baseline_i`
  - no absolute value in ranking.
- Bars now plot `score_i(v)` directly (not raw conditional means).
- Removed dashed baseline lines from plots.
- Figures force full value coverage (`0..v-1`, or `0..n-1` for level 0 when available), with `no data` panels when a value is missing.

## Current plotting semantics
For each token position and expected target `(level, j)`:
- Build one figure with one subplot per value.
- In each subplot:
  - x-axis: feature ids (top-k for that value)
  - y-axis: `E[f_i | value] - baseline_i`
  - zero line shown for reference.

## Important implementation details
- Matching target indices are sorted by target value.
- NaN rows are masked out before top-k selection.
- `top_k` is clamped by latent dimension and valid feature count.

## Validation performed
- File diagnostics: no errors.
- Syntax compile: `python -m py_compile scripts/sae_direct_analysis/plot_sae_analysis.py` passed.
- Plot command was run on real artifacts successfully (exit code 0).

## Useful command
`python scripts/sae_direct_analysis/plot_sae_analysis.py --artifact <path/to/*.feature_latent.pt> --out_dir <out_dir> --top_k 12`

## Suggested next optional tweaks
- Color bars by sign (positive vs negative score).
- Add optional mode to rank by absolute score while still plotting signed bars.
- Add fixed shared y-limits across subplots for easier visual comparison.
