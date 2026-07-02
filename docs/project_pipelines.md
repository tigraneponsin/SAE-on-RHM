# SAE on RHM: Complete Pipeline Reference

This document specifies the pipelines of the SAE-on-RHM project as of June 2026
(branch: circuit-tracing). It is intended as context for report writing: it covers
the data model, the architectures, the training procedures, the parameter ranges
(both what the code supports and what was actually run), and the analysis /
evaluation pipelines. All notation is defined inline. It is restricted to what
the report will actually use; legacy code paths (CLS-token classifier, causal
LM, FCN/CNN/LCN baselines, linear probes) exist in the repo but are omitted
here.

Project goal: study how deep transformers learn hierarchically compositional data
by training sparse autoencoders (SAEs) on the residual stream of transformers
trained on the Random Hierarchy Model (RHM), and by tracing per-input circuits
through those SAEs.

> **Level-numbering convention (important).** This document is written in the
> CODE convention: RHM levels run top-down, root = level 0 and leaves =
> level `L`, so `trees[l]` has shape `(N, s^l)` and a transformer block `k`
> resolves level `L-1-k`. The report and every figure use the flipped "report"
> convention (leaves = 0, root = `L`) via `report_level(l, L) = L - l` (see
> `scripts/common/notation.py`); the README states the RHM in that convention.
> All plotting/diagnostic CLIs accept `--report-notation` to relabel figures
> without changing any computed value. Formulas below are in the code
> convention unless a passage says otherwise.

---

## 1. The data: Random Hierarchy Model (RHM)

Implementation: `datasets/random_hierarchy_model.py`.

### 1.1 Generative model

The RHM is a probabilistic context-free grammar with L levels:

- `n`  : number of classes (root vocabulary size)
- `v`  : vocabulary size of every non-root level (including leaves)
- `m`  : number of synonymic production rules per symbol (multiplicity)
- `s`  : branching factor (tuple size; each symbol expands into s children)
- `L`  : number of levels; the input has `s^L` leaf tokens

Rules are sampled once per dataset (seeded by `seed_rules`), WITHOUT replacement
from the v^s possible s-tuples:

- Level 0 (root -> level 1): `n*m` distinct tuples, reshaped to `rules[0]` of
  shape `(n, m, s)`.
- Levels 1..L-1: `v*m` distinct tuples each, `rules[l]` of shape `(v, m, s)`.

Sampling a datum (seeded by `seed_samples`): draw a uniform class label, then at
every node independently pick one of its m rules uniformly and expand, down to
the leaves. The full derivation is kept: `trees[l]` has shape `(N_data, s^l)`
for l = 1..L and `(N_data,)` for l = 0 (the class label). `trees[L]` is the leaf
token sequence fed to the network.

Total number of distinct data: `n * m^((s^L - 1)/(s - 1))`.

Auxiliary machinery used by the analysis pipelines:

- `latent_prior(rules, n, v)`: exact marginal P(Z_{l,j} = z) for every
  (level l, slot j), propagated down from a uniform class prior. Returns, per
  level, a `(s^l, V_l)` tensor.
- `latent_entropy(prior)`: H(Z_{l,j}) in nats per (level, slot). Used as the
  theoretical reference entropy in the SAE entropy metrics.

### 1.2 Input encoding and task

Inputs are integer token indices (`input_format='long'`) into an embedding
table. The task is always classification (`mode='class'`): predict the root
class `trees[0]` from the leaf sequence `trees[L]`.

### 1.3 Transformer-layer <-> RHM-level correspondence (working hypothesis)

A natural expectation is that block k (0-based) resolves RHM level `L-1-k`
(bottom-up composition). Under that expectation, a feature at layer k and leaf
position p has a *matched* (default) latent to compare against:

- level: `L - 1 - k`
- ancestor position at that level: `p // s^(1+k)`
- label vocabulary: `v` for levels 1..L-1, `n` for level 0.

This matched latent is the default reference used by the streaming-eval target
layout, the tree reconstruction, and the circuit-tracing labels
(`scripts/circuit_tracing/labels.py`). It is a working hypothesis, not an
assumption the analysis leans on: the entropy diagnostics deliberately relax it,
also scoring each feature against other same-level cells, against every cell in
the tree, and against a reassigned parent (see section 5), so the matched latent
can be checked rather than taken for granted.

---

## 2. Architectures

### 2.1 Transformer variants (`models/transformer.py`)

All variants share the same trunk:

- Token embedding table (`nn.Embedding(v, d)`) + learned absolute position
  embeddings; real token i sits at sequence position i (no CLS token).
- `depth` pre-LayerNorm transformer blocks (`DecoderBlock`), each:
  `x = x + dropout(Attn(LN1(x)))` then `x = x + dropout(MLP(LN2(x)))`.
- Attention is bidirectional (no causal mask) multi-head; custom unbiased
  parameter matrices with 1/sqrt(dim) forward scalings (Q/K/V scaled by
  C^-1/2, output projection by out_dim^-1/2).
- The block MLP is a 1-hidden-layer ReLU MLP of width `ffwd_size * d`
  (its linear layers also use the 1/sqrt(fan_in) forward scaling, no bias).
- Final `ln_f` LayerNorm, then a pooling head and a linear classifier to `n`
  logits.

The four variants used differ in the pooling head and the residual wiring:

| model name | pooling after ln_f | residual |
|---|---|---|
| `transformer_meanclass` | uniform mean over positions | yes |
| `transformer_meanclass_nores` | uniform mean | NO residual (both skips removed, `NoResidualDecoderBlock`) |
| `transformer_freeclass` | learned softmax pooling | yes |
| `transformer_freeclass_nores` | learned softmax pooling | NO residual |

Learned pooling (freeclass): a single learnable vector `pool_logits` of length
`s^L`, zero-initialized (so the model is exactly meanclass at init);
`pooled = sum_p softmax(pool_logits)[p] * ln_f(x)[p]`. Models expose
`pool_weights()`; downstream code detects freeclass via
`getattr(model, 'pool_weights', None)` and falls back to uniform mean.

The no-residual variants exist because the per-layer SAE/circuit story is
cleaner without an identity path: each block must rewrite the representation,
so "layer k resolves level L-1-k" is testable layer by layer.

### 2.2 Sparse autoencoder (`models/sae.py`)

Single-hidden-layer SAE on a d-dimensional activation vector:

- Encoder: `z = ReLU(W_enc x + b_enc)`, `W_enc` of shape `(F, d)`.
- Decoder: `x_hat = W_dec z + b_dec`, `W_dec` of shape `(d, F)`.
- Initialization: decoder columns are random directions with fixed norm 0.1;
  encoder initialized as decoder transpose; biases zero.
- Loss: `MSE(x_hat, x) + lambda_1 * mean_batch( sum_i |z_i| * ||W_dec[:, i]||_2 )`
  i.e. a decoder-norm-weighted L1 penalty (Anthropic April-2024 style). This
  makes the penalty invariant to the encoder/decoder rescaling symmetry.
- `latent_dim F` is a free parameter (default `4*d`; main experiments use
  `20*512 = 10240`, smaller runs `4*512 = 2048`).

---

## 3. Transformer training pipeline

Entry point: `main.py` (Slurm wrapper: `slurm/transformer/Sbatch_trsf_for_SAE.sh`).

Procedure:

1. Build the RHM dataset (rules from `seed_rules`, samples from `seed_sample`),
   split into `train_size` (P) train and `test_size` test samples.
2. Train with cross-entropy on the class label, AdamW (`--optim adam`).
   Optional schedulers exist (`cosine`, `cosine-warmup`); the main runs use a
   constant learning rate.
3. Log/checkpoint on a log-spaced step grid. The model with the lowest TEST
   loss seen so far is kept as `best` alongside the final (`last`) weights.
4. Stop when running train loss <= `loss_threshold` (default 1e-3), or, with
   `--stop_on_test_loss --test_loss_threshold T`, when test loss <= T.
5. With `--save_models` (forced on by the SAE launcher) the output artifact
   `<outname>.pt` contains: `config`, and `output = {rules, init, best
   {step, loss, model}, model (last), dynamics, step}`. Saving `rules` is what
   lets every downstream stage regenerate exact RHM data.

Actually used in experiments (from the Slurm launchers):

- Fixed: `s=2`, `n=v`, `depth=L` (one transformer block per RHM level),
  `embedding_dim=512`, `num_heads=8`, `ffwd_size=4`, `test_size=32768`,
  AdamW, `loss_threshold=1e-3`, `max_epochs=20000`, seeds drawn at random per
  run and recorded in a manifest file next to the artifact.
- Common hyperparameters: `dropout=0.1` (sometimes 0), `weight_decay=1e-4`,
  `lr=1e-3`
  batch size 16..1280 depending on P.
- Main (v, L, m, P) configurations trained for SAE work:
  - v=16, L=3, m=4,  P=12160 or 32768
  - v=16, L=3, m=16, P=1048576
  - v=16, L=5, m=4,  P=256000
  - v=8,  L=4, m=8,  P=524288
  - models: `transformer_meanclass_nores` (historical default), plus matched
    `transformer_meanclass`, `transformer_freeclass`, and
    `transformer_freeclass_nores` runs (the freeclass pair most recently for
    v=8, L=4, m=8).

---

## 4. SAE training pipeline

Entry point: `train_sae.py` (post-hoc, frozen transformer).

### 4.1 Activation source

A forward hook captures activations; token modes used
(`sae_activation_source`):

- `all_tokens`: post-block residual stream at `model.blocks[layer_id]`, all
  token positions. This is the standard mode for layers 0..L-2 (and for the
  last layer in non-pooled setups).
- `mean_pooled`: hooks `model.ln_f` instead and pools the post-ln_f sequence
  with the model's own readout weights (uniform mean for meanclass, learned
  `pool_weights()` for freeclass). Only valid at the LAST block (forced), and
  only for meanclass/freeclass models. This gives an SAE on the exact vector
  the classifier reads; circuit tracing uses it as the pooled last layer.
- `one_token` (single position `sae_token_idx`) was used in early sweeps and
  is still supported.

### 4.2 Normalization

Per layer, a scalar `act_scale = sqrt(d) / E[||x||_2]` is computed on the SAE
training set so that scaled activations have expected norm sqrt(d). The SAE is
trained on `act_scale * x`; the scale is stored in the checkpoint
(`sae_training_setup.act_scale`) and every consumer divides it back out.

### 4.3 Data discipline

SAE training data is sampled from the SAME saved rules but with a sampling
seed forced to differ from the transformer's training seed (default:
transformer seed + 1); a further disjoint eval split (default: train seed + 1)
is used for periodic eval-loss logging. The checkpoint records all seeds and
disjointness flags under `sae_dataset_split`.

### 4.4 Optimization

- One SAE per layer, trained independently (`--sae_layer`).
- AdamW, weight_decay 0, lr `sae_lr`.
- `sae_steps` gradient steps, cycling over the training loader; each step
  takes one transformer forward batch of `sae_sample_batch_size` RHM samples
  (so `all_tokens` yields `batch * s^L` SAE tokens per step; `sae_batch_limit`
  optionally subsamples tokens per step).
- lambda_1 warmup (linear ramp over `sae_lambda_warmup_frac` of steps) and
  end-of-training LR decay (`sae_lr_decay_frac`) are supported; the sweeps set
  both to 0 (constant lambda, constant lr).
- SAE init seeded by `seed_sample + 1000 + layer_id` for reproducibility.
- Train/eval loss curves logged at ~`sae_log_points` log-spaced checkpoints.
- The SAE is trained against the `best` transformer weights by default
  (`--model_variant best`). The variant is recorded and enforced downstream.

### 4.5 Checkpoint schema (`*_sae.pt`)

`config` (transformer config + sae_* fields), `source` (paths, model_step,
model_variant), `sae_dataset_split` (rules source + all seeds + RHM params),
`sae_layers`, `sae_state[layer_id]`, `sae_metrics[layer_id]` (final losses,
active_fraction, dead_features, latent_dim), `sae_training_curves`,
`sae_eval_curves`, `sae_training_setup` (activation source, token idx,
batch sizes, warmup/decay fracs, per-layer act_scale).

### 4.6 Hyperparameter values

From the sweep generator (`scripts/sae_sweep/generate_sweep.py`) and the runs:

- `sae_latent_dim`: 20*512 = 10240 (main); a "Small_SAE" line uses 4*512 = 2048
- `sae_lambda_l1`: the principal sweep axis; grids spanning roughly 1e-4 .. 1,
  with zoom sweeps around the elbow (e.g. 1e-3 .. 2e-2)
- `sae_lr`: 1e-4 (tuned via Optuna, see 4.7)
- `sae_steps`: 2^17 = 131072
- `sae_sample_batch_size`: 2^7 = 128 (round-1 sweeps explored 32..512)
- `sae_train_size`: 2^14 = 16384; `sae_eval_size`: 2^14 (eval artifacts later
  use 32768 fresh samples)
- `sae_lambda_warmup_frac` = 0, `sae_lr_decay_frac` = 0

### 4.7 Optuna tuning (`optuna_tune_sae.py`)

Search space: `sae_lr` log-uniform in [1e-6, 1e-2], batch size categorical
{64, 128, 256, 512, 1024}; fixed lambda_1 = 0.01, latent 20*d, no
warmup/decay. Objective: final eval total loss. Result informed the
lr = 1e-4 / batch 128 choice.

### 4.8 Sweep orchestration

- `generate_sweep.py` builds the Cartesian product of the grid into
  `sweep_configs.json` (one entry per job; output filenames encode transformer
  tag + layer + activation tag + latent dim + lambda + lr + steps + batch +
  train size).
- `scripts/sae_sweep/run_one.py` runs configs[task_id] (one Slurm array task =
  one SAE).
- `slurm/sae/run_full_sweep.sh` is a one-shot submitter: (optionally generate
  configs) -> submit training array -> submit a dependent (afterok)
  eval+plots job. Outputs land in `<sweep_dir>/sae_checkpoints/`,
  `<sweep_dir>/analysis_files/` (`*.sae_eval.pt`, `sweep_metrics.csv`,
  per-position CSV), `<sweep_dir>/analysis_plots/` (lambda metrics, entropy
  plots).
- Sweeps are organized one folder per (transformer, layer, activation source),
  e.g. `.../v_8_L_4_m_8_wdecay_0.0001_dropout_0.1_freeclassnores/
  sweep_alltokens_layer3_lambda1_zoom/`.

---

## 5. SAE evaluation pipeline (streaming eval)

Entry point: `scripts/sae_eval/run.py` -> `scripts/sae_eval/streaming.py`.
This is the main evaluation entry point: one streaming pass of a fresh RHM eval set
(defaults: `eval_size=32768`, `eval_seed=99999`, batch 512, optional
de-duplication of repeated trees) through the frozen transformer + SAE,
accumulating metric blocks gated by flags. Output: one `*.sae_eval.pt`
artifact per SAE checkpoint, plus optional sweep-level CSVs.

Metric blocks (flags, default off; the pipeline runs `--with-all`):

1. Scalar activity aggregates: dead feature count/ratio, mean active features
   per token (L0), inverse participation ratio (IPR), and thresholded variants
   counting only features whose mean activation exceeds 1% / 10% of the max
   feature mean (these thresholded counts are the headline sparsity metrics).
2. Per-position versions of the above (`[P]` arrays over token positions) --
   this is what surfaces the dead-token effect seen in these runs (positions
   whose features die out layer by layer).
3. Per-feature: baseline mean/std of each feature at each position, firing
   counts/rates, mean co-firing, decoder norms.
4. Per-target conditional statistics: for every target latent (level,
   position, value) in the matched-layout (section 1.3): conditional mean
   activation, delta vs baseline, z-scores. Used for selectivity analysis.
5. Entropy block: joint fire counts P(latent value, feature fires), per-feature
   conditional entropy H_i = H(Z | f_i > 0) in nats against the position's
   matched latent, and position-level weighted aggregates
   `H_bar_fire` (weights = firing rate), `H_bar_dec` (weights = mean weighted
   activation), `H_bar_raw` (activation / decoder norm), each also normalized
   by the THEORETICAL entropy H(Z_{l,j}) from `latent_prior`. Normalized
   entropy closer to 0 indicates features more selective for single latent
   values; closer to 1, less informative about the latent.
6. Classification impact (second pass): splice the SAE into the forward pass
   (replace the hooked activation by its reconstruction) and measure
   classification error and cross-entropy vs the clean baseline;
   `norm_err = sae_err / (1 - 1/n)` normalizes by the chance error.

Plotting / diagnostics over a sweep (all in `scripts/sae_sweep/`):

- `plot_lambda_metrics.py`: ever-active and mean-active feature counts (three
  thresholds) and classification error vs lambda_1, per layer.
- `plot_lambda_metrics_per_token.py`: same but keeping the token-position
  axis (subset selection, per-token curves).
- `entropy_diag.py`: the unified entropy diagnostic, one CLI with three
  subcommands (all three weight schemes fire/raw/dec). `aggregate` plots the
  stored normalized entropy vs lambda_1 per position; `min` finds, per feature,
  the SAME-LEVEL latent group minimizing its conditional entropy and draws the
  parent/level/unconstrained curves plus the leakage fraction; `splitcheck`
  adds the feature-splitting reassignment (child pinned beyond the RHM leak ->
  reassigned up one level). Numeric helpers live in `entropy_core.py`; the
  per-feature labeling core shared with circuit tracing is `absorption.py`.
- `feature_splitting_alpha_sweep.py`: sweeps the split-check alpha threshold
  (reassign fraction and ratio histogram at a chosen lambda).
- `plot_rank_magnitude.py`: histograms of per-feature mean activation
  magnitudes (dominant features vs near-zero tail).
- `plot_loss_curves.py`: SAE training/eval loss curves for one sweep, or a
  cross-sweep comparison with `--sweep_dirs` (formerly `plot_multi_sweep.py`).
- `decoder_cosine_diag.py`: c_dec = mean |cos| over distinct decoder column
  pairs (Chanin & Garriga-Alonso, arXiv:2508.16560); its minimum over the
  lambda sweep flags the L0 with least feature mixing. Optionally restricted to
  active (ever-firing) latents.

---

## 6. Tree reconstruction from SAE features

Entry point: `scripts/sae_tree_reconstruction/run.py`. Requires one
`.sae_eval.pt` artifact per transformer layer (produced with
`--with-per-feature --with-conditional --with-entropy`).

For each eval input, reconstruct the ENTIRE latent tree:

1. One transformer forward, hooked at every layer.
2. Per layer, SAE-encode and weight by decoder norms:
   `f_act[p, i] = relu(W_enc (act_scale * x[p]) + b_enc)_i * ||W_dec[:, i]||`.
3. For each latent (l, j): take the leaf positions p in its subtree
   (`p // s^(1+k) == j`, k = L-1-l), gather the active alive features, and
   score each value z by the weighted average of the features' conditional
   distributions `P(Z = z | f_i fires)` (weights: activation, or uniform;
   both variants are produced). Average over positions, argmax = predicted
   latent value.
4. Compare against the true tree: per-(level, position) accuracy tables (CSV +
   .pt outputs, plots via `plot_results.py`).

Seed discipline: the script refuses to evaluate on seeds the transformer or
SAE saw during training (override: `--allow_seed_overlap`).

This probes how well the SAE feature dictionary, read through its conditional
statistics, can decode the internal RHM variables -- to what extent the SAEs
recover the generative parse. The per-(level, position) accuracy tables report
where this succeeds and where it does not, rather than asserting a result.

---

## 7. Direct interventions (no SAE)

Entry point: `scripts/intervention/ablate_tokens.py`. Replaces the post-block
residual stream at chosen (layer, positions) with a surrogate -- per-position
dataset mean, zeros, or a resample from another input in the batch -- in one
forward pass, possibly stacking interventions across layers, and reports
classification accuracy (plus residual-norm tables). Experiments are specified
in a JSON list; defaults eval_size 32768, seed 99999.

Purpose: a causal check of the dead-token effect -- if the transformer has
already dropped the information at a position/layer, ablating it should not hurt
accuracy.

`scripts/sae_direct_analysis/` complements this on the SAE side: per-value
top-K feature selectivity plots from the eval artifacts
(score_i(value) = E[f_i | value] - baseline_i), per layer and position.

---

## 8. Circuit tracing pipeline

Package: `scripts/circuit_tracing/`. The most recent pipeline; produces per-input
attribution graphs through the SAE feature bases, in the spirit of
attribution graphs / transcoder circuit work, adapted to this architecture.

### 8.1 Inputs

- A trained meanclass/freeclass transformer checkpoint (with rules).
- K = depth SAE checkpoints, one per layer, bottom-to-top, all trained on the
  same transformer variant ('best'/'last' is auto-detected and enforced).
  Layers 0..K-2 must be per-position (`all_tokens`) SAEs; the last layer is
  either per-position or, in "pooled" mode (auto-detected), a `mean_pooled`
  SAE living in the post-ln_f pooled space the classifier actually reads.
- K matching `.sae_eval.pt` artifacts (for feature labels).
- A single input: row `input_idx` of a freshly sampled eval set
  (`eval_seed`, `eval_size`).
- `scripts/sae_sweep/resolve_circuit_inputs.py` resolves all of these
  automatically from per-layer sweep folders given one target lambda_1 per
  layer (nearest match, with warnings).
- Hard consistency checks: every artifact must agree on the RHM rules (same
  saved rules object or same seed), latent dims must match, layer order must
  be bottom-to-top.

### 8.2 Linearization (`linearize.py`)

On one forward pass, capture "anchors" at every block: the post-softmax
attention pattern, the MLP ReLU mask, both LayerNorms' (mean, rstd), plus
ln_f stats, the embedding output, per-block residual streams, pooled vector,
and logits. Freezing these makes every block an affine function of its input;
the centered map `M_{k+1}(u) = LinBlock_{k+1}(x_anchor + u) -
LinBlock_{k+1}(x_anchor)` is exactly linear in the perturbation u and exact at
the anchor. Works for residual and no-residual blocks, uniform and learned
pooling.

### 8.3 Nodes and SAE splice

At each layer k the residual stream is decomposed through the SAE:
`x_k = x_hat_k + e_k` with `z_k` the feature activations. Node kinds:

- `('embed', p)`: token+position embedding input, one per position
- `('feat', k, p, i)`: SAE feature i at layer k, position p (only if firing)
- `('err', k, p)`: reconstruction-error node (what the SAE missed)
- `('logit', c)`: one per class

Two bit-identity checks gate the run: per-layer `x_hat + e == x` (tol 5e-4)
and a whole-pipeline splice (replacing every layer's activation by
reconstruction + error and propagating through the linearized model must
reproduce the logits to 1e-3).

### 8.4 Edge attribution (`attribution.py`)

All edges are exact linear attributions through the frozen maps:

- feature(k,p,i) -> feature(k+1,q,j): contribution of the source's decoder
  vector (scaled by z and act_scale conventions) through M_{k+1} to the
  target's encoder pre-activation.
- error(k,p) -> feature(k+1,q,j): same with e_k at position p as source.
- embed(p) -> feature(0,q,j): embedding rows through M_0.
- feature/error at K-1 -> logit(c): through the frozen ln_f linearization and
  the pooling weights w[p] (1/N uniform or learned softmax), per class.
- In pooled mode the K-2 -> K-1 edges target the single pooled SAE
  (`edges_layer_to_pooled`), and final-layer-to-logit edges skip ln_f/pooling
  (the pooled SAE already lives there).

### 8.5 Pruning (`prune.py`)

Indirect-influence pruning (two thresholds):

1. Build absolute, row-normalized adjacency A; indirect influence
   B = A + A^2 + ... + A^(K+2) (exact: the DAG is strict).
2. Sink weights at the logits: `softmax_logits` (softmax of the logits) or
   `true_class` (1 at the true class).
3. node_score = B @ sink; keep top feature nodes until cumulative score
   fraction >= `node_threshold` (default 0.8).
4. Recompute on the kept subgraph; keep top edges by
   A_sub[t,s] * node_score_sub[t] until fraction >= `edge_threshold`
   (default 0.98).
5. Trim to edges on some (embedding|error) -> logit path.

Diagnostics: completeness and replacement scores, node/edge counts by kind.

### 8.6 Fidelity metrics (`fidelity.py`)

- `per_layer_error_fraction`: share of k -> k+1 attribution mass carried by
  error (rather than feature) sources; > 0.2 is flagged.
- `feature_mediated_logit_fraction`: share of logit attribution mass from
  features vs errors (sink-weighted).
- Subtree alignment (pre- and post-prune): fraction of absolute edge mass
  whose source and destination positions share the RHM ancestor implied by
  the layer mapping (embed->L0 compares `p // s` vs `q // s`; feat k -> k+1
  compares `p // s^(2+k)` vs `q // s^(2+k)`). High alignment = attribution
  flows along the data's tree.

### 8.7 Grouping (`grouping.py`, newest stage)

Bottom-up signature-based collapse of the pruned graph: two layer-k feature
nodes merge iff they have the exact same incoming signature
`frozenset{(already-grouped source super-node, sign(weight))}` over pruned
feature/embed incoming edges (error edges excluded; empty signatures never
merge; cross-position merges allowed). Merges cascade upward because layer
k-1 is grouped before layer k. Each group node carries constituents,
positions, z-weighted value distribution (lifted to the full parent-level
vocabulary), argmax label, P(value|fire), and normalized entropy. Edges are
remapped to group endpoints, self-loops dropped, duplicates summed.
Motivation: RHM synonyms induce many functionally identical detectors;
grouping makes the circuit readable.

### 8.8 Labels (`labels.py`)

Each feature node is labeled from its layer's eval artifact:
`label_value = argmax_z P(Z = z | feature fires)` at the matched (level,
parent position), plus P(value|fire) and normalized conditional entropy
(drives the white->green confidence gradient in the visualizations).

### 8.9 Outputs, sweep, visualization

Per run (`circuit_trace.py`): `nodes.pt`, `edges.pt` (all + pruned),
`fidelity.pt`, `grouped_nodes.pt`, `grouped_edges.pt`, `group_membership.pt`,
`tree_for_input.pt`, `graph.gpickle` (networkx), `summary.json`.

`circuit_trace_sweep.py` runs the expensive threshold-independent stages once
and then loops over a (node_threshold x edge_threshold) grid, one subdir per
cell plus `sweep_summary.json`. Slurm wrapper:
`slurm/sae/run_circuit_trace_sweep.sh` (defaults: eval_size 1024, eval_seed 0,
sink softmax_logits; example grids: node 0.7/0.8/0.9 x edge 0.9/0.95/0.98).

Visualization: `visualize.py` (static layered plot), `visualize_interactive.py`
(single-run interactive HTML), `visualize_sweep.py` (one HTML with two
threshold sliders over the sweep grid, rendered automatically by the Slurm
wrapper).

---

## 9. End-to-end workflow summary

1. Train transformer on RHM (`main.py` via `Sbatch_trsf_for_SAE.sh`);
   artifact saves weights (best + last) and the RHM rules.
2. Generate + run an SAE sweep per layer, principally over lambda_1
   (`generate_sweep.py` + `run_full_sweep.sh`); each SAE saves act_scale,
   seeds, curves.
3. Streaming eval on every SAE (`scripts/sae_eval/run.py --with-all`) ->
   `.sae_eval.pt` artifacts + sweep CSVs.
4. Pick operating lambda_1 per layer using: sparsity/lambda plots, normalized
   entropy plots, classification impact, decoder cosine and min-entropy
   diagnostics.
5. Optional causal checks: token ablations; feature selectivity plots; full
   tree reconstruction from the SAE dictionaries.
6. Circuit-trace single inputs through the chosen per-layer SAEs
   (per-position layers + optionally a mean_pooled last layer), sweep the
   pruning thresholds, and inspect the grouped interactive graphs against the
   input's true RHM derivation tree.

## 10. Repository map (for reference)

- `main.py`, `init.py`, `measures.py`: transformer training.
- `datasets/`: RHM generator.
- `models/`: transformer variants, SAE.
- `train_sae.py`, `optuna_tune_sae.py`: SAE training/tuning.
- `scripts/sae_sweep/`: sweep generation/running, all sweep-level plots and
  diagnostics, circuit-input resolution.
- `scripts/sae_eval/`: streaming evaluation library + CLI.
- `scripts/sae_tree_reconstruction/`: tree decoding from SAE features.
- `scripts/intervention/`: residual-stream ablations.
- `scripts/sae_direct_analysis/`: per-value feature selectivity plots.
- `scripts/circuit_tracing/`: linearization, attribution, pruning, grouping, fidelity,
  visualization, single-run and sweep CLIs.
- `slurm/`: cluster launchers mirroring all of the above (these record the
  actually-used hyperparameter values).
