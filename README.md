# Sparse Autoencoders on the Random Hierarchy Model

How do deep transformers learn hierarchically compositional data? This repo
trains transformers to classify the **Random Hierarchy Model (RHM)**, then uses
**sparse autoencoders (SAEs)** and **per-input circuit tracing** to test whether
the transformer's internal representations recover the latent variables of the
data's generative tree.

For the complete technical reference (data model, architectures, every training
and analysis pipeline, and the exact hyperparameters used) see
[`docs/project_pipelines.md`](docs/project_pipelines.md). For the SAE evaluation
library specifically, see [`docs/sae_eval_guide.md`](docs/sae_eval_guide.md).

## Background: the Random Hierarchy Model

The RHM is a synthetic classification task with a tree-structured generative
process. Data is generated top-down, from a class label to input tokens:

- **Level 0**: a class label, one of `n` values (the root).
- **Level l**: `s^l` symbols. Each node at level `l-1` expands into `s` children
  via one of `m` synonymic production rules, each child drawn from a vocabulary
  of `v` values.
- **Level L**: the `s^L` leaf tokens fed to the network.

Parameters: `n` (classes), `v` (vocabulary per node), `m` (synonymic rules per
node), `s` (branching factor), `L` (levels). The full derivation is kept as
`trees[l]` (shape `(N, s^l)`, with `trees[0]` the labels and `trees[L]` the
leaves), so every downstream stage can check its predictions against the ground
truth at every level.

Deep networks learn this task with sample complexity polynomial in the input
dimension by building representations invariant to exchanging synonyms
(Cagnetta et al., *Phys. Rev. X* 14, 2024).

**Central hypothesis.** A depth-`L` transformer is expected to resolve the
hierarchy bottom-up: block `k` resolves RHM level `L-1-k` (block 0 groups leaves
into their level-`(L-1)` parents; the last block resolves the class). This
layer-to-level mapping is what the SAE, tree-reconstruction, and circuit-tracing
pipelines test.

## Repository map

| Path | Purpose |
|---|---|
| `main.py`, `init.py`, `measures.py` | Transformer training |
| `datasets/` | RHM generator (`random_hierarchy_model.py`) |
| `models/` | Transformer variants + `SparseAutoencoder` (plus CNN/FCN/LCN baselines) |
| `train_sae.py`, `optuna_tune_sae.py` | Post-hoc SAE training / lr tuning |
| `scripts/sae_sweep/` | Sweep generation/running, sweep plots and diagnostics |
| `scripts/sae_eval/` | Streaming SAE evaluation library + CLI |
| `scripts/sae_tree_reconstruction/` | Decode the full RHM tree from SAE features |
| `scripts/intervention/` | Residual-stream token ablations |
| `scripts/sae_direct_analysis/` | Per-value feature-selectivity plots |
| `scripts/circuit_tracing/` | Linearization, attribution, pruning, grouping, visualization |
| `slurm/` | Cluster launchers (one per pipeline stage; also record the hyperparameters actually used) |
| `docs/` | Full pipeline reference and SAE-eval guide |

## Requirements

Python 3.10+, `torch`, `numpy`, `matplotlib` (plots), `networkx` (circuit
tracing), and `optuna` (optional, for SAE lr tuning).

## Quickstart

Every stage runs standalone from the command line. Below are minimal shapes; the
`slurm/` launchers wrap the same commands with the exact hyperparameters used in
the experiments.

**1. Train a transformer on the RHM** (`n=v`, one block per level, mean-pooled
classifier):

```bash
python main.py \
    --model transformer_meanclass \
    --num_features 8 --num_classes 8 --num_synonyms 4 --tuple_size 2 \
    --num_layers 3 --num_tokens 8 \
    --mode class --input_format long \
    --embedding_dim 512 --num_heads 8 --ffwd_size 4 --depth 3 \
    --optim adam --lr 1e-3 --weight_decay 1e-4 --max_epochs 20000 \
    --save_models --outname runs/transformer
```

The checkpoint saves the RHM rules, so every later stage regenerates the exact
same data.

**2. Train an SAE** on a frozen layer's residual stream:

```bash
python train_sae.py \
    --train_output runs/transformer.pt \
    --sae_layer 0 --sae_activation_source all_tokens \
    --sae_latent_dim 10240 --sae_lambda_l1 1e-3 --sae_lr 1e-4 --sae_steps 131072 \
    --outname runs/sae_layer0
```

For a full lambda sweep per layer, use
`scripts/sae_sweep/generate_sweep.py` + `scripts/sae_sweep/run_one.py`
(orchestrated by `slurm/sae/run_full_sweep.sh`).

**3. Evaluate the SAE** (the canonical streaming eval, produces a
`*.sae_eval.pt` artifact):

```bash
python scripts/sae_eval/run.py --with-all \
    --sae_checkpoint runs/sae_layer0.pt
```

**4. Downstream analysis** (each needs the eval artifacts above):

- Decode the whole latent tree from SAE features:
  `python scripts/sae_tree_reconstruction/run.py ...`
- Causal token ablations:
  `python scripts/intervention/ablate_tokens.py --experiments scripts/intervention/experiments_dead_token.json ...`
- Per-input circuit tracing through the per-layer SAEs:
  `python -m scripts.circuit_tracing.circuit_trace --train_output ... --sae_ckpts L0.pt L1.pt L2.pt --sae_eval_artifacts ... --input_idx 42`
  (sweep pruning thresholds with `scripts.circuit_tracing.circuit_trace_sweep`).

See [`docs/project_pipelines.md`](docs/project_pipelines.md) for the full flag
set, artifact schemas, and the end-to-end workflow.

## Notes for reuse

The `slurm/` directory contains de-personalized launchers, one per pipeline
stage. They target a specific cluster (partitions, conda env, absolute paths) and
are meant as runnable references for the exact commands and hyperparameters, not
as portable scripts.
