# Sparse Autoencoders on the Random Hierarchy Model

This repository implements the Random Hierarchy Model (RHM), trains transformers
to classify it, and provides tooling to train and evaluate sparse autoencoders
(SAEs) on the trained transformers and to trace per-input circuits through them.

For the full technical reference (data model, architectures, training and
analysis pipelines, and the hyperparameters used) see
[`docs/project_pipelines.md`](docs/project_pipelines.md). For the SAE evaluation
library specifically, see [`docs/sae_eval_guide.md`](docs/sae_eval_guide.md).

## The Random Hierarchy Model

The RHM (Cagnetta et al., *Phys. Rev. X* 14, 2024) is a synthetic classification
task with a tree-structured generative process. Sampling runs top-down, from a
class label at the root to the input tokens at the leaves. Levels are numbered
bottom-up (leaves = 0, class = `L`):

- **Level `L`**: the class label, one of `n` values (a single symbol, the root).
- **Level `l`** (`0 < l < L`): `s^(L-l)` symbols. Each node at level `l` expands
  into `s` children at level `l-1` via one of `m` synonymic production rules,
  each child drawn from a vocabulary of `v` values.
- **Level 0**: the `s^L` leaf tokens fed to the network.

Parameters: `n` (classes), `v` (vocabulary per node), `m` (synonymic rules per
node), `s` (branching factor), `L` (levels). The full derivation is kept at every
level, so downstream analyses can compare against the ground-truth latents.

> **Level numbering.** This README and the figures number levels bottom-up
> (leaves = 0, class = `L`). The code uses the opposite convention
> (root = level 0, leaves = level `L`), related by `report_level = L - code_level`.
> [`docs/project_pipelines.md`](docs/project_pipelines.md) is written in the code
> convention.

## Repository map

| Path | Purpose |
|---|---|
| `main.py`, `init.py`, `measures.py` | Transformer training |
| `datasets/` | RHM generator (`random_hierarchy_model.py`) |
| `models/` | Transformer variants + `SparseAutoencoder` (plus CNN/FCN/LCN baselines) |
| `train_sae.py`, `optuna_tune_sae.py` | Post-hoc SAE training / lr tuning |
| `scripts/sae_sweep/` | Sweep generation/running, sweep plots and diagnostics |
| `scripts/sae_eval/` | Streaming SAE evaluation library + CLI |
| `scripts/sae_tree_reconstruction/` | Decode the RHM tree from SAE features |
| `scripts/intervention/` | Residual-stream token ablations |
| `scripts/sae_direct_analysis/` | Per-value feature-selectivity plots |
| `scripts/circuit_tracing/` | Linearization, attribution, pruning, grouping, visualization |
| `docs/` | Pipeline reference and SAE-eval guide |

## Requirements

Python 3.10+, `torch`, `numpy`, `matplotlib` (plots), `networkx` (circuit
tracing), and `optuna` (optional, for SAE lr tuning).

## Quickstart

Each stage runs standalone from the command line. Minimal shapes:

```bash
# 1. Train a transformer on the RHM
python main.py --model transformer_meanclass \
    --num_features 8 --num_classes 8 --num_synonyms 4 --tuple_size 2 \
    --num_layers 3 --num_tokens 8 --depth 3 --mode class --input_format long \
    --embedding_dim 512 --num_heads 8 --ffwd_size 4 \
    --optim adam --lr 1e-3 --weight_decay 1e-4 --save_models --outname runs/transformer

# 2. Train an SAE on a frozen layer's residual stream
python train_sae.py --train_output runs/transformer.pt \
    --sae_layer 0 --sae_activation_source all_tokens \
    --sae_latent_dim 10240 --sae_lambda_l1 1e-3 --sae_lr 1e-4 --sae_steps 131072 \
    --outname runs/sae_layer0

# 3. Evaluate the SAE (streaming eval -> *.sae_eval.pt artifact)
python scripts/sae_eval/run.py --with-all --ckpt runs/sae_layer0.pt
```

Downstream tooling (each consumes the eval artifacts): tree reconstruction
(`scripts/sae_tree_reconstruction/run.py`), token ablations
(`scripts/intervention/ablate_tokens.py`), and circuit tracing
(`python -m scripts.circuit_tracing.circuit_trace`). Per-layer SAE sweeps are
generated with `scripts/sae_sweep/generate_sweep.py`.

See [`docs/project_pipelines.md`](docs/project_pipelines.md) for flags, artifact
schemas, and the end-to-end workflow.

## Acknowledgments

This repository is built on the original Random Hierarchy Model codebase by
Cagnetta, Petrini, and collaborators, which accompanies Cagnetta et al.,
*Phys. Rev. X* 14, 2024. The RHM data generator and transformer-training
scaffolding derive from that project (see `LICENSE`); the SAE, circuit-tracing,
and analysis tooling are added here.

Original RHM repository: <ORIGINAL_RHM_REPO_URL>
