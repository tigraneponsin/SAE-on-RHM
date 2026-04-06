# Repository Reorganization Plan (Safe Migration)

This plan introduces a standard layout while keeping current workflows runnable during migration.

## Goals

- Improve discoverability for new collaborators.
- Keep training/eval commands stable while files move.
- Avoid breaking Slurm workflows in the middle of experiments.

## Target Layout

```
SAE-on-RHM/
  README.md
  LICENSE
  pyproject.toml                # phase 2
  src/
    rhm/
      __init__.py
      data/
        __init__.py
        random_hierarchy_model.py
        utils.py
      models/
        __init__.py
        cnn.py
        fcn.py
        lcn.py
        sae.py
        transformer.py
      training/
        __init__.py
        init.py
        measures.py
        transformer_train.py
        sae_train.py
      analysis/
        __init__.py
        latent_analysis.py
        linear_probe.py
  scripts/
    train_transformer.py        # wrapper entrypoint (imports src)
    train_sae.py                # wrapper entrypoint (imports src)
    run_probe.py                # wrapper entrypoint (imports src)
    eval_probe.py               # wrapper entrypoint (imports src)
    sweep/
      run_one.py
      run_one_probe.py
      eval_sweep.py
      eval_probe.py
      plot_loss_curves.py
      plot_lambda_metrics.py
      plot_multi_sweep.py
      plot_optuna.py
      plot_rank_magnitude.py
      inspect_ckpt.py
      generate_sweep.py
  slurm/
    transformer/
      sbatch_trsf_for_sae.sh
      sbatch_scaling_laws_transformer.sh
      sbatch_parameter_tuning.sh
      job_array_trsf_for_sae.sh
      job_array_scaling_laws_transformer.sh
      job_array_parameter_tuning.sh
    sae/
      run_sweep.sh
      run_eval.sh
    probe/
      run_probe.sh
      run_eval_probe.sh
    misc/
      sbatch_jupyter.sh
  notebooks/
    experiments/
      test_init_train.ipynb
      test_train_sae_from_transformer.ipynb
      test_train_sae_from_meantransformer.ipynb
  docs/
    linear_probe_handoff.md
    repo_reorganization_plan.md
    quickstart.md               # phase 1 output
  tests/
    smoke/
      test_probe_smoke.py       # phase 1/2, tiny CPU smoke
```

## Phase 1 (1-2 days): No Core Refactor

Keep imports as-is. Move wrappers, docs, and notebooks first.

### Step 1: Create directories only

- Create: `scripts/`, `scripts/sweep/`, `slurm/transformer/`, `slurm/sae/`, `slurm/probe/`, `slurm/misc/`, `notebooks/experiments/`.
- Do not move Python core modules yet (`datasets/`, `models/`, `init.py`, `measures.py`, `latent_analysis.py`, `linear_probe.py`).

### Step 2: Move Slurm files (with compatibility stubs)

Move files from root and `sae_sweep/` into `slurm/` groups.

After moving each file, leave a stub at the old path that forwards to the new location, for example:

```bash
#!/bin/bash
exec "$(dirname "$0")/../slurm/probe/run_probe.sh" "$@"
```

This keeps old commands working:
- `sbatch sae_sweep/run_probe.sh`
- `sbatch Sbatch_trsf_for_SAE.sh`

### Step 3: Move sweep helper scripts under scripts/sweep

Move scripts currently in `sae_sweep/` to `scripts/sweep/` but keep old-path wrappers in `sae_sweep/` that delegate to `scripts/sweep/`.

### Step 4: Move notebooks to notebooks/experiments

Move exploratory notebooks out of root. Keep short pointer notes in root only if needed.

### Step 5: Update README and add quickstart

- Update `README.md` with the new folder map.
- Add `docs/quickstart.md` with:
  - local tiny smoke commands
  - cluster run commands
  - where logs/checkpoints are written
  - common failure messages

### Step 6: Standardize output naming in all run wrappers

Replicate the descriptive naming style already used in probe slurm wrappers:
- include transformer tag, target layer/token, key hyperparameters, job id.

### Step 7: Add smoke checks (tiny, CPU-safe)

Add one script or test per official workflow:
- transformer tiny run
- SAE tiny run
- probe tiny run

## Phase 2 (1-2 weeks): Move Core Code Into src

Only begin after Phase 1 stabilizes.

### Step 1: Create `src/rhm` package and copy modules

Copy (not move) first:
- `datasets/*` -> `src/rhm/data/*`
- `models/*` -> `src/rhm/models/*`
- `init.py`, `measures.py` -> `src/rhm/training/*`
- `latent_analysis.py`, `linear_probe.py` -> `src/rhm/analysis/*`

### Step 2: Update script entrypoints to import from src

Update wrappers in `scripts/` to import only from `src/rhm`.

### Step 3: Keep backward compatibility for one transition window

At old module paths, keep thin import shims, for example:

```python
from src.rhm.analysis.linear_probe import *
```

### Step 4: Remove deprecated paths after validation

After all wrappers and jobs pass, remove old shims in one cleanup PR.

## Official Entrypoints (to document and support)

- `python main.py` (or future `python scripts/train_transformer.py`)
- `python train_sae.py` (or future `python scripts/train_sae.py`)
- `python scripts/sweep/run_one.py`
- `python scripts/sweep/run_one_probe.py`
- `python scripts/sweep/eval_probe.py`

## Definition of Done (Phase 1)

- New collaborator can run one tiny local command for each workflow from docs.
- Slurm jobs still work with both old and new script paths.
- Logs and checkpoints have descriptive names and known locations.
- Root directory is cleaner (fewer shell/notebook artifacts).

## Definition of Done (Phase 2)

- Core imports come from `src/rhm/*`.
- Legacy shims removed.
- README and quickstart reflect final paths only.
- Smoke tests pass.
