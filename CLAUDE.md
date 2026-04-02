## Before Writing Code
- Read all relevant files first. Never edit blind.
- Understand the full requirement before writing anything.

## While Writing Code
- Test after writing. Never leave code untested.
- Fix errors before moving on. Never skip failures.
- Prefer editing over rewriting whole files.
- Simplest working solution. No over-engineering.

## Before Declaring Done
- Run the code one final time to confirm it works.
- Never declare done without a passing test.

## Output
- No sycophantic openers or closing fluff.
- No em dashes, smart quotes, or Unicode. ASCII only.
- Be concise. If unsure, say so. Never guess.

## Override Rule
User instructions always override this file.

## Project Context
- This is a research project training Sparse Autoencoders (SAEs) on a Random Hierarchy Model (RHM).
- Entry points: `train_sae.py` (SAE training), `main.py` (transformer training).
- Sweep workflow: edit `sae_sweep/generate_sweep.py` -> run it -> `sbatch sae_sweep/run_sweep.sh`.
- Jobs run on Slurm (h100 partition, account pcsl). Never run heavy training locally.
- Checkpoints and sweep configs live under paths set in the sweep scripts -- check them before assuming locations.
