#!/bin/bash
# =============================================================================
# Slurm script to run eval_sweep.py on a completed SAE sweep.
#
# Usage:
#   sbatch sae_sweep/run_eval.sh
# =============================================================================

# ── Job metadata ─────────────────────────────────────────────────────────────
#SBATCH --job-name=eval_sweep
#SBATCH --chdir /home/ponsin
#SBATCH --account pcsl

# ── Resources ────────────────────────────────────────────────────────────────
#SBATCH --time=01:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --mem=90G
#SBATCH --cpus-per-task=16

# =============================================================================
# USER: set these paths before submitting
# =============================================================================
SWEEP_DIR=/work/pcsl/ponsin/Mean_Transformer/SAE/2nd_generation/sweep_onetok0_layer0_lambda1_nowarm_lr5e-5
REPO_DIR=/home/ponsin/SAE-on-RHM
# =============================================================================

#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# ── Environment setup ────────────────────────────────────────────────────────
source /home/ponsin/miniconda3/etc/profile.d/conda.sh
conda activate pcsl

# ── Redirect logs next to the CSV output ─────────────────────────────────────
exec > "${SWEEP_DIR}/eval_sweep.out" 2> "${SWEEP_DIR}/eval_sweep.err"

echo "======================================================================"
echo "Job:        ${SLURM_JOB_ID}"
echo "Node:       ${SLURMD_NODENAME}"
echo "SWEEP_DIR:  ${SWEEP_DIR}"
echo "======================================================================"

srun python "${REPO_DIR}/sae_sweep/eval_sweep.py" \
    --sweep_dir "${SWEEP_DIR}" \
    --outcsv "${SWEEP_DIR}/eval_results.csv"

EXIT_CODE=$?
echo "Eval finished with exit code ${EXIT_CODE}."
exit ${EXIT_CODE}
