#!/bin/bash
# =============================================================================
# Slurm script to evaluate linear probes on SAE-reconstructed activations.
#
# Usage:
#   sbatch sae_sweep/run_eval_probe.sh
#
# For each SAE checkpoint in SWEEP_DIR, this:
#   1. Trains a linear probe on clean activations at the SAE's (layer, token).
#   2. Evaluates the probe on clean eval data (upper bound).
#   3. Evaluates the same probe on SAE-reconstructed eval data.
#   4. Reports clean_acc vs recon_acc and the accuracy drop.
# =============================================================================

# -- Job metadata -------------------------------------------------------------
#SBATCH --job-name=eval_probe
#SBATCH --chdir /home/ponsin
#SBATCH --account pcsl

# -- Resources ----------------------------------------------------------------
#SBATCH --time=01:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --mem=90G
#SBATCH --cpus-per-task=16

# =============================================================================
# USER: set these paths before submitting
# =============================================================================
SWEEP_DIR=/work/pcsl/ponsin/Mean_Transformer/SAE/2nd_generation/sweep_onetok0_layer0_lambda1_nowarm_noscale
REPO_DIR=/home/ponsin/SAE-on-RHM
# =============================================================================

#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# -- Environment setup --------------------------------------------------------
source /home/ponsin/miniconda3/etc/profile.d/conda.sh
conda activate pcsl

# -- Redirect logs next to the CSV output -------------------------------------
exec > "${SWEEP_DIR}/eval_probe.out" 2> "${SWEEP_DIR}/eval_probe.err"

echo "======================================================================"
echo "Job:        ${SLURM_JOB_ID}"
echo "Node:       ${SLURMD_NODENAME}"
echo "SWEEP_DIR:  ${SWEEP_DIR}"
echo "======================================================================"

srun python "${REPO_DIR}/sae_sweep/eval_probe.py" \
    --sweep_dir "${SWEEP_DIR}" \
    --probe_train_size 8192 \
    --probe_eval_size 4096 \
    --probe_steps 2000 \
    --probe_lr 1e-3 \
    --outcsv "${SWEEP_DIR}/probe_results.csv"

EXIT_CODE=$?
echo "Eval probe finished with exit code ${EXIT_CODE}."
exit ${EXIT_CODE}
