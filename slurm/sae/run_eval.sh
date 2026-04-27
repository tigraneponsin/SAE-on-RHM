#!/bin/bash
# =============================================================================
# Slurm script to run scripts/sae_eval/run.py on a completed SAE sweep,
# emitting the sweep-summary CSV (scalar aggregates + classification impact +
# entropy aggregates) alongside per-checkpoint *.sae_eval.pt artifacts.
#
# Usage:
#   sbatch slurm/sae/run_eval.sh
# =============================================================================

# -- Job metadata -------------------------------------------------------------
#SBATCH --job-name=sae_eval_sweep
#SBATCH --chdir /home/ponsin
#SBATCH --account pcsl

# ── Resources ────────────────────────────────────────────────────────────────
#SBATCH --time=01:00:00
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# =============================================================================
# USER: set these paths before submitting
# =============================================================================
SWEEP_DIR=/work/pcsl/ponsin/Mean_Transformer/Small_SAE/latent_dim_4*512/v_16_L_3_m_4_wdecay_0.0001/sweep_alltokens_layer2_lambda1_zoom
REPO_DIR=/home/ponsin/SAE-on-RHM
# =============================================================================

#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# ── Environment setup ────────────────────────────────────────────────────────
source /home/ponsin/miniconda3/etc/profile.d/conda.sh
conda activate pcsl

# -- Redirect logs next to the CSV output -------------------------------------
exec > "${SWEEP_DIR}/sae_eval_sweep.out" 2> "${SWEEP_DIR}/sae_eval_sweep.err"

echo "======================================================================"
echo "Job:        ${SLURM_JOB_ID}"
echo "Node:       ${SLURMD_NODENAME}"
echo "SWEEP_DIR:  ${SWEEP_DIR}"
echo "======================================================================"

srun python "${REPO_DIR}/scripts/sae_eval/run.py" \
    --sweep_dir "${SWEEP_DIR}" \
    --out_dir "${SWEEP_DIR}/sae_eval_artifacts" \
    --outcsv "${SWEEP_DIR}/eval_results.csv" \
    --per_position_csv "${SWEEP_DIR}/eval_results_per_position.csv" \
    --with-all

EXIT_CODE=$?
echo "Eval finished with exit code ${EXIT_CODE}."
exit ${EXIT_CODE}
