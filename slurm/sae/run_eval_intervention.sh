#!/bin/bash
# =============================================================================
# Slurm script to run eval_sweep.py with intervention-specific flags:
#   --subset_positions  : restrict lambda metrics to a leaf position subset
#   --per_position_csv  : emit per-position breakdown (one row per ckpt x pos)
#
# Outputs land NEXT TO the sweep dir so they never collide with run_eval.sh:
#   ${SWEEP_DIR}/eval_results_intervention.csv
#   ${SWEEP_DIR}/per_position.csv
#   ${SWEEP_DIR}/eval_intervention.{out,err}
#
# Usage:
#   sbatch slurm/sae/run_eval_intervention.sh
#   (edit the USER section below before submitting)
# =============================================================================

# -- Job metadata -------------------------------------------------------------
#SBATCH --job-name=eval_sweep_intervention
#SBATCH --chdir /home/ponsin
#SBATCH --account pcsl

# -- Resources ----------------------------------------------------------------
#SBATCH --time=01:00:00
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# =============================================================================
# USER: set these variables before submitting
# =============================================================================
SWEEP_DIR=/work/pcsl/ponsin/Mean_Transformer/SAE/v_16_L_3_m_4_wdecay_0.0001/sweep_alltokens_layer0_lambda1_zoom
# Comma-separated leaf positions to compute subset_* metrics on (all_tokens SAEs only).
# Leave empty ("") to skip subset aggregation.
SUBSET_POSITIONS="0,2,4,6"
REPO_DIR=/home/ponsin/SAE-on-RHM
# Optional overrides (leave defaults to match run_eval.sh)
EVAL_SIZE=32768
BATCH_SIZE=256
# =============================================================================

# -- Environment setup --------------------------------------------------------
set +u
source /home/ponsin/miniconda3/etc/profile.d/conda.sh
conda activate pcsl
set -u

# -- Redirect logs next to sweep outputs --------------------------------------
exec > "${SWEEP_DIR}/eval_intervention.out" 2> "${SWEEP_DIR}/eval_intervention.err"

echo "======================================================================"
echo "Job:             ${SLURM_JOB_ID}"
echo "Node:            ${SLURMD_NODENAME}"
echo "SWEEP_DIR:       ${SWEEP_DIR}"
echo "SUBSET_POSITIONS:${SUBSET_POSITIONS}"
echo "EVAL_SIZE:       ${EVAL_SIZE}"
echo "BATCH_SIZE:      ${BATCH_SIZE}"
echo "======================================================================"

SUBSET_ARGS=()
if [[ -n "${SUBSET_POSITIONS}" ]]; then
    SUBSET_ARGS+=(--subset_positions "${SUBSET_POSITIONS}")
fi

srun python "${REPO_DIR}/scripts/sae_sweep/eval_sweep.py" \
    --sweep_dir "${SWEEP_DIR}" \
    --eval_size "${EVAL_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --outcsv "${SWEEP_DIR}/eval_results_intervention.csv" \
    --per_position_csv "${SWEEP_DIR}/per_position.csv" \
    "${SUBSET_ARGS[@]}"

EXIT_CODE=$?
echo "Eval finished with exit code ${EXIT_CODE}."
exit ${EXIT_CODE}
