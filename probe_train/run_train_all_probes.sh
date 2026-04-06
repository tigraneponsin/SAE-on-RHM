#!/bin/bash
# =============================================================================
# Slurm script to train linear probes for all (layer, token) pairs of a
# transformer checkpoint.
#
# Usage:
#   sbatch probe_train/run_train_all_probes.sh
#
# Probes are saved next to the transformer checkpoint as:
#   probe_layer{layer}_tok{token_idx}__{transformer_stem}.pt
# Already-existing probe files are skipped (safe to rerun).
# =============================================================================

# -- Job metadata -------------------------------------------------------------
#SBATCH --job-name=train_all_probes
#SBATCH --chdir /home/ponsin
#SBATCH --account pcsl

# -- Resources ----------------------------------------------------------------
#SBATCH --time=02:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# =============================================================================
# USER: set these before submitting
# =============================================================================
TRAIN_OUTPUT=/work/pcsl/ponsin/Mean_Transformer/Transformer_for_SAE/v_16_L_3_m_4/RESULT_TRFCLASS_v_16_L_3_m=4_P_12160_0_emb_512_h_8_lr_5e-3_dropout_0.1.pkl.pt
REPO_DIR=/home/ponsin/SAE-on-RHM
PROBE_TRAIN_SIZE=8192
PROBE_EVAL_SIZE=4096
PROBE_STEPS=2000
PROBE_LR=1e-3
# =============================================================================

TRF_STEM=$(basename "${TRAIN_OUTPUT}" .pt)
LOG_DIR=$(dirname "${TRAIN_OUTPUT}")

#SBATCH -o /home/ponsin/train_all_probes_%j.out
#SBATCH -e /home/ponsin/train_all_probes_%j.err

# -- Environment setup --------------------------------------------------------
source /home/ponsin/miniconda3/etc/profile.d/conda.sh
conda activate pcsl

# -- Redirect logs next to the transformer checkpoint ------------------------
exec > "${LOG_DIR}/train_all_probes__${TRF_STEM}.out" \
     2> "${LOG_DIR}/train_all_probes__${TRF_STEM}.err"

echo "======================================================================"
echo "Job:           ${SLURM_JOB_ID}"
echo "Node:          ${SLURMD_NODENAME}"
echo "TRAIN_OUTPUT:  ${TRAIN_OUTPUT}"
echo "PROBE_STEPS:   ${PROBE_STEPS}"
echo "PROBE_LR:      ${PROBE_LR}"
echo "======================================================================"

srun python "${REPO_DIR}/probe_train/train_all_probes.py" \
    --train_output "${TRAIN_OUTPUT}" \
    --probe_train_size "${PROBE_TRAIN_SIZE}" \
    --probe_eval_size "${PROBE_EVAL_SIZE}" \
    --probe_steps "${PROBE_STEPS}" \
    --probe_lr "${PROBE_LR}"

EXIT_CODE=$?
echo "train_all_probes finished with exit code ${EXIT_CODE}."
exit ${EXIT_CODE}
