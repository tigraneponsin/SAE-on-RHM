#!/bin/bash
# =============================================================================
# Slurm script to train a linear probe on a transformer checkpoint.
#
# Usage:
#   sbatch probe_train/run_probe.sh
#
# Trains a probe at the specified (layer, token) pair and saves it next to
# the transformer checkpoint as:
#   probe_layer{LAYER}_tok{TOKEN_IDX}__{transformer_stem}.pt
# =============================================================================

# -- Job metadata -------------------------------------------------------------
#SBATCH --job-name=probe_train
#SBATCH --chdir /home/ponsin
#SBATCH --account pcsl

# -- Resources ----------------------------------------------------------------
#SBATCH --time=00:30:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# =============================================================================
# USER: set these before submitting
# =============================================================================
TRAIN_OUTPUT=/work/pcsl/ponsin/Mean_Transformer/Transformer_for_SAE/v_16_L_3_m_4/RESULT_TRFCLASS_v_16_L_3_m=4_P_12160_0_emb_512_h_8_lr_5e-3_dropout_0.1.pkl.pt
LAYER=0
TOKEN_IDX=0
REPO_DIR=/home/ponsin/SAE-on-RHM
PROBE_TRAIN_SIZE=8192
PROBE_EVAL_SIZE=4096
PROBE_STEPS=2000
PROBE_LR=1e-3
# Optional: override output path (default: next to transformer checkpoint)
# OUTNAME=/path/to/probe_result.pt
# =============================================================================

TRF_STEM=$(basename "${TRAIN_OUTPUT}" .pt)
LOG_DIR=$(dirname "${TRAIN_OUTPUT}")

#SBATCH -o /home/ponsin/probe_train_%j.out
#SBATCH -e /home/ponsin/probe_train_%j.err

# -- Environment setup --------------------------------------------------------
source /home/ponsin/miniconda3/etc/profile.d/conda.sh
conda activate pcsl

# -- Redirect logs next to the transformer checkpoint ------------------------
exec > "${LOG_DIR}/probe_layer${LAYER}_tok${TOKEN_IDX}__${TRF_STEM}.out" \
     2> "${LOG_DIR}/probe_layer${LAYER}_tok${TOKEN_IDX}__${TRF_STEM}.err"

echo "======================================================================"
echo "Job:           ${SLURM_JOB_ID}"
echo "Node:          ${SLURMD_NODENAME}"
echo "TRAIN_OUTPUT:  ${TRAIN_OUTPUT}"
echo "LAYER:         ${LAYER}"
echo "TOKEN_IDX:     ${TOKEN_IDX}"
echo "Output probe:  probe_layer${LAYER}_tok${TOKEN_IDX}__${TRF_STEM}.pt"
echo "======================================================================"

CMD="srun python ${REPO_DIR}/probe_train/run_one_probe.py \
    --train_output ${TRAIN_OUTPUT} \
    --layer ${LAYER} \
    --token_idx ${TOKEN_IDX} \
    --probe_train_size ${PROBE_TRAIN_SIZE} \
    --probe_eval_size ${PROBE_EVAL_SIZE} \
    --probe_steps ${PROBE_STEPS} \
    --probe_lr ${PROBE_LR}"

if [ -n "${OUTNAME}" ]; then
    CMD="${CMD} --outname ${OUTNAME}"
fi

echo "Command: ${CMD}"
eval ${CMD}

EXIT_CODE=$?
echo "Probe training finished with exit code ${EXIT_CODE}."
exit ${EXIT_CODE}
