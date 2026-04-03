#!/bin/bash
# =============================================================================
# Slurm script to train a linear probe on a transformer checkpoint.
#
# Usage:
#   sbatch sae_sweep/run_probe.sh
#
# This trains a linear probe at the specified (layer, token) position to
# predict the intermediate RHM latent at level L-1-layer.
# =============================================================================

# -- Job metadata -------------------------------------------------------------
#SBATCH --job-name=probe_train
#SBATCH --chdir /work/pcsl/ponsin/Probes
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
LAYER=1
TOKEN_IDX=5
REPO_DIR=/home/ponsin/SAE-on-RHM
PROBES_DIR=/work/pcsl/ponsin/Probes
PROBE_TRAIN_SIZE=8192
PROBE_EVAL_SIZE=4096
PROBE_STEPS=5000
PROBE_LR=1e-3
# Optional: override output path (default: ${PROBES_DIR}/...)
# OUTNAME=${PROBES_DIR}/probe_result.pt
# =============================================================================

#SBATCH -o /work/pcsl/ponsin/Probes/%x_%j.out
#SBATCH -e /work/pcsl/ponsin/Probes/%x_%j.err

# -- Environment setup --------------------------------------------------------
source /home/ponsin/miniconda3/etc/profile.d/conda.sh
conda activate pcsl

mkdir -p "${PROBES_DIR}"

TRF_TAG=$(basename "${TRAIN_OUTPUT}" .pt)
TRF_TAG=$(echo "${TRF_TAG}" | sed 's/[^A-Za-z0-9._-]/_/g')
RUN_TAG="probe_trf-${TRF_TAG}_layer-${LAYER}_tok-${TOKEN_IDX}_trN-${PROBE_TRAIN_SIZE}_evN-${PROBE_EVAL_SIZE}_steps-${PROBE_STEPS}_lr-${PROBE_LR}_job-${SLURM_JOB_ID}"
LOG_OUT="${PROBES_DIR}/${RUN_TAG}.out"
LOG_ERR="${PROBES_DIR}/${RUN_TAG}.err"

# Keep the default Slurm logs, but also write parameter-rich logs.
exec > >(tee -a "${LOG_OUT}") 2> >(tee -a "${LOG_ERR}" >&2)

echo "======================================================================"
echo "Job:           ${SLURM_JOB_ID}"
echo "Node:          ${SLURMD_NODENAME}"
echo "TRAIN_OUTPUT:  ${TRAIN_OUTPUT}"
echo "LAYER:         ${LAYER}"
echo "TOKEN_IDX:     ${TOKEN_IDX}"
echo "PROBES_DIR:    ${PROBES_DIR}"
echo "RUN_TAG:       ${RUN_TAG}"
echo "LOG_OUT:       ${LOG_OUT}"
echo "LOG_ERR:       ${LOG_ERR}"
echo "======================================================================"

CMD="srun python ${REPO_DIR}/sae_sweep/run_one_probe.py \
    --train_output ${TRAIN_OUTPUT} \
    --layer ${LAYER} \
    --token_idx ${TOKEN_IDX} \
    --probe_train_size ${PROBE_TRAIN_SIZE} \
    --probe_eval_size ${PROBE_EVAL_SIZE} \
    --probe_steps ${PROBE_STEPS} \
    --probe_lr ${PROBE_LR}"

if [ -n "${OUTNAME}" ]; then
    CMD="${CMD} --outname ${OUTNAME}"
else
    OUTNAME="${PROBES_DIR}/${RUN_TAG}.pt"
    CMD="${CMD} --outname ${OUTNAME}"
fi

echo "Command: ${CMD}"
eval ${CMD}

EXIT_CODE=$?
echo "Probe training finished with exit code ${EXIT_CODE}."
exit ${EXIT_CODE}
