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
# Optional: override output path (default: auto-generated next to transformer)
# OUTNAME=/path/to/probe_result.pt
# =============================================================================

#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# -- Environment setup --------------------------------------------------------
source /home/ponsin/miniconda3/etc/profile.d/conda.sh
conda activate pcsl

echo "======================================================================"
echo "Job:           ${SLURM_JOB_ID}"
echo "Node:          ${SLURMD_NODENAME}"
echo "TRAIN_OUTPUT:  ${TRAIN_OUTPUT}"
echo "LAYER:         ${LAYER}"
echo "TOKEN_IDX:     ${TOKEN_IDX}"
echo "======================================================================"

CMD="srun python ${REPO_DIR}/sae_sweep/run_one_probe.py \
    --train_output ${TRAIN_OUTPUT} \
    --layer ${LAYER} \
    --token_idx ${TOKEN_IDX} \
    --probe_train_size 8192 \
    --probe_eval_size 4096 \
    --probe_steps 2000 \
    --probe_lr 1e-3"

if [ -n "${OUTNAME}" ]; then
    CMD="${CMD} --outname ${OUTNAME}"
fi

echo "Command: ${CMD}"
eval ${CMD}

EXIT_CODE=$?
echo "Probe training finished with exit code ${EXIT_CODE}."
exit ${EXIT_CODE}
