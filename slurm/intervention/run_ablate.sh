#!/bin/bash
# =============================================================================
# Slurm script to run ablate_tokens.py (direct residual intervention).
#
# Usage:
#   sbatch slurm/intervention/run_ablate.sh <TRAIN_OUTPUT> <EXPERIMENTS_JSON> <OUT_DIR> [EVAL_SIZE] [BATCH_SIZE]
#
# Examples:
#   sbatch slurm/intervention/run_ablate.sh \
#     /work/pcsl/ponsin/Mean_Transformer/Transformer_for_SAE/v_16_L_3_m_4_wdecay_0.0001/checkpoint.pt \
#     /home/ponsin/SAE-on-RHM/scripts/intervention/experiments_dead_token.json \
#     /work/pcsl/ponsin/Mean_Transformer/Intervention/v_16_L_3_m_4_wdecay_0.0001/ablation
#
#   sbatch slurm/intervention/run_ablate.sh \
#     /work/pcsl/ponsin/.../checkpoint.pt \
#     /home/ponsin/SAE-on-RHM/scripts/intervention/my_experiments.json \
#     /work/pcsl/ponsin/.../ablation \
#     65536 512
# =============================================================================

# -- Job metadata -------------------------------------------------------------
#SBATCH --job-name=ablate_tokens
#SBATCH --chdir /home/ponsin
#SBATCH --account pcsl

# -- Resources ----------------------------------------------------------------
#SBATCH --time=00:30:00
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

set -euo pipefail

REPO_DIR=/home/ponsin/SAE-on-RHM

if [[ $# -lt 3 ]]; then
    echo "Usage: sbatch slurm/intervention/run_ablate.sh <TRAIN_OUTPUT> <EXPERIMENTS_JSON> <OUT_DIR> [EVAL_SIZE] [BATCH_SIZE]"
    echo "  TRAIN_OUTPUT     : path to trained transformer .pt (output from main.py)"
    echo "  EXPERIMENTS_JSON : path to experiments JSON file"
    echo "  OUT_DIR          : directory for ablate.csv, norms.csv, and logs"
    echo "  EVAL_SIZE        : optional, default 32768"
    echo "  BATCH_SIZE       : optional, default 256"
    exit 1
fi

TRAIN_OUTPUT=$1
EXPERIMENTS_JSON=$2
OUT_DIR=$3
EVAL_SIZE=${4:-32768}
BATCH_SIZE=${5:-256}

mkdir -p "${OUT_DIR}"

# -- Environment setup --------------------------------------------------------
set +u
source /home/ponsin/miniconda3/etc/profile.d/conda.sh
conda activate pcsl
set -u

# -- Redirect logs next to outputs --------------------------------------------
exec > "${OUT_DIR}/ablate_tokens.out" 2> "${OUT_DIR}/ablate_tokens.err"

echo "======================================================================"
echo "Job:          ${SLURM_JOB_ID}"
echo "Node:         ${SLURMD_NODENAME}"
echo "TRAIN_OUTPUT: ${TRAIN_OUTPUT}"
echo "EXPERIMENTS:  ${EXPERIMENTS_JSON}"
echo "OUT_DIR:      ${OUT_DIR}"
echo "EVAL_SIZE:    ${EVAL_SIZE}"
echo "BATCH_SIZE:   ${BATCH_SIZE}"
echo "======================================================================"

START_EPOCH=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')
echo "START:        ${START_HUMAN}"

set +e
srun python "${REPO_DIR}/scripts/intervention/ablate_tokens.py" \
    --train_output "${TRAIN_OUTPUT}" \
    --experiments "${EXPERIMENTS_JSON}" \
    --eval_size "${EVAL_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --model_variant best \
    --outcsv "${OUT_DIR}/ablate.csv" \
    --norms_csv "${OUT_DIR}/norms.csv"
EXIT_CODE=$?
set -e

END_EPOCH=$(date +%s)
END_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')
ELAPSED_SEC=$((END_EPOCH - START_EPOCH))
ELAPSED_H=$((ELAPSED_SEC / 3600))
ELAPSED_M=$(((ELAPSED_SEC % 3600) / 60))
ELAPSED_S=$((ELAPSED_SEC % 60))

echo "END:          ${END_HUMAN}"
printf 'ELAPSED:      %02d:%02d:%02d (%ds)\n' "${ELAPSED_H}" "${ELAPSED_M}" "${ELAPSED_S}" "${ELAPSED_SEC}"
echo "Ablation finished with exit code ${EXIT_CODE}."
exit ${EXIT_CODE}
