#!/bin/bash
# =============================================================================
# Slurm script to run direct SAE feature-latent analysis.
#
# Usage:
#   sbatch slurm/sae/run_analysis.sh <SWEEP_DIR_OR_CKPT> <OUT_DIR> [EVAL_SIZE] [BATCH_SIZE] [DEVICE] [DEDUPE]
#
# Examples:
#   sbatch slurm/sae/run_analysis.sh \
#     /work/pcsl/ponsin/Mean_Transformer/SAE/my_sweep \
#     /work/pcsl/ponsin/Mean_Transformer/SAE/my_sweep/direct_analysis
#
#   sbatch slurm/sae/run_analysis.sh \
#     /work/pcsl/ponsin/Mean_Transformer/SAE/my_sweep/sae_layer0_tok0.pt \
#     /work/pcsl/ponsin/Mean_Transformer/SAE/single_ckpt_analysis
#
#   sbatch slurm/sae/run_analysis.sh \
#     /work/pcsl/ponsin/Mean_Transformer/SAE/my_sweep \
#     /work/pcsl/ponsin/Mean_Transformer/SAE/my_sweep/direct_analysis \
#     65536 1024 cuda 1
# =============================================================================

# -- Job metadata -------------------------------------------------------------
#SBATCH --job-name=sae_analysis
#SBATCH --chdir /home/ponsin
#SBATCH --account pcsl

# -- Resources ----------------------------------------------------------------
#SBATCH --time=00:10:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --mem=90G
#SBATCH --cpus-per-task=16

#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

set -euo pipefail

REPO_DIR=/home/ponsin/SAE-on-RHM

if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch slurm/sae/run_analysis.sh <SWEEP_DIR_OR_CKPT> <OUT_DIR> [EVAL_SIZE] [BATCH_SIZE] [DEVICE] [DEDUPE]"
    echo "  SWEEP_DIR_OR_CKPT: either"
    echo "                     - a directory containing SAE .pt checkpoints"
    echo "                     - a single SAE checkpoint .pt file"
    echo "  OUT_DIR   : destination directory for *.sae_eval.pt"
    echo "  EVAL_SIZE : optional, default 32768"
    echo "  BATCH_SIZE: optional, default 512"
    echo "  DEVICE    : optional, default cuda"
    echo "  DEDUPE    : optional, 0 or 1, default 1"
    echo ""
    echo "Runs the unified SAE streaming eval (scripts/sae_eval/run.py) with every"
    echo "--with-* flag enabled; produces *.sae_eval.pt per checkpoint containing"
    echo "scalar aggregates, per-position tensors, per-feature tensors, conditional"
    echo "stats, joint-fire entropy, and classification impact."
    exit 1
fi

INPUT_PATH=$1
OUT_DIR=$2
EVAL_SIZE=${3:-32768}
BATCH_SIZE=${4:-512}
DEVICE=${5:-cuda}
DEDUPE=${6:-1}

ANALYZE_TARGET_ARGS=()
TARGET_KIND=""

if [[ -d "${INPUT_PATH}" ]]; then
    TARGET_KIND="sweep_dir"
    ANALYZE_TARGET_ARGS+=(--sweep_dir "${INPUT_PATH}")
elif [[ -f "${INPUT_PATH}" ]]; then
    TARGET_KIND="ckpt"
    ANALYZE_TARGET_ARGS+=(--ckpt "${INPUT_PATH}")
else
    echo "ERROR: first argument must be an existing directory or file: ${INPUT_PATH}"
    exit 1
fi

mkdir -p "${OUT_DIR}"

# -- Environment setup --------------------------------------------------------
# Some conda activation scripts reference unset vars; disable nounset temporarily.
set +u
source /home/ponsin/miniconda3/etc/profile.d/conda.sh
conda activate pcsl
set -u

# -- Redirect logs next to the analysis outputs -------------------------------
exec > "${OUT_DIR}/sae_eval.out" 2> "${OUT_DIR}/sae_eval.err"

echo "======================================================================"
echo "Job:        ${SLURM_JOB_ID}"
echo "Node:       ${SLURMD_NODENAME}"
echo "TARGET_KIND:${TARGET_KIND}"
echo "INPUT_PATH: ${INPUT_PATH}"
echo "OUT_DIR:    ${OUT_DIR}"
echo "EVAL_SIZE:  ${EVAL_SIZE}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "DEVICE:     ${DEVICE}"
echo "DEDUPE:     ${DEDUPE}"
echo "======================================================================"

START_EPOCH=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')
echo "START:      ${START_HUMAN}"

DEDUPE_ARGS=()
if [[ "${DEDUPE}" == "1" ]]; then
    DEDUPE_ARGS+=(--dedupe)
fi

set +e
srun python "${REPO_DIR}/scripts/sae_eval/run.py" \
    "${ANALYZE_TARGET_ARGS[@]}" \
    --out_dir "${OUT_DIR}" \
    --eval_size "${EVAL_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --with-all \
    --outcsv "${OUT_DIR}/sweep_metrics.csv" \
    "${DEDUPE_ARGS[@]}"
EXIT_CODE=$?
set -e

END_EPOCH=$(date +%s)
END_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')
ELAPSED_SEC=$((END_EPOCH - START_EPOCH))
ELAPSED_H=$((ELAPSED_SEC / 3600))
ELAPSED_M=$(((ELAPSED_SEC % 3600) / 60))
ELAPSED_S=$((ELAPSED_SEC % 60))

echo "END:        ${END_HUMAN}"
printf 'ELAPSED:    %02d:%02d:%02d (%ds)\n' "${ELAPSED_H}" "${ELAPSED_M}" "${ELAPSED_S}" "${ELAPSED_SEC}"
echo "Analysis finished with exit code ${EXIT_CODE}."
exit ${EXIT_CODE}