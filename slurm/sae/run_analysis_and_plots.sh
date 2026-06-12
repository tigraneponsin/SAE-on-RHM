#!/bin/bash
# =============================================================================
# Slurm script: run the SAE feature-latent eval AND the lambda/entropy plots
# for one sweep directory, in a single job.
#
# This is a superset of run_analysis.sh: it runs the same streaming eval, then
# additionally calls plot_lambda_metrics.py and plot_entropy_lambda.py. Outputs
# land in two fixed subfolders of the sweep dir:
#   <SWEEP_DIR>/analysis_files/   -- *.sae_eval.pt, sweep_metrics.csv, csvs
#   <SWEEP_DIR>/analysis_plots/   -- lambda_metrics.png, entropy_lambda_layer*.png
#
# run_analysis.sh and the plot scripts remain usable standalone; this script is
# an extra convenience (and the dependent step submitted by run_full_sweep.sh).
#
# Usage:
#   sbatch slurm/sae/run_analysis_and_plots.sh <SWEEP_DIR> \
#       [EVAL_SIZE] [BATCH_SIZE] [DEVICE] [DEDUPE] [XLIM_MIN] [XLIM_MAX] [ERR_TOL]
#
# Examples:
#   sbatch slurm/sae/run_analysis_and_plots.sh \
#     /work/pcsl/ponsin/Mean_Transformer/SAE/my_layer0_sweep
#
#   sbatch slurm/sae/run_analysis_and_plots.sh \
#     /work/pcsl/ponsin/Mean_Transformer/SAE/my_layer0_sweep \
#     65536 1024 cuda 1 1e-3 1e-1 0.01
# =============================================================================

# -- Job metadata -------------------------------------------------------------
#SBATCH --job-name=sae_analysis_plots
#SBATCH --chdir /home/ponsin
#SBATCH --account pcsl

# -- Resources ----------------------------------------------------------------
#SBATCH --time=01:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --mem=90G
#SBATCH --cpus-per-task=16

#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

set -euo pipefail

REPO_DIR=/home/ponsin/SAE-on-RHM

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch slurm/sae/run_analysis_and_plots.sh <SWEEP_DIR> [EVAL_SIZE] [BATCH_SIZE] [DEVICE] [DEDUPE] [XLIM_MIN] [XLIM_MAX] [ERR_TOL]"
    echo "  SWEEP_DIR : directory containing the SAE .pt checkpoints for the sweep"
    echo "  EVAL_SIZE : optional, default 32768"
    echo "  BATCH_SIZE: optional, default 512"
    echo "  DEVICE    : optional, default cuda"
    echo "  DEDUPE    : optional, 0 or 1, default 1"
    echo "  XLIM_MIN  : optional lambda-axis lower bound for the plots"
    echo "  XLIM_MAX  : optional lambda-axis upper bound for the plots"
    echo "  ERR_TOL   : optional error tolerance for entropy threshold line, default 0.01"
    echo ""
    echo "Runs the unified SAE streaming eval (scripts/sae_eval/run.py --with-all)"
    echo "into <SWEEP_DIR>/analysis_files, then plots lambda metrics and entropy"
    echo "into <SWEEP_DIR>/analysis_plots."
    exit 1
fi

SWEEP_DIR=$1
EVAL_SIZE=${2:-32768}
BATCH_SIZE=${3:-512}
DEVICE=${4:-cuda}
DEDUPE=${5:-1}
XLIM_MIN=${6:-}
XLIM_MAX=${7:-}
ERR_TOL=${8:-0.01}

if [[ ! -d "${SWEEP_DIR}" ]]; then
    echo "ERROR: SWEEP_DIR is not an existing directory: ${SWEEP_DIR}"
    exit 1
fi

ANALYSIS_DIR="${SWEEP_DIR}/analysis_files"
PLOTS_DIR="${SWEEP_DIR}/analysis_plots"
mkdir -p "${ANALYSIS_DIR}" "${PLOTS_DIR}"

# Checkpoints may live in <sweep>/sae_checkpoints/ (new layout) or flat in the
# sweep dir (old layout). Prefer the subfolder when it exists and holds *.pt.
CKPT_DIR="${SWEEP_DIR}"
if compgen -G "${SWEEP_DIR}/sae_checkpoints/*.pt" > /dev/null 2>&1; then
    CKPT_DIR="${SWEEP_DIR}/sae_checkpoints"
fi

# -- Environment setup --------------------------------------------------------
# Some conda activation scripts reference unset vars; disable nounset temporarily.
set +u
source /home/ponsin/miniconda3/etc/profile.d/conda.sh
conda activate pcsl
set -u

# -- Redirect logs next to the analysis outputs -------------------------------
exec > "${ANALYSIS_DIR}/analysis_and_plots.out" 2> "${ANALYSIS_DIR}/analysis_and_plots.err"

echo "======================================================================"
echo "Job:         ${SLURM_JOB_ID:-NA}"
echo "Node:        ${SLURMD_NODENAME:-NA}"
echo "SWEEP_DIR:   ${SWEEP_DIR}"
echo "CKPT_DIR:    ${CKPT_DIR}"
echo "ANALYSIS_DIR:${ANALYSIS_DIR}"
echo "PLOTS_DIR:   ${PLOTS_DIR}"
echo "EVAL_SIZE:   ${EVAL_SIZE}"
echo "BATCH_SIZE:  ${BATCH_SIZE}"
echo "DEVICE:      ${DEVICE}"
echo "DEDUPE:      ${DEDUPE}"
echo "XLIM:        ${XLIM_MIN:-<none>} ${XLIM_MAX:-<none>}"
echo "ERR_TOL:     ${ERR_TOL}"
echo "======================================================================"

START_EPOCH=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')
echo "START:       ${START_HUMAN}"

DEDUPE_ARGS=()
if [[ "${DEDUPE}" == "1" ]]; then
    DEDUPE_ARGS+=(--dedupe)
fi

# Optional lambda-axis limits shared by both plots.
XLIM_ARGS=()
if [[ -n "${XLIM_MIN}" && -n "${XLIM_MAX}" ]]; then
    XLIM_ARGS+=(--xlim "${XLIM_MIN}" "${XLIM_MAX}")
fi

# REPORT_NOTATION=1 relabels every plot in report style (SAE k 1-based, bottom-up
# RHM levels). Display only; no recompute. Default unset -> current behavior.
REPORT_NOTATION="${REPORT_NOTATION:-0}"
RN_ARGS=()
if [[ "${REPORT_NOTATION}" == "1" ]]; then
    RN_ARGS+=(--report-notation)
fi
echo "REPORT_NOTATION: ${REPORT_NOTATION}"

CSV_PATH="${ANALYSIS_DIR}/sweep_metrics.csv"

# -- Step 1: streaming eval (same call as run_analysis.sh) --------------------
echo ""
echo "[1/3] Streaming eval ${CKPT_DIR} -> ${ANALYSIS_DIR}"
set +e
srun python "${REPO_DIR}/scripts/sae_eval/run.py" \
    --sweep_dir "${CKPT_DIR}" \
    --out_dir "${ANALYSIS_DIR}" \
    --eval_size "${EVAL_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --with-all \
    --outcsv "${CSV_PATH}" \
    --per_position_csv "${ANALYSIS_DIR}/per_position_metrics.csv" \
    "${DEDUPE_ARGS[@]}"
EVAL_EXIT=$?
set -e

if [[ ${EVAL_EXIT} -ne 0 ]]; then
    echo "ERROR: eval step failed with exit code ${EVAL_EXIT}; skipping plots."
    exit ${EVAL_EXIT}
fi

# -- Step 2: lambda metrics plot ----------------------------------------------
echo ""
echo "[2/3] Lambda metrics plot -> ${PLOTS_DIR}/lambda_metrics.png"
set +e
python "${REPO_DIR}/scripts/sae_sweep/plot_lambda_metrics.py" \
    --csv "${CSV_PATH}" \
    --outfile "${PLOTS_DIR}/lambda_metrics.png" \
    ${XLIM_ARGS[@]+"${XLIM_ARGS[@]}"} \
    ${RN_ARGS[@]+"${RN_ARGS[@]}"}
LAMBDA_EXIT=$?
set -e

# -- Step 3: entropy vs lambda plots (one per layer) --------------------------
echo ""
echo "[3/3] Entropy-lambda plots -> ${PLOTS_DIR}/entropy_lambda_layer*.png"
set +e
python "${REPO_DIR}/scripts/sae_sweep/plot_entropy_lambda.py" \
    --artifacts_dir "${ANALYSIS_DIR}" \
    --outfile_prefix "${PLOTS_DIR}/entropy_lambda" \
    --err_tolerance "${ERR_TOL}" \
    ${XLIM_ARGS[@]+"${XLIM_ARGS[@]}"} \
    ${RN_ARGS[@]+"${RN_ARGS[@]}"}
ENTROPY_EXIT=$?
set -e

END_EPOCH=$(date +%s)
END_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')
ELAPSED_SEC=$((END_EPOCH - START_EPOCH))
ELAPSED_H=$((ELAPSED_SEC / 3600))
ELAPSED_M=$(((ELAPSED_SEC % 3600) / 60))
ELAPSED_S=$((ELAPSED_SEC % 60))

echo ""
echo "END:         ${END_HUMAN}"
printf 'ELAPSED:     %02d:%02d:%02d (%ds)\n' "${ELAPSED_H}" "${ELAPSED_M}" "${ELAPSED_S}" "${ELAPSED_SEC}"
echo "eval exit=${EVAL_EXIT}  lambda_plot exit=${LAMBDA_EXIT}  entropy_plot exit=${ENTROPY_EXIT}"

# Eval already succeeded above; surface a nonzero exit if any plot failed.
FINAL_EXIT=0
[[ ${LAMBDA_EXIT} -ne 0 ]] && FINAL_EXIT=${LAMBDA_EXIT}
[[ ${ENTROPY_EXIT} -ne 0 ]] && FINAL_EXIT=${ENTROPY_EXIT}
echo "Analysis+plots finished with exit code ${FINAL_EXIT}."
exit ${FINAL_EXIT}
