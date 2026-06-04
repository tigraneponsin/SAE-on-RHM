#!/bin/bash
# =============================================================================
# One-shot SAE sweep: train -> eval -> plots, as a single submission.
#
# This is a thin SUBMITTER (run it on the login node, not via sbatch). It:
#   1. submits the training array (slurm/sae/run_sweep.sh) sized to the sweep,
#   2. submits a dependent (afterok) analysis+plots job
#      (slurm/sae/run_analysis_and_plots.sh) on the same sweep dir.
#
# The training array runs in parallel exactly as before; the eval+plots step
# only starts once every array task has finished successfully. All the existing
# entry points (run_sweep.sh, run_analysis.sh, the plot scripts) still work
# standalone -- this just chains them.
#
# Usage:
#   bash slurm/sae/run_full_sweep.sh --sweep_configs /path/to/sweep_configs.json
#   bash slurm/sae/run_full_sweep.sh --sweep_dir /path/to/sweep_dir
#
# Optional eval/plot knobs (forwarded to run_analysis_and_plots.sh):
#   --eval_size N     [32768]
#   --batch_size N    [512]
#   --device D        [cuda]
#   --dedupe 0|1      [1]
#   --xlim MIN MAX    [none]    lambda-axis limits shared by both plots
#   --err_tolerance F [0.01]    entropy threshold tolerance
#
# Example:
#   bash slurm/sae/run_full_sweep.sh \
#       --sweep_configs /work/.../my_sweep/sweep_configs.json \
#       --eval_size 65536 --xlim 1e-3 1e-1
# =============================================================================

set -euo pipefail

REPO_DIR=/home/ponsin/SAE-on-RHM
RUN_SWEEP="${REPO_DIR}/slurm/sae/run_sweep.sh"
RUN_ANALYSIS_PLOTS="${REPO_DIR}/slurm/sae/run_analysis_and_plots.sh"

# -- Defaults -----------------------------------------------------------------
SWEEP_CONFIGS=""
SWEEP_DIR=""
EVAL_SIZE=32768
BATCH_SIZE=512
DEVICE=cuda
DEDUPE=1
XLIM_MIN=""
XLIM_MAX=""
ERR_TOL=0.01

usage() {
    cat <<'EOF'
Usage: bash slurm/sae/run_full_sweep.sh (--sweep_configs PATH | --sweep_dir PATH)
         [--eval_size N] [--batch_size N] [--device D] [--dedupe 0|1]
         [--xlim MIN MAX] [--err_tolerance F]

Either --sweep_configs (path to sweep_configs.json) or --sweep_dir (the dir that
contains sweep_configs.json) is required.
EOF
}

# -- Argument parsing ---------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sweep_configs)  SWEEP_CONFIGS=$2; shift 2 ;;
        --sweep_dir)      SWEEP_DIR=$2;     shift 2 ;;
        --eval_size)      EVAL_SIZE=$2;     shift 2 ;;
        --batch_size)     BATCH_SIZE=$2;    shift 2 ;;
        --device)         DEVICE=$2;        shift 2 ;;
        --dedupe)         DEDUPE=$2;        shift 2 ;;
        --xlim)           XLIM_MIN=$2; XLIM_MAX=$3; shift 3 ;;
        --err_tolerance)  ERR_TOL=$2;       shift 2 ;;
        -h|--help)        usage; exit 0 ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

# -- Resolve sweep_configs / sweep_dir ----------------------------------------
if [[ -z "${SWEEP_CONFIGS}" && -z "${SWEEP_DIR}" ]]; then
    echo "ERROR: provide --sweep_configs or --sweep_dir" >&2
    usage >&2
    exit 1
fi

if [[ -z "${SWEEP_CONFIGS}" ]]; then
    SWEEP_CONFIGS="${SWEEP_DIR%/}/sweep_configs.json"
fi
if [[ -z "${SWEEP_DIR}" ]]; then
    SWEEP_DIR=$(dirname "${SWEEP_CONFIGS}")
fi

if [[ ! -f "${SWEEP_CONFIGS}" ]]; then
    echo "ERROR: sweep_configs.json not found: ${SWEEP_CONFIGS}" >&2
    exit 1
fi

# -- Determine array size from the config file --------------------------------
N=$(python3 -c "
import json, sys
configs = json.load(open('${SWEEP_CONFIGS}'))
print(len(configs))
")
if [[ -z "${N}" || "${N}" -lt 1 ]]; then
    echo "ERROR: sweep_configs.json has no configs: ${SWEEP_CONFIGS}" >&2
    exit 1
fi
ARRAY_RANGE="0-$((N - 1))"

echo "======================================================================"
echo "Full sweep submission"
echo "  SWEEP_CONFIGS: ${SWEEP_CONFIGS}"
echo "  SWEEP_DIR:     ${SWEEP_DIR}"
echo "  configs (N):   ${N}  -> --array=${ARRAY_RANGE}"
echo "  eval_size=${EVAL_SIZE} batch_size=${BATCH_SIZE} device=${DEVICE} dedupe=${DEDUPE}"
echo "  xlim=${XLIM_MIN:-<none>} ${XLIM_MAX:-<none>}  err_tolerance=${ERR_TOL}"
echo "======================================================================"

# -- Step 1: submit the training array ----------------------------------------
# Pass SWEEP_CONFIGS through the environment (run_sweep.sh honors the override)
# and set the array range on the CLI (overrides the #SBATCH --array directive).
TRAIN_JOB_ID=$(sbatch --parsable \
    --array="${ARRAY_RANGE}" \
    --export="ALL,SWEEP_CONFIGS=${SWEEP_CONFIGS}" \
    "${RUN_SWEEP}")
echo "Submitted training array: job ${TRAIN_JOB_ID} (--array=${ARRAY_RANGE})"

# -- Step 2: submit the dependent analysis+plots job --------------------------
# afterok on the whole array: runs only if every array task succeeds.
XLIM_PASS=()
if [[ -n "${XLIM_MIN}" && -n "${XLIM_MAX}" ]]; then
    XLIM_PASS=("${XLIM_MIN}" "${XLIM_MAX}")
else
    # run_analysis_and_plots.sh reads positional XLIM_MIN/XLIM_MAX; pass empties
    # so ERR_TOL lands in the right slot.
    XLIM_PASS=("" "")
fi

ANALYSIS_JOB_ID=$(sbatch --parsable \
    --dependency="afterok:${TRAIN_JOB_ID}" \
    "${RUN_ANALYSIS_PLOTS}" \
    "${SWEEP_DIR}" \
    "${EVAL_SIZE}" \
    "${BATCH_SIZE}" \
    "${DEVICE}" \
    "${DEDUPE}" \
    "${XLIM_PASS[0]}" \
    "${XLIM_PASS[1]}" \
    "${ERR_TOL}")
echo "Submitted analysis+plots: job ${ANALYSIS_JOB_ID} (afterok:${TRAIN_JOB_ID})"

echo ""
echo "Outputs when complete:"
echo "  ${SWEEP_DIR}/analysis_files/   (sweep_metrics.csv, *.sae_eval.pt)"
echo "  ${SWEEP_DIR}/analysis_plots/   (lambda_metrics.png, entropy_lambda_layer*.png)"
echo ""
echo "Track with:  squeue -j ${TRAIN_JOB_ID},${ANALYSIS_JOB_ID}"
