#!/bin/bash
# =============================================================================
# Slurm script to run a per-input circuit-trace SWEEP over (node_threshold,
# edge_threshold) pairs. The expensive parts of circuit_trace (transformer
# forward, SAE forwards, full attribution) run ONCE; only pruning + grouping
# + per-config save happen per cell.
#
# Outputs are written under --out_dir as one subdir per cell named
#   node_<x>__edge_<y>/
# plus a top-level sweep_summary.json that the interactive HTML viewer reads.
#
# Required flags:
#   --train_output PATH
#   --sae_ckpts PATH [PATH ...]
#   --sae_eval_artifacts PATH [PATH ...]
#   --out_dir PATH
#   --input_idx N
#   --node_thresholds F [F ...]
#   --edge_thresholds F [F ...]
#
# Optional flags (defaults in brackets):
#   --eval_seed N         [0]
#   --eval_size N         [1024]
#   --mat_threshold N     [8192]
#   --sink_mode MODE      [softmax_logits]  one of {softmax_logits, true_class}
#   --device cuda|cpu     [cuda]
#   --model_variant best|last [auto]
#
# Example (3x3 grid):
#   sbatch slurm/sae/run_circuit_trace_sweep.sh \
#       --train_output /work/.../transformer.pt \
#       --sae_ckpts /work/.../L0.pt /work/.../L1.pt /work/.../L2.pt \
#       --sae_eval_artifacts /work/.../L0.sae_eval.pt /work/.../L1.sae_eval.pt /work/.../L2.sae_eval.pt \
#       --out_dir /work/.../circuit_sweep/input42 \
#       --input_idx 42 \
#       --node_thresholds 0.7 0.8 0.9 \
#       --edge_thresholds 0.9 0.95 0.98
# =============================================================================

# -- Job metadata -------------------------------------------------------------
#SBATCH --job-name=circuit_trace_sweep
#SBATCH --chdir /home/ponsin
#SBATCH --account pcsl

# -- Resources ----------------------------------------------------------------
#SBATCH --time=00:30:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --mem=90G
#SBATCH --cpus-per-task=16

#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

set -euo pipefail

REPO_DIR=/home/ponsin/SAE-on-RHM

# -- Defaults -----------------------------------------------------------------
TRAIN_OUTPUT=""
OUT_DIR=""
INPUT_IDX=""
EVAL_SEED=0
EVAL_SIZE=1024
MAT_THRESHOLD=8192
SINK_MODE=softmax_logits
DEVICE=cuda
MODEL_VARIANT=""
SAE_CKPTS=()
EVAL_ARTS=()
NODE_THRESHOLDS=()
EDGE_THRESHOLDS=()

usage() {
    cat <<'EOF'
Usage: sbatch slurm/sae/run_circuit_trace_sweep.sh \
         --train_output PATH \
         --sae_ckpts PATH [PATH ...] \
         --sae_eval_artifacts PATH [PATH ...] \
         --out_dir PATH \
         --input_idx N \
         --node_thresholds F [F ...] \
         --edge_thresholds F [F ...] \
         [--eval_seed N] [--eval_size N] [--mat_threshold N] \
         [--sink_mode softmax_logits|true_class] \
         [--device cuda|cpu] [--model_variant best|last]

--sae_ckpts, --sae_eval_artifacts, --node_thresholds, and --edge_thresholds
each take one or more values; pass each list terminated by either the next
flag or end-of-line. Sweeping a single axis is valid (pass a 1-value list).
EOF
}

# Strip leading/trailing whitespace from a string (defensive against pasted
# paths like ' /work/...').
_strip() {
    local x="$1"
    # Remove leading whitespace, then trailing whitespace.
    x="${x#"${x%%[![:space:]]*}"}"
    x="${x%"${x##*[![:space:]]}"}"
    printf '%s' "$x"
}

# -- Argument parsing ---------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --train_output)    TRAIN_OUTPUT=$(_strip "$2");  shift 2 ;;
        --out_dir)         OUT_DIR=$(_strip "$2");       shift 2 ;;
        --input_idx)       INPUT_IDX=$2;                 shift 2 ;;
        --eval_seed)       EVAL_SEED=$2;                 shift 2 ;;
        --eval_size)       EVAL_SIZE=$2;                 shift 2 ;;
        --mat_threshold)   MAT_THRESHOLD=$2;             shift 2 ;;
        --sink_mode)       SINK_MODE=$2;                 shift 2 ;;
        --device)          DEVICE=$2;                    shift 2 ;;
        --model_variant)   MODEL_VARIANT=$2;             shift 2 ;;
        --sae_ckpts)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SAE_CKPTS+=("$(_strip "$1")")
                shift
            done
            ;;
        --sae_eval_artifacts)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                EVAL_ARTS+=("$(_strip "$1")")
                shift
            done
            ;;
        --node_thresholds)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                NODE_THRESHOLDS+=("$1")
                shift
            done
            ;;
        --edge_thresholds)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                EDGE_THRESHOLDS+=("$1")
                shift
            done
            ;;
        -h|--help)         usage; exit 0 ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

# -- Validate required arguments ---------------------------------------------
missing=()
[[ -z "${TRAIN_OUTPUT}" ]]            && missing+=("--train_output")
[[ -z "${OUT_DIR}" ]]                 && missing+=("--out_dir")
[[ -z "${INPUT_IDX}" ]]               && missing+=("--input_idx")
[[ ${#SAE_CKPTS[@]} -eq 0 ]]          && missing+=("--sae_ckpts")
[[ ${#EVAL_ARTS[@]} -eq 0 ]]          && missing+=("--sae_eval_artifacts")
[[ ${#NODE_THRESHOLDS[@]} -eq 0 ]]    && missing+=("--node_thresholds")
[[ ${#EDGE_THRESHOLDS[@]} -eq 0 ]]    && missing+=("--edge_thresholds")
if [[ ${#missing[@]} -gt 0 ]]; then
    echo "ERROR: missing required arguments: ${missing[*]}" >&2
    usage >&2
    exit 1
fi

if [[ ${#SAE_CKPTS[@]} -ne ${#EVAL_ARTS[@]} ]]; then
    echo "ERROR: --sae_ckpts (${#SAE_CKPTS[@]}) and --sae_eval_artifacts (${#EVAL_ARTS[@]}) must have the same length" >&2
    exit 1
fi

if [[ ! -f "${TRAIN_OUTPUT}" ]]; then
    echo "ERROR: --train_output does not exist: ${TRAIN_OUTPUT}" >&2
    exit 1
fi
for f in "${SAE_CKPTS[@]}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: SAE checkpoint does not exist: ${f}" >&2
        exit 1
    fi
done
for f in "${EVAL_ARTS[@]}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: eval artifact does not exist: ${f}" >&2
        exit 1
    fi
done

mkdir -p "${OUT_DIR}"

# -- Environment setup --------------------------------------------------------
set +u
source /home/ponsin/miniconda3/etc/profile.d/conda.sh
conda activate pcsl
set -u

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# -- Redirect logs next to the outputs ----------------------------------------
exec > "${OUT_DIR}/circuit_trace_sweep.out" 2> "${OUT_DIR}/circuit_trace_sweep.err"

echo "======================================================================"
echo "Job:           ${SLURM_JOB_ID:-NA}"
echo "Node:          ${SLURMD_NODENAME:-NA}"
echo "TRAIN_OUTPUT:  ${TRAIN_OUTPUT}"
echo "OUT_DIR:       ${OUT_DIR}"
echo "INPUT_IDX:     ${INPUT_IDX}"
echo "EVAL_SEED:     ${EVAL_SEED}"
echo "EVAL_SIZE:     ${EVAL_SIZE}"
echo "MAT_THRESHOLD: ${MAT_THRESHOLD}"
echo "SINK_MODE:     ${SINK_MODE}"
echo "NODE_THRESHOLDS: ${NODE_THRESHOLDS[*]}"
echo "EDGE_THRESHOLDS: ${EDGE_THRESHOLDS[*]}"
echo "DEVICE:        ${DEVICE}"
echo "MODEL_VARIANT: ${MODEL_VARIANT:-auto (detect from SAE checkpoints)}"
echo "----------------------------------------------------------------------"
echo "SAE checkpoints (${#SAE_CKPTS[@]}, layer 0 first):"
for f in "${SAE_CKPTS[@]}"; do echo "  ${f}"; done
echo "Eval artifacts (${#EVAL_ARTS[@]}, layer 0 first):"
for f in "${EVAL_ARTS[@]}"; do echo "  ${f}"; done
echo "======================================================================"

START_EPOCH=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')
echo "START:         ${START_HUMAN}"

EXTRA_ARGS=()
if [[ -n "${MODEL_VARIANT}" ]]; then
    EXTRA_ARGS+=(--model_variant "${MODEL_VARIANT}")
fi

set +e
srun python -m circuit_tracing.circuit_trace_sweep \
    --train_output "${TRAIN_OUTPUT}" \
    --sae_ckpts "${SAE_CKPTS[@]}" \
    --sae_eval_artifacts "${EVAL_ARTS[@]}" \
    --input_idx "${INPUT_IDX}" \
    --eval_seed "${EVAL_SEED}" \
    --eval_size "${EVAL_SIZE}" \
    --output_dir "${OUT_DIR}" \
    --mat_threshold "${MAT_THRESHOLD}" \
    --sink_mode "${SINK_MODE}" \
    --node_thresholds "${NODE_THRESHOLDS[@]}" \
    --edge_thresholds "${EDGE_THRESHOLDS[@]}" \
    --device "${DEVICE}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
EXIT_CODE=$?
set -e

END_EPOCH=$(date +%s)
END_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')
ELAPSED_SEC=$((END_EPOCH - START_EPOCH))
ELAPSED_H=$((ELAPSED_SEC / 3600))
ELAPSED_M=$(((ELAPSED_SEC % 3600) / 60))
ELAPSED_S=$((ELAPSED_SEC % 60))

echo "END:           ${END_HUMAN}"
printf 'ELAPSED:       %02d:%02d:%02d (%ds)\n' "${ELAPSED_H}" "${ELAPSED_M}" "${ELAPSED_S}" "${ELAPSED_SEC}"
echo "Circuit-trace sweep finished with exit code ${EXIT_CODE}."
exit ${EXIT_CODE}
