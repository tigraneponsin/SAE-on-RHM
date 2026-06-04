#!/bin/bash
# =============================================================================
# Slurm script to run a per-input circuit-trace SWEEP over (node_threshold,
# edge_threshold) pairs. The expensive parts of circuit_trace (transformer
# forward, SAE forwards, full attribution) run ONCE; only pruning + grouping
# + per-config save happen per cell.
#
# Inputs are resolved automatically from a parent directory that holds one
# per-layer SAE sweep folder (each with its analysis_files/), plus one lambda1
# per layer. The matching SAE checkpoint (nearest lambda1) and its .sae_eval.pt
# are picked per layer, and the transformer train_output is auto-detected from
# the selected checkpoints (see scripts/sae_sweep/resolve_circuit_inputs.py).
#
# Outputs are written under --out_dir as one subdir per cell named
#   node_<x>__edge_<y>/
# plus a top-level sweep_summary.json. After the trace cells are built, the
# interactive 2-slider plot is rendered automatically to
#   <out_dir>/sweep_circuit.html
# (via circuit_tracing.visualize_sweep; no separate step needed).
#
# Required flags:
#   --parent_dir PATH        dir containing the per-layer SAE sweep folders
#   --lambda1 L:V [L:V ...]   one LAYER:VALUE pair per layer (e.g. 0:0.01 1:0.02)
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
# Example (3x3 grid over layers 0,1,2 at lambda1=0.01 each):
#   sbatch slurm/sae/run_circuit_trace_sweep.sh \
#       --parent_dir /work/.../v_16_L_3_m_16_.../ \
#       --lambda1 0:0.01 1:0.01 2:0.01 \
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
PARENT_DIR=""
OUT_DIR=""
INPUT_IDX=""
EVAL_SEED=0
EVAL_SIZE=1024
MAT_THRESHOLD=8192
SINK_MODE=softmax_logits
DEVICE=cuda
MODEL_VARIANT=""
LAMBDA1=()
NODE_THRESHOLDS=()
EDGE_THRESHOLDS=()

usage() {
    cat <<'EOF'
Usage: sbatch slurm/sae/run_circuit_trace_sweep.sh \
         --parent_dir PATH \
         --lambda1 L:V [L:V ...] \
         --out_dir PATH \
         --input_idx N \
         --node_thresholds F [F ...] \
         --edge_thresholds F [F ...] \
         [--eval_seed N] [--eval_size N] [--mat_threshold N] \
         [--sink_mode softmax_logits|true_class] \
         [--device cuda|cpu] [--model_variant best|last]

--parent_dir holds one per-layer SAE sweep folder (each with analysis_files/).
--lambda1 takes one LAYER:VALUE pair per layer, contiguous from 0
(e.g. 0:0.01 1:0.02 2:0.005). The nearest-lambda checkpoint and its eval
artifact are selected per layer; train_output is auto-detected.

--lambda1, --node_thresholds, and --edge_thresholds each take one or more
values; pass each list terminated by either the next flag or end-of-line.
EOF
}

# Strip leading/trailing whitespace from a string (defensive against pasted
# paths like ' /work/...').
_strip() {
    local x="$1"
    x="${x#"${x%%[![:space:]]*}"}"
    x="${x%"${x##*[![:space:]]}"}"
    printf '%s' "$x"
}

# -- Argument parsing ---------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --parent_dir)      PARENT_DIR=$(_strip "$2");    shift 2 ;;
        --out_dir)         OUT_DIR=$(_strip "$2");       shift 2 ;;
        --input_idx)       INPUT_IDX=$2;                 shift 2 ;;
        --eval_seed)       EVAL_SEED=$2;                 shift 2 ;;
        --eval_size)       EVAL_SIZE=$2;                 shift 2 ;;
        --mat_threshold)   MAT_THRESHOLD=$2;             shift 2 ;;
        --sink_mode)       SINK_MODE=$2;                 shift 2 ;;
        --device)          DEVICE=$2;                    shift 2 ;;
        --model_variant)   MODEL_VARIANT=$2;             shift 2 ;;
        --lambda1)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                LAMBDA1+=("$(_strip "$1")")
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
[[ -z "${PARENT_DIR}" ]]              && missing+=("--parent_dir")
[[ -z "${OUT_DIR}" ]]                 && missing+=("--out_dir")
[[ -z "${INPUT_IDX}" ]]               && missing+=("--input_idx")
[[ ${#LAMBDA1[@]} -eq 0 ]]            && missing+=("--lambda1")
[[ ${#NODE_THRESHOLDS[@]} -eq 0 ]]    && missing+=("--node_thresholds")
[[ ${#EDGE_THRESHOLDS[@]} -eq 0 ]]    && missing+=("--edge_thresholds")
if [[ ${#missing[@]} -gt 0 ]]; then
    echo "ERROR: missing required arguments: ${missing[*]}" >&2
    usage >&2
    exit 1
fi

if [[ ! -d "${PARENT_DIR}" ]]; then
    echo "ERROR: --parent_dir does not exist: ${PARENT_DIR}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

# -- Environment setup --------------------------------------------------------
set +u
source /home/ponsin/miniconda3/etc/profile.d/conda.sh
conda activate pcsl
set -u

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# -- Redirect logs next to the outputs ----------------------------------------
exec > "${OUT_DIR}/circuit_trace_sweep.out" 2> "${OUT_DIR}/circuit_trace_sweep.err"

# -- Resolve SAE checkpoints / eval artifacts / train_output ------------------
# resolve_circuit_inputs.py emits a JSON object; parse it into bash arrays with
# python (handles arbitrary paths safely). Warnings from the resolver go to its
# stderr, which we surface below.
echo "Resolving circuit-trace inputs from ${PARENT_DIR} with lambda1=${LAMBDA1[*]}"
set +e
RESOLVE_JSON=$(python "${REPO_DIR}/scripts/sae_sweep/resolve_circuit_inputs.py" \
    --parent_dir "${PARENT_DIR}" \
    --lambda1 "${LAMBDA1[@]}" \
    --emit json)
RESOLVE_RC=$?
set -e
if [[ ${RESOLVE_RC} -ne 0 ]]; then
    echo "ERROR: resolve_circuit_inputs.py failed (rc=${RESOLVE_RC}); see stderr above." >&2
    exit ${RESOLVE_RC}
fi

TRAIN_OUTPUT=$(printf '%s' "${RESOLVE_JSON}" | python -c "import json,sys;print(json.load(sys.stdin)['train_output'])")
mapfile -t SAE_CKPTS < <(printf '%s' "${RESOLVE_JSON}" | python -c "import json,sys;[print(p) for p in json.load(sys.stdin)['sae_ckpts']]")
mapfile -t EVAL_ARTS < <(printf '%s' "${RESOLVE_JSON}" | python -c "import json,sys;[print(p) for p in json.load(sys.stdin)['sae_eval_artifacts']]")

# -- Validate resolved paths --------------------------------------------------
if [[ ${#SAE_CKPTS[@]} -ne ${#EVAL_ARTS[@]} ]]; then
    echo "ERROR: resolved #sae_ckpts (${#SAE_CKPTS[@]}) != #eval_artifacts (${#EVAL_ARTS[@]})" >&2
    exit 1
fi
if [[ ${#SAE_CKPTS[@]} -eq 0 ]]; then
    echo "ERROR: resolver returned no SAE checkpoints" >&2
    exit 1
fi
if [[ ! -f "${TRAIN_OUTPUT}" ]]; then
    echo "ERROR: resolved train_output does not exist: ${TRAIN_OUTPUT}" >&2
    exit 1
fi
for f in "${SAE_CKPTS[@]}"; do
    [[ -f "${f}" ]] || { echo "ERROR: SAE checkpoint does not exist: ${f}" >&2; exit 1; }
done
for f in "${EVAL_ARTS[@]}"; do
    [[ -f "${f}" ]] || { echo "ERROR: eval artifact does not exist: ${f}" >&2; exit 1; }
done

echo "======================================================================"
echo "Job:           ${SLURM_JOB_ID:-NA}"
echo "Node:          ${SLURMD_NODENAME:-NA}"
echo "PARENT_DIR:    ${PARENT_DIR}"
echo "LAMBDA1:       ${LAMBDA1[*]}"
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

# -- Plot the sweep into an interactive HTML (visualize_sweep.py) -------------
# Only if the trace sweep succeeded; a plot failure is reported but does not
# mask the (successful) trace exit code. Writes <OUT_DIR>/sweep_circuit.html.
VIZ_EXIT=0
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo ""
    echo "Plotting sweep -> ${OUT_DIR}/sweep_circuit.html"
    set +e
    srun python -m circuit_tracing.visualize_sweep \
        --sweep_dir "${OUT_DIR}"
    VIZ_EXIT=$?
    set -e
    if [[ ${VIZ_EXIT} -ne 0 ]]; then
        echo "WARNING: visualize_sweep.py failed (rc=${VIZ_EXIT}); trace outputs are intact."
    fi
else
    echo "Skipping plot: circuit_trace_sweep exited with ${EXIT_CODE}."
fi

END_EPOCH=$(date +%s)
END_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %Z')
ELAPSED_SEC=$((END_EPOCH - START_EPOCH))
ELAPSED_H=$((ELAPSED_SEC / 3600))
ELAPSED_M=$(((ELAPSED_SEC % 3600) / 60))
ELAPSED_S=$((ELAPSED_SEC % 60))

echo "END:           ${END_HUMAN}"
printf 'ELAPSED:       %02d:%02d:%02d (%ds)\n' "${ELAPSED_H}" "${ELAPSED_M}" "${ELAPSED_S}" "${ELAPSED_SEC}"
echo "Circuit-trace sweep finished with exit code ${EXIT_CODE} (plot exit ${VIZ_EXIT})."
exit ${EXIT_CODE}
