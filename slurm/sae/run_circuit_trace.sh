#!/bin/bash
# =============================================================================
# Slurm script to run per-input circuit tracing.
#
# You select ONE specific SAE checkpoint per transformer layer (and the matching
# .sae_eval.pt artifact for that exact checkpoint). There is no auto-discovery
# from a sweep -- the sweep contains many lambdas per layer and the script does
# not know which one you want.
#
# Required flags:
#   --train_output PATH
#       Path to the transformer checkpoint (.pt).
#
#   --sae_ckpts PATH [PATH ...]
#       K SAE checkpoint paths, one per transformer layer, in bottom-to-top
#       order (layer 0 first). Each must be a single SAE checkpoint, not a
#       sweep directory.
#
#   --sae_eval_artifacts PATH [PATH ...]
#       K .sae_eval.pt artifact paths matching --sae_ckpts (same order, same
#       length). Each artifact must have been produced from the corresponding
#       SAE checkpoint with --with-all (so it carries joint_fire_count and
#       firing_count for label argmax).
#
#   --out_dir PATH
#       Destination for nodes.pt / edges.pt / fidelity.pt / graph.gpickle /
#       summary.json. Created if absent.
#
#   --input_idx N
#       Row index into the freshly sampled eval set.
#
# Optional flags (defaults in brackets):
#   --eval_seed N         [0]
#   --eval_size N         [1024]
#   --mat_threshold N     [8192]    materialize M when N*d <= threshold
#   --sink_mode MODE      [softmax_logits]  one of {softmax_logits, true_class}
#   --node_threshold F    [0.8]     indirect-influence node-prune threshold
#   --edge_threshold F    [0.98]    indirect-influence edge-prune threshold
#   --device cuda|cpu     [cuda]
#   --model_variant best|last [auto]   auto-detected from SAE checkpoints if unset
#
# Example (3-layer transformer):
#   sbatch slurm/sae/run_circuit_trace.sh \
#       --train_output /work/.../transformer.pt \
#       --sae_ckpts \
#           /work/.../sweep/sae_layer0_lambda3e-3.pt \
#           /work/.../sweep/sae_layer1_lambda1e-3.pt \
#           /work/.../sweep/sae_layer2_lambda5e-4.pt \
#       --sae_eval_artifacts \
#           /work/.../eval/sae_layer0_lambda3e-3.sae_eval.pt \
#           /work/.../eval/sae_layer1_lambda1e-3.sae_eval.pt \
#           /work/.../eval/sae_layer2_lambda5e-4.sae_eval.pt \
#       --out_dir /work/.../circuit_tracing/input42 \
#       --input_idx 42
# =============================================================================

# -- Job metadata -------------------------------------------------------------
#SBATCH --job-name=circuit_trace
#SBATCH --chdir /home/ponsin
#SBATCH --account pcsl

# -- Resources ----------------------------------------------------------------
#SBATCH --time=00:10:00
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

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
NODE_THRESHOLD=0.8
EDGE_THRESHOLD=0.98
DEVICE=cuda
# Empty = let circuit_trace.py auto-detect from the SAE checkpoints
# (must agree across layers). Override with --model_variant best|last.
MODEL_VARIANT=""
SAE_CKPTS=()
EVAL_ARTS=()

usage() {
    cat <<'EOF'
Usage: sbatch slurm/sae/run_circuit_trace.sh \
         --train_output PATH \
         --sae_ckpts PATH [PATH ...] \
         --sae_eval_artifacts PATH [PATH ...] \
         --out_dir PATH \
         --input_idx N \
         [--eval_seed N] [--eval_size N] \
         [--mat_threshold N] \
         [--sink_mode softmax_logits|true_class] \
         [--node_threshold F] [--edge_threshold F] \
         [--device cuda|cpu] [--model_variant best|last]

Required:
  --train_output         transformer checkpoint (.pt)
  --sae_ckpts            K SAE checkpoint paths, bottom-to-top
  --sae_eval_artifacts   K matching .sae_eval.pt paths, same order
  --out_dir              destination directory
  --input_idx            row index into the eval set

Optional:
  --eval_seed       default 0
  --eval_size       default 1024
  --mat_threshold   default 8192             (materialize M when N*d <= threshold)
  --sink_mode       default softmax_logits   (or true_class)
  --node_threshold  default 0.8              (indirect-influence node prune)
  --edge_threshold  default 0.98             (indirect-influence edge prune)
  --device          default cuda
  --model_variant   default auto             (one of {best, last}; if unset,
                                              circuit_trace.py auto-detects from
                                              the SAE checkpoints, which must
                                              all agree on the variant.)

--sae_ckpts and --sae_eval_artifacts each take one or more paths; pass each
list terminated by either the next flag or end-of-line. They must be the same
length (one per transformer layer).
EOF
}

# -- Argument parsing ---------------------------------------------------------
# We support multi-value flags (--sae_ckpts, --sae_eval_artifacts) by reading
# values until the next argument starting with '--' or end of args.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --train_output)    TRAIN_OUTPUT=$2;     shift 2 ;;
        --out_dir)         OUT_DIR=$2;          shift 2 ;;
        --input_idx)       INPUT_IDX=$2;        shift 2 ;;
        --eval_seed)       EVAL_SEED=$2;        shift 2 ;;
        --eval_size)       EVAL_SIZE=$2;        shift 2 ;;
        --mat_threshold)   MAT_THRESHOLD=$2;    shift 2 ;;
        --sink_mode)       SINK_MODE=$2;        shift 2 ;;
        --node_threshold)  NODE_THRESHOLD=$2;   shift 2 ;;
        --edge_threshold)  EDGE_THRESHOLD=$2;   shift 2 ;;
        --prune_fraction)
            echo "ERROR: --prune_fraction has been removed." >&2
            echo "  Use --sink_mode {softmax_logits|true_class} with --node_threshold and --edge_threshold instead." >&2
            exit 1
            ;;
        --device)          DEVICE=$2;           shift 2 ;;
        --model_variant)   MODEL_VARIANT=$2;    shift 2 ;;
        --sae_ckpts)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SAE_CKPTS+=("$1")
                shift
            done
            ;;
        --sae_eval_artifacts)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                EVAL_ARTS+=("$1")
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
[[ -z "${TRAIN_OUTPUT}" ]]    && missing+=("--train_output")
[[ -z "${OUT_DIR}" ]]         && missing+=("--out_dir")
[[ -z "${INPUT_IDX}" ]]       && missing+=("--input_idx")
[[ ${#SAE_CKPTS[@]} -eq 0 ]]  && missing+=("--sae_ckpts")
[[ ${#EVAL_ARTS[@]} -eq 0 ]]  && missing+=("--sae_eval_artifacts")
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

# Ensure `python -m circuit_tracing.circuit_trace` resolves the package: the
# slurm job's CWD is /home/ponsin (per --chdir), not the repo, so we have to
# add the repo to PYTHONPATH explicitly.
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# -- Redirect logs next to the outputs ----------------------------------------
exec > "${OUT_DIR}/circuit_trace.out" 2> "${OUT_DIR}/circuit_trace.err"

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
echo "NODE_THRESHOLD:${NODE_THRESHOLD}"
echo "EDGE_THRESHOLD:${EDGE_THRESHOLD}"
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

# Pass --model_variant only when the user set it explicitly; otherwise let
# circuit_trace.py auto-detect from the SAE checkpoints.
EXTRA_ARGS=()
if [[ -n "${MODEL_VARIANT}" ]]; then
    EXTRA_ARGS+=(--model_variant "${MODEL_VARIANT}")
fi

set +e
srun python -m circuit_tracing.circuit_trace \
    --train_output "${TRAIN_OUTPUT}" \
    --sae_ckpts "${SAE_CKPTS[@]}" \
    --sae_eval_artifacts "${EVAL_ARTS[@]}" \
    --input_idx "${INPUT_IDX}" \
    --eval_seed "${EVAL_SEED}" \
    --eval_size "${EVAL_SIZE}" \
    --output_dir "${OUT_DIR}" \
    --mat_threshold "${MAT_THRESHOLD}" \
    --sink_mode "${SINK_MODE}" \
    --node_threshold "${NODE_THRESHOLD}" \
    --edge_threshold "${EDGE_THRESHOLD}" \
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
echo "Circuit trace finished with exit code ${EXIT_CODE}."
exit ${EXIT_CODE}
