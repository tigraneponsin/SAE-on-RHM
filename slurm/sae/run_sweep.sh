#!/bin/bash
# =============================================================================
# Slurm job array script for SAE hyperparameter sweep.
#
# Workflow:
#   1. Edit constants in sae_sweep/generate_sweep.py, then run:
#      python sae_sweep/generate_sweep.py
#   2. Set SWEEP_CONFIGS and #SBATCH --array below, then:
#      sbatch slurm/sae/run_sweep.sh
# =============================================================================

# ── Job metadata ──────────────────────────────────────────────────────────────
#SBATCH --job-name=sae_sweep
#SBATCH --chdir /home/ponsin
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --account pcsl

# ── Array size: set to 0-<N-1> where N = number printed by generate_sweep.py ─
#SBATCH --array=0-24

# ── Resources ─────────────────────────────────────────────────────────────────
#SBATCH --time=01:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --mem=90G
#SBATCH --cpus-per-task=16

# =============================================================================
# USER: set these two paths
# =============================================================================
SWEEP_CONFIGS=/work/pcsl/ponsin/Mean_Transformer/Small_SAE/v_16_L_3_m_4_wdecay_0.0001/sweep_onetok0_layer0_lambda1_zoom/sweep_configs.json 
REPO_DIR=/home/ponsin/SAE-on-RHM
# =============================================================================

# ── Environment setup ─────────────────────────────────────────────────────────
if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Submit this script with sbatch, not directly."
    exit 1
fi

source /home/ponsin/miniconda3/etc/profile.d/conda.sh
conda activate pcsl

# ── Redirect logs to a file named after the SAE output artifact ───────────────
# Read the outname for this task from sweep_configs.json, then use exec to
# redirect stdout/stderr to <outname>.out / <outname>.err so each log is
# co-located with its .pt file and carries the full hyperparameter name.
OUTNAME=$(python3 -c "
import json, sys
configs = json.load(open('${SWEEP_CONFIGS}'))
idx = int('${SLURM_ARRAY_TASK_ID}')
if idx < 0 or idx >= len(configs):
    sys.exit(1)
print(configs[idx]['outname'])
")
LOG_BASE="${OUTNAME%.pt}"
exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

echo "======================================================================"
echo "Job array:  ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}]"
echo "Node:       ${SLURMD_NODENAME}"
echo "SWEEP_CONFIGS: ${SWEEP_CONFIGS}"
echo "Output:     ${OUTNAME}"
echo "======================================================================"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun python "${REPO_DIR}/scripts/sae_sweep/run_one.py" \
    --sweep_configs "${SWEEP_CONFIGS}" \
    --task_id "${SLURM_ARRAY_TASK_ID}"

EXIT_CODE=$?
echo "Job finished with exit code ${EXIT_CODE}."
exit ${EXIT_CODE}
