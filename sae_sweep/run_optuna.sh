#!/bin/bash
# =============================================================================
# Single Slurm job for Optuna SAE hyperparameter tuning.
#
# Workflow:
#   1. Set the USER constants below (CHECKPOINT, SAE_LAYER, OUTDIR, N_TRIALS).
#   2. Submit:
#        sbatch sae_sweep/run_optuna.sh
#
# The study is backed by a SQLite DB at <OUTDIR>/optuna_layer<N>.db.
# To resume an interrupted run, re-submit with the same OUTDIR and SAE_LAYER
# -- Optuna will load the existing study and add more trials.
#
# Results are saved to <OUTDIR>/optuna_layer<N>.pt.
# Logs go to <OUTDIR>/optuna_layer<N>.out / .err.
# =============================================================================

# ── Job metadata ──────────────────────────────────────────────────────────────
#SBATCH --job-name=sae_optuna
#SBATCH --chdir /home/ponsin
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --account pcsl

# ── Resources ─────────────────────────────────────────────────────────────────
#SBATCH --time=3:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --mem=90G
#SBATCH --cpus-per-task=16

# =============================================================================
# USER: set these constants
# =============================================================================
CHECKPOINT=/work/pcsl/ponsin/Mean_Transformer/Transformer_for_SAE/v_16_L_3_m_4/RESULT_TRFCLASS_v_16_L_3_m=4_P_12160_0_emb_512_h_8_lr_5e-3_dropout_0.1.pkl.pt
SAE_LAYER=0
N_TRIALS=25
OUTDIR=/work/pcsl/ponsin/Mean_Transformer/SAE/2nd_generation/optuna_layer${SAE_LAYER}_nowarm
REPO_DIR=/home/ponsin/SAE-on-RHM
# =============================================================================

mkdir -p "${OUTDIR}"

OUTNAME="${OUTDIR}/optuna_layer${SAE_LAYER}"
exec > "${OUTNAME}.out" 2> "${OUTNAME}.err"

echo "======================================================================"
echo "Job:        ${SLURM_JOB_ID}"
echo "Node:       ${SLURMD_NODENAME}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Layer:      ${SAE_LAYER}"
echo "Trials:     ${N_TRIALS}"
echo "Output:     ${OUTNAME}.pt"
echo "Study DB:   ${OUTNAME}.db"
echo "======================================================================"

source /home/ponsin/miniconda3/etc/profile.d/conda.sh
conda activate pcsl

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun python "${REPO_DIR}/optuna_tune_sae.py" \
    --train_output    "${CHECKPOINT}" \
    --sae_layer       "${SAE_LAYER}" \
    --n_trials        "${N_TRIALS}" \
    --study_name      "sae_layer${SAE_LAYER}" \
    --outname         "${OUTNAME}"

EXIT_CODE=$?
echo "Job finished with exit code ${EXIT_CODE}."
exit ${EXIT_CODE}
