#!/bin/bash
# =============================================================================
# Slurm script: tree reconstruction from SAE features.
#
# What it does:
#   For one trained transformer with L layers, take ONE SAE eval artifact per
#   layer (the .feature_latent.pt / .sae_eval.pt produced by
#   scripts/sae_eval/run.py), then reconstruct the full RHM latent tree for
#   each input by aggregating per-feature evidence.
#
# What you need to provide:
#   ARTIFACTS — exactly L paths to .feature_latent.pt (or .sae_eval.pt) files,
#               one per layer 0..L-1, all from the SAME trained transformer.
#               They typically live in different sweep directories (one per
#               layer), so you list them explicitly here. Order does not
#               matter; the script reads layer_id from each artifact and
#               validates that 0..L-1 are covered exactly.
#
# Example layout (from your /work tree):
#   .../sweep_alltokens_layer0_lambda1_zoom/sae_..._layer0_..._l10.027_....feature_latent.pt
#   .../sweep_alltokens_layer1_lambda1_zoom/sae_..._layer1_..._l10.033_....feature_latent.pt
#   .../sweep_alltokens_layer2_lambda1_zoom/sae_..._layer2_..._l10.03_....feature_latent.pt
#
# Usage:
#   sbatch slurm/sae/run_tree_reconstruction.sh
#   sbatch slurm/sae/run_tree_reconstruction.sh
# =============================================================================

# -- Job metadata -------------------------------------------------------------
#SBATCH --job-name=sae_tree_recon
#SBATCH --chdir /home/ponsin
#SBATCH --account pcsl

# Resources
#SBATCH --time=00:10:00
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# =============================================================================
# USER: fill these in before submitting
# =============================================================================

# One artifact per transformer layer. List them in any order.
ARTIFACTS=(
    /work/pcsl/ponsin/Mean_Transformer/Small_SAE/latent_dim_4*512/v_16_L_3_m_4_wdecay_0.0001/sweep_alltokens_layer0_lambda1_zoom/analysis_files/new_analysis/sae_v16_L3_m4_P12160_emb512_layer0_alltok_ldim2048_l10.023_lr1e-4_steps131072_bs128_ts16384.sae_eval.pt
   /work/pcsl/ponsin/Mean_Transformer/Small_SAE/latent_dim_4*512/v_16_L_3_m_4_wdecay_0.0001/sweep_alltokens_layer1_lambda1_zoom/analysis_files/sae_v16_L3_m4_P12160_emb512_layer1_alltok_ldim2048_l10.023_lr1e-4_steps131072_bs128_ts16384.sae_eval.pt
   /work/pcsl/ponsin/Mean_Transformer/Small_SAE/latent_dim_4*512/v_16_L_3_m_4_wdecay_0.0001/sweep_alltokens_layer2_lambda1_zoom/analysis_files/sae_v16_L3_m4_P12160_emb512_layer2_alltok_ldim2048_l10.023_lr1e-4_steps131072_bs128_ts16384.sae_eval.pt
)

# Where to write outputs (Z_hat, scores, CSV, log files).
OUT_DIR=/work/pcsl/ponsin/Mean_Transformer/Small_SAE/latent_dim_4*512/v_16_L_3_m_4_wdecay_0.0001/tree_reconstruction

# Generalization split for f_i(x). Leave empty to default to
# (artifact eval_seed + 1) and the artifact's eval_size. The script refuses
# any seed already used by transformer / SAE training / cond_prob estimation,
# so the default is safe.
EVAL_SEED_RECON=
EVAL_SIZE_RECON=

REPO_DIR=/home/ponsin/SAE-on-RHM
BATCH_SIZE=2048
# =============================================================================

#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# Environment setup
source /home/ponsin/miniconda3/etc/profile.d/conda.sh
conda activate pcsl

mkdir -p "${OUT_DIR}"

# Redirect logs next to the output
exec > "${OUT_DIR}/tree_reconstruction.out" 2> "${OUT_DIR}/tree_reconstruction.err"

echo "======================================================================"
echo "Job:      ${SLURM_JOB_ID}"
echo "Node:     ${SLURMD_NODENAME}"
echo "OUT_DIR:  ${OUT_DIR}"
echo "Artifacts (${#ARTIFACTS[@]}):"
for a in "${ARTIFACTS[@]}"; do
    echo "  ${a}"
    if [ ! -f "${a}" ]; then
        echo "ERROR: file does not exist: ${a}" >&2
        exit 2
    fi
done
echo "======================================================================"

EXTRA_ARGS=()
if [ -n "${EVAL_SEED_RECON}" ]; then
    EXTRA_ARGS+=(--eval_seed_recon "${EVAL_SEED_RECON}")
fi
if [ -n "${EVAL_SIZE_RECON}" ]; then
    EXTRA_ARGS+=(--eval_size_recon "${EVAL_SIZE_RECON}")
fi

srun python "${REPO_DIR}/scripts/sae_tree_reconstruction/run.py" \
    --artifacts "${ARTIFACTS[@]}" \
    --out_dir "${OUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --device cuda \
    --sanity \
    "${EXTRA_ARGS[@]}"

EXIT_CODE=$?
echo "Tree reconstruction finished with exit code ${EXIT_CODE}."
exit ${EXIT_CODE}
