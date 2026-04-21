#!/bin/bash
# Submit per-token layer-0 ablation jobs for all 5 transformer seeds.
# Run from the repo root: bash slurm/intervention/Job_ablate_per_token_L0.sh

CKPT_DIR=/work/pcsl/ponsin/Transformers_for_stats/v_16_L_3_m_4_wdecay_0.0001
EXPERIMENTS=/home/ponsin/SAE-on-RHM/scripts/intervention/experiments_per_token_L0.json
CKPT_PREFIX=RESULT_TRFCLASS_v_16_L_3_m=4_P_12160
CKPT_SUFFIX=_emb_512_h_8_lr_5e-3_dropout_0.1_wd_0.0001.pkl.pt

for SEED in 0 1 2 3 4; do
    TRAIN_OUTPUT="${CKPT_DIR}/${CKPT_PREFIX}_${SEED}${CKPT_SUFFIX}"
    OUT_DIR="${CKPT_DIR}/ablation_per_token_L0/seed_${SEED}"
    sbatch slurm/intervention/run_ablate.sh "$TRAIN_OUTPUT" "$EXPERIMENTS" "$OUT_DIR"
done
