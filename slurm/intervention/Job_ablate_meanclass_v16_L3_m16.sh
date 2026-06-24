#!/bin/bash
# Per-token ablation for the meanclass transformer: v=16 L=3 m=16 wd=1e-4 dropout=0.1
# Run from the repo root: bash slurm/intervention/Job_ablate_meanclass_v16_L3_m16.sh

TRAIN_OUTPUT=/work/pcsl/ponsin/Mean_Transformer/Transformer_for_SAE_meanclass/v_16_L_3_m_16_wdecay_0.0001_dropout_0.1/RESULT_TRFCLASS_v_16_L_3_m=16_P_1048576_0_emb_512_h_8_lr_0.001_dropout_0.1_wd_0.0001.pkl.pt
EXP_DIR=/home/ponsin/SAE-on-RHM/scripts/intervention
OUT_BASE=/work/pcsl/ponsin/Mean_Transformer/Intervention/v_16_L_3_m_16_wdecay_0.0001_dropout_0.1_meanclass

for LAYER in 0 1 2; do
    EXPERIMENTS="${EXP_DIR}/experiments_per_token_L${LAYER}.json"
    OUT_DIR="${OUT_BASE}/ablation_L${LAYER}"
    sbatch slurm/intervention/run_ablate.sh "$TRAIN_OUTPUT" "$EXPERIMENTS" "$OUT_DIR"
done
