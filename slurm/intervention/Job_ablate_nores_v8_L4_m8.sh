#!/bin/bash
# Per-token ablation for the nores transformer: v=8 L=4 m=8 wd=1e-4 dropout=0.1
# Run from the repo root: bash slurm/intervention/Job_ablate_nores_v8_L4_m8.sh

TRAIN_OUTPUT=/work/pcsl/ponsin/Mean_Transformer/Transformer_for_SAE_nores/v_8_L_4_m_8_wdecay_0.0001_dropout_0.1/RESULT_TRFCLASS_v_8_L_4_m=8_P_524288_0_emb_512_h_8_lr_0.001_dropout_0.1_wd_0.0001.pkl.pt
EXP_DIR=/home/ponsin/SAE-on-RHM/scripts/intervention
OUT_BASE=/work/pcsl/ponsin/Mean_Transformer/Intervention/v_8_L_4_m_8_wdecay_0.0001_dropout_0.1_nores

for LAYER in 0 1 2 3; do
    EXPERIMENTS="${EXP_DIR}/experiments_per_token_v8_L4_layer${LAYER}.json"
    OUT_DIR="${OUT_BASE}/ablation_L${LAYER}"
    sbatch slurm/intervention/run_ablate.sh "$TRAIN_OUTPUT" "$EXPERIMENTS" "$OUT_DIR"
done
