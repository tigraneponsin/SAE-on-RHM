#!/bin/bash
# Submit per-token, per-layer ablation jobs for the v=16 L=5 m=4 wd=1e-4
# transformer (transformer_meanclass, tuple_size=2 -> 32 leaf tokens, 5 blocks).
# Uses --model_variant best (hardcoded in run_ablate.sh).
# Run from the repo root: bash slurm/intervention/Job_ablate_per_token_v16_L5_m4.sh

TRAIN_OUTPUT=/work/pcsl/ponsin/Mean_Transformer/Transformer_for_SAE/v_16_L_5_m_4_wdecay_0.0001/RESULT_TRFCLASS_v_16_L_5_m=4_P_128000_0_emb_512_h_8_lr_5e-3_dropout_0.1_wd_0.0001.pkl.pt
EXP_DIR=/home/ponsin/SAE-on-RHM/scripts/intervention
OUT_BASE=/work/pcsl/ponsin/Mean_Transformer/Intervention/v_16_L_5_m_4_wdecay_0.0001/ablation_per_token_best

for LAYER in 0 1 2 3 4; do
    EXPERIMENTS="${EXP_DIR}/experiments_per_token_v16_L5_m4_layer${LAYER}.json"
    OUT_DIR="${OUT_BASE}/layer_${LAYER}"
    sbatch slurm/intervention/run_ablate.sh "$TRAIN_OUTPUT" "$EXPERIMENTS" "$OUT_DIR"
done
