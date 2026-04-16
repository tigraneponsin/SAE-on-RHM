TRAIN_OUTPUT=/work/pcsl/ponsin/Mean_Transformer/Transformer_for_SAE/v_16_L_3_m_4_wdecay_0/RESULT_TRFCLASS_v_16_L_3_m=4_P_12160_0_emb_512_h_8_lr_5e-3_dropout_0.1.pkl.pt
EXPERIMENTS=/home/ponsin/SAE-on-RHM/scripts/intervention/experiments_dead_token.json
OUT_DIR=/work/pcsl/ponsin/Mean_Transformer/Intervention/v_16_L_3_m_4_wdecay_0/ablation

sbatch slurm/intervention/run_ablate.sh "$TRAIN_OUTPUT" "$EXPERIMENTS" "$OUT_DIR"
