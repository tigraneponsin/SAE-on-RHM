TRAIN_OUTPUT=/work/pcsl/ponsin/Mean_Transformer/Transformer_for_SAE/v_16_L_3_m_4_wdecay_0.0001/RESULT_TRFCLASS_v_16_L_3_m=4_P_12160_0_emb_512_h_8_lr_5e-3_dropout_0.1_wd_0.0001.pkl.pt
EXPERIMENTS=/home/ponsin/SAE-on-RHM/scripts/intervention/experiments_per_token_L2.json
OUT_DIR=/work/pcsl/ponsin/Mean_Transformer/Intervention/v_16_L_3_m_4_wdecay_0.0001/ablation_L2

sbatch slurm/intervention/run_ablate.sh "$TRAIN_OUTPUT" "$EXPERIMENTS" "$OUT_DIR"
