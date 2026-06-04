# v=16
# L=3
# m=2
# batch_size=8
# for P in 4096; do
#     sbatch Sbatch_trsf_for_SAE.sh "$P" "$v" "$L" "$m" "$batch_size"
# done

# v=16
# L=3
# m=4
# batch_size=16
# for P in  12160; do
#     sbatch Sbatch_trsf_for_SAE.sh "$P" "$v" "$L" "$m" "$batch_size" 
#  done


# v=16
# L=3
# m=4
# batch_size=16
# for P in 12160; do
#     sbatch Sbatch_trsf_for_SAE.sh "$P" "$v" "$L" "$m" "$batch_size" "" "" "" "" "" "" "" "0.01"
#  done


# v=16
# L=3
# m=4
# batch_size=128
# for P in 12160; do
#     sbatch slurm/transformer/Sbatch_trsf_for_SAE.sh "$P" "$v" "$L" "$m" "$batch_size" "" "" "" "" "" "" "" "0.0001"
# done

# v=16
# L=3
# m=16
# batch_size=256
# P=1048576
# lr=0.001
# DROPOUT=0.1
# for wd in 0.0001; do
#     sbatch slurm/transformer/Sbatch_trsf_for_SAE.sh "$P" "$v" "$L" "$m" "$batch_size" "" "" "$lr" "" "" "$DROPOUT" "" "$wd"
#  done

# v=16
# L=5
# m=4
# batch_size=1280
# lr=0.0001
# for P in 256000; do
#     sbatch slurm/transformer/Sbatch_trsf_for_SAE.sh "$P" "$v" "$L" "$m" "$batch_size" "" "" "$lr"
# done

# V=16
# L=3
# M=4
# BATCH_SIZE=32
# lr=0.001
# DROPOUT=0
# wd=0.0001
# TEST_LOSS_THRESHOLD=0.001
# for P in 32768; do
#     sbatch slurm/transformer/Sbatch_trsf_for_SAE.sh "$P" "$V" "$L" "$M" "$BATCH_SIZE" "" "" "$lr" "" "" "$DROPOUT" "" "$wd" "" "" "$TEST_LOSS_THRESHOLD" "1"
#  done

# v=16
# L=5
# m=4
# batch_size=256
# lr=0.001
# DROPOUT=0.1
# wd=0.0001
# for P in 256000; do
#     sbatch slurm/transformer/Sbatch_trsf_for_SAE.sh "$P" "$v" "$L" "$m" "$batch_size" "" "" "$lr" "" "" "$DROPOUT" "" "$wd"
# done

# v=8
# L=4
# m=8
# batch_size=256
# P=524288
# lr=0.001
# DROPOUT=0.1
# for wd in 0.0001; do
#     sbatch slurm/transformer/Sbatch_trsf_for_SAE.sh "$P" "$v" "$L" "$m" "$batch_size" "" "" "$lr" "" "" "$DROPOUT" "" "$wd"
#  done

# Free-pooled (learned pooling head) variants. The model is passed as $18.
# Empty positional args ($14-$17: warmup/decay/test_loss_threshold/stop) are
# kept blank here. With residual stream use transformer_freeclass; without,
# transformer_freeclass_nores.
# v=16
# L=3
# m=16
# batch_size=256
# P=1048576
# lr=0.001
# DROPOUT=0.1
# wd=0.0001
# for MODEL in transformer_freeclass transformer_freeclass_nores; do
#     sbatch slurm/transformer/Sbatch_trsf_for_SAE.sh "$P" "$v" "$L" "$m" "$batch_size" "" "" "$lr" "" "" "$DROPOUT" "" "$wd" "" "" "" "" "$MODEL"
# done

# Free-pooled (learned pooling head) variants. The model is passed as $18.
# Empty positional args ($14-$17: warmup/decay/test_loss_threshold/stop) are
# kept blank here. With residual stream use transformer_freeclass; without,
# transformer_freeclass_nores.
v=8
L=4
m=8
batch_size=256
P=524288
lr=0.001
DROPOUT=0.1
wd=0.0001
for MODEL in transformer_freeclass transformer_freeclass_nores; do
    sbatch slurm/transformer/Sbatch_trsf_for_SAE.sh "$P" "$v" "$L" "$m" "$batch_size" "" "" "$lr" "" "" "$DROPOUT" "" "$wd" "" "" "" "" "$MODEL"
done