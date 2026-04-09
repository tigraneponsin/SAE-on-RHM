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
# batch_size=16
# for P in 12160; do
#     sbatch slurm/transformer/Sbatch_trsf_for_SAE.sh "$P" "$v" "$L" "$m" "$batch_size" "" "" "" "" "" "" "" "0.0001"
#  done

v=16
L=3
m=4
batch_size=16
P=12160
for wd in 0.00008 0.00009 0.00011 0.00012; do
    sbatch slurm/transformer/Sbatch_trsf_for_SAE.sh "$P" "$v" "$L" "$m" "$batch_size" "" "" "" "" "" "" "" "$wd"
 done
