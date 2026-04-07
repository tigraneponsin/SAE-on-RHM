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


v=16
L=3
m=4
batch_size=16
for P in 12160; do
    sbatch Sbatch_trsf_for_SAE.sh "$P" "$v" "$L" "$m" "$batch_size" "" "" "" "" "" "" "" "0.01"
 done

