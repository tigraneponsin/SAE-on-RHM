v=16
L=3
m=4
batch_size=16
P=12160
sbatch --array=0-4 slurm/transformer/Sbatch_trsf_for_stats.sh "$P" "$v" "$L" "$m" "$batch_size" "" "" "" "" "" "" "" "0.0001"