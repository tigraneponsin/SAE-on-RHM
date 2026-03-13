#v=16
#L=2
#m=8
#batch_size=256
#ffwd_size=20

#for P in 512 768 1024 2048 2816 3584 4864 6144 7680 9984 12544 16128 20480 26112 33280 42240 54016 68608; do
#    sbatch Sbatch_fine_tuning.sh "$P" "$v" "$L" "$m" "$batch_size" "" "" "" "" "$ffwd_size"
#done

#v=16
#L=2
#m=8
#batch_size=256
#n_h=256

#for P in 512 768 1024 2048 2816 3584 4864 6144 7680 9984 12544 16128 20480 26112 33280 42240 54016 68608; do
#    sbatch Sbatch_fine_tuning.sh "$P" "$v" "$L" "$m" "$batch_size" "" "$n_h"
#done

#v=16
#L=2
#m=8
#batch_size=256
#n_h=8

#for P in 512 768 1024 2048 2816 3584 4864 6144 7680 9984 12544 16128 20480 26112 33280 42240 54016 68608; do
#    sbatch Sbatch_fine_tuning.sh "$P" "$v" "$L" "$m" "$batch_size" "" "$n_h"
#done

#v=8
#L=4
#m=8
#batch_size=256
#n_h=4
#N_REAL=5

#for P in 10240 15360 20480 30720 51200 61440 81920 122880 163840 204800; do
#    sbatch --array=0-$((N_REAL-1))%$N_REAL Sbatch_fine_tuning.sh "$P" "$v" "$L" "$m" "$batch_size" "" "$n_h"
#done

#v=16
#L=3
#m=9
#P=89600
#for batch_size in 32 64 128 256 512; do
#    sbatch Sbatch_fine_tuning.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=16
#L=3
#m=9
#P=89600

#for batch_size in 1280; do
#    sbatch Sbatch_fine_tuning.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=16
#L=3
#m=5
#batch_size=16
#for P in   4224 6016 8576 12160 17280 24576 34944 49664  ; do
#    sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done


v=16
L=3
m=5
for P in 3840 4480 5120 5760 6400 7040 7680 8320 8960 10240; do #16640 21760 25600
   for batch_size in 64; do
       sbatch Sbatch_parameter_tuning.sh "$P" "$v" "$L" "$m" "$batch_size" "" "" "" "" "" "0.05"
   done
done

# v=16
# L=3
# m=3
# for P in 256 512 640 896 1280 1792 2560 3840; do
#     for batch_size in 4, 8, 16, 32, 64, 128; do
#         sbatch Sbatch_parameter_tuning.sh "$P" "$v" "$L" "$m" "$batch_size"
#     done
# done


# v=16
# L=3
# m=9
# for P in 20480 25600 30720 40960; do #50176 70656 76800 82944 90112 95232 100352 128000 192512; do
#     for batch_size in 32 64 128 256 1024; do
#         sbatch Sbatch_parameter_tuning.sh "$P" "$v" "$L" "$m" "$batch_size"
#     done
# done

# v=16
# L=3
# m=9
# for P in 20480 25600 30720 40960 50176 70656 76800 82944; do
#     for batch_size in 128; do
#         sbatch Sbatch_parameter_tuning.sh "$P" "$v" "$L" "$m" "$batch_size" "" "" "" "" "" "0.1"
#     done
# done

