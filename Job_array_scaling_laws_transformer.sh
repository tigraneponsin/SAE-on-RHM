# P=$((128*50))
# m=4
# L=4
# lr=0.01
# max_epoch=20000

m=4
L=4
for P in 384 512 640 896 1152 1408 1792 2304 2944 3712 4736 6016 7680 9856 12544 16000 20352 25984 33152 42240 53888; do
    sbatch Sbatch_example_TP.sh "$P" "$m" "$L" "$lr" "$max_epoch"
done

#m=5
#L=3
#lr=0.01
#max_epoch=200000

#for P in 128 256 384 512 640 896 1152 1408 1792 2304 2944 3712 4736 6016 7680 9856 12544 16000 20352 ; do
#    sbatch Sbatch_example_TP.sh "$P" "$m" "$L" "$lr" "$max_epoch"
#done

#m=8
#L=3
#lr=0.01
#max_epoch=200000

#for P in 1024 1536 2048 2560 3072 4096 5120 6144 7680 9216 10752 12288 13824 15360 20480 25600 35840  ; do
    #sbatch Sbatch_example_TP.sh "$P" "$m" "$L" "$lr" "$max_epoch"
#done


m=10
L=3
lr=0.01
max_epoch=200000

for P in 3072 4096 5120 6144 7680 9216 10752 12288 13824 15360 16896 18432 20480 25600 35840 39680 47360 60160 ; do
    sbatch Sbatch_example_TP.sh "$P" "$m" "$L" "$lr" "$max_epoch"
done



