#m=4
#L=4
#for P in 2944 3712 4736 6016 7680 9856 12544 16000 20352 25984 33152 42240 53888 68480 87680 111360 142592 181120 231808 294400 376832 478592 612608; do
   # sbatch Sbatch_scaling_laws_transformer.sh "$P" "$m" "$L" 
#done

#m=5
#L=3


#for P in 512 640 896 1152 1408 1792 2304 2944 3712 4736 6016 7680 9856 12544 16000 20352 25984 33152 42240 53888 68480 87680 111360 142592 181120 231808; do
    #sbatch Sbatch_scaling_laws_transformer.sh "$P" "$m" "$L" 
#done

#m=8
#L=3


#for P in  3072 4096 5120 6144 7680 9216 10752 12288 13824 15360 20480 25600 35840 45696 58368 74496 95104 121344 154112 196608 250880 320128 408448; do
    #sbatch Sbatch_scaling_laws_transformer.sh "$P" "$m" "$L"
#done


#m=10
#L=3


#for P in  9216 10752 12288 13824 15360 16896 18432 20480 25600 35840 39680 47360 60160 76800 98048 125056 159616 203648 259840 331136 422144 538240 686336; do
    #sbatch Sbatch_scaling_laws_transformer.sh "$P" "$m" "$L" 
#done

#m=12
#L=3

#for P in 18432 20480 25600 35840 40960 47360 60160 76800 98048 125056 159616 203648 259840 331136 422144 538240 686336 875136 1115776 1422592 1813760 2312576; do
    #sbatch Sbatch_scaling_laws_transformer.sh "$P" "$m" "$L" 
#done

#m=16
#L=3

#for P in 60160 76800 98048 125056 159616 203648 259840 331136 422144 538240 686336 875136 1115776 1422592 1813760 2312576 2948480 3759360 4793216 6111360 7792000 9934848 12666880 16150272; do
    #sbatch Sbatch_scaling_laws_transformer.sh "$P" "$m" "$L" 
#done

#m=20
#L=3

#for P in 125056 159616 203648 259840 331136 422144 538240 686336 875136 1115776 1422592 1813760 2312576 2948480 3759360 4793216 6111360 7792000 9934848 12666880 16150272; do
    #sbatch Sbatch_scaling_laws_transformer.sh "$P" "$m" "$L" 
#done

#m=6
#L=2

#for P in 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048 2304 2560 2816 3072 3456 3840 4224 4608 5120 5760 6400 7040 7680 8576 9472 10752 12288; do
    #sbatch Sbatch_scaling_laws_transformer.sh "$P" "$m" "$L"
#done


#m=8
#L=2

#for P in 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048 2304 2560 2816 3072 3456 3840 4224 4608 5120 5760 6400 7040 7680 8576 9472 10752 12288 13824 15360 16896 18432 20480; do
    #sbatch Sbatch_scaling_laws_transformer.sh "$P" "$m" "$L"
#done

#m=10
#L=2

#for P in 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048 2304 2560 2816 3072 3456 3840 4224 4608 5120 5760 6400 7040 7680 8576 9472 10752 12288 13824 15360 16896 18432 20480 25600 35840 39680; do
    #sbatch Sbatch_scaling_laws_transformer.sh "$P" "$m" "$L"
#done

#m=16
#L=2

#for P in 4096 5120 6144 7680 9216 10752 12288 13824 15360 20480 25600 35840 45696 58368 74496 95104 121344 154112 196608 250880; do
    #sbatch Sbatch_scaling_laws_transformer.sh "$P" "$m" "$L"
#done

#v=4
#L=2
#m=2
#batch_size=4
#for P in 4 8 16 32 64 96 128 152 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048; do
    #sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=4
#L=2
#m=4
#batch_size=4

#for P in 4 8 16 32 64 96 128 152 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048; do
   # sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=4
#L=3
#m=2
#batch_size=4

#for P in 4 8 16 32 64 96 128 152 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048; do
    #sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=4
#L=3
#m=4
#batch_size=32

#for P in 32 64 96 128 160 192 256 352 480 672 960 1344 1888 2688 3744 5280 7424 10432 14656 20480; do
    #sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=4
#L=4
#m=2
#batch_size=4
#for P in 4 8 16 32 64 96 128 152 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048; do
   #sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=4
#L=4
#m=4
#batch_size=128
#for P in 1024 2048 2944 3712 4736 6016 7680 9856 12544 16000 20352 25984 33152 42240 53888 68480 87680 111360 142592 181120 231808; do
    #sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=8
#L=2
#m=2
#batch_size=4
#for P in 4 8 16 32 64 96 128 152 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048; do
    #sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=8
#L=2
#m=4
#batch_size=16
#for P in 16 32 48 64 80 96 128 160 192 256 352 480 672 960 1344 1888 2688 3744 5280 7424 10432 14656 20480 ; do
    #sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=8
#L=2
#m=8
#batch_size=128
#for P in 128 256 512 768 1024 2048 2944 3712 4736 6016 7680 9856 12544 16000 20352 25984 33152 42240 53888 68480; do
    #sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=8
#L=3
#m=2
#batch_size=4
#for P in 4 8 16 32 64 96 128 152 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048; do
    #sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=8
#L=3
#m=4
#batch_size=128
#for P in 128 256 512 768 1024 2048 2944 3712 4736 6016 7680 9856 12544 16000 20352 25984 33152 42240 53888 68480; do
 #   sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=8
#L=3
#m=8
#batch_size=128
#for P in 1024 2048 3072 4096 5120 6144 7680 9216 10752 12288 13824 15360 20480 25600 35840 45696 58368 74496 95104 121344 154112 196608 250880 ; do
    #sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=8
#L=4
#m=2
#batch_size=16
#for P in 16 32 48 64 80 96 128 160 192 256 352 480 672 960 1344 1888 2688 3744 5280 7424 10432 14656 20480 ; do
    #sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=8
#L=4
#m=4
#batch_size=128
#for P in 1024 2048 3072 4096 5120 6144 7680 9216 10752 12288 13824 15360 20480 25600 35840 45696 58368 74496 95104 121344 154112 196608 25088; do
   #sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=8
#L=4
#m=4
#batch_size=128
#for P in 10752 12288 13824 15360 20480 25600 35840 45696 58368 74496 95104 121344 154112 196608 25088 35840 45696 58368 74496 95104 121344 154112 196608 250880 331136 422144 538240 686336 875136; do
   #sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=16
#L=2
#m=2
#batch_size=4
#for P in 4 8 16 32 64 96 128 152 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048; do
#    sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done   

#v=16
#L=2
#m=4
#batch_size=32
#for P in 32 64 96 128 160 192 256 352 480 672 960 1344 1888 2688 3744 5280 7424 10432 14656 20480; do
#    sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=16
#L=2
#m=8
#batch_size=128
#for P in 1024 2048 2944 3712 4736 6016 7680 9856 12544 16000 20352 25984 33152 42240 53888 68480 87680 111360 142592 181120 231808; do
#   sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=16
#L=3
#m=2
#batch_size=16
#for P in 16 32 48 64 80 96 128 160 192 256 352 480 672 960 1344 1888 2688 3744 5280 7424 10432 14656 20480 ; do
#    sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=16
#L=3
#m=4
#batch_size=128
#for P in 1024 2048 2944 3712 4736 6016 7680 9856 12544 16000 20352 25984 33152 42240 53888 68480 87680 111360 142592 181120 231808 ; do
#    sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=16
#L=3
#m=8
#batch_size=128
#for P in 10752 12288 13824 15360 20480 25600 35840 45696 58368 74496 95104 121344 154112 196608 25088 35840 45696 58368 74496 95104 121344 154112 196608 250880; do
#    sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=16
#L=3
#m=2
#batch_size=16
#for P in 16 32 48 64 80 96 128 160 192 256 352 480 672 960 1344 1888 2688 3744 5280 7424; do
#    sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=16
#L=3
#m=3
#batch_size=16
#or P in 16 32 48 64 80 96 128 160 192 256 352 480 672 960 1344 1888 2688 3744 5280 7424; do
 #   sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=16
#L=3
#m=4
#batch_size=128
#for P in 128 256 384 512 640 768 1024 1536 2176 2944 4224 6016 8576 12160 17280 24576 34944 49664 70400 99968; do
#    sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=16
#L=3
#m=5
#batch_size=128
#for P in  512 640 768 1024 1536 2176 2944 4224 6016 8576 12160 17280 24576 34944 49664 70400 99968 128000 192000 256000 384000 ; do
#    sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=16
#L=3
#m=6
#batch_size=128
#for P in  2944 4224 6016 8576 12160 17280 24576 34944 38400 42880 44800 4800 49664 70400 99968 128000 192000 ; do
#    sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=16
#L=3
#m=7
#batch_size=128
#for P in    8576 12160 17280 24576 34944 49664 51200 55040 58880 62720 66560 70400 99968 128000 192000  ; do
#    sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done


#v=16
#L=3
#m=8
#batch_size=128
#for P in   2176 2944 4224 6016 8576 12160 17280 24576 34944 49664 55040 62720 66560 70400 99968 128000 192000 256000 384000 ; do
    #sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=16
#L=3
#m=8
#batch_size=128
#for P in 55040 62720 66560; do
#    sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=16
#L=3
#m=9
#batch_size=128
#for P in   2176 2944 4224 6016 8576 12160 17280 24576 34944 49664 70400 76800 83200 89600 94720 99968 128000 192000 256000 384000 ; do
    #sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=16
#L=3
#m=9
#batch_size=128
#for P in 76800 83200 89600 94720; do
#    sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=16
##L=3
#m=10
#batch_size=1280
#for P in 26880 32000 38400 47360 57600 69120 84480 102400 124160 151040 184320 224000 271360 330240; do
#    sbatch Sbatch_scaling_laws_transformer.sh "$P" "$v" "$L" "$m" "$batch_size"
#done

#v=16
#L=3
#m=12
#batch_size=1280
#for P in 


