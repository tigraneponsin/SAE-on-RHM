#!/bin/sh
# Default duration in hours if not provided
DURATION=${1:-9}

# Default cluster if not provided
CLUSTER=${2:-h100}

MEM=46G
NUM_THREADS=8
# Set partition based on cluster
if [ "$CLUSTER" = "l40s" ]; then
    PARTITION="l40s"
    QOS="--qos=debug"
elif [ "$CLUSTER" = "h100" ]; then
    PARTITION="h100"
    QOS=""
elif [ "$CLUSTER" = "mig" ]; then
    PARTITION="mig24gb"
    QOS=""
    MEM=24G
    NUM_THREADS=5
elif [ "$CLUSTER" = "any" ]; then
    PARTITION="mig24gb,l40s,h100"
    QOS=""
    MEM=24G
    NUM_THREADS=5
else
    echo "Error: Invalid cluster '$CLUSTER'. Use 'h100' or 'l40s'"
    exit 1
fi

# Convert hours to the format required by srun (0-hours:00:00)
TIME_FORMAT="0-${DURATION}:00:00"
srun -p ${PARTITION}  --mem=${MEM} --time=${TIME_FORMAT} -n 1 --gpus-per-node=1  --cpus-per-task=$NUM_THREADS --job-name=interactive --pty bash  #for kuma

# srun  --mem=80G --partition=bigmem --time=${TIME_FORMAT} -n $NUM_THREADS  --ntasks-per-node=$NUM_THREADS --job-name=interactive --pty bash  #for jed
# srun -p l40s --qos=debug  --mem=16G --time=0-01:00:00 -n 1 --gpus-per-node=1  --pty bash  #for kuma
# srun --partition=build --qos=build --time=0-02:00:00 -n 1 --mem=90G --pty bash  #works on izar.
# srun -p l40s --qos=debug  --pty bash  #for kuma
# h100 or l40s