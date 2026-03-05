#!/bin/bash

#SBATCH --job-name test_rhm_trfclass
#SBATCH --chdir /home/ponsin
#SBATCH -o /home/ponsin/results_scale_law_trfclass/%x_%A_%a.out
#SBATCH -e /home/ponsin/results_scale_law_trfclass/%x_%A_%a.err

#SBATCH --partition h100
#SBATCH --time 10:00:00
#SBATCH --mem 90G
#SBATCH --cpus-per-task=16
#SBATCH --account pcsl
#SBATCH --array=0-4%5

# Arguments:
# $1=train_size
# $2=num_features
# $3=num_layers
# $4=embedding_dim      (default: 512)
# $5=num_heads          (default: 8)
# $6=lr                 (default: 1e-3)
# $7=max_epochs         (default: 20000)
# $8=ffwd_size          (default: 4)
# $9=dropout            (default: 0.0)

DEVICE="cuda"
MODE="class"

TRAIN_SIZE=$1
NUM_FEATURES=$2
NUM_CLASSES=$2
NUM_SYNONYMS=$2
TUPLE_SIZE=2

NUM_LAYERS=$3
DEPTH=$3
NUM_TOKENS=$(( TUPLE_SIZE ** NUM_LAYERS ))

EMBEDDING_DIM=${4:-512}
NUM_HEADS=${5:-8}
LEARNING_RATE=${6:-1e-3}
MAX_EPOCHS=${7:-20000}
FFWD_SIZE=${8:-4}
DROPOUT=${9:-0.0}

# Keep width for compatibility/logging in existing training code
WIDTH=$EMBEDDING_DIM

seed1=$(od -An -N3 -i /dev/random | tr -d ' ')
seed2=$(od -An -N3 -i /dev/random | tr -d ' ')
seed3=$(od -An -N3 -i /dev/random | tr -d ' ')

BATCH_SIZE=128
TEST_SIZE=32768
ACCUMULATION=1
INPUT_FORMAT="long"
WHITENING=0

MODEL="transformer_class"
OPTIM="adam"
MOMENTUM=0.0
PRINT_FREQ=32768
SAVE_FREQ=2
LOSS_THRESHOLD=0.001

OUTNAME="RESULT_TRFCLASS_v_${NUM_CLASSES}_L_${NUM_LAYERS}_P_${TRAIN_SIZE}_${SLURM_ARRAY_TASK_ID}_emb_${EMBEDDING_DIM}_h_${NUM_HEADS}_lr_${LEARNING_RATE}.pkl"

RESULTS_DIR="/home/ponsin/results_scale_law_trfclass"

mkdir -p "$RESULTS_DIR"

exec > "${RESULTS_DIR}/${OUTNAME%.pkl}.out" 2> "${RESULTS_DIR}/${OUTNAME%.pkl}.err"

path="${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$path"
cd "$path" || exit 1

echo STARTING AT
date

srun python /home/ponsin/random-hierarchy-model/main.py \
    --device "$DEVICE" \
    --mode "$MODE" \
    --num_features "$NUM_FEATURES" \
    --num_classes "$NUM_CLASSES" \
    --num_synonyms "$NUM_SYNONYMS" \
    --tuple_size "$TUPLE_SIZE" \
    --num_layers "$NUM_LAYERS" \
    --num_tokens "$NUM_TOKENS" \
    --seed_rules "$seed1" \
    --train_size "$TRAIN_SIZE" \
    --test_size "$TEST_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --seed_sample "$seed2" \
    --input_format "$INPUT_FORMAT" \
    --whitening "$WHITENING" \
    --model "$MODEL" \
    --depth "$DEPTH" \
    --width "$WIDTH" \
    --embedding_dim "$EMBEDDING_DIM" \
    --num_heads "$NUM_HEADS" \
    --ffwd_size "$FFWD_SIZE" \
    --dropout "$DROPOUT" \
    --seed_model "$seed3" \
    --lr "$LEARNING_RATE" \
    --optim "$OPTIM" \
    --accumulation "$ACCUMULATION" \
    --momentum "$MOMENTUM" \
    --max_epochs "$MAX_EPOCHS" \
    --print_freq "$PRINT_FREQ" \
    --save_freq "$SAVE_FREQ" \
    --loss_threshold "$LOSS_THRESHOLD" \
    --outname "$OUTNAME"

echo FINISHED AT
date