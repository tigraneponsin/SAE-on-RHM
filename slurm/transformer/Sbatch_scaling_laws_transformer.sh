#!/bin/bash

#SBATCH --job-name test_rhm_trfclass
#SBATCH --chdir /home/ponsin
#SBATCH -o /dev/null
#SBATCH -e /dev/null

#SBATCH --partition h100
#SBATCH --time 10:00:00
#SBATCH --mem 90G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account pcsl
#SBATCH --array=0-2%3

# Arguments:
# $1=train_size
# $2=num_features
# $3=num_layers
# $4=num_synonyms       (optional, default: num_features)
# $5=batch_size         (default: 32)
# $6=embedding_dim      (default: 512)
# $7=num_heads          (default: 8)
# $8=lr                 (default: 1e-3)
# $9=max_epochs         (default: 20000)
# $10=ffwd_size         (default: 4)
# $11=dropout           (default: 0.1)
# $12=save_models       (default: 0, set to 1 to save model weights for post-hoc SAE)

DEVICE="cuda"
MODE="class"

TRAIN_SIZE=$1
NUM_FEATURES=$2
NUM_CLASSES=$2
NUM_SYNONYMS=${4:-$NUM_FEATURES}
TUPLE_SIZE=2

NUM_LAYERS=$3
DEPTH=$3
NUM_TOKENS=$(( TUPLE_SIZE ** NUM_LAYERS ))

EMBEDDING_DIM=${6:-512}
NUM_HEADS=${7:-8}
LEARNING_RATE=${8:-5e-3}
MAX_EPOCHS=${9:-20000}
FFWD_SIZE=${10:-4}
DROPOUT=${11:-0.1}
BATCH_SIZE=${5:-32}
SAVE_MODELS=${12:-0}

# Keep width for compatibility/logging in existing training code
WIDTH=$EMBEDDING_DIM

seed1=$(od -An -N3 -i /dev/random | tr -d ' ')
seed2=$(od -An -N3 -i /dev/random | tr -d ' ')
seed3=$(od -An -N3 -i /dev/random | tr -d ' ')

TEST_SIZE=32768
ACCUMULATION=1
INPUT_FORMAT="long"
WHITENING=0

MODEL="transformer_meanclass"
OPTIM="adam"
MOMENTUM=0.0
PRINT_FREQ=32768
SAVE_FREQ=2
LOSS_THRESHOLD=0.001

SAVE_MODEL_ARGS=()
if [[ "$SAVE_MODELS" == "1" ]]; then
    SAVE_MODEL_ARGS+=(--save_models)
fi

OUTNAME="RESULT_TRFCLASS_v_${NUM_CLASSES}_L_${NUM_LAYERS}_m=${NUM_SYNONYMS}_P_${TRAIN_SIZE}_${SLURM_ARRAY_TASK_ID}_emb_${EMBEDDING_DIM}_h_${NUM_HEADS}_lr_${LEARNING_RATE}_dropout_${DROPOUT}.pkl"

RESULTS_DIR="/work/pcsl/ponsin/Mean_Transformer/Transformer_scaling_laws/results_scale_law_varying_m/v_${NUM_FEATURES}_L_${NUM_LAYERS}_m_${NUM_SYNONYMS}/"

mkdir -p "$RESULTS_DIR"

exec > "${RESULTS_DIR}/${OUTNAME%.pkl}.out" 2> "${RESULTS_DIR}/${OUTNAME%.pkl}.err"

# Work directly in results directory
cd "$RESULTS_DIR" || exit 1

echo STARTING AT
date

srun python /home/ponsin/SAE-on-RHM/main.py \
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
    --outname "$OUTNAME" \
    "${SAVE_MODEL_ARGS[@]}"

echo FINISHED AT
date