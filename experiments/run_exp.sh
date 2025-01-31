#!/bin/bash

# Set strict mode for better error handling
set -euo pipefail

# Navigate to the experiment directory (one level up from the script's location)
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Assign input arguments with default values
GPU_IDS=${1:-0}
BENCHMARK_TYPE=${2:-comm} # choose from "comm" and "math"
MODEL_NAME_OR_PATH=${3:-"meta-llama/Llama-3.1-8B"}
OUTPUT=${4:-celora_llama3}
WANDB_PROJECT=${5:-CELORA_EXP}
TRAIN_MODE=${6:-celora}

EVAL_MODEL_PATH=${EXPERIMENT_DIR}/results/${BENCHMARK_TYPE}/train/${OUTPUT}/last_model

if [[ "$BENCHMARK_TYPE" == "comm" ]]; then
    ${EXPERIMENT_DIR}/train_script/finetune_commensense.sh ${GPU_IDS} ${OUTPUT} ${WANDB_PROJECT} ${TRAIN_MODE} ${MODEL_NAME_OR_PATH}
elif [[ "$BENCHMARK_TYPE" == "math" ]]; then
    ${EXPERIMENT_DIR}/train_script/finetune_math.sh ${GPU_IDS} ${OUTPUT} ${WANDB_PROJECT} ${TRAIN_MODE} ${MODEL_NAME_OR_PATH}
fi

${EXPERIMENT_DIR}/eval_script/evaluate.sh ${GPU_IDS} ${BENCHMARK_TYPE} ${EVAL_MODEL_PATH} ${OUTPUT} ${WANDB_PROJECT}