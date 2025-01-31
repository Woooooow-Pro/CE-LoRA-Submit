#!/bin/bash

# Set strict mode for better error handling
set -euo pipefail

# Function to display usage information
usage() {
    echo "Usage: $0 [GPU_IDS] [BENCHMARK_TYPE] [MODEL_NAME_OR_PATH] [OUTPUT] [WANDB_PROJECT]"
    echo "  GPU_IDS            Comma-separated GPU IDs to use (default: 0)"
    echo "  BENCHMARK_TYPE     Type of benchmark: 'comm' or 'math' (default: comm)"
    echo "  MODEL_NAME_OR_PATH Path to the model or model identifier (default: '${EXPERIMENT_DIR}/results/${BENCHMARK_TYPE}/train/celora_llama3')"
    echo "  OUTPUT             Output directory name (default: celora_llama3)"
    echo "  WANDB_PROJECT      Weights & Biases project name (default: CELORA_EXP)"
    exit 1
}

# Check for excessive arguments
if [ "$#" -gt 5 ]; then
    usage
fi

# Navigate to the experiment directory (one level up from the script's location)
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."

# Assign input arguments with default values
GPU_IDS=${1:-0}
BENCHMARK_TYPE=${2:-comm} # choose from "comm" and "math"
MODEL_NAME_OR_PATH=${3:-"${EXPERIMENT_DIR}/results/${BENCHMARK_TYPE}/train/celora_llama3"}
OUTPUT=${4:-celora_llama3}
WANDB_PROJECT=${5:-CELORA_EXP}
BATCH_SIZE=8
DATA_PATH="$HOME/workspace/LLM-Adapters/dataset"

# Generate a random master port
MASTER_PORT=$((RANDOM % 5000 + 20000))

# Validate BENCHMARK_TYPE
if [[ "$BENCHMARK_TYPE" != "comm" && "$BENCHMARK_TYPE" != "math" ]]; then
    echo "Error: Invalid BENCHMARK_TYPE '$BENCHMARK_TYPE'. Choose either 'comm' or 'math'."
    usage
fi

# Create the output directory
OUTPUT="${EXPERIMENT_DIR}/results/${BENCHMARK_TYPE}/test/${OUTPUT}"
mkdir -p "$OUTPUT"

# Parse GPU IDs into an array and count the number of GPUs
IFS=',' read -r -a GPU_ARRAY <<<"$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

# Validate GPU IDs
for GPU_ID in "${GPU_ARRAY[@]}"; do
    if ! nvidia-smi -i "$GPU_ID" &>/dev/null; then
        echo "Error: GPU ID $GPU_ID is not available or does not exist."
        exit 1
    fi
done

# Set dataset
DATASETS=()

if [[ "$BENCHMARK_TYPE" == "comm" ]]; then
    DATASETS+=(boolq piqa social_i_qa ARC-Challenge ARC-Easy openbookqa hellaswag winogrande)
elif [[ "$BENCHMARK_TYPE" == "math" ]]; then
    DATASETS+=(MultiArith gsm8k AddSub AQuA SingleEq SVAMP mawps)
fi

# Execute the evaluation script with logging
echo "Starting evaluation with the following settings:"
echo "GPU_IDS: $GPU_IDS"
echo "BENCHMARK_TYPE: $BENCHMARK_TYPE"
echo "MODEL_NAME_OR_PATH: $MODEL_NAME_OR_PATH"
echo "OUTPUT_DIR: $OUTPUT"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "Datasets: ${DATASETS[@]}"

accelerate launch --main_process_port $MASTER_PORT \
    --gpu_ids ${GPU_IDS} \
    --num_processes ${NUM_GPUS} \
    ${EXPERIMENT_DIR}/eval/evaluate.py \
    --data_path $DATA_PATH \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --per_device_eval_batch_size $BATCH_SIZE \
    --seed 1234 \
    --dtype bf16 \
    --datasets ${DATASETS[@]} \
    --output_dir "$OUTPUT" \
    --wandb_project $WANDB_PROJECT \
    2> >(tee "$OUTPUT/err.log" >&2) | tee "$OUTPUT/eval_info.log"
