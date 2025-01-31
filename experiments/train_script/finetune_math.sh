#!/bin/bash

# Set strict mode for better error handling
set -euo pipefail

# Function to display usage information
usage() {
    echo "Usage: $0 [GPU_IDS] [OUTPUT] [WANDB_PROJECT] [TRAIN_MODE]"
    echo "  GPU_IDS            Comma-separated GPU IDs to use (default: 0)"
    echo "  OUTPUT             Output directory name (default: celora_llama3)"
    echo "  WANDB_PROJECT      Weights & Biases project name (default: CELORA_EXP)"
    echo "  TRAIN_MODE         Train with LoRA or CELoRA: 'lora' or 'celora' (default: celora)"
    echo "  MODEL_NAME_OR_PATH Path to the model or model identifier (default: '${EXPERIMENT_DIR}/results/${BENCHMARK_TYPE}/train/celora_llama3')"
    exit 1
}

if [ "$#" -gt 5 ]; then
    usage
fi

# Navigate to the experiment directory (one level up from the script's location)
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
DATA_PATH="$HOME/workspace/LLM-Adapters/ft-training_set/math_10k.json"

# Assign input arguments with default values
GPU_IDS=${1:-0}
OUTPUT=${2:-celora_llama3}
WANDB_PROJECT=${3:-CELORA_EXP}
TRAIN_MODE=${4:-celora} # 'lora' or 'celora'
MODEL_NAME_OR_PATH=${5:-"meta-llama/Llama-3.2-1B"}

# Parse GPU IDs into an array and count the number of GPUs
IFS=',' read -r -a GPU_ARRAY <<<"$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

# Validate GPU IDs
for GPU_ID in "${GPU_ARRAY[@]}"; do
    if ! nvidia-smi -L | grep -q "GPU $GPU_ID:"; then
        echo "Error: GPU ID $GPU_ID is not available."
        exit 1
    fi
done

# Create the output directory
OUTPUT=${EXPERIMENT_DIR}/results/math/train/${OUTPUT}
mkdir -p "$OUTPUT"

# Generate a random master port
MASTER_PORT=$((RANDOM % 5000 + 20000))

# Common arguments for both single and multiple GPU runs
COMMON_ARGS=(
    --model_name_or_path "$MODEL_NAME_OR_PATH"
    --data_path "$DATA_PATH"
    --output_dir "$OUTPUT"
    --num_train_epochs 1
    --train_batch_size 64
    --train_micro_batch_size_per_gpu 8
    --dtype bf16
    --seed 42
    --instruction_type single
    --learning_rate 1e-4
    --weight_decay 0.0
    --lr_scheduler_type linear
    --warmup_ratio 0.005
    --eval_batch_size 8
    --eval_set_size 120
    --eval_steps 80
    --eval_delay 0.5
    --max_seq_len 5120
    --enable_wandb
    --wandb_project "$WANDB_PROJECT"
    --lora
    --lora_r 16
    --lora_alpha 32
    --lora_dropout 0.05
    --task_type CAUSAL_LM
    --target_modules q_proj k_proj v_proj o_proj up_proj down_proj gate_proj
    --use_rslora
)

# Conditionally add CeLoRA arguments if TRAIN_MODE is 'celora'
if [[ "$TRAIN_MODE" == "celora" ]]; then
    COMMON_ARGS+=(
        --celora
        --celora_pattern "${EXPERIMENT_DIR}/train_script/configs/celora_config.json"
        --recompute_counter 200
    )
fi

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Running on a single GPU: ${GPU_ARRAY[0]}"
    # Export CUDA_VISIBLE_DEVICES for single GPU
    export CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}"
    # Execute the Python script without DeepSpeed
    python "${EXPERIMENT_DIR}/train/finetune.py" "${COMMON_ARGS[@]}" 2> >(tee "$OUTPUT/err.log" >&2) | tee "$OUTPUT/training_info.log"
else
    echo "Running on multiple GPUs: ${GPU_IDS}"
    # Export CUDA_VISIBLE_DEVICES for multiple GPUs
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    # Add DeepSpeed configuration for multi-GPU
    DEEPSPEED_ARGS=(--deepspeed "${EXPERIMENT_DIR}/train_script/configs/deepspeed.json")
    # Execute the torchrun command with DeepSpeed
    torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" "${EXPERIMENT_DIR}/train/finetune.py" "${DEEPSPEED_ARGS[@]}" "${COMMON_ARGS[@]}" 2> >(tee "$OUTPUT/err.log" >&2) | tee "$OUTPUT/training_info.log"
fi
