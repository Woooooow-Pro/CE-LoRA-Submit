# CE-LoRA: Computational-Efficient LoRA Fine-Tuning for Language Models

## Installation

### 1. Environment Setup

Follow these steps to set up the virtual environment and install the necessary dependencies:

```bash
# Set Python version and create a virtual environment
pyenv local 3.12
python -m venv .venv 
source .venv/bin/activate

# Set CUDA environment variables
export CUDA_HOME="/usr/local/cuda"
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install required Python packages and CELoRA
pip install --upgrade torch scipy scikit-learn transformers datasets peft deepspeed triton \
  'huggingface_hub[cli,torch]' wandb
pip install flash-attn --no-build-isolation
pip install .
```

**Note:**

You must run flash-attn with gradient checkpointing enabled. This is necessary to manage memory efficiently during training.

### 2. Install Dataset

Clone the required dataset repository:

```bash
mkdir "$HOME/workspace"
git clone https://github.com/AGI-Edgerunners/LLM-Adapters.git ~/workspace/LLM-Adapters
```

### 3. Fine-tune & Evaluation

To reproduce CE-LoRA's results on the commonsense and arithmetic reasoning tasks please run `experiments/run_exp.sh` script. This scripts accept the following parameters.

The script parameters are as follows:
```
./script.sh [GPU_IDS] [OUTPUT] [WANDB_PROJECT] [TRAIN_MODE]
    GPU_IDS        Comma-separated GPU IDs to use (default: 0)
    BENCHMARK_TYPE      Type of benchmark: 'comm' or 'math' (default: comm)
    MODEL_NAME_OR_PATH  Path to the model or model identifier
    OUTPUT         Output directory name (default: celora_llama3)
    WANDB_PROJECT  Weights & Biases project name (default: CELORA_EXP)
    TRAIN_MODE     Training mode: 'lora' or 'celora' (default: celora)
```

#### Example

Fine-tune Llama3 using the CELoRA method on the commonsense benchmark:

```shell
./experiments/run_exp.sh 0 comm meta-llama/Llama-3.1-8B celora_llama3 COMM_BENCH celora
```

**Description:**

This command will:

- Save training outputs to `experiments/results/comm/train/celora_llama3`
- Create a `COMM_BENCH` project under your Weights & Biases account.

**Customization:**

To adjust CELoRA settings, modify the `experiments/train_script/configs/celora_config.json` file. Follow the format below:

```json
{
    "target_module_1": {
        "sample_ratio": 0.5, // Must be present and within [0, 1].
        "svd_rank": 10 // Optional: Specify only if customizing the SVD rank.
    }
    // Add additional modules as needed
}
```