import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import transformers
import wandb
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    HfArgumentParser,
    IntervalStrategy,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from utils.data_utils import SupervisedDataset, DataCollatorForSupervisedDataset
from utils.model_utils import (
    create_hf_model,
    get_latest_checkpoint,
    load_hf_tokenizer,
    make_model_gradient_checkpointing_compatible,
)
from utils.celora_utils import convert_PEFT_to_CELoRA, update_lora_rank_pattern
from utils.utils import set_random_seed

LAST_MODEL_DIR = "last_model"

# Setup logging
# TODO: There is multi-process problem in logging. Author 
# will create a root logger to fix this issue later.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
transformers.logging.set_verbosity_info()


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models."
        }
    )
    dropout: float = field(default=0.0, metadata={"help": "Dropout rate of the model."})
    use_flash_attn: bool = field(
        default=True, metadata={"help": "Use Flash Attention."}
    )


@dataclass
class DataTrainingArguments:
    data_path: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Path to the training dataset. Expected to be a JSON file."
        },
    )
    max_seq_len: int = field(
        default=4096, metadata={"help": "The maximum sequence length."}
    )
    instruction_type: str = field(
        default="single",
        metadata={"help": "`single` for single-round instructions."},
    )


@dataclass
class TrainingConfig:
    num_train_epochs: int = field(
        default=1, metadata={"help": "Total number of training epochs to perform."}
    )
    train_batch_size: int = field(
        default=64, metadata={"help": "Total batch size for training."}
    )
    train_micro_batch_size_per_gpu: int = field(
        default=8, metadata={"help": "Micro batch size per GPU."}
    )
    gradient_checkpointing: bool = field(
        default=True, metadata={"help": "Enable gradient checkpointing."}
    )
    dtype: str = field(
        default="bf16",
        metadata={
            "help": "Training data type",
            "choices": ["fp16", "bf16", "fp32"],
        },
    )
    seed: int = field(
        default=42, metadata={"help": "A seed for reproducible training."}
    )
    learning_rate: float = field(
        default=5e-5, metadata={"help": "Initial learning rate."}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay."}
    )
    lr_scheduler_type: str = field(
        default="linear",
        metadata={
            "help": "The scheduler type to use.",
            "choices": [
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
            ],
        },
    )
    warmup_ratio: float = field(
        default=0.005, metadata={"help": "Warmup ratio for learning rate scheduler."}
    )
    eval_batch_size: int = field(
        default=16, metadata={"help": "Batch size for evaluation."}
    )
    eval_set_size: int = field(
        default=0,
        metadata={
            "help": "Size of the validation set. If 0, no validation set is used."
        },
    )
    eval_steps: int = field(
        default=80, metadata={"help": "Evaluate every n steps."}
    )
    eval_delay: float = field(
        default=0, metadata={"help": "Evaluate after certain steps or ratio of steps."}
    )
    deepspeed: Optional[str] = field(
        default=None, metadata={"help": "Path to Deepspeed config file."}
    )
    # LoRA related configs
    lora: bool = field(
        default=False, metadata={"help": "Use LoRA for efficient training."}
    )
    lora_r: int = field(
        default=32, metadata={"help": "LoRA rank."}
    )
    lora_alpha: int = field(
        default=32, metadata={"help": "LoRA alpha parameter."}
    )
    lora_dropout: float = field(
        default=0.0, metadata={"help": "LoRA dropout."}
    )
    task_type: str = field(
        default="CAUSAL_LM",
        metadata={
            "help": "Task type for PEFT.",
            "choices": [
                "CAUSAL_LM",
                "SEQ_CLS",
                "SEQ_2_SEQ_LM",
                "TOKEN_CLS",
                "QUESTION_ANS",
                "FEATURE_EXTRACTION",
            ],
        },
    )
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        metadata={"help": "Modules to apply the adapter to."},
    )
    use_rslora: bool = field(
        default=False, metadata={"help": "Use Rank-Stabilized LoRA."}
    )
    # CELoRA related configs
    celora: bool = field(
        default=False, metadata={"help": "Use CELoRA for efficient training."}
    )
    svd_rank: Optional[int] = field(
        default=None, metadata={"help": "SVD rank for CELoRA."}
    )
    recompute_counter: int = field(
        default=200, metadata={"help": "Recompute counter for CELoRA."}
    )
    celora_pattern: Optional[str] = field(
        default=None,
        metadata={"help": "Path to CELoRA pattern JSON file."},
    )
    # Logging Related
    enable_wandb: bool = field(
        default=False, metadata={"help": "Enable Weights & Biases logging."}
    )
    wandb_project: str = field(
        default="CELORA", metadata={"help": "WandB project name."}
    )
    output_dir: str = field(
        default="./results",
        metadata={"help": "Where to store the model and checkpoints."},
    )
    save_steps: int = field(
        default=500, metadata={"help": "Save checkpoint every n update steps."}
    )

class CustomWandBCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.accumulated_loss = 0.0
        self.total_steps = 0
        self.start_time = None
        self.samples = 0
        self.start = 0.0
        self.time_counter = 0.0
        self.steps = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        # Reset metrics
        self.accumulated_loss = 0.0
        self.total_steps = 0
        self.samples = 0
        self.time_counter = 0.0
        self.steps = 0

    def on_step_begin(self, args, state, control, **kwargs):
        self.start = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        self.time_counter += time.time() - self.start
        self.steps += 1

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Calculate latency since last log
        step_latency = None
        if self.steps != 0:
            step_latency = self.time_counter / self.steps
        self.time_counter = 0.0
        self.steps = 0

        # Accumulate loss
        loss = logs.get("loss")
        if loss is not None:
            self.accumulated_loss += loss
            self.total_steps += 1

        # Update GPU memory usage
        current_memory = self.get_total_gpu_memory()

        # Update samples processed
        # Assuming batch size remains constant; adjust if dynamic
        batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        self.samples += batch_size

        # Compute moving average loss
        moving_avg_loss = self.accumulated_loss / \
            self.total_steps if self.total_steps > 0 else 0

        # Estimate TFLOPs (Placeholder)
        # tflops = self.estimate_tflops()

        # Log custom metrics to wandb
        logging_envents = {
            "train_logs/moving_avg_loss": moving_avg_loss,
            "train_logs/gpu_memory_GB": current_memory,
            "train_logs/samples_processed": self.samples
        }
        if step_latency:
            logging_envents["train_logs/latency"] = step_latency
        wandb.log(logging_envents, step=state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            # Log evaluation metrics
            wandb.log(metrics, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        wandb.log({"train_summary/total_training_time_sec": total_time},
                  step=state.global_step)

    def get_total_gpu_memory(self):
        if torch.cuda.is_available():
            total_memory = 0
            for device in range(torch.cuda.device_count()):
                mem = torch.cuda.memory_allocated(
                    device) / (1024 ** 3)  # Convert to MB
                total_memory += mem
            return total_memory
        return 0

def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingConfig))
    model_args, data_args, training_config = parser.parse_args_into_dataclasses()

    # Log initial information
    logger.warning(
        f"Process rank: {0}, device: {torch.device("cuda" if torch.cuda.is_available(
        ) else "cpu")}, n_gpu: {torch.cuda.device_count()}"
    )
    logger.info(f"Training/evaluation parameters {training_config}")
    
    # Set celora_pattern as early as possible
    if training_config.lora and training_config.celora and training_config.celora_pattern:
        with open(training_config.celora_pattern, "r") as f:
            celora_pattern = json.load(f)
    else:
        celora_pattern = None

    # Set seed for reproducibility
    set_random_seed(training_config.seed)

    # Load tokenizer
    tokenizer = load_hf_tokenizer(model_args.model_name_or_path, fast_tokenizer=True)
    tokenizer.model_max_length = data_args.max_seq_len
    logger.info(f"Tokenizer max length: {tokenizer.model_max_length}")

    # Load model
    logger.info(f"Loading model from {model_args.model_name_or_path}")
    model = create_hf_model(
        AutoModelForCausalLM,
        model_args.model_name_or_path,
        tokenizer,
        dropout=model_args.dropout,
        use_flash_attn=model_args.use_flash_attn,
        dtype=training_config.dtype,
    )

    # Apply LoRA/CELoRA if enabled
    if training_config.lora:

        # Update LoRA rank pattern if CELoRA is used
        lora_rank_pattern, celora_pattern = update_lora_rank_pattern(
            training_config.lora_r, training_config.target_modules, celora_pattern
        )
        

        lora_config = LoraConfig(
            r=training_config.lora_r,
            lora_alpha=training_config.lora_alpha,
            lora_dropout=training_config.lora_dropout,
            task_type=training_config.task_type,
            target_modules=training_config.target_modules,
            use_rslora=training_config.use_rslora,
            rank_pattern=lora_rank_pattern,
        )
        model = get_peft_model(model, lora_config)

        if training_config.celora and celora_pattern is not None:
            convert_PEFT_to_CELoRA(
                model=model,
                celora_pattern=celora_pattern,
                recompute_counter=training_config.recompute_counter
            )
        model = make_model_gradient_checkpointing_compatible(model)
    
    # Initialize W&B if enabled
    run_name = training_config.output_dir.split("/")
    run_name = f"{run_name[-3]}-{run_name[-1]}" if len(run_name) >= 3 else run_name[-1]
    training_config.celora_pattern = celora_pattern
    if training_config.enable_wandb:
        wandb.init(
            project=training_config.wandb_project,
            name=run_name,
            config=vars(training_config),
            save_code=True,
        )

    # # # # # # # # # # # # # # #
    #       Prepare Datasets     #
    # # # # # # # # # # # # # # #
    logger.info("------ Preparing datasets ------")

    # Initialize the SupervisedDataset with the first data path
    train_dataset = SupervisedDataset(
        data_path=data_args.data_path[0],
        tokenizer=tokenizer,
        instruction_type=data_args.instruction_type,
        model_name_or_path=model_args.model_name_or_path,
    )

    # If multiple data paths are provided, you can concatenate datasets
    if len(data_args.data_path) > 1:
        additional_datasets = [
            SupervisedDataset(
                data_path=path,
                tokenizer=tokenizer,
                instruction_type=data_args.instruction_type,
                model_name_or_path=model_args.model_name_or_path,
            )
            for path in data_args.data_path[1:]
        ]
        train_dataset = torch.utils.data.ConcatDataset(
            [train_dataset] + additional_datasets)
        logger.info(f"Concatenated {len(data_args.data_path)} datasets.")

    eval_dataset = None
    if training_config.eval_set_size > 0:
        logger.info(
            f"Splitting dataset into train and validation sets with eval_set_size={training_config.eval_set_size}"
        )
        total_size = len(train_dataset)
        train_size = total_size - training_config.eval_set_size
        val_size = training_config.eval_set_size
        train_dataset, eval_dataset = torch.utils.data.random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(training_config.seed),
        )
        logger.info(f"Train set size: {train_size}, Validation set size: {val_size}")
    else:
        logger.info("No validation set will be used.")

    # Setup Data Collator
    logger.info("------ Setting up data collator ------")
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # Define TrainingArguments
    gradient_accumulation_steps = training_config.train_batch_size // training_config.train_micro_batch_size_per_gpu
    need_evaluate = training_config.eval_set_size > 0
    
    training_args = TrainingArguments(
        output_dir=training_config.output_dir,
        overwrite_output_dir=False,
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.train_micro_batch_size_per_gpu,
        per_device_eval_batch_size=training_config.eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy=IntervalStrategy.STEPS if need_evaluate else IntervalStrategy.NO,
        save_strategy=IntervalStrategy.STEPS,
        save_steps=training_config.eval_steps if need_evaluate else training_config.save_steps,
        save_total_limit=1,
        eval_steps=training_config.eval_steps if need_evaluate else None,
        logging_steps=5,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        lr_scheduler_type=training_config.lr_scheduler_type,
        warmup_ratio=training_config.warmup_ratio,
        fp16=training_config.dtype == "fp16",
        bf16=training_config.dtype == "bf16",
        deepspeed=training_config.deepspeed,
        gradient_checkpointing=training_config.gradient_checkpointing,
        report_to=["wandb"] if training_config.enable_wandb else [],
        run_name=run_name,
        seed=training_config.seed,
        load_best_model_at_end=need_evaluate,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_grad_norm=0.9,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[CustomWandBCallback()] if training_config.enable_wandb else [],
    )
    
    # Resume from checkpoint if available
    logger.info("*** Starting training ***")
    checkpoint_path = get_latest_checkpoint(training_config.output_dir)
    if checkpoint_path:
        logger.info(f"Found checkpoint at {checkpoint_path}. Resuming training from this checkpoint.")
    else:
        logger.info("No checkpoint found. Starting training from scratch.")
    try:
        trainer.train(resume_from_checkpoint=checkpoint_path)
    except FileNotFoundError as e:
        logger.error("The best checkpoint saving failed, but it will not hurt training.")
        logger.error(e)
    
     # Save the final model
    if training_args.output_dir is not None:
        last_model_dir = os.path.join(training_args.output_dir, LAST_MODEL_DIR)
        trainer_state_file = os.path.join(last_model_dir, "trainer_state.json")
        trainer.save_model(last_model_dir)
        tokenizer.save_pretrained(last_model_dir)
        if training_config.lora:
            model.save_pretrained(last_model_dir)
        trainer.state.save_to_json(trainer_state_file)
        logger.info("Last model saved.")

    # Evaluation
    if eval_dataset is not None:
        logger.info("*** Evaluating ***")
        eval_results = trainer.evaluate()
        logger.info(json.dumps(eval_results, indent=4))

    # Finish W&B run
    if training_config.enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
