import json
import logging
import os
import re
import sys
import torch
from dataclasses import dataclass, field
from typing import List, Union

import wandb
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
    HfArgumentParser,
)

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))

from utils.model_utils import create_hf_model, load_hf_tokenizer
from utils.generation_utils import generate_completions
from utils.utils import set_random_seed

TEST_FILE = "test.json"

INPUT_PROMPT = '''<s> Below is an instruction that describes a task. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Response:
'''

@dataclass
class EvaluationConfig:
    data_path: str = field(
        default="",
        metadata={"help": "Path to the data directory containing datasets."}
    )
    datasets: List[str] = field(
        default_factory=lambda: ["boolq", "piqa", "social_i_qa", "ARC-Challenge",
                                 "ARC-Easy", "openbookqa", "hellaswag", "winogrande"],
        metadata={"help": "List of dataset names to evaluate on."}
    )
    output_dir: str = field(
        default="./results",
        metadata={"help": "Directory to save evaluation outputs."}
    )
    model_name_or_path: str = field(
        default="",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models."
        }
    )
    seed: int = field(
        default=1234,
        metadata={"help": "A seed for reproducible evaluation."}
    )
    dtype: str = field(
        default="bf16",
        metadata={
            "help": "Training data type",
            "choices": ["fp16", "bf16", "fp32"],
        },
    )
    use_flash_attn: bool = field(
        default=False, 
        metadata={"help": "Use Flash Attention."}
    )
    per_device_eval_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for evaluation."}
    )
    wandb_project: str = field(
        default="Evaluation",
        metadata={"help": "WandB project name."}
    )

# TODO: Move logger settup to utils dir
def setup_logging(accelerator: Accelerator):
    """
    Configures the logging based on the process rank.
    Only rank 0 will emit logs to the console.
    """

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if accelerator.is_main_process else logging.WARNING)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO if accelerator.is_main_process else logging.WARNING)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt="%m/%d/%Y %H:%M:%S"
    )
    handler.setFormatter(formatter)

    # Remove all existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(handler)
    return logger

def extract_answer(dataset: str, sentence: str) -> Union[float, str]:
    """
    Extracts the predicted answer from the generated sentence based on the dataset.
    """
    comm_datasets_pattern = {
        'boolq': r'true|false',
        'piqa': r'solution1|solution2',
        'social_i_qa': r'answer1|answer2|answer3|answer4|answer5',
        'ARC-Challenge': r'answer1|answer2|answer3|answer4|answer5',
        'ARC-Easy': r'answer1|answer2|answer3|answer4|answer5',
        'openbookqa': r'answer1|answer2|answer3|answer4|answer5',
        'hellaswag': r'ending1|ending2|ending3|ending4',
        'winogrande': r'option1|option2',
    }
    math_datasets = {"MultiArith", "gsm8k", "AddSub", "SingleEq", "SVAMP", "mawps"}
    math_letter_datasets = {"AQuA"}
    
    if dataset in comm_datasets_pattern:
        matches = re.findall(comm_datasets_pattern[dataset], sentence.lower().strip())
        return matches[-1] if matches else ""

    elif dataset in math_datasets:
        numbers = [s for s in re.findall(r'-?\d+\.?\d*', sentence.replace(',', ''))]
        if not numbers:
            return float('inf')
        try:
            return float(numbers[-1])
        except ValueError:
            return float('inf')
    
    elif dataset in math_letter_datasets:
        matches = re.findall(r'[ABCDE]', sentence.strip())
        return matches[-1] if matches else ""

    else:
        raise NotImplementedError(' not support dataset: {}'.format(dataset))


@torch.no_grad()
def main(config: EvaluationConfig):
    set_random_seed(config.seed)
    accelerator = Accelerator()
    
    # Setup logger first
    logger = setup_logging(accelerator)
    if accelerator.is_main_process:
        run_name_parts = config.output_dir.strip("/").split("/")
        run_name = f"test-{run_name_parts[-3]}-{run_name_parts[-1]}" if len(run_name_parts) >= 3 else run_name_parts[-1]
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config=vars(config),
            job_type="evaluation"
        )
        wandb_table = wandb.Table(columns=["Method"] + config.datasets)

    logger.info("Loading model and tokenizer...")
    tokenizer = load_hf_tokenizer(config.model_name_or_path, fast_tokenizer=True)
    tokenizer.padding_side = "left"
    logger.info(f"Tokenizer pad side: {tokenizer.padding_side}")

    model = create_hf_model(
        AutoModelForCausalLM,
        config.model_name_or_path,
        tokenizer,
        dtype=config.dtype,
        use_flash_attn=config.use_flash_attn,
    )

    device = accelerator.device
    model = model.to(device)

    model.eval()
    logger.info(f'Model dtype: {model.dtype}')

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        top_k=40,
        num_beams=4,
        max_new_tokens=512,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        bos_token_id=model.config.bos_token_id,
    )

    accelerator.wait_for_everyone()

    results = []
    for dataset in config.datasets:
        logger.info(f"***** Evaluating on dataset: {dataset} *****")

        data_file_path = os.path.join(config.data_path, dataset)
        test_file = os.path.join(data_file_path, TEST_FILE)

        with open(test_file, "r", encoding="utf8") as f:
            t_test_data = json.load(f)

        prompts = []
        for example in t_test_data:
            prompt = INPUT_PROMPT.format_map(example)
            prompts.append(prompt)
        logger.info(f"Sample prompt: {prompts[0]}")

        # Split prompts among processes
        with accelerator.split_between_processes(prompts) as split_prompts:
            model_outputs = generate_completions(
                model=model,
                device=device,
                logger=logger,
                tokenizer=tokenizer,
                prompts=split_prompts,
                batch_size=config.per_device_eval_batch_size,
                stop_id_sequences=[[tokenizer.eos_token_id]],
                generation_config=generation_config
            )

        # Gather outputs from all processes
        outputs = gather_object(model_outputs)

        save_outputs = []
        correct = 0
        for example, output in zip(t_test_data, outputs):
            example['raw_output'] = output
            target = example["answer"].lower()
            predict = extract_answer(dataset, output)
            example['prediction'] = predict
            if (isinstance(predict, float) and abs(float(target) - predict) <= 1e-4) or \
                (isinstance(predict, str) and predict.lower() == target):
                correct += 1
            else:
                save_outputs.append(example)

        logger.info(f"Saving outputs to {config.output_dir}")

        weighted_acc = (correct / len(t_test_data)) * 100
        logger.info(
            f"Dataset:\t{dataset}\nResult:\t{weighted_acc:.1f}\ntotal:\t{len(t_test_data)}"
        )

        results.append(weighted_acc)

        # Save incorrect predictions
        os.makedirs(config.output_dir, exist_ok=True)
        with open(os.path.join(config.output_dir, f"{dataset}.jsonl"), "w", encoding='utf-8') as fout:
            for example in save_outputs:
                fout.write(json.dumps(example) + "\n")

    if accelerator.is_main_process:
        row = [run_name_parts[-1]] + results
        wandb_table.add_data(*row)
        wandb.log({"test/evaluation_results_table": wandb_table})
        wandb.finish()

if __name__ == "__main__":
    parser = HfArgumentParser((EvaluationConfig))
    eval_config, = parser.parse_args_into_dataclasses()

    main(eval_config)
