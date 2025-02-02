# TODO Rewrite this god damn code
import torch
import tqdm
from typing import List
from transformers import (
    StoppingCriteria,
    PreTrainedTokenizerBase
)


class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(
            stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequences_should_be_stopped.append(True)
                    break
            sequences_should_be_stopped.append(False)
        return all(sequences_should_be_stopped)


@torch.no_grad()
def generate_completions(
    model,
    device,
    logger,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    batch_size: int = 1,
    stop_id_sequences=None,
    **generation_kwargs
):
    generations = []
    logger.info("-----Model Generation Args-----")
    try:
        logger.info(f"{model.generation_config}")
    except Exception:
        logger.info(f"{model.module.generation_config}")

    if generation_kwargs:
        logger.info(
            "-----GenerationConfig-----\n"
            f"{generation_kwargs}"
        )

    progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        tokenized_prompts = tokenizer(
            batch_prompts, 
            padding="longest",
            return_tensors="pt"
        ).to(device)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        try:
            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=[KeyWordsCriteria(stop_id_sequences)] if stop_id_sequences else None,
                **generation_kwargs
            )
            batch_outputs = batch_outputs.detach().cpu()

            # the stopping criteria is applied at batch level, so if other examples are not stopped, the entire batch will continue to generate.
            # so some outputs still have the stop sequence, which we need to remove.
            if stop_id_sequences:
                for output_idx in range(batch_outputs.shape[0]):
                    for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                        if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                            batch_outputs[output_idx,
                                          token_idx:] = tokenizer.pad_token_id
                            break

            # in case piece id out of range
            # batch_outputs[batch_outputs >= tokenizer.vocab_size] = tokenizer.unk_token_id
            # batch_outputs[batch_outputs == -1] = tokenizer.unk_token_id

            # remove the prompt from the output
            # we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
            # we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
            # space is important for some tasks (e.g., code completion).
            batch_outputs = tokenizer.batch_decode(
                batch_outputs, skip_special_tokens=True)
            batch_prompts = tokenizer.batch_decode(
                batch_input_ids, skip_special_tokens=True)
            # duplicate the prompts to match the number of return sequences
            batch_prompts = [
                prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
            batch_generations = [
                output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
            ]
        except Exception as e:
            logger.error(
                "Error when generating completions for batch:\n"
                f"Error message:\n{e}\n"
                "Use empty string as the completion."
            )
            batch_generations = [""] * \
                len(batch_prompts) * num_return_sequences

        generations += batch_generations

        progress.update(len(batch_prompts)//num_return_sequences)

    # DEBUG Info
    # assert len(generations) == len(
    #     prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations
