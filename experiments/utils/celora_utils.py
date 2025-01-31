import copy

from typing import List, Dict, Tuple
from torch.nn import Linear

from celora import CeLoRALinear


SVD_RANK_NAME = "svd_rank"
SAMPLE_RATIO_NAME = "sample_ratio"


def get_balanced_rank(lora_rank: int, svd_rank: int = None) -> int:
    """Returns the balanced rank based on the provided conditions."""
    if svd_rank is None:
        return int(lora_rank * 7 / 8)
    return lora_rank - int(svd_rank / 7)


def update_lora_rank_pattern(
    lora_rank: int,
    target_modules: List[str],
    celora_pattern: Dict[str, Dict[str, int]]
) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    """Updates LoRA rank patterns for the specified modules and adjusts the celora pattern accordingly."""
    lora_rank_pattern = {module: lora_rank for module in target_modules}

    if celora_pattern is None:
        return lora_rank_pattern, None

    for module in target_modules:
        if module in celora_pattern:
            pattern = celora_pattern[module]
            if SVD_RANK_NAME in pattern:
                # Update LoRA rank based on the provided SVD rank in the pattern
                lora_rank_pattern[module] = get_balanced_rank(
                    lora_rank, pattern[SVD_RANK_NAME]
                )
            else:
                # If SVD rank is not specified, use default calculation
                lora_rank_pattern[module] = get_balanced_rank(lora_rank)
                # Store the balanced rank as SVD rank in the pattern
                pattern[SVD_RANK_NAME] = lora_rank_pattern[module]

    return lora_rank_pattern, celora_pattern


def convert_PEFT_to_CELoRA(model, recompute_counter, celora_pattern: dict):
    target_modules = set(celora_pattern.keys())

    def is_in_target_modules(name):
        # Check if any layer from the target modules is in the name (use set lookup for efficiency)
        return next((layer for layer in target_modules if layer in name), None)

    def replace_frozen_layer(module, layer_name=None):
        for child_name, child_module in list(module.named_children()):
            if layer_name is None:
                layer_name = is_in_target_modules(child_name)

            # Only replace lora related layers.
            # TODO Support replacing all frozen layers.
            if "base_layer" in child_name and isinstance(child_module, Linear) and layer_name is not None:
                checkpoint = copy.deepcopy(child_module.state_dict())

                celora_layer = CeLoRALinear(
                    in_features=child_module.in_features,
                    out_features=child_module.out_features,
                    bias=child_module.bias,
                    device=next(child_module.parameters()).device,
                    dtype=next(child_module.parameters()).dtype,
                    sample_ratio=celora_pattern[layer_name][SAMPLE_RATIO_NAME],
                    recompute_counter=recompute_counter,
                )
                celora_layer.load_state_dict(checkpoint, strict=False)
                celora_layer.sparse_weight(
                    celora_pattern[layer_name][SVD_RANK_NAME])
                setattr(module, child_name, celora_layer)

                del child_module
                del checkpoint
            else:
                replace_frozen_layer(child_module, layer_name)

    replace_frozen_layer(model)
