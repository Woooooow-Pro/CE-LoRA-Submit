import torch
import torch.nn as nn
from torch.nn import Linear

from peft import LoraConfig, get_peft_model

from celora import CeLoRALinear

device = "cuda:0"
dtype = torch.bfloat16

ce_sample_ratio = 1/16
ce_rank = 56
recompute_counter = 200
steps = 500

lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=["linear"],
    lora_dropout=0.05,
    bias="none"
)


in_features = 4096
out_features = 4096

micro_batch_size = 8 * 2
sequence_len = 4096 * 2

inputs = torch.rand(micro_batch_size * sequence_len * in_features, device=device, dtype=dtype)
outputs_grad = torch.rand(micro_batch_size * sequence_len * out_features, device=device, dtype=dtype)

inputs = inputs.reshape(micro_batch_size, sequence_len, in_features).requires_grad_(True)
outputs_grad = outputs_grad.reshape(micro_batch_size, sequence_len, out_features)
torch.backends.cuda.matmul.allow_tf32 = True

def replace_base_layer(model, ce_layer):
    for child_name, child_module in list(model.named_children()):
        if "base_layer" in child_name and isinstance(child_module, Linear):
            setattr(model, child_name, ce_layer)
        else:
            replace_base_layer(child_module, ce_layer)


class WrapModel(nn.Module):
    def __init__(self):
        super().__init__()

    def set_linear(self, linear):
        self.linear = linear

    def forward(self, x):
        return self.linear(x)


if __name__ == "__main__":
    ce_layer = CeLoRALinear(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        device=device,
        dtype=dtype,
        sample_ratio=ce_sample_ratio,
        recompute_counter=recompute_counter
    )
    ce_layer.reset_parameters()
    ce_layer.sparse_weight(ce_rank)

    linear_layer = Linear(
        in_features=in_features,
        out_features=in_features,
        bias=False,
        device=device,
        dtype=dtype,
    )

    wrap1 = WrapModel()
    wrap2 = WrapModel()

    wrap1.set_linear(linear_layer)
    wrap1 = get_peft_model(wrap1, lora_config)
    replace_base_layer(wrap1, ce_layer)

    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    cnt = 0

    torch.cuda.empty_cache()

    ce_timer = 0
    wrap1.train()
    for i in range(steps):
        if i >= 10:
            cnt += 1
            begin.record()
        outputs = wrap1(inputs)
        outputs.backward(outputs_grad)

        if i >= 10:
            end.record()
            torch.cuda.synchronize()
            ce_timer += begin.elapsed_time(end)

        wrap1.zero_grad()
        if inputs.grad is not None:
            inputs.grad.zero_()
    print(f"One Layer CELoRA Time (Avg): {ce_timer/cnt}(ms)")

    del wrap1
    torch.cuda.empty_cache()

    wrap2.set_linear(linear_layer)
    wrap2 = get_peft_model(wrap2, lora_config)
    
    cnt = 0
    lora_timer = 0
    wrap2.train()
    for i in range(steps):
        if i >= 10:
            cnt += 1
            begin.record()
        outputs = wrap2(inputs)
        outputs.backward(outputs_grad)
        if i >= 10:
            end.record()
            torch.cuda.synchronize()
            lora_timer += begin.elapsed_time(end)

        wrap2.zero_grad()
        if inputs.grad is not None:
            inputs.grad.zero_()

    print(f"One Layer LoRA Time (Avg): {lora_timer/cnt}(ms)")
