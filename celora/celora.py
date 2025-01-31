import math
from typing import Optional

import torch
import torch.nn as nn

from torch import Tensor
from torch.autograd import Function
from torch.nn.parameter import Parameter


class CeLinearFunction(Function):
    @staticmethod
    def forward(
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        A0: Tensor, B0: Tensor,
        k: int,
        index: Optional[Tensor] = None,
        recompute_index: bool = False,
        callback: callable = None
    ) -> Tensor:
        output = input @ weight.t()
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight,  bias, A0, B0, k, index, recompute_index, callback = inputs
        ctx.save_for_backward(input, weight, bias, A0, B0)
        ctx.k = k
        ctx.index = index
        ctx.recompute_index = recompute_index
        ctx.callback = callback

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, A0, B0 = ctx.saved_tensors
        grad_input = None

        if ctx.needs_input_grad[0]:
            out_features, in_features = weight.size()
            grad_output = grad_output.reshape(-1, out_features)
            grad_input = torch.empty(grad_output.size(0), in_features)
            if ctx.k >= weight.size(0):
                grad_input = grad_output @ weight
                return grad_input.reshape(input.shape), None, None, None, None, None, None, None, None

            sorted_index = ctx.index

            sparse_weight_selected = torch.addmm(weight[sorted_index], B0[sorted_index], A0, alpha=-1)
            grad_input = (grad_output @ B0) @ A0
            grad_input.addmm_(grad_output[..., sorted_index], sparse_weight_selected)

            if ctx.recompute_index:
                # Compute norms using simplified math:
                # norms[i] = sqrt( (sum_{b,s} grad_output[b,s,i]^2)*(sum_j sparse_weight[i,j]^2) )
                sparse_weight = torch.addmm(input=weight, mat1=B0, mat2=A0, alpha=-1)
                G_sums = (grad_output**2).sum(dim=[0, 1])  # shape (O,)
                W_sums = (sparse_weight**2).sum(dim=1)     # shape (O,)
                norms = torch.sqrt(G_sums * W_sums)
                # sorted index can make index select much more faster
                sorted_index = torch.topk(norms, ctx.k, largest=True, sorted=False)[1].sort()[0]
                # Callback to update index
                ctx.callback(sorted_index)

        return grad_input.reshape(input.shape), None, None, None, None, None, None, None, None


class CeLoRALinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None, dtype=None,
        sample_ratio: float = None,
        recompute_counter: int = 200,
    ) -> None:
        super().__init__()
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs), requires_grad=False
        )

        if bias:
            self.bias = Parameter(
                torch.empty(out_features, **factory_kwargs), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

        # CELoRA parameter
        if sample_ratio > 1 or sample_ratio < 0:
            raise ValueError("Sampling ratio should choose between 0 to 1")
        self.sample_ratio = sample_ratio
        
        if sample_ratio < 1:
            self.k = math.ceil(out_features * sample_ratio)
            self.register_buffer(
                "index", torch.LongTensor([i for i in range(out_features)])
            )
            self.index = self.index.to(device)
        else:
            self.k = out_features
            self.index = None

        self.recompute_counter = recompute_counter
        self.counter = recompute_counter + 1

    def sparse_weight(self, svd_rank: int = None):
        factory_kwargs = {
            "device": self.weight.device,
            "dtype": self.weight.dtype,
        }

        gpu_weight = self.weight.clone().to('cuda').to(torch.float32)

        U, S, Vh = torch.linalg.svd(gpu_weight, full_matrices=False)
        svd_rank = min(len(S), 0 if svd_rank == None else svd_rank)
        S = torch.sqrt(S[:svd_rank])

        self.A0 = Parameter(
            (torch.diag(S) @ Vh[:svd_rank, :]).to(**factory_kwargs),
            requires_grad=False
        )
        self.B0 = Parameter(
            (U[:, :svd_rank] @ torch.diag(S)).to(**factory_kwargs),
            requires_grad=False
        )

        del gpu_weight
        del U
        del S
        del Vh

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def set_index(self, index):
        self.index = index

    def forward(self, input: Tensor) -> Tensor:
        recompute_index = False
        if self.counter > self.recompute_counter:
            self.counter = 0
            recompute_index = True
        self.counter += 1
        return CeLinearFunction.apply(input, self.weight, self.bias, self.A0, self.B0, self.k, self.index, recompute_index, self.set_index)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, r={self.sample_ratio},'
