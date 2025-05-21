import operator

import torch
from lxt import functional as lfunctional
from lxt import rules as lrules
from torch import nn
from transformers.activations import (
    AccurateGELUActivation,
    ClippedGELUActivation,
    FastGELUActivation,
    GELUActivation,
    LaplaceActivation,
    LinearActivation,
    MishActivation,
    NewGELUActivation,
    PytorchGELUTanh,
    QuickGELUActivation,
    ReLUSquaredActivation,
)
from zennit import types as ztypes

from ..model import activations
from .propagator_lxt import LFPEpsilon, ParameterizableComposite, RuleGenerator

# Breakdown of Rules / Caveats when applying LFP to ViT
# ViTEmbeddings --> Ignore when fintuning? (i.e. set to nograd and dont update)

# ViTEncoder --> Fine
# ViTLayer --> Take care of Residual Sums
# ViTAttention --> fine
# ViTSdpaAttention --> Replace with ViTAttention (equivalent),
#   otherwise functionality is hidden behind a torch function
# ViTSelfAttention -->
#    q, k, v are linear layer, so fine.
#    q, k matmul --> LXT Eps and Uniform rule
#    division --> LXT Identity (div rule)
#    softmax --> LXT softmax DTD at x
#    Uniform OR Identity/grad? dropout/masking --> gradient and identity.
#    v matmul --> LXT Eps and Uniform rule
# ViTSelfOutput --> Linear and dropout, simple LFP-eps
# ViTIntermediate --> Linear into activation, so simple LFP-Eps + Identity
# ViTOutput --> Linear and dropout and residual sum, so LFP-eps + vanilla-grad + taking care of sum

# LayerNorm --> LFP-Eps, similar to BatchNorm

# ViTPooler --> Standard Grad, linear layer is taken care of

# classifier --> Standard Eps-Rule

# linear layers --> Standard Eps Rule
# activations --> standard identity rule. CAREFUL, ACT2FN may have new activation types


# Questions:
# - ResSums/Matmul/softmax in LXT? --> operator
# - Uniform Rule vs. Identity/grad?
# - SDPA Attention vs Attention
# - Train Embeddings?


# Div2 Function for LXT
class div2_fn(lfunctional.Function):
    """
    Uniform LRP rule for elementwise division (along all dimensions) of two tensors according to Proposition 3.2 of the
    paper
    'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'

    If one of the inputs is a constant or does not require gradients, the relevance is distributed 100% to the other
    input.

    Parameters:
    -----------
    input_a: torch.Tensor
        The first input tensor
    input_b: torch.Tensor
        The second input tensor
    inplace: bool
        Whether to perform the operation in place during the backward pass, will overwrite the relevance at the output
    """

    @staticmethod
    def forward(ctx, input_a, input_b, inplace=False):
        ctx.requires_grads = [
            i for i, inp in enumerate((input_a, input_b)) if isinstance(inp, torch.Tensor) and inp.requires_grad
        ]
        ctx.inplace = inplace

        return input_a / input_b

    @staticmethod
    @lfunctional.conservation_check_wrap
    def backward(ctx, *out_relevance):
        n_required = len(ctx.requires_grads)

        if ctx.inplace:
            out_relevance = out_relevance[0].div_(n_required)
        else:
            out_relevance = out_relevance[0] / n_required

        # only return relevance at requires_grad indices else None
        return tuple(out_relevance if i in ctx.requires_grads else None for i in range(2)) + (None,)


class TransformersActivation(metaclass=ztypes.SubclassMeta):
    """Abstract base class that describes activation modules defined in transformers package."""

    __subclass__ = (
        PytorchGELUTanh,
        NewGELUActivation,
        GELUActivation,
        FastGELUActivation,
        QuickGELUActivation,
        ClippedGELUActivation,
        AccurateGELUActivation,
        MishActivation,
        LinearActivation,
        LaplaceActivation,
        ReLUSquaredActivation,
    )


class LFPEpsilonComposite(ParameterizableComposite):
    def __init__(self, norm_backward=False, epsilon=1):
        layer_map = {
            ztypes.Activation: lrules.IdentityRule,
            TransformersActivation: lrules.IdentityRule,
            activations.Step: lrules.IdentityRule,
            ztypes.AvgPool: RuleGenerator(
                LFPEpsilon,
                epsilon=epsilon,
            ),
            ztypes.Linear: RuleGenerator(
                LFPEpsilon,
                epsilon=epsilon,
            ),
            ztypes.BatchNorm: RuleGenerator(
                LFPEpsilon,
                epsilon=epsilon,
            ),
            nn.LayerNorm: RuleGenerator(
                LFPEpsilon,
                epsilon=epsilon,
            ),
            operator.add: lfunctional.add2,
            operator.truediv: div2_fn,
            operator.matmul: lfunctional.matmul,
            nn.functional.softmax: lfunctional.softmax,
        }

        super().__init__(layer_map=layer_map)
