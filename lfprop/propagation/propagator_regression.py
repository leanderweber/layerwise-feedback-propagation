import torch
from lxt import rules as lrules
from zennit import core as zcore
from zennit import types as ztypes

from ..model import activations
from ..model.custom_resnet import Sum
from .propagator_lxt import LFPEpsilon, ParameterizableComposite, RuleGenerator


class LFPRegressionLastLayer(lrules.EpsilonRule):
    """
    LFP Rule for Regression Last Layer
    """

    def __init__(
        self,
        module,
        epsilon=1e-6,
        inplace=True,
    ):
        super(LFPRegressionLastLayer, self).__init__(module, epsilon)
        self.inplace = inplace

    def forward(self, *inputs):
        return lfp_regression_last_layer.apply(
            self.module,
            self.epsilon,
            self.inplace,
            *inputs,
        )


# TODO: This is not really working atm... The reference point chosen as target may not be good as well...
class lfp_regression_last_layer(lrules.epsilon_lrp_fn):
    """
    LFP Epsilon Rule for Regression Last Layer
    """

    @staticmethod
    def forward(ctx, fn, epsilon, inplace, *inputs):
        # create boolean mask for inputs requiring gradients
        requires_grads = [True if inp.requires_grad else False for inp in inputs]

        # detach inputs to avoid overwriting gradients if same input is used as multiple arguments
        # (like in self-attention)
        inputs = tuple(inp.detach().requires_grad_() if inp.requires_grad else inp for inp in inputs)

        # get parameters to store for backward. Here, we want to accumulate reward, so we do not detach
        params = [param for _, param in fn.named_parameters(recurse=False) if param.requires_grad]

        with torch.enable_grad():
            outputs = fn(*inputs)

        (
            ctx.epsilon,
            ctx.requires_grads,
            ctx.inplace,
        ) = (
            epsilon,
            requires_grads,
            inplace,
        )
        # save only inputs requiring gradients
        inputs = tuple(inputs[i] for i in range(len(inputs)) if requires_grads[i])
        ctx.save_for_backward(*inputs, *params, outputs)

        ctx.n_inputs, ctx.n_params = len(inputs), len(params)
        ctx.fn = fn

        return outputs.detach()

    @staticmethod
    def backward(ctx, *incoming_target):
        outputs = ctx.saved_tensors[-1]
        inputs = ctx.saved_tensors[: ctx.n_inputs]
        params = ctx.saved_tensors[ctx.n_inputs : ctx.n_inputs + ctx.n_params]

        # Assume we get the target, NOT the reward
        incoming_reward = (incoming_target[0] - outputs) * torch.where(
            outputs.sign() == 0, 1.0, outputs.sign()
        )  # Compute L1 Reward

        # We use the target in the denominator
        normed_reward = incoming_reward / zcore.stabilize(
            (outputs - incoming_target[0]),
            ctx.epsilon,
            clip=False,
            norm_scale=False,
            dim=None,
        )

        z_target = incoming_target[0] * normed_reward

        # compute param reward (used to update parameters)
        for param in params:
            if not isinstance(param, tuple):
                param = (param,)  # noqa: PLW2901
            param_grads_1 = torch.autograd.grad(
                outputs, param, normed_reward, retain_graph=True
            )  # a_i * 1/(o_c-y_c) * r_c
            param_grads_2 = torch.autograd.grad(
                outputs, param, z_target, retain_graph=True
            )  # a_i * y_c/(o_c-y_c) * r_c
            param_grads_3 = torch.autograd.grad(outputs, param, torch.ones_like(outputs), retain_graph=True)  # a_i
            param_reward = tuple(
                param_grads_1[i] * param[i].abs()
                - torch.where(
                    param_grads_3[i] != 0,
                    param_grads_2[i] / param_grads_3[i],
                    0.0,
                )
                for i in range(len(param))
            )  # (|w_ic|*a_i - y_c)/(o_c-y_c) * r_c

            if torch.isnan(param_reward[0]).sum() > 0:
                # print(incoming_target)
                # print(outputs)
                # print(incoming_reward)
                # print(normed_reward)
                # print(z_target)
                print(param_grads_1)
                print(param_grads_2)
                print(param_grads_3)
                print(param_grads_1[0] * param[0].abs())
                print(
                    torch.where(
                        param_grads_2[0] != 0,
                        param_grads_2[0] / param_grads_3[0],
                        0.0,
                    )
                )
                # print(param_reward)
                exit

            for i in range(len(param)):
                param[i].feedback = param_reward[i]

        # compute input reward (= outgoing reward to propagate)
        input_grads_1 = torch.autograd.grad(
            outputs, inputs, normed_reward, retain_graph=True
        )  # w_ic * 1/(o_c-y_c) * r_c
        input_grads_2 = torch.autograd.grad(outputs, inputs, z_target, retain_graph=True)  # w_ic * y_c/(o_c-y_c) * r_c
        input_grads_3 = torch.autograd.grad(outputs, inputs, torch.ones_like(outputs), retain_graph=False)  # w_ic

        outgoing_reward = tuple(
            (
                input_grads_1[i] * inputs[i]
                - torch.where(
                    input_grads_3[i] != 0,
                    input_grads_2[i] / input_grads_3[i],
                    0.0,
                )
                if ctx.requires_grads[i]
                else None
            )
            for i in range(len(ctx.requires_grads))
        )  # (z_ic-y_c)/(o_c-y_c) * r_c

        # return relevance at requires_grad indices else None
        return (None, None, None) + outgoing_reward


class LFPEpsilonRegressionComposite(ParameterizableComposite):
    def __init__(self, epsilon=1e-6):
        layer_map = {
            "last": RuleGenerator(
                LFPRegressionLastLayer,
                epsilon=epsilon,
            ),
            ztypes.Activation: lrules.IdentityRule,
            activations.Step: lrules.IdentityRule,
            Sum: RuleGenerator(LFPEpsilon, epsilon=epsilon, inplace=False),
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
        }

        super().__init__(layer_map=layer_map)
