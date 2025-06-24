import torch
from lxt import rules as lrules
from zennit import core as zcore
from zennit import types as ztypes

from ..model import activations
from ..model.custom_resnet import Sum
from .propagator_lxt import LFPEpsilon, ParameterizableComposite, RuleGenerator


class LFPRegressionLastLayer(lrules.EpsilonRule):
    """
    Custom LFP (Layer-wise Feedback Propagation) Epsilon Rule for the last layer in regression tasks.
    This rule modifies the relevance propagation for the regression output layer.
    """

    def __init__(
        self,
        module,
        epsilon=1e-6,
        inplace=True,
    ):
        """
        Initialize the LFPRegressionLastLayer.

        Args:
            module: The module (layer) to which this rule is applied.
            epsilon (float): Stabilization term to avoid division by zero.
            inplace (bool): Whether to perform operations in-place.
        """
        super(LFPRegressionLastLayer, self).__init__(module, epsilon)
        self.inplace = inplace

    def forward(self, *inputs):
        """
        Forward pass for the custom regression last layer rule.

        Args:
            *inputs: Input tensors to the module.

        Returns:
            Output tensor after applying the custom rule.
        """
        return lfp_regression_last_layer.apply(
            self.module,
            self.epsilon,
            self.inplace,
            *inputs,
        )


# TODO: This is not really working atm... The reference point chosen as target may not be good as well...
class lfp_regression_last_layer(lrules.epsilon_lrp_fn):
    """
    Custom autograd Function implementing the LFP Epsilon Rule for the last layer in regression.
    Handles both forward and backward passes for relevance propagation.
    """

    @staticmethod
    def forward(ctx, fn, epsilon, inplace, *inputs):
        """
        Forward pass for the custom regression last layer rule.

        Args:
            ctx: Context object for saving information for backward computation.
            fn: The module (layer) function.
            epsilon (float): Stabilization term.
            inplace (bool): Whether to perform operations in-place.
            *inputs: Input tensors.

        Returns:
            Output tensor detached from the computation graph.
        """
        # Track which inputs require gradients
        requires_grads = [True if inp.requires_grad else False for inp in inputs]

        # Detach inputs to avoid overwriting gradients if same input is used multiple times
        inputs = tuple(inp.detach().requires_grad_() if inp.requires_grad else inp for inp in inputs)

        # Get parameters to store for backward (do not detach, as we want to accumulate reward)
        params = [param for _, param in fn.named_parameters(recurse=False) if param.requires_grad]

        with torch.enable_grad():
            outputs = fn(*inputs)

        # Save context for backward
        (
            ctx.epsilon,
            ctx.requires_grads,
            ctx.inplace,
        ) = (
            epsilon,
            requires_grads,
            inplace,
        )
        # Save only inputs requiring gradients
        inputs = tuple(inputs[i] for i in range(len(inputs)) if requires_grads[i])
        ctx.save_for_backward(*inputs, *params, outputs)

        ctx.n_inputs, ctx.n_params = len(inputs), len(params)
        ctx.fn = fn

        return outputs.detach()

    @staticmethod
    def backward(ctx, *incoming_target):
        """
        Backward pass for custom relevance propagation.

        Args:
            ctx: Context object with saved tensors and parameters.
            *incoming_target: Incoming target tensor(s) for relevance propagation.

        Returns:
            Tuple of gradients for each input to the forward function.
        """
        outputs = ctx.saved_tensors[-1]
        inputs = ctx.saved_tensors[: ctx.n_inputs]
        params = ctx.saved_tensors[ctx.n_inputs : ctx.n_inputs + ctx.n_params]

        # Compute incoming reward as L1 difference between target and output, scaled by sign
        incoming_reward = (incoming_target[0] - outputs) * torch.where(
            outputs.sign() == 0, 1.0, outputs.sign()
        )  # Compute L1 Reward

        # Normalize reward using stabilized denominator
        normed_reward = incoming_reward / zcore.stabilize(
            (outputs - incoming_target[0]),
            ctx.epsilon,
            clip=False,
            norm_scale=False,
            dim=None,
        )

        z_target = incoming_target[0] * normed_reward

        # Compute parameter reward (used to update parameters)
        for param in params:
            if not isinstance(param, tuple):
                param = (param,)  # Noqa: PLW2901
            # Compute gradients for different reward terms
            param_grads_1 = torch.autograd.grad(outputs, param, normed_reward, retain_graph=True)
            param_grads_2 = torch.autograd.grad(outputs, param, z_target, retain_graph=True)
            param_grads_3 = torch.autograd.grad(outputs, param, torch.ones_like(outputs), retain_graph=True)

            # Combine gradients to compute parameter reward
            param_reward = tuple(
                param_grads_1[i] * param[i].abs()
                - torch.where(
                    param_grads_3[i] != 0,
                    param_grads_2[i] / param_grads_3[i],
                    0.0,
                )
                for i in range(len(param))
            )

            # Debugging: print if NaNs are detected
            if torch.isnan(param_reward[0]).sum() > 0:
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
                exit

            # Store feedback in parameter
            for i in range(len(param)):
                param[i].feedback = param_reward[i]

        # Compute input reward (outgoing reward to propagate)
        input_grads_1 = torch.autograd.grad(outputs, inputs, normed_reward, retain_graph=True)
        input_grads_2 = torch.autograd.grad(outputs, inputs, z_target, retain_graph=True)
        input_grads_3 = torch.autograd.grad(outputs, inputs, torch.ones_like(outputs), retain_graph=False)

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
        )

        # Return None for non-tensor arguments, followed by outgoing rewards
        return (None, None, None) + outgoing_reward


class LFPEpsilonRegressionComposite(ParameterizableComposite):
    """
    Composite rule for LFP Epsilon propagation in regression models.
    Maps different layer types to their corresponding propagation rules.
    """

    def __init__(self, epsilon=1e-6):
        """
        Initialize the composite rule with a mapping from layer types to rules.

        Args:
            epsilon (float): Stabilization term for all epsilon-based rules.
        """
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
