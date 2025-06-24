"""
Propagator for training SNN with LFP
"""

import torch
from lxt import rules as lrules
from zennit import core as zcore

from ..model.spiking_networks import SpikingLayer
from .propagator_lxt import ParameterizableComposite, RuleGenerator


class LFPEpsilonSNN(lrules.EpsilonRule):
    """
    LFP Epsilon Rule for Spiking Neural Networks (SNN).

    This rule implements the epsilon-LRP (Layer-wise Relevance Propagation) for SNNs,
    adapted for use with the LFP (Layer-wise Feedback Propagation) framework.
    """

    def __init__(
        self,
        module,
        epsilon=1e-6,
        inplace=True,
    ):
        """
        Initialize the LFPEpsilonSNN rule.

        Args:
            module: The module (layer) to which this rule is applied.
            epsilon (float): Stabilizer for denominator to avoid division by zero.
            inplace (bool): Whether to perform operations in-place.
        """
        super(LFPEpsilonSNN, self).__init__(module, epsilon)
        self.inplace = inplace

        # For compatibility with certain modules (e.g., transformers' modeling_vit.py)
        if hasattr(module, "weight"):
            self.weight = module.weight

    def forward(self, *inputs):
        """
        Forward pass for the LFP epsilon rule.

        Args:
            *inputs: Input tensors to the module.

        Returns:
            Output(s) of the wrapped module, detached from the computation graph.
        """
        return epsilon_lfp_snn_fn.apply(
            self.module,
            self.epsilon,
            self.inplace,
            *inputs,
        )


class epsilon_lfp_snn_fn(lrules.epsilon_lrp_fn):
    """
    Custom autograd function for LFP Epsilon Rule in SNNs.

    This function handles both the forward and backward passes for relevance propagation
    through spiking layers, making assumptions about the wrapped module (e.g., SpikingLayer).
    """

    @staticmethod
    def forward(ctx, fn: SpikingLayer, epsilon, inplace, *inputs):
        """
        Forward pass for the LFP epsilon rule.

        Args:
            ctx: Context object for saving information for backward computation.
            fn (SpikingLayer): The spiking layer module.
            epsilon (float): Stabilizer for denominator.
            inplace (bool): Whether to perform operations in-place.
            *inputs: Input tensors.

        Returns:
            Output(s) of the spiking layer, detached from the computation graph.
        """
        assert isinstance(fn, SpikingLayer)

        # Track which inputs require gradients
        requires_grads = [True if inp.requires_grad else False for inp in inputs]

        # Detach inputs to avoid overwriting gradients if reused
        inputs = tuple(inp.detach().requires_grad_() if inp.requires_grad else inp for inp in inputs)

        # Gather parameters that require gradients
        params = [param for _, param in fn.parameterized_layer.named_parameters(recurse=True) if param.requires_grad]

        # Reset feedback and internal reward attributes
        for param in params:
            if hasattr(param, "feedback"):
                del param.feedback
        if hasattr(fn, "internal_reward"):
            del fn.internal_reward

        # Store membrane potential before forward pass
        fn.spike_mechanism.mem = fn.spike_mechanism.mem.detach().requires_grad_()
        u_t = fn.spike_mechanism.mem

        with torch.enable_grad():
            outputs = fn(*inputs)  # Forward pass through the spiking layer

        # Store membrane potential after forward pass
        u_tnew = fn.spike_mechanism.mem

        # Compute reverse reset matrix based on reset mechanism
        if fn.spike_mechanism.reset_mechanism_val == 0:  # reset by subtraction
            reverse_reset = fn.spike_mechanism.reset * fn.spike_mechanism.threshold
        elif fn.spike_mechanism.reset_mechanism_val == 1:  # reset to zero
            raise NotImplementedError()

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
        ctx.save_for_backward(*inputs, *params, u_t, u_tnew, reverse_reset)

        ctx.n_inputs, ctx.n_params = (
            len(inputs),
            len(params),
        )
        ctx.fn = fn

        # Return outputs, detached from computation graph
        if isinstance(outputs, tuple):
            return outputs[0].detach(), outputs[1].detach()
        else:
            return outputs.detach()

    @staticmethod
    def backward(ctx, *incoming_reward):
        """
        Backward pass for the LFP epsilon rule.

        Args:
            ctx: Context object with saved tensors and attributes.
            *incoming_reward: Incoming relevance/reward tensors.

        Returns:
            Tuple of gradients for each input to the forward function.
        """
        # Filter out None rewards and aggregate
        valid_incoming_rewards = [in_reward for in_reward in incoming_reward if in_reward is not None]
        aggregated_incoming_reward = torch.stack(valid_incoming_rewards).sum(dim=0)

        # Add any stored internal reward
        if hasattr(ctx.fn, "internal_reward"):
            for in_reward in ctx.fn.internal_reward:
                aggregated_incoming_reward += in_reward

        # Retrieve saved tensors
        inputs = ctx.saved_tensors[: ctx.n_inputs]
        params = ctx.saved_tensors[ctx.n_inputs : ctx.n_inputs + ctx.n_params]
        u_t = ctx.saved_tensors[-3]
        u_tnew = ctx.saved_tensors[-2]
        reverse_reset = ctx.saved_tensors[-1]

        # Compute difference in membrane potential, correcting for reset
        if u_t.shape == u_tnew.shape:
            u_diff = (u_tnew + reverse_reset - u_t).abs()
            denom = u_t.abs() + u_diff
        else:
            u_diff = (u_tnew + reverse_reset).abs()
            denom = u_diff

        # Normalize reward by stabilized denominator
        normed_reward = aggregated_incoming_reward / zcore.stabilize(
            denom, ctx.epsilon, clip=False, norm_scale=False, dim=None
        )

        # Compute parameter reward (feedback for parameters)
        for param in params:
            if not isinstance(param, tuple):
                param = (param,)  # Noqa: PLW2901
            param_grads = torch.autograd.grad((u_tnew,), param, normed_reward, retain_graph=True)
            if ctx.inplace:
                param_reward = tuple(param_grads[i].mul_(param[i].abs()) for i in range(len(param)))
            else:
                param_reward = tuple(param_grads[i] * param[i].abs() for i in range(len(param)))
            for i in range(len(param)):
                if not hasattr(param[i], "feedback"):
                    param[i].feedback = param_reward[i]
                else:
                    param[i].feedback += param_reward[i]

        # Compute internal reward for membrane potential if applicable
        if u_t.shape == u_tnew.shape:
            ut_grads = torch.autograd.grad((u_tnew,), (u_t,), normed_reward, retain_graph=True)
            if ctx.inplace:
                internal_reward = ((ut_grads[0].mul_(u_t) if u_t.requires_grad else None),)
            else:
                internal_reward = tuple(
                    (ut_grads[0] * u_t if u_t.requires_grad else None),
                )
            ctx.fn.internal_reward = internal_reward

        # Compute outgoing reward for inputs
        input_grads = torch.autograd.grad((u_tnew,), inputs, normed_reward, retain_graph=False)

        if ctx.inplace:
            outgoing_reward = tuple(
                input_grads[i].mul_(inputs[i]) if ctx.requires_grads[i] else None
                for i in range(len(ctx.requires_grads))
            )
        else:
            outgoing_reward = tuple(
                input_grads[i] * inputs[i] if ctx.requires_grads[i] else None for i in range(len(ctx.requires_grads))
            )

        # Return None for non-tensor arguments, followed by outgoing rewards
        return (
            None,
            None,
            None,
        ) + outgoing_reward


class LFPSNNEpsilonComposite(ParameterizableComposite):
    """
    Composite rule for applying the LFP Epsilon Rule to SNN layers.

    This composite maps SpikingLayer modules to the LFPEpsilonSNN rule.
    """

    def __init__(self, epsilon=1e-6):
        """
        Initialize the composite rule.

        Args:
            epsilon (float): Stabilizer for denominator in epsilon rule.
        """
        layer_map = {
            SpikingLayer: RuleGenerator(
                LFPEpsilonSNN,
                epsilon=epsilon,
            ),
        }

        super().__init__(layer_map=layer_map)
        super().__init__(layer_map=layer_map)
