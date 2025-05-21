"""
Propagator for training SNN with LFP
"""

import torch
from lxt import rules as lrules
from zennit import core as zcore
from zennit import types as ztypes

from ..model import activations
from ..model.spiking_networks import SpikingLayer
from .propagator_lxt import ParameterizableComposite, RuleGenerator


class LFPEpsilonSNN(lrules.EpsilonRule):
    """
    LFP Epsilon Rule for SNN
    """

    def __init__(
        self,
        module,
        epsilon=1e-6,
        inplace=True,
    ):
        super(LFPEpsilonSNN, self).__init__(module, epsilon)
        self.inplace = inplace

        # This is needed for compatibility with L631 in modeling_vit.py from transformers library
        if hasattr(module, "weight"):
            self.weight = module.weight

    def forward(self, *inputs):
        return epsilon_lfp_snn_fn.apply(
            self.module,
            self.epsilon,
            self.inplace,
            *inputs,
        )


class epsilon_lfp_snn_fn(lrules.epsilon_lrp_fn):
    """
    LFP Epsilon Rule for SNN
    Makes some assumption about the wrapped module "fn",
    e.g., that it is of type "SpikingLayer"

    Note: This function assumes mostly default parameter from snn.Leaky
    (cf. parameters defined in lfprop.model.spiking_networks)
    """

    @staticmethod
    def forward(ctx, fn: SpikingLayer, epsilon, inplace, *inputs):
        assert isinstance(fn, SpikingLayer)

        # create boolean mask for inputs requiring gradients
        requires_grads = [True if inp.requires_grad else False for inp in inputs]

        # detach inputs to avoid overwriting gradients if same input is used as multiple arguments
        # (like in self-attention)
        inputs = tuple(inp.detach().requires_grad_() if inp.requires_grad else inp for inp in inputs)

        # get parameters to store for backward. Here, we want to accumulate reward, so we do not detach
        params = [param for _, param in fn.parameterized_layer.named_parameters(recurse=False) if param.requires_grad]
        # Reset feedback:
        for param in params:
            if hasattr(param, "feedback"):
                del param.feedback
        # Reset Internal Reward
        if hasattr(fn, "internal_reward"):
            del fn.internal_reward

        # get U_[t] to store for backward
        fn.spike_mechanism.mem = fn.spike_mechanism.mem.detach().requires_grad_()
        u_t = fn.spike_mechanism.mem

        with torch.enable_grad():
            outputs = fn(*inputs)  # Note: this can be either spikes, or (spikes, mem)

        # get U_[t+1] to store for backward
        u_tnew = fn.spike_mechanism.mem

        # Get reverse reset matrix
        if fn.spike_mechanism.reset_mechanism_val == 0:  # reset by subtraction
            reverse_reset = fn.spike_mechanism.reset * fn.spike_mechanism.threshold
        elif fn.spike_mechanism.reset_mechanism_val == 1:  # reset to zero
            raise NotImplementedError()

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
        ctx.save_for_backward(
            *inputs, *params, u_t, u_tnew, reverse_reset
        )  # u_tnew is stored in place of outputs, compared to ann version of LFP

        ctx.n_inputs, ctx.n_params = (
            len(inputs),
            len(params),
        )
        ctx.fn = fn

        if isinstance(outputs, tuple):
            # if layer is not hidden
            return outputs[0].detach(), outputs[1].detach()
        else:
            return outputs.detach()

    @staticmethod
    def backward(ctx, *incoming_reward):
        # Handle incoming reward being for spikes and mem or just for spikes
        valid_incoming_rewards = [  # TODO check these
            in_reward for in_reward in incoming_reward if in_reward is not None
        ]
        # Aggregate rewards via sum. may use something else in future
        aggregated_incoming_reward = torch.stack(valid_incoming_rewards).sum(dim=0)

        # Add the old internal reward (i.e., r_[u_tnew->ut])
        if hasattr(ctx.fn, "internal_reward"):
            for in_reward in ctx.fn.internal_reward:
                aggregated_incoming_reward += in_reward

        # Get stored tensors
        inputs = ctx.saved_tensors[: ctx.n_inputs]
        params = ctx.saved_tensors[ctx.n_inputs : ctx.n_inputs + ctx.n_params]
        u_t = ctx.saved_tensors[-3]  # Assume there is only one u_t
        u_tnew = ctx.saved_tensors[-2]  # Assume there is only one u_tnew
        reverse_reset = ctx.saved_tensors[-1]  # Assume there is only one reverse_reset

        if u_t.shape == u_tnew.shape:
            u_diff = (
                u_tnew + reverse_reset - u_t
            ).abs()  # Correct u_tnew by reverse reset. We are interested in pre-activation that
            # CAUSED the spike
            # u_diff is basically abs(W.T*X)
            denom = u_t.abs() + u_diff  # Basically u_tnew, assuming all contributions are positive only
        else:
            u_diff = (u_tnew + reverse_reset).abs()
            denom = u_diff

        normed_reward = aggregated_incoming_reward / zcore.stabilize(
            denom, ctx.epsilon, clip=False, norm_scale=False, dim=None
        )  # *u_tnew.sign() #TODO: sign worsens performance. investigate why.

        # compute param reward (used to update parameters)
        for param in params:
            if not isinstance(param, tuple):
                param = (param,)  # noqa: PLW2901
            param_grads = torch.autograd.grad(
                (u_tnew,), param, normed_reward, retain_graph=True
            )  # a_i * r_j/(z_j+eps).
            # Note: we can use u_tnew here as we only use it for getting a_i in the correct shape.
            if ctx.inplace:
                param_reward = tuple(param_grads[i].mul_(param[i].abs()) for i in range(len(param)))
            else:
                param_reward = tuple(param_grads[i] * param[i].abs() for i in range(len(param)))
            for i in range(len(param)):
                if not hasattr(param[i], "feedback"):
                    param[i].feedback = param_reward[i]
                else:
                    param[i].feedback += param_reward[i]

        # In the first foward pass, snntorch initializes u_t.
        # We cannot pass reward since the u_t before that is not used in the graph
        if u_t.shape == u_tnew.shape:
            # compute U[t] Reward
            ut_grads = torch.autograd.grad((u_tnew,), (u_t,), normed_reward, retain_graph=True)

            if ctx.inplace:
                internal_reward = ((ut_grads[0].mul_(u_t) if u_t.requires_grad else None),)
            else:
                internal_reward = tuple(
                    (ut_grads[0] * u_t if u_t.requires_grad else None),
                )

            # Store U[t] reward. TODO find nicer implementation for this?
            ctx.fn.internal_reward = internal_reward

        # compute input reward (= outgoing reward to propagate)
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

        # return relevance at requires_grad indices else None
        return (
            None,
            None,
            None,
        ) + outgoing_reward


class LFPSNNEpsilonComposite(ParameterizableComposite):
    def __init__(self, epsilon=1e-6):
        layer_map = {
            ztypes.Activation: lrules.IdentityRule,
            activations.Step: lrules.IdentityRule,
            ztypes.Activation: lrules.IdentityRule,
            activations.Step: lrules.IdentityRule,
            SpikingLayer: RuleGenerator(
                LFPEpsilonSNN,
                epsilon=epsilon,
            ),
            # snn.SpikingNeuron: lrules.StopRelevanceRule,
            # ztypes.Linear: lrules.StopRelevanceRule,
        }

        super().__init__(layer_map=layer_map)
