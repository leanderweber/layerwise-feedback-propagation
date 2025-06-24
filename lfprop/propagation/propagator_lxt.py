import torch
from lxt import core as lcore
from lxt import rules as lrules
from lxt.modules import INIT_MODULE_MAPPING
from torch import nn
from zennit import core as zcore
from zennit import rules as zrules
from zennit import types as ztypes

from ..model import activations
from ..model.custom_resnet import Sum


class DummyCompositeContext:
    """
    Context manager for registering and removing a composite from a module.

    This class is adapted from the 'zennit' library for neural network interpretability.
    It allows for registering a composite to a module within a context and ensures
    that hooks and canonizers are properly removed afterwards.

    Parameters
    ----------
    module: torch.nn.Module
        The module to which the composite should be registered.
    composite: Composite
        The composite to register.
    verbose: bool
        Whether to print verbose output during registration.
    **register_kwargs:
        Additional keyword arguments to pass to the composite's register method.
    """

    def __init__(self, module, composite, verbose, **register_kwargs):
        self.module = module
        self.composite = composite
        self.verbose = verbose
        self.register_kwargs = register_kwargs

    def __enter__(self):
        self.composite.register(parent=self.module, verbose=self.verbose, **self.register_kwargs)
        return self.module

    def __exit__(self, exc_type, exc_value, traceback):
        self.composite.remove()
        return False


class ParameterizableComposite(lcore.Composite):
    """
    Composite class that allows for more parameterization of rules.

    This class extends the base Composite to support parameterized rules and
    handles activation canonization (removing inplace operations).
    """

    def __init__(self, layer_map, canonizers=[], zennit_composite=None) -> None:
        super(ParameterizableComposite, self).__init__(layer_map, canonizers, zennit_composite)
        self.original_inplace_states = []

    def _canonize_inplace_activations(self, parent: nn.Module):
        """
        Recursively set the 'inplace' attribute of activation modules to False.

        This is necessary because inplace activations can interfere with feedback propagation.
        """
        for name, child in parent.named_children():
            if hasattr(child, "inplace"):
                self.original_inplace_states.append((child, child.inplace))
                child.inplace = False
            self._canonize_inplace_activations(child)

    def register(
        self,
        parent: nn.Module,
        dummy_inputs: dict = None,
        tracer=lcore.HFTracer,
        verbose=False,
        no_grad=False,  # Set no_grad to false to allow param gradients
    ) -> None:
        # Activation "Canonization": remove inplace ops from activations, they don't work with LFP
        self._canonize_inplace_activations(parent)
        super(ParameterizableComposite, self).register(parent, dummy_inputs, tracer, verbose, no_grad)

    def remove(self):
        """
        Restore original inplace states and remove the composite.
        """
        for module, inplace in self.original_inplace_states:
            module.inplace = inplace
        self.original_inplace_states = []
        super(ParameterizableComposite, self).remove()

    def _attach_module_rule(self, child, parent, name, rule_dict):
        """
        Attach a rule to the child module if it matches the rule_dict.

        If the rule is a subclass of WrapModule, the module is wrapped.
        If the rule is a torch.nn.Module, the module is replaced.
        If the rule is a RuleGenerator, it is called with the child module.
        """
        for layer_type, rule in rule_dict.items():
            check = False
            if isinstance(layer_type, str):
                if layer_type == name:
                    check = True
            elif isinstance(child, layer_type):
                check = True

            if check:
                if isinstance(rule, type) and issubclass(rule, lrules.WrapModule):
                    # Replace module with LXT.rules.WrapModule and attach to parent
                    xai_module = rule(child)
                elif not isinstance(rule, type) and isinstance(rule, RuleGenerator):
                    # RuleGenerator can be parameterized more
                    xai_module = rule(child)
                elif isinstance(rule, type) and issubclass(rule, nn.Module):
                    # Replace module with LXT.module and attach to parent
                    xai_module = INIT_MODULE_MAPPING[rule](child, rule)
                    child = xai_module
                else:
                    raise ValueError(f"Rule {rule} must be a subclass of WrapModule or a torch.nn.Module")

                setattr(parent, name, xai_module)
                # Save original module to revert the composite in self.remove()
                self.original_modules.append((parent, name, child))
                return child

        # No rule found for the module
        return child

    def context(self, module, verbose=False, **register_kwargs):
        """
        Return a context manager for registering and removing the composite.
        """
        return DummyCompositeContext(module, self, verbose, **register_kwargs)


class RuleGenerator:
    """
    Helper class to generate parameterized rules.

    Stores a rule class and keyword arguments, and calls the rule with the module and arguments.
    """

    def __init__(self, rule, **kwargs):
        self.rule = rule
        self.rule_kwargs = kwargs

    def __call__(self, module):
        return self.rule(module, **self.rule_kwargs)


class LFPEpsilon(lrules.EpsilonRule):
    """
    LFP Epsilon Rule.

    If hebbian=True, a Hebbian update (d_w = |w_ij| * a_i*z_j) is used instead of the original LFP update.
    If use_oja=True, Oja's rule replaces a_i*z_j with z_j*(a_i-z_j*w_ij).
    The reward backward pass is not affected by these parameters, only the local weight updates.

    Parameters
    ----------
    module: torch.nn.Module
        The module to which the rule is applied.
    epsilon: float
        Stabilization parameter.
    inplace: bool
        Whether to perform inplace operations.
    hebbian: bool
        Whether to use Hebbian update.
    use_oja: bool
        Whether to use Oja's rule.
    reverse_pos: bool
        Whether to reverse positive outputs.
    reverse_neg: bool
        Whether to reverse negative outputs.
    """

    def __init__(
        self,
        module,
        epsilon=1e-6,
        inplace=True,
        hebbian=False,
        use_oja=False,
        reverse_pos=False,
        reverse_neg=False,
    ):
        super(LFPEpsilon, self).__init__(module, epsilon)
        self.inplace = inplace
        self.hebbian = hebbian
        self.use_oja = use_oja
        self.reverse_pos = reverse_pos
        self.reverse_neg = reverse_neg

        # For compatibility with some transformer models
        if hasattr(module, "weight"):
            self.weight = module.weight

    def forward(self, *inputs):
        return epsilon_lfp_fn.apply(
            self.module,
            self.epsilon,
            self.inplace,
            self.hebbian,
            self.use_oja,
            self.reverse_pos,
            self.reverse_neg,
            *inputs,
        )


class LFPGamma(lrules.EpsilonRule):
    """
    LFP Gamma Rule.

    Parameters
    ----------
    module: torch.nn.Module
        The module to which the rule is applied.
    epsilon: float
        Stabilization parameter.
    gamma: float
        Gamma parameter for the rule.
    inplace: bool
        Whether to perform inplace operations.
    """

    def __init__(self, module, epsilon=1e-6, gamma=0.25, inplace=True):
        super(LFPGamma, self).__init__(module, epsilon)
        self.gamma = gamma
        self.inplace = inplace

        # For compatibility with some transformer models
        if hasattr(module, "weight"):
            self.weight = module.weight

    def forward(self, *inputs):
        return gamma_lfp_fn.apply(self.module, self.epsilon, self.gamma, self.inplace, *inputs)


class epsilon_lfp_fn(lrules.epsilon_lrp_fn):
    """
    Custom autograd function for LFP Epsilon Rule.

    Implements the forward and backward passes for the LFP Epsilon rule,
    supporting a more hebbian-like update and optional application of Oja's rule.
    """

    @staticmethod
    def forward(ctx, fn, epsilon, inplace, hebbian, use_oja, reverse_pos, reverse_neg, *inputs):
        # Track which inputs require gradients
        requires_grads = [True if inp.requires_grad else False for inp in inputs]

        # Detach inputs to avoid overwriting gradients if same input is used as multiple arguments
        inputs = tuple(inp.detach().requires_grad_() if inp.requires_grad else inp for inp in inputs)

        # Get parameters to store for backward (do not detach)
        params = [param for _, param in fn.named_parameters(recurse=False) if param.requires_grad]

        with torch.enable_grad():
            outputs = fn(*inputs)

        (
            ctx.epsilon,
            ctx.requires_grads,
            ctx.inplace,
            ctx.hebbian,
            ctx.use_oja,
            ctx.reverse_pos,
            ctx.reverse_neg,
        ) = (
            epsilon,
            requires_grads,
            inplace,
            hebbian,
            use_oja,
            reverse_pos,
            reverse_neg,
        )
        # Save only inputs requiring gradients
        inputs = tuple(inputs[i] for i in range(len(inputs)) if requires_grads[i])
        ctx.save_for_backward(*inputs, *params, outputs)

        ctx.n_inputs, ctx.n_params = len(inputs), len(params)
        ctx.fn = fn

        return outputs.detach()

    @staticmethod
    def backward(ctx, *incoming_reward):
        outputs = ctx.saved_tensors[-1]
        inputs = ctx.saved_tensors[: ctx.n_inputs]
        params = ctx.saved_tensors[ctx.n_inputs : ctx.n_inputs + ctx.n_params]

        print_debug = False

        if print_debug:
            if hasattr(ctx.fn, "tmpname"):
                print(ctx.fn.tmpname)
            if hasattr(ctx.fn, "weight"):
                print(ctx.fn.weight.abs().min(), ctx.fn.weight.abs().max())

        # Optionally reverse reward for positive/negative outputs
        if ctx.reverse_pos:
            incoming_reward = tuple(
                torch.where(outputs > 0, -incoming_reward[i], incoming_reward[i]) for i in range(len(incoming_reward))
            )
        if ctx.reverse_neg:
            incoming_reward = tuple(
                torch.where(outputs < 0, -incoming_reward[i], incoming_reward[i]) for i in range(len(incoming_reward))
            )

        # Normalize reward
        normed_reward = incoming_reward[0] / zcore.stabilize(
            outputs, ctx.epsilon, clip=False, norm_scale=False, dim=None
        )

        # Compute parameter reward (for weight updates)
        if ctx.hebbian:
            hebb_reward = incoming_reward[0] * outputs

            if ctx.use_oja:
                oja_placeholder_outgrad = outputs**3

            for param in params:
                if not isinstance(param, tuple):
                    param = (param,)  # Noqa: PLW2901
                param_grads = torch.autograd.grad(outputs, param, hebb_reward, retain_graph=True)  # a_i * z_j * r_j

                if ctx.use_oja:
                    oja_factor_num = torch.autograd.grad(outputs, param, oja_placeholder_outgrad, retain_graph=True)
                    oja_factor_denom = torch.autograd.grad(outputs, param, outputs, retain_graph=True)
                    oja_factor = tuple(
                        torch.where(
                            oja_factor_denom[i] != 0,
                            oja_factor_num[i] / oja_factor_denom[i],
                            oja_factor_num[i],
                        )
                        * param[i]
                        for i in range(len(param))
                    )  # z_j^2*w_ij

                    param_reward = tuple(
                        param_grads[i] - oja_factor[i] for i in range(len(param))
                    )  # z_j * (a_i - z_j*w_ij) * r_j
                else:
                    param_reward = tuple(param_grads)

                param_reward = tuple(param_reward[i] * param[i].abs() for i in range(len(param)))

                for i in range(len(param)):
                    # Store feedback for each parameter
                    param[i].feedback = param_reward[i]
        else:
            for param in params:
                if not isinstance(param, tuple):
                    param = (param,)  # Noqa: PLW2901
                param_grads = torch.autograd.grad(
                    outputs, param, normed_reward, retain_graph=True
                )  # a_i * r_j/(z_j+eps)
                if ctx.inplace:
                    param_reward = tuple(param_grads[i].mul_(param[i].abs()) for i in range(len(param)))
                else:
                    param_reward = tuple(param_grads[i] * param[i].abs() for i in range(len(param)))
                for i in range(len(param)):
                    param[i].feedback = param_reward[i]

        # Compute input reward (outgoing reward to propagate)
        input_grads = torch.autograd.grad(outputs, inputs, normed_reward, retain_graph=False)

        if ctx.inplace:
            outgoing_reward = tuple(
                input_grads[i].mul_(inputs[i]) if ctx.requires_grads[i] else None
                for i in range(len(ctx.requires_grads))
            )
        else:
            outgoing_reward = tuple(
                input_grads[i] * inputs[i] if ctx.requires_grads[i] else None for i in range(len(ctx.requires_grads))
            )

        # Return relevance at requires_grad indices else None
        return (None, None, None, None, None, None, None) + outgoing_reward


# TODO: Something is wrong with this. Check backward pass computation.
class gamma_lfp_fn(lrules.epsilon_lrp_fn):
    """
    Custom autograd function for LFP Gamma Rule.

    Gamma is only applied to backpropagate reward. Weights are updated using epsilon.
    """

    @staticmethod
    def forward(ctx, fn, epsilon, gamma, inplace, *inputs):
        # Prepare modifications for gamma rule
        mod_kwargs = {"zero_params": None}
        mod_kwargs_nobias = {"zero_params": zrules.zero_bias(None)}

        # Track which inputs and params require gradients
        requires_grads = [True if inp.requires_grad else False for inp in inputs]
        requires_grads_params = [
            True if param.requires_grad else False for _, param in fn.named_parameters(recurse=False)
        ]

        # Prepare modified inputs for gamma rule
        mod_inputs = [
            tuple(inp.clamp(min=0) for inp in inputs),
            tuple(inp.clamp(max=0) for inp in inputs),
            tuple(inp.clamp(min=0) for inp in inputs),
            tuple(inp.clamp(max=0) for inp in inputs),
            inputs,
        ]

        # Detach inputs as needed
        mod_inputs = [
            tuple(
                (mod_input[i].detach().requires_grad_() if requires_grads[i] else mod_input[i])
                for i in range(len(mod_input))
            )
            for mod_input in mod_inputs
        ]

        # Get original parameters (do not detach)
        orig_params = [param for _, param in fn.named_parameters(recurse=False) if param.requires_grad]

        # Prepare parameter modifications for gamma rule
        param_mods = [
            zrules.GammaMod(gamma, min=0.0, **mod_kwargs),
            zrules.GammaMod(gamma, max=0.0, **mod_kwargs_nobias),
            zrules.GammaMod(gamma, max=0.0, **mod_kwargs),
            zrules.GammaMod(gamma, min=0.0, **mod_kwargs_nobias),
            zrules.NoMod(),
        ]

        # Compute outputs for each parameter modification
        mod_outputs = []
        with torch.enable_grad():
            for p, param_mod in enumerate(param_mods):
                with zcore.ParamMod.ensure(param_mod)(fn) as mod_fn:
                    # Ensure requires_grad for the last mod only
                    for i, (name, param) in enumerate(mod_fn.named_parameters(recurse=False)):
                        if p == len(param_mods) - 1 and requires_grads_params[i]:
                            getattr(mod_fn, name).data = param.detach().requires_grad_()
                        else:
                            param.requires_grad_(False)

                    mod_outputs.append(mod_fn(*mod_inputs[p]))

                    # Save the last parameter mod for backward
                    if p == len(param_mods) - 1:
                        mod_params = [
                            param for _, param in mod_fn.named_parameters(recurse=False) if param.requires_grad
                        ]

        ctx.epsilon, ctx.gamma, ctx.requires_grads, ctx.inplace = (
            epsilon,
            gamma,
            requires_grads,
            inplace,
        )
        # Save only inputs requiring gradients
        mod_inputs_backward = []
        for mod_input in mod_inputs:
            tmp = [mod_input[i] for i in range(len(mod_input)) if requires_grads[i]]
            n_inputs = len(tmp)
            mod_inputs_backward += tmp
        mod_inputs_backward = tuple(mod_inputs_backward)
        ctx.save_for_backward(*orig_params, *mod_params, *mod_outputs, *mod_inputs_backward)

        ctx.n_mods, ctx.n_inputs, ctx.n_params = (
            len(mod_outputs),
            n_inputs,
            len(orig_params),
        )
        ctx.fn = fn

        return mod_outputs[4].detach()

    @staticmethod
    def backward(ctx, *incoming_reward):
        print_debug = False

        if print_debug:
            if hasattr(ctx.fn, "tmpname"):
                print(ctx.fn.tmpname, ctx.epsilon, ctx.gamma, ctx.fn)
            if hasattr(ctx.fn, "weight"):
                print(ctx.fn.weight.shape)
            if isinstance(incoming_reward, tuple):
                print("IN", [inc.abs().mean() for inc in incoming_reward])
            else:
                print("IN", incoming_reward.abs().mean())

        params = ctx.saved_tensors[: ctx.n_params]
        mod_params = ctx.saved_tensors[ctx.n_params : 2 * ctx.n_params]
        mod_outputs = ctx.saved_tensors[2 * ctx.n_params : 2 * ctx.n_params + ctx.n_mods]
        mod_inputs = []
        for i in range(ctx.n_mods):
            mod_inputs += [
                ctx.saved_tensors[
                    2 * ctx.n_params + ctx.n_mods + i * ctx.n_inputs : 2 * ctx.n_params
                    + ctx.n_mods
                    + (i + 1) * ctx.n_inputs
                ]
            ]

        # Compute parameter reward (for weight updates) using standard epsilon
        normed_reward = incoming_reward[0] / zcore.stabilize(
            mod_outputs[-1], ctx.epsilon, clip=False, norm_scale=True, dim=None
        )
        for m_param, orig_param in zip(mod_params, params):
            if not isinstance(m_param, tuple):
                m_param = (m_param,)  # Noqa: PLW2901
            if not isinstance(orig_param, tuple):
                orig_param = (orig_param,)  # Noqa: PLW2901
            param_grads = torch.autograd.grad(mod_outputs[-1], m_param, normed_reward, retain_graph=True)
            if ctx.inplace:
                param_reward = tuple(param_grads[i].mul_(m_param[i].abs()) for i in range(len(m_param)))
            else:
                param_reward = tuple(param_grads[i] * m_param[i].abs() for i in range(len(m_param)))
            for i in range(len(m_param)):
                orig_param[i].feedback = param_reward[i]

        # Compute input reward (outgoing reward to propagate) using gamma rule
        mod_normed_rewards = [
            mod_outputs[-1]
            > 0
            * incoming_reward[0]
            / zcore.stabilize(sum(mod_outputs[:2]), ctx.epsilon, clip=False, norm_scale=True, dim=None),
            mod_outputs[-1]
            > 0
            * incoming_reward[0]
            / zcore.stabilize(sum(mod_outputs[:2]), ctx.epsilon, clip=False, norm_scale=True, dim=None),
            mod_outputs[-1]
            < 0
            * incoming_reward[0]
            / zcore.stabilize(sum(mod_outputs[:2]), ctx.epsilon, clip=False, norm_scale=True, dim=None),
            mod_outputs[-1]
            < 0
            * incoming_reward[0]
            / zcore.stabilize(sum(mod_outputs[:2]), ctx.epsilon, clip=False, norm_scale=True, dim=None),
        ]

        mod_input_grads = [
            torch.autograd.grad(mod_outputs[i], mod_inputs[i], mod_normed_rewards[i], retain_graph=False)
            for i in range(ctx.n_mods - 1)
        ]

        reduced = []
        for n in range(len(ctx.requires_grads)):
            if not ctx.requires_grads[n]:
                reduced.append(None)
            else:
                if ctx.inplace:
                    reduced.append(sum(mod_input_grads[m][n].mul_(mod_inputs[m][n]) for m in range(ctx.n_mods - 1)))
                else:
                    reduced.append(sum(mod_input_grads[m][n] * mod_inputs[m][n] for m in range(ctx.n_mods - 1)))
        outgoing_reward = tuple(reduced)

        if print_debug:
            if isinstance(outgoing_reward, tuple):
                print(
                    "OUT",
                    [inc.abs().mean() if inc is not None else None for inc in outgoing_reward],
                )
            else:
                print("OUT", outgoing_reward.abs().mean())

        # Return relevance at requires_grad indices else None
        return (None, None, None, None) + outgoing_reward


class LFPEpsilonComposite(ParameterizableComposite):
    """
    Composite for LFP Epsilon rule.

    Applies the LFPEpsilon rule to supported layer types.
    """

    def __init__(self, epsilon=1e-6):
        layer_map = {
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


class LFPHebbianEpsilonComposite(ParameterizableComposite):
    """
    Composite for LFP Hebbian Epsilon rule.

    Applies the LFPEpsilon rule with Hebbian and optional Oja updates to supported layer types.
    Essentially, this tries to implement a more hebbian-like version of LFP.
    Does not perform that well in practice.
    """

    def __init__(self, epsilon=1e-6, use_oja=False):
        layer_map = {
            ztypes.Activation: lrules.IdentityRule,
            activations.Step: lrules.IdentityRule,
            Sum: RuleGenerator(
                LFPEpsilon,
                epsilon=epsilon,
                inplace=False,
                hebbian=True,
                use_oja=use_oja,
            ),
            ztypes.AvgPool: RuleGenerator(LFPEpsilon, epsilon=epsilon, hebbian=True, use_oja=use_oja),
            ztypes.Linear: RuleGenerator(LFPEpsilon, epsilon=epsilon, hebbian=True, use_oja=use_oja),
            ztypes.BatchNorm: RuleGenerator(LFPEpsilon, epsilon=epsilon, hebbian=True, use_oja=use_oja),
        }

        super().__init__(layer_map=layer_map)


# TODO: This is not working properly.
class LFPGammaComposite(ParameterizableComposite):
    """
    Composite for LFP Gamma rule.

    Applies the LFPGamma rule to supported layer types.
    """

    def __init__(self, epsilon=1e-6, gamma=0.25):
        layer_map = {
            ztypes.Activation: lrules.IdentityRule,
            activations.Step: lrules.IdentityRule,
            Sum: RuleGenerator(LFPEpsilon, epsilon=epsilon, inplace=False),
            ztypes.AvgPool: RuleGenerator(
                LFPEpsilon,
                epsilon=epsilon,
            ),
            ztypes.Linear: RuleGenerator(LFPGamma, epsilon=epsilon, gamma=gamma),
            ztypes.BatchNorm: RuleGenerator(
                LFPEpsilon,
                epsilon=epsilon,
            ),
        }

        super().__init__(layer_map=layer_map)
