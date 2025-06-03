import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.nn.parameter import Parameter

from . import functional as sf
from .utils import to_pair


class Convolution(nn.Module):
    r"""Performs a 2D convolution over an input spike-wave composed of several input
    planes. Current version only supports stride of 1 with no padding.

    The input is a 4D tensor with the size :math:`(T, C_{{in}}, H_{{in}}, W_{{in}})` and the crresponsing output
    is of size :math:`(T, C_{{out}}, H_{{out}}, W_{{out}})`,
    where :math:`T` is the number of time steps, :math:`C` is the number of feature maps (channels), and
    :math:`H`, and :math:`W` are the hight and width of the input/output planes.

    * :attr:`in_channels` controls the number of input planes (channels/feature maps).

    * :attr:`out_channels` controls the number of feature maps in the current layer.

    * :attr:`kernel_size` controls the size of the convolution kernel. It can be a single integer or a tuple of two integers.

    * :attr:`weight_mean` controls the mean of the normal distribution used for initial random weights.

    * :attr:`weight_std` controls the standard deviation of the normal distribution used for initial random weights.

    .. note::

        Since this version of convolution does not support padding, it is the user responsibility to add proper padding
        on the input before applying convolution.

    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        weight_mean (float, optional): Mean of the initial random weights. Default: 0.8
        weight_std (float, optional): Standard deviation of the initial random weights. Default: 0.02
    """

    def __init__(self, in_channels, out_channels, kernel_size, weight_mean=0.8, weight_std=0.02):
        super(Convolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = to_pair(kernel_size)
        # self.weight_mean = weight_mean
        # self.weight_std = weight_std

        # For future use
        self.stride = 1
        self.bias = None
        self.dilation = 1
        self.groups = 1
        self.padding = 0

        # Parameters
        self.weight = Parameter(torch.Tensor(self.out_channels, self.in_channels, *self.kernel_size))
        self.weight.requires_grad_(False)  # We do not use gradients
        self.reset_weight(weight_mean, weight_std)

    def reset_weight(self, weight_mean=0.8, weight_std=0.02):
        """Resets weights to random values based on a normal distribution.

        Args:
            weight_mean (float, optional): Mean of the random weights. Default: 0.8
            weight_std (float, optional): Standard deviation of the random weights. Default: 0.02
        """
        self.weight.normal_(weight_mean, weight_std)

    def load_weight(self, target):
        """Loads weights with the target tensor.

        Args:
            target (Tensor=): The target tensor.
        """
        self.weight.copy_(target)

    def forward(self, input):
        return fn.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Linear(nn.Module):
    r"""Performs a linear transformation over an input spike-wave.

    The input is a 3D tensor with the size :math:`(T, C_{{in}}, N)` where:
    - :math:`T` is the number of time steps.
    - :math:`C_{{in}}` is the number of input neurons (channels/units).
    - :math:`N` is the size of the input layer (the number of features).

    The corresponding output is of size :math:`(T, C_{{out}}, N_{{out}})`, where:
    - :math:`C_{{out}}` is the number of output neurons (output channels).
    - :math:`N_{{out}}` is the size of the output layer (the number of output features).

    Args:
        in_features (int): Number of input features (neurons).
        out_features (int): Number of output features (neurons).
        weight_mean (float, optional): Mean of the initial random weights. Default: 0.8.
        weight_std (float, optional): Standard deviation of the initial random weights. Default: 0.02.
    """

    def __init__(self, in_features, out_features, weight_mean=0.8, weight_std=0.02):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Parameters
        self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight.requires_grad_(False)  # We do not use gradients in this spiking layer
        self.reset_weight(weight_mean, weight_std)

    def reset_weight(self, weight_mean=0.8, weight_std=0.02):
        """Resets weights to random values based on a normal distribution.

        Args:
            weight_mean (float, optional): Mean of the random weights. Default: 0.8.
            weight_std (float, optional): Standard deviation of the random weights. Default: 0.02.
        """
        self.weight.normal_(weight_mean, weight_std)

    def load_weight(self, target):
        """Loads weights with the target tensor.

        Args:
            target (Tensor): The target tensor.
        """
        self.weight.copy_(target)

    def forward(self, input):
        r"""Performs the forward pass of the linear transformation on the input spike-wave.

        Args:
            input (Tensor): Input spike-wave of size :math:`(T, C_{{in}}, N)`.

        Returns:
            Tensor: Output spike-wave of size :math:`(T, C_{{out}}, N_{{out}})`.
        """
        return torch.matmul(input, self.weight.T)


class Flatten(nn.Module):
    r"""Flattens the input tensor into a 2D tensor.

    Args:
        start_dim (int, optional): The dimension to start flattening. Default: 1.
        end_dim (int, optional): The dimension to end flattening. Default: -1.
    """

    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)


class Pooling(nn.Module):
    r"""Performs a 2D max-pooling over an input signal (spike-wave or potentials) composed of several input
    planes.

    .. note::

        Regarding the structure of the spike-wave tensors, application of max-pooling over spike-wave tensors results
        in propagation of the earliest spike within each pooling window.

    The input is a 4D tensor with the size :math:`(T, C, H_{{in}}, W_{{in}})` and the crresponsing output
    is of size :math:`(T, C, H_{{out}}, W_{{out}})`,
    where :math:`T` is the number of time steps, :math:`C` is the number of feature maps (channels), and
    :math:`H`, and :math:`W` are the hight and width of the input/output planes.

    * :attr:`kernel_size` controls the size of the pooling window. It can be a single integer or a tuple of two integers.

    * :attr:`stride` controls the stride of the pooling. It can be a single integer or a tuple of two integers. If the value is None, it does pooling with full stride.

    * :attr:`padding` controls the amount of padding. It can be a single integer or a tuple of two integers.

    Args:
        kernel_size (int or tuple): Size of the pooling window
        stride (int or tuple, optional): Stride of the pooling window. Default: None
        padding (int or tuple, optional): Size of the padding. Default: 0
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super(Pooling, self).__init__()
        self.kernel_size = to_pair(kernel_size)
        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = to_pair(stride)
        self.padding = to_pair(padding)

        # For future use
        self.dilation = 1
        self.return_indices = False
        self.ceil_mode = False

    def forward(self, input):
        return sf.pooling(input, self.kernel_size, self.stride, self.padding)


class STDP(nn.Module):
    r"""Performs STDP learning rule over synapses of a convolutional or linear layer.

    The rule is the same as previously defined, but the class adapts to both
    convolutional and linear layers.
    """

    def __init__(self, layer, learning_rate, use_stabilizer=True, lower_bound=0, upper_bound=1):
        super(STDP, self).__init__()
        self.layer = layer  # Can be either convolutional or linear
        if isinstance(learning_rate, list):
            self.learning_rate = learning_rate
        else:
            if isinstance(layer, Convolution):
                self.learning_rate = [learning_rate] * layer.out_channels
            else:  # For linear layers
                self.learning_rate = [learning_rate] * layer.out_features

        for i in range(len(self.learning_rate)):
            self.learning_rate[i] = (
                Parameter(torch.tensor([self.learning_rate[i][0]])),
                Parameter(torch.tensor([self.learning_rate[i][1]])),
            )
            self.register_parameter("ltp_" + str(i), self.learning_rate[i][0])
            self.register_parameter("ltd_" + str(i), self.learning_rate[i][1])
            self.learning_rate[i][0].requires_grad_(False)
            self.learning_rate[i][1].requires_grad_(False)

        self.use_stabilizer = use_stabilizer
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_pre_post_ordering(self, input_spikes, output_spikes, winners):
        if isinstance(self.layer, Convolution):
            return self._get_pre_post_ordering_conv(input_spikes, output_spikes, winners)
        else:
            return self._get_pre_post_ordering_linear(input_spikes, output_spikes, winners)

    def _get_pre_post_ordering_conv(self, input_spikes, output_spikes, winners):
        """STDP for convolutional layers."""
        input_latencies = torch.sum(input_spikes, dim=0)
        output_latencies = torch.sum(output_spikes, dim=0)
        result = []
        for winner in winners:
            out_tensor = torch.ones(*self.layer.kernel_size, device=output_latencies.device) * output_latencies[winner]
            in_tensor = input_latencies[
                :,
                winner[-2] : winner[-2] + self.layer.kernel_size[-2],
                winner[-1] : winner[-1] + self.layer.kernel_size[-1],
            ]
            result.append(torch.ge(in_tensor, out_tensor))
        return result

    def _get_pre_post_ordering_linear(self, input_spikes, output_spikes, winners):
        """STDP for linear layers."""
        input_latencies = torch.sum(input_spikes, dim=0)
        output_latencies = torch.sum(output_spikes, dim=0)
        result = []
        for winner in winners:
            out_tensor = output_latencies[winner]
            in_tensor = input_latencies
            result.append(in_tensor <= out_tensor)  # Simple pre-post ordering for linear layers
        return result

    def forward(self, input_spikes, potentials, output_spikes, winners=None, kwta=1, inhibition_radius=0):
        if winners is None:
            winners = sf.get_k_winners(potentials, kwta, inhibition_radius, output_spikes)
        pairings = self.get_pre_post_ordering(input_spikes, output_spikes, winners)

        # Adjust learning rates for the layer type
        lr = torch.zeros_like(self.layer.weight)
        for i in range(len(winners)):
            if isinstance(self.layer, Convolution):
                f = winners[i][0]
            else:
                f = winners[i][0]  # For linear layers

            lr[f] = torch.where(pairings[i], self.learning_rate[f][0], self.learning_rate[f][1])

        # Update weights with stabilizer term if required
        self.layer.weight += lr * (
            (self.layer.weight - self.lower_bound) * (self.upper_bound - self.layer.weight)
            if self.use_stabilizer
            else 1
        )
        self.layer.weight.clamp_(self.lower_bound, self.upper_bound)

    def update_learning_rate(self, feature, ap, an):
        """Updates learning rate for a specific feature."""
        self.learning_rate[feature][0][0] = ap
        self.learning_rate[feature][1][0] = an

    def update_all_learning_rate(self, ap, an):
        """Updates learning rates of all the features to the same value."""
        for feature in range(len(self.learning_rate)):
            self.learning_rate[feature][0][0] = ap
            self.learning_rate[feature][1][0] = an
