try:
    import snntorch as snn
except ImportError:
    print(
        "The SNN functionality of this package requires extra dependencies ",
        "which can be installed via pip install lfprop[snn] (or lfprop[full] for all dependencies).",
    )
    raise ImportError("snntorch required; reinstall lfprop with option `snn` (pip install lfprop[snn])")


import torch
from torch import nn as tnn

from .activations import Step

# Model definitions

# Mapping of spike gradient names to their corresponding surrogate gradient functions
SPIKE_GRAD_MAP = {
    "step": Step,
    "atan": snn.surrogate.atan,
    "fast_sigmoid": snn.surrogate.fast_sigmoid,
}


class NoisyWrapper(tnn.Module):
    """
    Module wrapper that adds Gaussian noise to its input during training.

    Args:
        module (nn.Module): The wrapped module.
        noise_size (float): Standard deviation of the Gaussian noise.
        apply_noise (bool): Whether to apply noise during training.
    """

    def __init__(self, module, noise_size, apply_noise, *args, **kwargs):
        super().__init__()

        self.noise_size = noise_size
        self.apply_noise = apply_noise
        self.module = module
        self.zeros_ratio = 0

    def forward(self, x):
        # Track the ratio of zeros in the input
        self.zeros_ratio = ((x == 0).sum() / x.numel()).item()
        # Add noise if in training mode and apply_noise is True
        if self.training and self.apply_noise:
            noise = torch.randn_like(x) * self.noise_size
            x = x + noise

        return self.module.forward(x)


class Interpolate(tnn.Module):
    """
    Module for resizing tensors using interpolation.

    Args:
        size (tuple): Output spatial size.
        mode (str): Interpolation mode.
    """

    def __init__(self, size, mode="bilinear", *args, **kwargs):
        super().__init__()

        self.size = size
        self.mode = mode

    def forward(self, x):
        # Resize input tensor
        x = tnn.functional.interpolate(
            x,
            size=self.size,
            mode=self.mode,
        )

        return x


class SpikingLayer(tnn.Module):
    """
    Wrapper for a parameterized layer (e.g., Linear, Conv) and a spiking neuron mechanism.

    Args:
        parameterized_layer (nn.Module): The linear or convolutional layer.
        spike_mechanism (snn.SpikingNeuron): The spiking neuron mechanism.
    """

    def __init__(self, parameterized_layer: tnn.Module, spike_mechanism: snn.SpikingNeuron):
        super().__init__()

        self.parameterized_layer = parameterized_layer
        self.spike_mechanism = spike_mechanism

    def forward(self, x):
        # Forward through the parameterized layer, then the spiking mechanism
        x = self.parameterized_layer(x)
        x = self.spike_mechanism(x)

        return x


class LifMLP(tnn.Module):
    """
    Multi-layer perceptron using Leaky-Integrate-and-Fire (LIF) neurons.

    Args:
        n_channels (int): Number of input features.
        n_outputs (int): Number of output classes.
        beta (float): LIF neuron decay parameter.
        surrogate_disable (bool): Disable surrogate gradient.
        spike_grad (str or callable): Surrogate gradient function.
        noise_size (float): Noise standard deviation.
        apply_noise (bool): Whether to apply noise.
        reset_delay (bool): Whether to use reset delay in LIF neurons.
    """

    def __init__(
        self,
        n_channels,
        n_outputs,
        beta,
        surrogate_disable=False,
        spike_grad=snn.surrogate.atan,
        noise_size=1e-6,
        apply_noise=True,
        reset_delay=True,
        **kwargs,
    ):
        super().__init__()

        self.apply_noise = apply_noise
        self.noise_size = noise_size
        kwargs.pop("n_linear_inputs", None)  # Remove n_linear_inputs if present

        # Classifier: 3-layer MLP with LIF neurons
        self.classifier = tnn.Sequential(
            SpikingLayer(
                NoisyWrapper(tnn.Linear(n_channels, 1000), self.noise_size, self.apply_noise),
                snn.Leaky(
                    beta=beta,
                    init_hidden=True,
                    surrogate_disable=surrogate_disable,
                    spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                    reset_delay=reset_delay,
                    **kwargs,
                ),
            ),
            SpikingLayer(
                NoisyWrapper(tnn.Linear(1000, 1000), self.noise_size, self.apply_noise),
                snn.Leaky(
                    beta=beta,
                    init_hidden=True,
                    surrogate_disable=surrogate_disable,
                    spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                    reset_delay=reset_delay,
                    **kwargs,
                ),
            ),
            SpikingLayer(
                NoisyWrapper(tnn.Linear(1000, n_outputs), self.noise_size, self.apply_noise),
                snn.Leaky(
                    beta=beta,
                    init_hidden=True,
                    output=True,
                    surrogate_disable=surrogate_disable,
                    spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                    reset_delay=reset_delay,
                ),
            ),
        )

        self.reset()

    def reset(self):
        """
        Reset and detach hidden states of all LIF neurons.
        """
        snn.Leaky.reset_hidden()
        snn.Leaky.detach_hidden()

    def forward(self, x):
        """
        Forward input through the MLP.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SmallLifMLP(LifMLP):
    """
    Smaller MLP using LIF neurons (2 layers).
    """

    def __init__(
        self,
        n_channels,
        n_outputs,
        beta,
        surrogate_disable=False,
        spike_grad=snn.surrogate.atan,
        noise_size=1e-6,
        apply_noise=True,
        reset_delay=True,
        **kwargs,
    ):
        super().__init__(
            n_channels,
            n_outputs,
            beta,
            surrogate_disable=surrogate_disable,
            spike_grad=spike_grad,
            reset_delay=reset_delay,
            noise_size=noise_size,
            apply_noise=apply_noise,
            **kwargs,
        )
        kwargs.pop("n_linear_inputs", None)  # Remove n_linear_inputs if present

        # Classifier: 2-layer MLP with LIF neurons
        self.classifier = tnn.Sequential(
            SpikingLayer(
                NoisyWrapper(tnn.Linear(n_channels, 1000), self.noise_size, self.apply_noise),
                snn.Leaky(
                    beta=beta,
                    init_hidden=True,
                    surrogate_disable=surrogate_disable,
                    spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                    reset_delay=reset_delay,
                    **kwargs,
                ),
            ),
            SpikingLayer(
                NoisyWrapper(tnn.Linear(1000, n_outputs), self.noise_size, self.apply_noise),
                snn.Leaky(
                    beta=beta,
                    init_hidden=True,
                    output=True,
                    surrogate_disable=surrogate_disable,
                    spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                    reset_delay=reset_delay,
                ),
            ),
        )


class LifCNN(LifMLP):
    """
    Simple convolutional neural network using LIF neurons.

    Args:
        n_channels (int): Number of input channels.
        n_outputs (int): Number of output classes.
        beta (float): LIF neuron decay parameter.
        surrogate_disable (bool): Disable surrogate gradient.
        spike_grad (str or callable): Surrogate gradient function.
        noise_size (float): Noise standard deviation.
        apply_noise (bool): Whether to apply noise.
        reset_delay (bool): Whether to use reset delay in LIF neurons.
    """

    def __init__(
        self,
        n_channels,
        n_outputs,
        beta,
        surrogate_disable=False,
        spike_grad=snn.surrogate.atan,
        noise_size=1e-6,
        apply_noise=True,
        reset_delay=True,
        **kwargs,
    ):
        super().__init__(
            n_channels,
            n_outputs,
            beta,
            surrogate_disable=surrogate_disable,
            spike_grad=spike_grad,
            reset_delay=reset_delay,
            noise_size=noise_size,
            apply_noise=apply_noise,
        )
        kwargs.pop("n_linear_inputs", None)  # Remove n_linear_inputs if present

        # Classifier: 2 conv layers + 1 linear layer, all with LIF neurons
        self.classifier = tnn.Sequential(
            SpikingLayer(
                NoisyWrapper(tnn.Conv2d(n_channels, 12, 5), self.noise_size, self.apply_noise),
                snn.Leaky(
                    beta=beta,
                    init_hidden=True,
                    surrogate_disable=surrogate_disable,
                    spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                    reset_delay=reset_delay,
                    **kwargs,
                ),
            ),
            tnn.MaxPool2d(2),
            SpikingLayer(
                NoisyWrapper(tnn.Conv2d(12, 64, 5), self.noise_size, self.apply_noise),
                snn.Leaky(
                    beta=beta,
                    init_hidden=True,
                    surrogate_disable=surrogate_disable,
                    spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                    reset_delay=reset_delay,
                    **kwargs,
                ),
            ),
            tnn.MaxPool2d(2),
            tnn.Flatten(),
            SpikingLayer(
                NoisyWrapper(
                    tnn.Linear(1600 if n_channels == 3 else 64 * 4 * 4, n_outputs),
                    self.noise_size,
                    self.apply_noise,
                ),
                snn.Leaky(
                    beta=beta,
                    init_hidden=True,
                    output=True,
                    surrogate_disable=surrogate_disable,
                    spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                    reset_delay=reset_delay,
                ),
            ),
        )

    def forward(self, x):
        """
        Forward input through the CNN.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        x = self.classifier(x)
        return x


class DeeperSNN(LifCNN):
    """
    Deeper SNN with more convolutional layers.
    Similar to Deeper2024 from https://github.com/aidinattar/snn

    Args:
        n_channels (int): Number of input channels.
        n_outputs (int): Number of output classes.
        beta (float): LIF neuron decay parameter.
        n_linear_inputs (int): Number of inputs to the final linear layer.
    """

    def __init__(
        self,
        n_channels,
        n_outputs,
        beta,
        surrogate_disable=False,
        spike_grad=snn.surrogate.atan,
        noise_size=1e-6,
        apply_noise=True,
        reset_delay=True,
        n_linear_inputs=2500,
        **kwargs,
    ):
        super().__init__(
            n_channels,
            n_outputs,
            beta,
            surrogate_disable=surrogate_disable,
            spike_grad=spike_grad,
            reset_delay=reset_delay,
            noise_size=noise_size,
            apply_noise=apply_noise,
        )

        self.n_linear_inputs = n_linear_inputs

        # Classifier: 5 conv layers + 1 linear layer, all with LIF neurons
        self.classifier = tnn.Sequential(
            SpikingLayer(
                NoisyWrapper(
                    tnn.Conv2d(n_channels, 30, 3, padding=2, bias=False),
                    self.noise_size,
                    self.apply_noise,
                ),
                snn.Leaky(
                    beta=beta,
                    init_hidden=True,
                    surrogate_disable=surrogate_disable,
                    spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                    reset_delay=reset_delay,
                    **kwargs,
                ),
            ),
            tnn.MaxPool2d(kernel_size=2, stride=2),
            SpikingLayer(
                NoisyWrapper(
                    tnn.Conv2d(30, 150, 3, padding=1, bias=False),
                    self.noise_size,
                    self.apply_noise,
                ),
                snn.Leaky(
                    beta=beta,
                    init_hidden=True,
                    surrogate_disable=surrogate_disable,
                    spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                    reset_delay=reset_delay,
                    **kwargs,
                ),
            ),
            tnn.MaxPool2d(kernel_size=2, stride=2),
            SpikingLayer(
                NoisyWrapper(
                    tnn.Conv2d(150, 250, 3, padding=1, bias=False),
                    self.noise_size,
                    self.apply_noise,
                ),
                snn.Leaky(
                    beta=beta,
                    init_hidden=True,
                    surrogate_disable=surrogate_disable,
                    spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                    reset_delay=reset_delay,
                    **kwargs,
                ),
            ),
            tnn.MaxPool2d(kernel_size=2, stride=2),
            SpikingLayer(
                NoisyWrapper(
                    tnn.Conv2d(250, 200, 3, padding=2, bias=False),
                    self.noise_size,
                    self.apply_noise,
                ),
                snn.Leaky(
                    beta=beta,
                    init_hidden=True,
                    surrogate_disable=surrogate_disable,
                    spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                    reset_delay=reset_delay,
                    **kwargs,
                ),
            ),
            tnn.MaxPool2d(kernel_size=2, stride=2),
            SpikingLayer(
                NoisyWrapper(
                    tnn.Conv2d(200, 100, 3, padding=2, bias=False),
                    self.noise_size,
                    self.apply_noise,
                ),
                snn.Leaky(
                    beta=beta,
                    init_hidden=True,
                    surrogate_disable=surrogate_disable,
                    spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                    reset_delay=reset_delay,
                    **kwargs,
                ),
            ),
            tnn.Flatten(),
            SpikingLayer(
                NoisyWrapper(
                    tnn.Linear(self.n_linear_inputs, n_outputs, bias=False),
                    self.noise_size,
                    self.apply_noise,
                ),
                snn.Leaky(
                    beta=beta,
                    init_hidden=True,
                    output=True,
                    surrogate_disable=surrogate_disable,
                    spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                    reset_delay=reset_delay,
                ),
            ),
        )


class ResNet(LifCNN):
    """
    ResNet-like architecture using LIF neurons.
    Similar to ResSNN from https://github.com/aidinattar/snn

    Args:
        n_channels (int): Number of input channels.
        n_outputs (int): Number of output classes.
        beta (float): LIF neuron decay parameter.
        n_linear_inputs (int): Number of inputs to the final linear layer.
    """

    def __init__(
        self,
        n_channels,
        n_outputs,
        beta,
        surrogate_disable=False,
        spike_grad=snn.surrogate.atan,
        noise_size=1e-6,
        apply_noise=True,
        reset_delay=True,
        n_linear_inputs=1800,
        **kwargs,
    ):
        super().__init__(
            n_channels,
            n_outputs,
            beta,
            surrogate_disable=surrogate_disable,
            spike_grad=spike_grad,
            reset_delay=reset_delay,
            noise_size=noise_size,
            apply_noise=apply_noise,
        )

        self.n_linear_inputs = n_linear_inputs

        # Remove the default classifier from LifCNN
        del self.classifier

        # Define ResNet-like blocks with skip connection
        self.block1 = SpikingLayer(
            NoisyWrapper(
                tnn.Conv2d(n_channels, 30, 5, padding=2, bias=False),
                self.noise_size,
                self.apply_noise,
            ),
            snn.Leaky(
                beta=beta,
                init_hidden=True,
                surrogate_disable=surrogate_disable,
                spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                reset_delay=reset_delay,
                **kwargs,
            ),
        )
        self.pool1 = tnn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = SpikingLayer(
            NoisyWrapper(
                tnn.Conv2d(30, 150, 3, padding=1, bias=False),
                self.noise_size,
                self.apply_noise,
            ),
            snn.Leaky(
                beta=beta,
                init_hidden=True,
                surrogate_disable=surrogate_disable,
                spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                reset_delay=reset_delay,
                **kwargs,
            ),
        )
        self.pool2 = tnn.MaxPool2d(kernel_size=2, stride=2)
        self.block3 = SpikingLayer(
            NoisyWrapper(
                tnn.Conv2d(150, 250, 3, padding=1, bias=False),
                self.noise_size,
                self.apply_noise,
            ),
            snn.Leaky(
                beta=beta,
                init_hidden=True,
                surrogate_disable=surrogate_disable,
                spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                reset_delay=reset_delay,
                **kwargs,
            ),
        )
        self.pool3 = tnn.MaxPool2d(kernel_size=3, stride=3)
        self.block4 = SpikingLayer(
            NoisyWrapper(
                tnn.Conv2d(250, 200, 4, padding=2, bias=False),
                self.noise_size,
                self.apply_noise,
            ),
            snn.Leaky(
                beta=beta,
                init_hidden=True,
                surrogate_disable=surrogate_disable,
                spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                reset_delay=reset_delay,
                **kwargs,
            ),
        )
        # Skip connection from block2 to block3
        self.skip_connection = SpikingLayer(
            tnn.Sequential(
                NoisyWrapper(
                    tnn.Conv2d(150, 250, 1, bias=False),
                    self.noise_size,
                    self.apply_noise,
                ),
                Interpolate(size=(8, 8), mode="nearest"),
            ),
            snn.Leaky(
                beta=beta,
                init_hidden=True,
                surrogate_disable=surrogate_disable,
                spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                reset_delay=reset_delay,
                **kwargs,
            ),
        )
        self.flatten = tnn.Flatten()
        self.fc = SpikingLayer(
            NoisyWrapper(
                tnn.Linear(self.n_linear_inputs, n_outputs, bias=False),
                self.noise_size,
                self.apply_noise,
            ),
            snn.Leaky(
                beta=beta,
                init_hidden=True,
                output=True,
                surrogate_disable=surrogate_disable,
                spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                reset_delay=reset_delay,
            ),
        )

    def forward(self, x):
        """
        Forward input through the ResNet-like SNN.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x_skip = self.skip_connection(x)
        x = self.block3(x)
        # Residual connection: element-wise max
        x = torch.max(x, x_skip).float()
        x = self.pool3(x)
        x = self.block4(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
