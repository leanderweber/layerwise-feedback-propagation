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

SPIKE_GRAD_MAP = {
    "step": Step,
    "atan": snn.surrogate.atan,
    "fast_sigmoid": snn.surrogate.fast_sigmoid,
}


class NoisyWrapper(tnn.Module):
    def __init__(self, module, noise_size, apply_noise, *args, **kwargs):
        super().__init__()

        self.noise_size = noise_size
        self.apply_noise = apply_noise
        self.module = module
        self.zeros_ratio = 0

    def forward(self, x):
        self.zeros_ratio = ((x == 0).sum() / x.numel()).item()
        if self.training and self.apply_noise:
            noise = torch.randn_like(x) * self.noise_size
            x = x + noise

        return self.module.forward(x)


class SpikingLayer(tnn.Module):
    """
    Wrapper for parameterized layer (e.g., Linear, Conv) + a Spiking Mechanism (e.g., LIF)
    """

    def __init__(self, parameterized_layer: tnn.Module, spike_mechanism: snn.SpikingNeuron):
        super().__init__()

        self.parameterized_layer = parameterized_layer
        self.spike_mechanism = spike_mechanism

    def forward(self, x):
        x = self.parameterized_layer(x)
        x = self.spike_mechanism(x)

        return x


class LifMLP(tnn.Module):
    """
    Simple MLP using Leaky-Integrate-And-Fire Neurons
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

        # Classifier
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
        snn.Leaky.reset_hidden()
        snn.Leaky.detach_hidden()

    def forward(self, x):
        """
        Forwards input through network
        """

        x = torch.flatten(x, 1)
        x = self.classifier(x)

        # Return output
        return x


class SmallLifMLP(LifMLP):
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

        # Classifier
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
    Simple CNN using Leaky-Integrate-And-Fire Neurons
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

        # Classifier
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
        Forwards input through network
        """

        x = self.classifier(x)

        # Return output
        return x


class DeeperSNN(LifCNN):
    """
    Deeper SNN with more layers
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

        # Classifier
        self.classifier = tnn.Sequential(
            SpikingLayer(
                NoisyWrapper(tnn.Conv2d(n_channels, 30, 3), self.noise_size, self.apply_noise),
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
                NoisyWrapper(tnn.Conv2d(30, 150, 3), self.noise_size, self.apply_noise),
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
                NoisyWrapper(tnn.Conv2d(150, 250, 3), self.noise_size, self.apply_noise),
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
                NoisyWrapper(tnn.Conv2d(250, 200, 3), self.noise_size, self.apply_noise),
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
                NoisyWrapper(tnn.Conv2d(200, 100, 3), self.noise_size, self.apply_noise),
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


class ResNet(LifCNN):
    """
    ResNet-like architecture using Leaky-Integrate-And-Fire Neurons
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

        # Classifier
        self.block1 = SpikingLayer(
            NoisyWrapper(tnn.Conv2d(n_channels, 30, 5), self.noise_size, self.apply_noise),
            snn.Leaky(
                beta=beta,
                init_hidden=True,
                surrogate_disable=surrogate_disable,
                spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                reset_delay=reset_delay,
                **kwargs,
            ),
        )
        self.block2 = SpikingLayer(
            NoisyWrapper(tnn.Conv2d(30, 150, 3), self.noise_size, self.apply_noise),
            snn.Leaky(
                beta=beta,
                init_hidden=True,
                surrogate_disable=surrogate_disable,
                spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                reset_delay=reset_delay,
                **kwargs,
            ),
        )
        self.block3 = SpikingLayer(
            NoisyWrapper(tnn.Conv2d(150, 250, 3), self.noise_size, self.apply_noise),
            snn.Leaky(
                beta=beta,
                init_hidden=True,
                surrogate_disable=surrogate_disable,
                spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                reset_delay=reset_delay,
                **kwargs,
            ),
        )
        self.block4 = SpikingLayer(
            NoisyWrapper(tnn.Conv2d(250, 200, 4), self.noise_size, self.apply_noise),
            snn.Leaky(
                beta=beta,
                init_hidden=True,
                surrogate_disable=surrogate_disable,
                spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                reset_delay=reset_delay,
                **kwargs,
            ),
        )
        # self.block5 = SpikingLayer(
        #     NoisyWrapper(tnn.Conv2d(200, 100, 3), self.noise_size, self.apply_noise),
        #     snn.Leaky(
        #         beta=beta,
        #         init_hidden=True,
        #         surrogate_disable=surrogate_disable,
        #         spike_grad=SPIKE_GRAD_MAP[spike_grad](),
        #         reset_delay=reset_delay,
        #         **kwargs,
        #     ),
        # )

        self.skip_connection = snn.Convolution(
            in_channels=150,
            out_channels=250,
            kernel_size=1,
            weight_mean=0.8,
            weight_std=0.05,
        )
