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
        self, n_channels, n_outputs, beta, surrogate_disable=False, spike_grad=Step, reset_delay=True, **kwargs
    ):
        super().__init__()

        # Classifier
        self.classifier = tnn.Sequential(
            SpikingLayer(
                tnn.Linear(n_channels, 1000),
                snn.Leaky(
                    beta=beta,
                    init_hidden=True,
                    surrogate_disable=surrogate_disable,
                    spike_grad=spike_grad(),
                    reset_delay=reset_delay,
                    **kwargs,
                ),
            ),
            SpikingLayer(
                tnn.Linear(1000, 1000),
                snn.Leaky(
                    beta=beta,
                    init_hidden=True,
                    surrogate_disable=surrogate_disable,
                    spike_grad=spike_grad(),
                    reset_delay=reset_delay,
                    **kwargs,
                ),
            ),
            SpikingLayer(
                tnn.Linear(1000, n_outputs),
                snn.Leaky(
                    beta=beta,
                    init_hidden=True,
                    output=True,
                    surrogate_disable=surrogate_disable,
                    spike_grad=spike_grad(),
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
        self, n_channels, n_outputs, beta, surrogate_disable=False, spike_grad=Step, reset_delay=True, **kwargs
    ):
        super().__init__(
            n_channels,
            n_outputs,
            beta,
            surrogate_disable=surrogate_disable,
            spike_grad=spike_grad,
            reset_delay=reset_delay,
            **kwargs,
        )

        # Classifier
        self.classifier = tnn.Sequential(
            SpikingLayer(
                tnn.Linear(n_channels, 1000),
                snn.Leaky(
                    beta=beta,
                    init_hidden=True,
                    surrogate_disable=surrogate_disable,
                    spike_grad=spike_grad(),
                    reset_delay=reset_delay,
                    **kwargs,
                ),
            ),
            SpikingLayer(
                tnn.Linear(1000, n_outputs),
                snn.Leaky(
                    beta=beta,
                    init_hidden=True,
                    output=True,
                    surrogate_disable=surrogate_disable,
                    spike_grad=spike_grad(),
                    reset_delay=reset_delay,
                ),
            ),
        )


class LifCNN(LifMLP):
    """
    Simple CNN using Leaky-Integrate-And-Fire Neurons
    """

    def __init__(
        self, n_channels, n_outputs, beta, surrogate_disable=False, spike_grad=Step, reset_delay=True, **kwargs
    ):
        super().__init__(
            n_channels,
            n_outputs,
            beta,
            surrogate_disable=surrogate_disable,
            spike_grad=spike_grad,
            reset_delay=reset_delay,
        )

        # Classifier
        self.classifier = tnn.Sequential(
            SpikingLayer(
                tnn.Conv2d(n_channels, 12, 5),
                snn.Leaky(
                    beta=beta,
                    init_hidden=True,
                    surrogate_disable=surrogate_disable,
                    spike_grad=spike_grad(),
                    reset_delay=reset_delay,
                    **kwargs,
                ),
            ),
            tnn.MaxPool2d(2),
            SpikingLayer(
                tnn.Conv2d(12, 64, 5),
                snn.Leaky(
                    beta=beta,
                    init_hidden=True,
                    surrogate_disable=surrogate_disable,
                    spike_grad=spike_grad(),
                    reset_delay=reset_delay,
                    **kwargs,
                ),
            ),
            tnn.MaxPool2d(2),
            tnn.Flatten(),
            SpikingLayer(
                tnn.Linear(64 * 4 * 4, n_outputs),
                snn.Leaky(
                    beta=beta,
                    init_hidden=True,
                    output=True,
                    surrogate_disable=surrogate_disable,
                    spike_grad=spike_grad(),
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


# Helper functions

MODEL_MAP = {
    "lifmlp": LifMLP,
    "smalllifmlp": SmallLifMLP,
    "lifcnn": LifCNN,
}


def init_uniform(m):
    if isinstance(m, tnn.Linear):
        torch.nn.init.uniform_(m.weight, 0.0, 1.0)


def get_model(model_name, n_channels, n_outputs, device, **kwargs):
    """
    Gets the correct model
    """

    # Check if model_name is supported
    if model_name not in MODEL_MAP:
        raise ValueError("Model '{}' is not supported.".format(model_name))

    # Build model
    if model_name in MODEL_MAP:
        model = MODEL_MAP[model_name](
            n_channels=n_channels,
            n_outputs=n_outputs,
            **kwargs,
        )

    model.reset()  # necessary for SNNs (see snntorch documentation)
    # Return model on correct device
    return model.to(device)
