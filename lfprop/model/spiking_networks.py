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

SPIKE_GRAD_MAP = {"step": Step, "atan": snn.surrogate.atan, "fast_sigmoid": snn.surrogate.fast_sigmoid}


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
        reset_delay=True,
        **kwargs,
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
                    spike_grad=SPIKE_GRAD_MAP[spike_grad](),
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
                    spike_grad=SPIKE_GRAD_MAP[spike_grad](),
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
                    spike_grad=SPIKE_GRAD_MAP[spike_grad](),
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
        )

        # Classifier
        self.classifier = tnn.Sequential(
            SpikingLayer(
                tnn.Conv2d(n_channels, 12, 5),
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
                tnn.Conv2d(12, 64, 5),
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
                tnn.Linear(1600 if n_channels == 3 else 64 * 4 * 4, n_outputs),
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


class LifVGG16(LifMLP):
    """
    VGG-16 CNN using Leaky-Integrate-And-Fire Neurons
    """

    def __init__(
        self,
        n_channels,
        n_outputs,
        beta,
        surrogate_disable=False,
        spike_grad=snn.surrogate.atan,
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
        )

        # Features (VGG-16 style)
        self.features = tnn.Sequential(
            SpikingLayer(
                tnn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
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
                tnn.Conv2d(64, 64, kernel_size=3, padding=1),
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
                tnn.Conv2d(64, 128, kernel_size=3, padding=1),
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
                tnn.Conv2d(128, 128, kernel_size=3, padding=1),
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
                tnn.Conv2d(128, 256, kernel_size=3, padding=1),
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
                tnn.Conv2d(256, 256, kernel_size=3, padding=1),
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
                tnn.Conv2d(256, 256, kernel_size=3, padding=1),
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
                tnn.Conv2d(256, 512, kernel_size=3, padding=1),
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
                tnn.Conv2d(512, 512, kernel_size=3, padding=1),
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
                tnn.Conv2d(512, 512, kernel_size=3, padding=1),
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
                tnn.Conv2d(512, 512, kernel_size=3, padding=1),
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
                tnn.Conv2d(512, 512, kernel_size=3, padding=1),
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
                tnn.Conv2d(512, 512, kernel_size=3, padding=1),
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
        )

        # Avgpool
        self.avgpool = tnn.AdaptiveAvgPool2d((7, 7))

        # Classifier (VGG-16 style, but with SpikingLayers)
        self.classifier = tnn.Sequential(
            SpikingLayer(
                tnn.Linear(512 * 7 * 7, 4096),
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
                tnn.Linear(4096, 4096),
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
                tnn.Linear(4096, n_outputs),
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

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        # Return output
        return x


class Sum(tnn.Module):
    def forward(self, x, y):
        return x + y


class LifResNet18(LifMLP):
    """
    ResNet-18 CNN using Leaky-Integrate-And-Fire Neurons
    """

    def __init__(
        self,
        n_channels,
        n_outputs,
        beta,
        surrogate_disable=False,
        spike_grad=snn.surrogate.atan,
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
        )

        def conv3x3(in_planes, out_planes, stride=1):
            return tnn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )

        class BasicBlock(tnn.Module):
            expansion = 1

            def __init__(self, in_planes, planes, stride=1, downsample=None):
                super().__init__()
                self.conv1 = conv3x3(in_planes, planes, stride)
                self.bn1 = SpikingLayer(
                    tnn.BatchNorm2d(planes),
                    snn.Leaky(
                        beta=beta,
                        init_hidden=True,
                        surrogate_disable=surrogate_disable,
                        spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                        reset_delay=reset_delay,
                        **kwargs,
                    ),
                )
                self.conv2 = conv3x3(planes, planes)
                self.bn2 = SpikingLayer(
                    tnn.BatchNorm2d(planes),
                    snn.Leaky(
                        beta=beta,
                        init_hidden=True,
                        surrogate_disable=surrogate_disable,
                        spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                        reset_delay=reset_delay,
                        **kwargs,
                    ),
                )
                self.downsample = downsample
                # Wrap Sum in SpikingLayer
                self.sum = SpikingLayer(
                    Sum(),
                    snn.Leaky(
                        beta=beta,
                        init_hidden=True,
                        surrogate_disable=surrogate_disable,
                        spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                        reset_delay=reset_delay,
                        **kwargs,
                    ),
                )

            def forward(self, x):
                identity = x

                out = self.conv1(x)
                out = self.bn1(out)

                out = self.conv2(out)
                out = self.bn2(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out = self.sum(out, identity)
                return out

        def make_layer(in_planes, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or in_planes != planes * BasicBlock.expansion:
                downsample = tnn.Sequential(
                    tnn.Conv2d(
                        in_planes,
                        planes * BasicBlock.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    SpikingLayer(
                        tnn.BatchNorm2d(planes * BasicBlock.expansion),
                        snn.Leaky(
                            beta=beta,
                            init_hidden=True,
                            surrogate_disable=surrogate_disable,
                            spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                            reset_delay=reset_delay,
                            **kwargs,
                        ),
                    ),
                )

            layers = []
            layers.append(BasicBlock(in_planes, planes, stride, downsample))
            for _ in range(1, blocks):
                layers.append(BasicBlock(planes * BasicBlock.expansion, planes))
            return tnn.Sequential(*layers)

        self.in_planes = 64

        self.conv1 = tnn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = SpikingLayer(
            tnn.BatchNorm2d(64),
            snn.Leaky(
                beta=beta,
                init_hidden=True,
                surrogate_disable=surrogate_disable,
                spike_grad=SPIKE_GRAD_MAP[spike_grad](),
                reset_delay=reset_delay,
                **kwargs,
            ),
        )
        self.maxpool = tnn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = make_layer(64, 64, 2)
        self.layer2 = make_layer(64, 128, 2, stride=2)
        self.layer3 = make_layer(128, 256, 2, stride=2)
        self.layer4 = make_layer(256, 512, 2, stride=2)

        self.avgpool = tnn.AdaptiveAvgPool2d((1, 1))
        self.fc = SpikingLayer(
            tnn.Linear(512 * BasicBlock.expansion, n_outputs),
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Helper functions

# MODEL_MAP = {
#     "lifmlp": LifMLP,
#     "smalllifmlp": SmallLifMLP,
#     "lifcnn": LifCNN,
#     "lifvgg16": LifVGG16,
#     "lifresnet18": LifResNet18,
# }


# def init_uniform(m):
#     if isinstance(m, tnn.Linear):
#         torch.nn.init.uniform_(m.weight, 0.0, 1.0)


# def get_model(model_name, n_channels, n_outputs, device, **kwargs):
#     """
#     Gets the correct model
#     """

#     # Check if model_name is supported
#     if model_name not in MODEL_MAP:
#         raise ValueError("Model '{}' is not supported.".format(model_name))

#     # Build model
#     if model_name in MODEL_MAP:
#         model = MODEL_MAP[model_name](
#             n_channels=n_channels,
#             n_outputs=n_outputs,
#             **kwargs,
#         )

#     model.reset()  # necessary for SNNs (see snntorch documentation)
#     # Return model on correct device
#     return model.to(device)
