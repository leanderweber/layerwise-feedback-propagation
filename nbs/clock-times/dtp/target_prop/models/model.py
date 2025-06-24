from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypedDict, TypeVar

import torch
from torch import Tensor, nn
from torchmetrics.classification.accuracy import Accuracy

from ..networks.network import Network

PhaseStr = Literal["train", "val", "test"]


class RequiredStepOutputs(TypedDict):
    """The dictionary format that is minimally required to be returned from
    `training/val/test_step`."""

    logits: Tensor
    """The un-normalized logits."""

    y: Tensor
    """ The class labels. """


class StepOutputDict(RequiredStepOutputs, total=False):
    """The dictionary format that is expected to be returned from `training/val/test_step`."""

    loss: Tensor
    """ Optional loss tensor that can be returned by those methods."""

    log: dict[str, Tensor | Any]
    """ Optional dictionary of things to log at each step."""


NetworkType = TypeVar("NetworkType", bound=Network)


class Model(ABC, Generic[NetworkType]):
    """Base class for all the models (a.k.a. learning algorithms) of the repo.

    The networks themselves are created separately.
    """

    def __init__(
        self,
        network: NetworkType,
        config=None,
    ):
        super().__init__()
        # IDEA: Could actually implement a `self.HParams` instance method that would choose the
        # default value contextually, based on the choice of datamodule! However Hydra already
        # kinda does that for us already.
        # NOTE: Can't exactly set the `hparams` attribute because it's a special property of PL.
        self.hp = hparams or self.HParams()
        self.net_hp = network.hparams
        self.config = config

        assert isinstance(network, nn.Module)
        self.network: NetworkType = network.to(self.config.device)
        # NOTE: Setting this property allows PL to infer the shapes and number of params.
        self.example_input_array = torch.rand(  # type: ignore
            [config["batch_size"], *config["dims"]], device=config["device"]
        )

        # IDEA: Could use a dict of metrics from torchmetrics instead of just accuracy:
        # self.supervised_metrics: dist[str, Metrics]
        self.accuracy = Accuracy()

        _ = self.network(self.example_input_array)
        print(f"Forward net: ")
        print(self.network)

    @property
    def forward_net(self) -> NetworkType:
        return self.network

    @forward_net.setter
    def forward_net(self, net: NetworkType) -> None:
        self.network = net

    def predict(self, x: Tensor) -> Tensor:
        """Predict the classification labels."""
        return self.forward_net(x).argmax(-1)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_net(x)
