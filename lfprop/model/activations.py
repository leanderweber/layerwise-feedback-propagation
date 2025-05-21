import torch
from torch import nn as tnn


class Step(tnn.Module):
    """
    Step activation
    """

    def __init__(self):
        # self.fn = StepFunction.apply
        super().__init__()

    def forward(self, inp):
        # return self.fn(input)
        step = torch.where(inp > 0, torch.sign(inp), inp * 0)
        return step


class NegTanh(tnn.Module):
    """
    Negative Tanh activation
    """

    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return -torch.nn.functional.tanh(inp)


class NegReLU(tnn.Module):
    """
    Negative ReLU activation
    """

    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return -torch.nn.functional.relu(inp)


class NegInnerReLU(tnn.Module):
    """
    Negative Inner ReLU activation
    """

    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return torch.nn.functional.relu(-inp)


class NegStep(tnn.Module):
    """
    Negative Step activation
    """

    def __init__(self):
        super().__init__()

    def forward(self, inp):
        step = torch.where(inp > 0, torch.sign(inp), inp * 0)
        return -step
