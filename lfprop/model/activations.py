import torch
from torch import nn as tnn


class Step(tnn.Module):
    """
    Step activation function.

    Applies a step function to the input tensor. For each element,
    returns its sign if it is positive, otherwise returns zero.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inp):
        """
        Forward pass for the Step activation.

        Args:
            inp (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the step function.
        """
        step = torch.where(inp > 0, torch.sign(inp), inp * 0)
        return step


class NegTanh(tnn.Module):
    """
    Negative Tanh activation function.

    Applies the negative hyperbolic tangent function to the input tensor.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inp):
        """
        Forward pass for the Negative Tanh activation.

        Args:
            inp (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying -tanh.
        """
        return -torch.nn.functional.tanh(inp)


class NegReLU(tnn.Module):
    """
    Negative ReLU activation function.

    Applies the negative rectified linear unit function to the input tensor.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inp):
        """
        Forward pass for the Negative ReLU activation.

        Args:
            inp (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying -ReLU.
        """
        return -torch.nn.functional.relu(inp)


class NegInnerReLU(tnn.Module):
    """
    Negative Inner ReLU activation function.

    Applies the ReLU function to the negated input tensor.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inp):
        """
        Forward pass for the Negative Inner ReLU activation.

        Args:
            inp (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying ReLU to -inp.
        """
        return torch.nn.functional.relu(-inp)


class NegStep(tnn.Module):
    """
    Negative Step activation function.

    Applies a negative step function to the input tensor.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inp):
        """
        Forward pass for the Negative Step activation.

        Args:
            inp (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the negative step function.
        """
        step = torch.where(inp > 0, torch.sign(inp), inp * 0)
        return -step
