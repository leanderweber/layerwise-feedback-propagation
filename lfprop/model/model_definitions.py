import torch
import torch.nn as tnn


class LeNet(tnn.Module):
    """
    LeNet-like small convolutional neural network.

    Args:
        n_channels (int): Number of input channels.
        n_outputs (int): Number of output classes.
        activation (nn.Module): Activation function class (default: nn.ReLU).
    """

    def __init__(self, n_channels, n_outputs, activation=tnn.ReLU):
        super().__init__()

        # Feature extractor: two convolutional layers with activation and pooling
        self.features = tnn.Sequential(
            tnn.Conv2d(n_channels, 16, 5),
            activation(),
            tnn.MaxPool2d(2, 2),
            tnn.Conv2d(16, 16, 5),
            activation(),
            tnn.MaxPool2d(2, 2),
        )

        # Classifier: three fully connected layers with activation and dropout
        self.classifier = tnn.Sequential(
            tnn.Linear(256 if n_channels == 1 else 400, 120),
            activation(),
            tnn.Dropout(),
            tnn.Linear(120, 84),
            activation(),
            tnn.Dropout(),
            tnn.Linear(84, n_outputs),
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output logits.
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CifarVGGLike(tnn.Module):
    """
    VGG-like convolutional neural network for CIFAR datasets.

    Args:
        n_channels (int): Number of input channels.
        n_outputs (int): Number of output classes.
        activation (nn.Module): Activation function class (default: nn.ReLU).
    """

    def __init__(self, n_channels, n_outputs, activation=tnn.ReLU):
        super().__init__()

        # Feature extractor: multiple convolutional layers with activation and pooling
        self.features = tnn.Sequential(
            tnn.Conv2d(n_channels, 32, 3, padding=1),
            activation(),
            tnn.Conv2d(32, 64, 3, padding=1),
            activation(),
            tnn.MaxPool2d(2, 2),
            tnn.Conv2d(64, 128, 3, padding=1),
            activation(),
            tnn.Conv2d(128, 128, 3, padding=1),
            activation(),
            tnn.MaxPool2d(2, 2),
            tnn.Conv2d(128, 256, 3, padding=1),
            activation(),
            tnn.Conv2d(256, 256, 3, padding=1),
            activation(),
            tnn.MaxPool2d(2, 2),
        )

        # Classifier: three fully connected layers with activation and dropout
        self.classifier = tnn.Sequential(
            tnn.Linear(256 * 4 * 4 if n_channels == 3 else 256 * 3 * 3, 1024),
            activation(),
            tnn.Dropout(),
            tnn.Linear(1024, 512),
            activation(),
            tnn.Dropout(),
            tnn.Linear(512, n_outputs),
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output logits.
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CifarVGGLikeBN(tnn.Module):
    """
    VGG-like convolutional neural network with Batch Normalization for CIFAR datasets.

    Args:
        n_channels (int): Number of input channels.
        n_outputs (int): Number of output classes.
        activation (nn.Module): Activation function class (default: nn.ReLU).
    """

    def __init__(self, n_channels, n_outputs, activation=tnn.ReLU):
        super().__init__()

        # Feature extractor: convolutional layers with batch normalization, activation, and pooling
        self.features = tnn.Sequential(
            tnn.Conv2d(n_channels, 32, 3, padding=1),
            tnn.BatchNorm2d(32),
            activation(),
            tnn.Conv2d(32, 64, 3, padding=1),
            tnn.BatchNorm2d(64),
            activation(),
            tnn.MaxPool2d(2, 2),
            tnn.Conv2d(64, 128, 3, padding=1),
            tnn.BatchNorm2d(128),
            activation(),
            tnn.Conv2d(128, 128, 3, padding=1),
            tnn.BatchNorm2d(128),
            activation(),
            tnn.MaxPool2d(2, 2),
            tnn.Conv2d(128, 256, 3, padding=1),
            tnn.BatchNorm2d(256),
            activation(),
            tnn.Conv2d(256, 256, 3, padding=1),
            tnn.BatchNorm2d(256),
            activation(),
            tnn.MaxPool2d(2, 2),
        )

        # Classifier: three fully connected layers with activation and dropout
        self.classifier = tnn.Sequential(
            tnn.Linear(256 * 4 * 4, 1024),
            activation(),
            tnn.Dropout(),
            tnn.Linear(1024, 512),
            activation(),
            tnn.Dropout(),
            tnn.Linear(512, n_outputs),
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output logits.
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class DenseOnly(tnn.Module):
    """
    Small fully connected (dense) neural network.

    Args:
        n_channels (int): Number of input features.
        n_outputs (int): Number of output classes.
        activation (nn.Module): Activation function class (default: nn.LeakyReLU).
        activation_kwargs (dict, optional): Keyword arguments for activation function.
    """

    def __init__(self, n_channels, n_outputs, activation=tnn.LeakyReLU, activation_kwargs=None):
        super().__init__()

        if activation_kwargs is None:
            activation_kwargs = {}

        # Classifier: four fully connected layers with activation and dropout
        self.classifier = tnn.Sequential(
            tnn.Linear(n_channels, 512),
            activation(**activation_kwargs),
            tnn.Dropout(),
            tnn.Linear(512, 256),
            activation(**activation_kwargs),
            tnn.Dropout(),
            tnn.Linear(256, 128),
            activation(**activation_kwargs),
            tnn.Dropout(),
            tnn.Linear(128, n_outputs),
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output logits.
        """
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ToyDataDense(tnn.Module):
    """
    Small dense neural network for toy datasets.

    Args:
        n_channels (int): Number of input features.
        n_outputs (int): Number of output classes.
        activation (nn.Module): Activation function class (default: nn.ReLU).
        activation_kwargs (dict, optional): Keyword arguments for activation function.
    """

    def __init__(self, n_channels, n_outputs, activation=tnn.ReLU, activation_kwargs=None):
        super().__init__()

        if activation_kwargs is None:
            activation_kwargs = {}

        # Classifier: three fully connected layers with activation
        self.classifier = tnn.Sequential(
            tnn.Linear(n_channels, 32),
            activation(**activation_kwargs),
            tnn.Linear(32, 16),
            activation(**activation_kwargs),
            tnn.Linear(16, n_outputs),
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output logits.
        """
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
