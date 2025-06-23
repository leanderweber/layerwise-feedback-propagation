import torch


class MaximizeSingleNeuron:
    """
    Reward function to maximize the activation of a single neuron.
    """

    def __init__(self, device, **kwargs):
        """
        Args:
            device: The torch device to use.
            **kwargs: Additional arguments (unused).
        """
        self.device = device
        self.saved_rewards = []

    def __call__(self, logits, labels):
        """
        Computes the reward for maximizing the neuron activation.

        Args:
            logits (torch.Tensor): The output logits.
            labels (torch.Tensor): The target labels.

        Returns:
            torch.Tensor: The computed reward.
        """
        reward = logits.sign() * (1.0 - torch.sigmoid(logits))
        return reward


class MinimizeSingleNeuron:
    """
    Reward function to minimize the activation of a single neuron.
    """

    def __init__(self, device, **kwargs):
        """
        Args:
            device: The torch device to use.
            **kwargs: Additional arguments (unused).
        """
        self.device = device
        self.saved_rewards = []

    def __call__(self, logits, labels):
        """
        Computes the reward for minimizing the neuron activation.

        Args:
            logits (torch.Tensor): The output logits.
            labels (torch.Tensor): The target labels.

        Returns:
            torch.Tensor: The computed reward.
        """
        reward = logits.sign() * -(1.0 - torch.sigmoid(logits))
        return reward


class BinarySigmoidLossReward:
    """
    Reward function based on the binary sigmoid loss.
    """

    def __init__(self, device, **kwargs):
        """
        Args:
            device: The torch device to use.
            **kwargs: Additional arguments (unused).
        """
        self.device = device
        self.saved_rewards = []

    def __call__(self, logits, labels):
        """
        Computes the reward using binary sigmoid loss.

        Args:
            logits (torch.Tensor): The output logits.
            labels (torch.Tensor): The target labels.

        Returns:
            torch.Tensor: The computed reward.
        """
        reward = logits * (labels.view_as(logits) - torch.sigmoid(logits))
        return reward


class SigmoidLossReward:
    """
    Reward function based on the sigmoid loss for multi-class classification.
    """

    def __init__(self, device, **kwargs):
        """
        Args:
            device: The torch device to use.
            **kwargs: Additional arguments (unused).
        """
        self.device = device
        self.saved_rewards = []

    def __call__(self, logits, labels):
        """
        Computes the reward using sigmoid loss with one-hot labels.

        Args:
            logits (torch.Tensor): The output logits.
            labels (torch.Tensor): The target labels.

        Returns:
            torch.Tensor: The computed reward.
        """
        # Prepare one-hot labels
        eye = torch.eye(logits.size()[1], device=self.device)
        one_hot = eye[labels]
        reward = logits * (one_hot - torch.sigmoid(logits))
        return reward


class SoftmaxLossReward:
    """
    Reward function based on the softmax loss for multi-class classification.
    """

    def __init__(self, device, **kwargs):
        """
        Args:
            device: The torch device to use.
            **kwargs: Additional arguments (unused).
        """
        self.device = device

    def __call__(self, logits, labels):
        """
        Computes the reward using softmax loss with one-hot labels.

        Args:
            logits (torch.Tensor): The output logits.
            labels (torch.Tensor): The target labels.

        Returns:
            torch.Tensor: The computed reward.
        """
        # Prepare one-hot labels
        eye = torch.eye(logits.size()[1], device=self.device)
        one_hot = eye[labels]
        # Compute reward
        reward = logits * (one_hot - torch.nn.functional.softmax(logits, dim=1))
        return reward


class BoundedSoftmaxReward:
    """
    Reward function based on softmax loss with bounds on the softmax output.
    """

    def __init__(self, device, lower_bound=0.0, higher_bound=1.0, logit_sign_only=False, **kwargs):
        """
        Args:
            device: The torch device to use.
            lower_bound (float): Lower bound for softmax output.
            higher_bound (float): Upper bound for softmax output.
            logit_sign_only (bool): Whether to use only the sign of logits.
            **kwargs: Additional arguments (unused).
        """
        self.device = device
        self.lower_bound = lower_bound
        self.higher_bound = higher_bound
        self.logit_sign_only = logit_sign_only

    def __call__(self, logits, labels):
        """
        Computes the reward using bounded softmax loss.

        Args:
            logits (torch.Tensor): The output logits.
            labels (torch.Tensor): The target labels.

        Returns:
            torch.Tensor: The computed reward.
        """
        eye = torch.eye(logits.size()[1], device=self.device)
        one_hot = eye[labels]

        # Regularize softmax output to be within bounds
        regularized_softmax = torch.where(
            torch.nn.functional.softmax(logits, dim=1) > self.higher_bound,
            1.0,
            torch.nn.functional.softmax(logits, dim=1),
        )
        regularized_softmax = torch.where(regularized_softmax < self.lower_bound, 0.0, regularized_softmax)

        # Compute reward
        if self.logit_sign_only:
            reward = logits.sign() * (one_hot - regularized_softmax)
        else:
            reward = logits * (one_hot - regularized_softmax)
        return reward


class CorrectclassificationReward:
    """
    Reward function that gives a reward for correct classification.
    """

    def __init__(self, device, **kwargs):
        """
        Args:
            device: The torch device to use.
            **kwargs: Additional arguments (unused).
        """
        self.device = device
        self.saved_rewards = []

    def __call__(self, logits, labels):
        """
        Computes the reward for correct classification.

        Args:
            logits (torch.Tensor): The output logits.
            labels (torch.Tensor): The target labels.

        Returns:
            torch.Tensor: The computed reward.
        """
        reward = torch.zeros_like(logits)
        # Set all correct classifications to 1
        for lab, label in enumerate(labels):
            if logits[lab].amax() == logits[lab][label]:
                reward[lab][label] = 1
        # Correct Sign
        reward *= logits.sign()
        return reward


class MisclassificationReward:
    """
    Reward function that penalizes misclassification.
    """

    def __init__(self, device, **kwargs):
        """
        Args:
            device: The torch device to use.
            **kwargs: Additional arguments (unused).
        """
        self.device = device
        self.saved_rewards = []

    def __call__(self, logits, labels):
        """
        Computes the reward for misclassification.

        Args:
            logits (torch.Tensor): The output logits.
            labels (torch.Tensor): The target labels.

        Returns:
            torch.Tensor: The computed reward.
        """
        reward = (
            torch.where(
                torch.stack([logits[lab] > logits[lab][label] for lab, label in enumerate(labels)]),
                -1,
                0,
            )
            * logits.sign()
        )
        return reward


# rewards for spiking neural networks (install required dependencies via `pip install lfprop[snn]`)


class SnnCorrectClassRewardSpikesRateCoded:
    """
    Reward function for spiking neural networks using rate-coded spikes.
    """

    def __init__(self, device):
        """
        Args:
            device: The torch device to use.
        """
        self.device = device
        self.saved_rewards = []

    def __call__(self, spikes: torch.Tensor, labels: torch.Tensor):
        """
        Computes the reward for correct class in SNNs using rate coding.

        Args:
            spikes (torch.Tensor): Shape (n_timesteps, batchsize, output_shape).
            labels (torch.Tensor): Shape (batchsize).

        Returns:
            torch.Tensor: The computed reward.
        """
        # Prepare one-hot labels for each timestep
        eye = torch.eye(spikes.size()[2], device=self.device)
        one_hot = torch.stack([eye[labels] for _ in range(spikes.shape[0])], dim=0)

        pos_sum = spikes.sum(dim=0, keepdims=True)
        neg_sum = (spikes - 1).abs().sum(dim=0, keepdims=True)

        pos_decay = 1 - torch.sigmoid(pos_sum - spikes.shape[0] // 2)
        neg_decay = 1 - torch.sigmoid(neg_sum - spikes.shape[0] // 2)

        # Compute reward: encourage spikes for true class, penalize for others
        reward = torch.where(
            one_hot == 1,
            torch.ones_like(spikes) * pos_decay,
            -spikes * neg_decay,
        )
        return reward

    def get_predictions(self, spikes: torch.Tensor):
        """
        Computes predictions from spikes by summing over time and taking argmax.

        Args:
            spikes (torch.Tensor): Shape (n_timesteps, batchsize, output_shape).

        Returns:
            torch.Tensor: Predicted class indices.
        """
        return spikes.sum(0).argmax(-1)
