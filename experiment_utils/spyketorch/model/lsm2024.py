################################################################################
# Title:            lsm2024.py                                                 #
# Description:      Implementation of a Liquid State Machine (LSM) for the     #
#                   NMNIST dataset.                                            #
# Author:           Aidin Attar                                                #
# Date:             2024-10-15                                                 #
# Version:          0.1                                                        #
# Usage:            None                                                       #
# Notes:            None                                                       #
# Python version:   3.11.7                                                     #
################################################################################

import numpy as np
import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from tqdm import tqdm


class LSM(nn.Module):
    def __init__(
        self,
        input_size: int,
        reservoir_size: int,
        output_size: int,
        n_partitions: int = 3,
        weight_in: np.ndarray = None,
        weight_res: np.ndarray = None,
        sparsity: float = 0.1,
        alpha: float = 0.9,
        beta: float = 0.9,
        threshold: float = 1.0,
    ):
        super(LSM, self).__init__()

        # Input layer (fully connected)
        self.input_fc = nn.Linear(input_size, reservoir_size)
        if weight_in is not None:
            self.input_fc.weight = nn.Parameter(torch.from_numpy(weight_in))

        # Define the reservoir layer
        self.reservoir_size = reservoir_size
        self.sparsity = sparsity
        self.n_partitions = n_partitions

        # Synaptic integration dynamics
        self.lsm = snn.RSynaptic(
            alpha=alpha, beta=beta, all_to_all=True, linear_features=reservoir_size, threshold=threshold
        )
        if weight_res is not None:
            self.lsm.recurrent.weight = nn.Parameter(torch.from_numpy(weight_res))

        # Output (readout) layer
        self.readout = nn.Linear(reservoir_size * n_partitions, output_size)

        # Freeze the weights of the input and reservoir layers
        self._freeze_weights()

        # Convert the model to double precision
        # self.double()
        self.to(torch.float32)

    def forward(self, x):
        # x = x.double()
        # Flatten the input data
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))

        num_steps = x.size(0)  # Number of time steps (assuming spike data is already encoded)
        spk, syn, mem = self.lsm.init_rsynaptic()  # Initialize membrane potential and synapse states

        spk_rec = []

        for step in range(num_steps):
            # Input to reservoir transformation
            current_input = self.input_fc(x[step])

            # Reservoir update
            spk, syn, mem = self.lsm(current_input, spk, syn, mem)

            spk_rec.append(spk)

        # Stack spikes over time
        spk_rec_out = torch.stack(spk_rec)

        # Partition the reservoir output
        partition_steps = num_steps // self.n_partitions
        lsm_parts = []

        # Aggregate spikes over each partition
        for partition in range(self.n_partitions):
            lsm_parts.append(
                torch.mean(spk_rec_out[partition * partition_steps : (partition + 1) * partition_steps], dim=0)
            )
        readout_input = torch.cat(lsm_parts, dim=1)

        # Pool the spike responses from the reservoir to pass to the readout layer
        output = self.readout(readout_input)

        # Softmax activation for classification
        output = F.softmax(output, dim=1)

        return output, spk_rec_out

    def _freeze_weights(self):
        """
        Freezes the weights of the input and reservoir layers.
        """
        for param in self.input_fc.parameters():
            param.requires_grad = False
        for param in self.lsm.recurrent.parameters():
            param.requires_grad = False


class LSM_partition(nn.Module):
    def __init__(
        self,
        input_size: int,
        reservoir_size: int,
        output_size: int,
        weight_in: np.ndarray = None,
        weight_lin: np.ndarray = None,
        weight_res: np.ndarray = None,
        n_partitions: int = 3,
        sparsity: float = 0.1,
        alpha: float = 0.9,
        beta: float = 0.9,
        threshold: float = 1.0,
        device: str = "cuda",
    ):
        super(LSM_partition, self).__init__()

        # Input layer (fully connected)
        self.input_fc = nn.Linear(input_size, reservoir_size)
        if weight_in is not None:
            self.input_fc.weight = nn.Parameter(torch.from_numpy(weight_in[0]))

        # Define the reservoir layer
        self.reservoir_size = reservoir_size
        self.sparsity = sparsity
        self.n_partitions = n_partitions
        self.weight_in = weight_in
        self.device = device

        # Initialize random sparse connectivity for the reservoir
        self.reservoir = nn.Linear(reservoir_size, reservoir_size, bias=False)
        if weight_lin is not None:
            self.reservoir.weight = nn.Parameter(torch.from_numpy(weight_lin))
        else:
            self.init_reservoir_connections()

        # Synaptic integration dynamics
        self.lsm = snn.RSynaptic(
            alpha=alpha, beta=beta, all_to_all=True, linear_features=reservoir_size, threshold=threshold
        )
        if weight_res is not None:
            self.lsm.recurrent.weight = nn.Parameter(torch.from_numpy(weight_res))

        # Output (readout) layer
        self.readout = nn.Linear(reservoir_size * n_partitions, output_size)

        # Freeze the weights of the input and reservoir layers
        self._freeze_weights()
        self.to(torch.float32)

    def _freeze_weights(self):
        """
        Freezes the weights of the input and reservoir layers.
        """
        for param in self.input_fc.parameters():
            param.requires_grad = False
        for param in self.reservoir.parameters():
            param.requires_grad = False
        for param in self.lsm.recurrent.parameters():
            param.requires_grad = False

    def init_reservoir_connections(self):
        # Randomly mask connections based on sparsity
        with torch.no_grad():
            mask = torch.rand_like(self.reservoir.weight) < self.sparsity
            self.reservoir.weight *= mask.float()

    def forward(self, x):
        # Flatten the input data
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))

        num_steps = x.size(0)  # Number of time steps (assuming spike data is already encoded)
        spk, syn, mem = self.lsm.init_rsynaptic()  # Initialize membrane potential and synapse states

        spk_rec = []

        partition_steps = num_steps // self.n_partitions
        partition_idx = 0

        for step in range(num_steps):
            if step % partition_steps == 0:
                self.input_fc.weight = nn.Parameter(torch.from_numpy(self.weight_in[partition_idx]).to(self.device))
                partition_idx = (partition_idx + 1) % self.n_partitions

            # Input to reservoir transformation
            current_input = self.input_fc(x[step])
            if step > partition_steps:
                current_input += self.reservoir(spk_rec[step - partition_steps])

            spk, syn, mem = self.lsm(current_input, spk, syn, mem)

            spk_rec.append(spk)

        # Stack spikes over time
        spk_rec_out = torch.stack(spk_rec)
        lsm_parts = []

        # Aggregate spikes over each partition
        for partition in range(self.n_partitions):
            lsm_parts.append(
                torch.mean(spk_rec_out[partition * partition_steps : (partition + 1) * partition_steps], dim=0)
            )
        readout_input = torch.cat(lsm_parts, dim=1)

        # Pool the spike responses from the reservoir to pass to the readout layer
        output = self.readout(readout_input)

        # Softmax activation for classification
        output = F.softmax(output, dim=1)

        return output, spk_rec_out


class LSM_radar(nn.Module):
    def __init__(
        self,
        input_size: int,
        reservoir_size: int,
        output_size: int,
        weight_in: np.ndarray = None,
        weight_lin: np.ndarray = None,
        weight_res: np.ndarray = None,
        n_partitions: int = 3,
        sparsity: float = 0.1,
        alpha: float = 0.9,
        beta: float = 0.9,
        threshold: float = 1.0,
        device: str = "cuda",
        dropout: float = 0.5,
    ):
        super(LSM_radar, self).__init__()

        # Input layer (fully connected)
        self.input_fc = nn.Linear(input_size, reservoir_size)
        if weight_in is not None:
            pass
            self.input_fc.weight = nn.Parameter(torch.from_numpy(weight_in[0]))

        # Define the reservoir layer
        self.reservoir_size = reservoir_size
        self.sparsity = sparsity
        self.n_partitions = n_partitions
        self.weight_in = weight_in
        self.device = device
        self.dropout = dropout

        # Initialize random sparse connectivity for the reservoir
        self.reservoir = nn.Linear(reservoir_size, reservoir_size, bias=False)
        if weight_lin is not None:
            self.reservoir.weight = nn.Parameter(torch.from_numpy(weight_lin))
        else:
            self.init_reservoir_connections()

        # Synaptic integration dynamics
        self.lsm = snn.RSynaptic(
            alpha=alpha, beta=beta, all_to_all=True, linear_features=reservoir_size, threshold=threshold
        )
        if weight_res is not None:
            self.lsm.recurrent.weight = nn.Parameter(torch.from_numpy(weight_res))

        # Output (readout) layer
        self.readout = nn.Sequential(
            nn.Linear(reservoir_size * n_partitions, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

        # Freeze the weights of the input and reservoir layers
        self._freeze_weights()
        self.to(torch.float32)  # Apply the weight initialization
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        """Initialize weights for the model layers."""
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)  # Xavier initialization
            if module.bias is not None:
                module.bias.data.fill_(0.0)  # Set bias to zero

    def _freeze_weights(self):
        """
        Freezes the weights of the input and reservoir layers.
        """
        for param in self.input_fc.parameters():
            param.requires_grad = False
        for param in self.reservoir.parameters():
            param.requires_grad = False
        for param in self.lsm.recurrent.parameters():
            param.requires_grad = False

    def init_reservoir_connections(self):
        # Randomly mask connections based on sparsity
        with torch.no_grad():
            mask = torch.rand_like(self.reservoir.weight) < self.sparsity
            self.reservoir.weight *= mask.float()

    def forward(self, x, lengths):
        max_length = max(lengths)
        spk, syn, mem = self.lsm.init_rsynaptic()

        spk_rec = []
        partition_steps = max_length // self.n_partitions
        partition_idx = 0

        # Dynamically iterate over time steps based on sequence length
        for step in range(max_length):
            # Change input weights for each partition
            if step % partition_steps == 0 and partition_idx < self.n_partitions:
                self.input_fc.weight = nn.Parameter(torch.from_numpy(self.weight_in[partition_idx]).to(self.device))
                partition_idx = (partition_idx + 1) % self.n_partitions

            # Apply input transformation and add previous reservoir spike
            current_input = torch.zeros((x.size(0), self.reservoir_size), device=self.device)
            for i, length in enumerate(lengths):
                if step < length:
                    current_input[i] = self.input_fc(x[i, step])

            if step >= partition_steps:
                current_input += self.reservoir(spk_rec[step - partition_steps])

            spk, syn, mem = self.lsm(current_input, spk, syn, mem)
            spk_rec.append(spk)

        # Stack spikes over time and compute average response for each partition
        spk_rec_out = torch.stack(spk_rec)
        lsm_parts = []
        for partition in range(self.n_partitions):
            start = partition * partition_steps
            end = start + partition_steps
            partition_data = [spk_rec_out[start:end, i].mean(0) for i, length in enumerate(lengths)]
            lsm_parts.append(torch.stack(partition_data))
        readout_input = torch.cat(lsm_parts, dim=1)

        # Apply dropout to the readout input
        readout_input = F.dropout(readout_input, p=self.dropout, training=self.training)

        # Pass aggregated response to readout layer and apply softmax
        output = self.readout(readout_input)
        # output = F.softmax(output, dim=1)
        output = F.log_softmax(output, dim=1)

        return output, spk_rec_out


class ConvLSM(nn.Module):
    def __init__(
        self,
        input_channels: int,
        input_height: int,
        input_width: int,
        reservoir_size: int,
        output_size: int,
        n_partitions: int = 3,
        weight_in: np.ndarray = None,
        weight_res: np.ndarray = None,
        kernel_size: int = 10,
        sparsity: float = 0.1,
        alpha: float = 0.9,
        beta: float = 0.9,
        threshold: float = 1.0,
    ):
        super(ConvLSM, self).__init__()

        # Input layer (convolutional)
        self.input_conv = nn.Conv2d(input_channels, reservoir_size, kernel_size=kernel_size, stride=1, padding=2)
        if weight_in is not None:
            pass
            # self.input_conv.weight = nn.Parameter(torch.from_numpy(weight_in))

        # Define the reservoir layer as recurrent convolutional layer
        self.reservoir_size = reservoir_size
        self.sparsity = sparsity
        self.n_partitions = n_partitions

        # Synaptic integration dynamics
        self.lsm = snn.RSynaptic(
            alpha=alpha,
            beta=beta,
            all_to_all=True,
            conv2d_channels=reservoir_size,
            kernel_size=kernel_size,
            threshold=threshold,
        )
        if weight_res is not None:
            pass
            self.lsm.recurrent.weight = nn.Parameter(torch.from_numpy(weight_res))

        # Output (readout) layer
        self.readout = nn.Sequential(
            nn.Conv2d(reservoir_size * n_partitions, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * input_height * input_width, output_size),
        )
        # Freeze the weights of the input and reservoir layers
        self._freeze_weights()

        # Ensure model uses float32 precision
        self.to(torch.float32)

    def forward(self, x):
        # Expecting input shape as [time_steps, batch_size, channels, height, width]
        num_steps = x.size(0)

        spk, syn, mem = self.lsm.init_rsynaptic()  # Initialize membrane potential and synapse states

        spk_rec = []

        for step in range(num_steps):
            # Input to reservoir transformation (convolution instead of linear)
            current_input = self.input_conv(x[step])

            # Reservoir update (convolutional dynamics within the reservoir)
            spk, syn, mem = self.lsm(current_input, spk, syn, mem)

            spk_rec.append(spk)

        # Stack spikes over time
        spk_rec_out = torch.stack(spk_rec)

        # Partition the reservoir output
        partition_steps = num_steps // self.n_partitions
        lsm_parts = []

        # Aggregate spikes over each partition
        for partition in range(self.n_partitions):
            lsm_parts.append(
                torch.mean(spk_rec_out[partition * partition_steps : (partition + 1) * partition_steps], dim=0)
            )
        readout_input = torch.cat(lsm_parts, dim=1)

        # Flatten the readout input
        # readout_input = readout_input.view(readout_input.size(0), -1)

        # Pool the spike responses from the reservoir to pass to the readout layer
        output = self.readout(readout_input)

        # Softmax activation for classification
        output = F.softmax(output, dim=1)

        return output, spk_rec_out

    def _freeze_weights(self):
        """
        Freezes the weights of the input and reservoir layers.
        """
        for param in self.input_conv.parameters():
            param.requires_grad = False
        for param in self.lsm.recurrent.parameters():
            param.requires_grad = False


class LSM_conv(nn.Module):
    def __init__(
        self,
        input_channels: int,
        reservoir_channels: int,
        output_size: int,
        weight_in=None,
        weight_res=None,
        n_partitions: int = 3,
        sparsity: float = 0.1,
        alpha: float = 0.9,
        beta: float = 0.9,
        threshold: float = 1.0,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
    ):
        super(LSM_conv, self).__init__()

        # Input layer (convolutional)
        self.input_conv = nn.Conv2d(
            input_channels, reservoir_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )
        if weight_in is not None:
            pass
            self.input_conv.weight = nn.Parameter(torch.from_numpy(weight_in))

        # Define the reservoir layer as recurrent convolutional layer
        self.reservoir_channels = reservoir_channels
        self.sparsity = sparsity
        self.n_partitions = n_partitions

        # Synaptic integration dynamics
        self.lsm = snn.RSynaptic(
            alpha=alpha,
            beta=beta,
            all_to_all=True,
            linear_features=None,
            conv2d_channels=reservoir_channels,
            threshold=threshold,
            kernel_size=kernel_size,
        )
        if weight_res is not None:
            pass
            self.lsm.recurrent.weight = nn.Parameter(torch.from_numpy(weight_res))

        # Output (readout) layer - this could also be a fully connected layer or another conv layer depending on the design
        self.readout1 = nn.Conv2d(
            reservoir_channels * n_partitions, output_size, kernel_size=kernel_size, stride=stride, padding=padding
        )

        self.readout2 = nn.Linear(output_size * 34 * 34, output_size)

        # Freeze the weights of the input and reservoir layers
        self._freeze_weights()

        # Ensure model uses float32 precision
        self.to(torch.float32)

    def forward(self, x):
        # Expecting input shape as [time_steps, batch_size, channels, height, width]
        num_steps = x.size(0)  # Number of time steps (assuming spike data is already encoded)
        spk, syn, mem = self.lsm.init_rsynaptic()  # Initialize membrane potential and synapse states

        spk_rec = []

        for step in range(num_steps):
            # Input to reservoir transformation (convolution instead of linear)
            current_input = self.input_conv(x[step])

            # Reservoir update (convolutional dynamics within the reservoir)
            spk, syn, mem = self.lsm(current_input, spk, syn, mem)

            spk_rec.append(spk)

        # Stack spikes over time
        spk_rec_out = torch.stack(spk_rec)

        # Partition the reservoir output
        partition_steps = num_steps // self.n_partitions
        lsm_parts = []

        # Aggregate spikes over each partition
        for partition in range(self.n_partitions):
            lsm_parts.append(
                torch.mean(spk_rec_out[partition * partition_steps : (partition + 1) * partition_steps], dim=0)
            )

        # Concatenate partitions along the channel dimension
        readout_input = torch.cat(lsm_parts, dim=1)

        # Pool the spike responses from the reservoir to pass to the readout layer
        output = self.readout1(readout_input)
        output = self.readout2(output.view(output.size(0), -1))

        # Softmax activation for classification
        output = F.softmax(output, dim=1)

        return output, spk_rec_out

    def _freeze_weights(self):
        """
        Freezes the weights of the input and reservoir layers.
        """
        for param in self.input_conv.parameters():
            param.requires_grad = False
        for param in self.lsm.recurrent.parameters():
            param.requires_grad = False


def initialize_weights(
    input_weight_scale,
    reservoir_weight_scale,
    input_connection_density,
    input_size,
    decay_factor=9,
    inhibitory_fraction=0.2,
    grid_x=10,
    grid_y=10,
    grid_z=10,
    initialize_reservoir_weights=True,
    reservoir_weights=None,
):
    """
    Initializes input-to-reservoir and reservoir-to-reservoir weights for an LSM.

    Parameters
    ---------
    input_weight_scale : float
        Scaling factor for input-to-reservoir connections.
    reservoir_weight_scale : float
        Scaling factor for reservoir-to-reservoir connections.
    input_connection_density : float
        Fraction of reservoir neurons connected to each input neuron.
    input_size : int
        Number of input neurons.
    decay_factor : float
        Spatial decay factor for reservoir connections.
    inhibitory_fraction : float
        Fraction of neurons in the reservoir that are inhibitory.
    grid_x : int
        Size of the reservoir grid in the x-dimension.
    grid_y : int
        Size of the reservoir grid in the y-dimension.
    grid_z : int
        Size of the reservoir grid in the z-dimension.
    initialize_reservoir_weights : bool
        Whether to initialize reservoir weights.
    reservoir_weights : np.ndarray or None
        Pre-initialized reservoir weights, if available.

    Returns
    -------
    Tuple
    - input_weights (np.ndarray): Input-to-reservoir weight matrix.
    - reservoir_weights (np.ndarray): Reservoir-to-reservoir weight matrix.
    """
    num_reservoir_neurons = grid_x * grid_y * grid_z
    input_weights = np.zeros((input_size, num_reservoir_neurons))
    num_input_connections = np.int32(num_reservoir_neurons * input_connection_density)

    # Initialize input-to-reservoir weights
    for input_neuron in range(input_size):
        neuron_indices = np.arange(num_reservoir_neurons)
        np.random.shuffle(neuron_indices)
        positive_connections = neuron_indices[:num_input_connections]
        negative_connections = neuron_indices[-num_input_connections:]

        input_weights[input_neuron, positive_connections] = input_weight_scale
        input_weights[input_neuron, negative_connections] = -input_weight_scale

    # Reservoir neuron inhibition/excitation setup
    neuron_indices = np.arange(num_reservoir_neurons)
    np.random.shuffle(neuron_indices)
    inhibitory_count = np.int32(inhibitory_fraction * num_reservoir_neurons)

    # Initialize reservoir-to-reservoir weights (if needed)
    if initialize_reservoir_weights:
        reservoir_weights = np.zeros((num_reservoir_neurons, num_reservoir_neurons))

        iterator = tqdm(
            range(num_reservoir_neurons),
            desc="Initializing reservoir weights",
            unit="neuron",
            position=0,
            leave=True,
            dynamic_ncols=True,
            total=num_reservoir_neurons,
            initial=0,
            ascii=True,
        )
        for post_idx in iterator:
            post_neuron = neuron_indices[post_idx]
            z_post = post_neuron // (grid_x * grid_y)
            y_post = (post_neuron - z_post * grid_x * grid_y) // grid_x
            x_post = post_neuron % grid_x

            for pre_idx in range(num_reservoir_neurons):
                pre_neuron = neuron_indices[pre_idx]
                z_pre = pre_neuron // (grid_x * grid_y)
                y_pre = (pre_neuron - z_pre * grid_x * grid_y) // grid_x
                x_pre = pre_neuron % grid_x

                distance_squared = (x_post - x_pre) ** 2 + (y_post - y_pre) ** 2 + (z_post - z_pre) ** 2
                probability = np.exp(-distance_squared / decay_factor)

                if post_idx < inhibitory_count and pre_idx < inhibitory_count:
                    # Inhibitory-to-inhibitory connection (II)
                    connection_prob = 0.3 * probability
                    if np.random.uniform() < connection_prob:
                        reservoir_weights[pre_neuron, post_neuron] = -reservoir_weight_scale
                elif post_idx < inhibitory_count and pre_idx >= inhibitory_count:
                    # Excitatory-to-inhibitory connection (EI)
                    connection_prob = 0.1 * probability
                    if np.random.uniform() < connection_prob:
                        reservoir_weights[pre_neuron, post_neuron] = reservoir_weight_scale
                elif post_idx >= inhibitory_count and pre_idx < inhibitory_count:
                    # Inhibitory-to-excitatory connection (IE)
                    connection_prob = 0.05 * probability
                    if np.random.uniform() < connection_prob:
                        reservoir_weights[pre_neuron, post_neuron] = -reservoir_weight_scale
                else:
                    # Excitatory-to-excitatory connection (EE)
                    connection_prob = 0.2 * probability
                    if np.random.uniform() < connection_prob:
                        reservoir_weights[pre_neuron, post_neuron] = reservoir_weight_scale

        # Ensure no self-connections
        np.fill_diagonal(reservoir_weights, 0)

    return input_weights.T, reservoir_weights.T  # Transposed for compatibility with PyTorch nn.Linear


def initialize_partitioned_weights(
    input_weight_scale,
    local_reservoir_weight_scale,
    long_distance_inhibitory_weight_scale,
    input_connection_density,
    input_size,
    num_partitions,
    decay_factor=9,
    inhibitory_fraction=0.2,
    grid_x=10,
    grid_y=10,
    grid_z=10,
    initialize_reservoir_weights=True,
    reservoir_weights=None,
):
    """
    Initializes input-to-reservoir and reservoir-to-reservoir weights for a partitioned LSM with cross-partition inhibition.

    Parameters
    ----------
    input_weight_scale : float
        Scaling factor for input-to-reservoir connections.
    local_reservoir_weight_scale : float
        Scaling factor for reservoir-to-reservoir local connections.
    long_distance_inhibitory_weight_scale : float
        Scaling factor for cross-partition inhibition connections.
    input_connection_density : float
        Fraction of reservoir neurons connected to each input neuron.
    input_size : int
        Number of input neurons.
    num_partitions : int
        Number of partitions for the reservoir.
    decay_factor : float
        Spatial decay factor for reservoir connections.
    inhibitory_fraction : float
        Fraction of neurons in the reservoir that are inhibitory.
    grid_x : int
        Size of the reservoir grid in the x-dimension.
    grid_y : int
        Size of the reservoir grid in the y-dimension.
    grid_z : int
        Size of the reservoir grid in the z-dimension.
    initialize_reservoir_weights : bool
        Whether to initialize reservoir weights.
    reservoir_weights : np.ndarray or None
        Pre-initialized reservoir weights, if available.

    Returns
    -------
    Tuple
    - input_weights (list of np.ndarray): Input-to-reservoir weight matrices for each partition.
    - reservoir_weights (np.ndarray): Reservoir-to-reservoir local weight matrix.
    - long_distance_inhibitory_weights (np.ndarray): Cross-partition long-distance inhibitory connections.
    """
    partition_size_z = grid_z // num_partitions
    total_neurons = grid_x * grid_y * grid_z
    partition_neurons = grid_x * grid_y * partition_size_z

    input_connection_range = np.int32(partition_neurons * input_connection_density)
    input_weights_partition = np.zeros((input_size, partition_neurons))
    input_weights = []

    # Initialize input-to-reservoir weights for each partition
    for input_neuron in range(input_size):
        neuron_indices = np.arange(partition_neurons)
        np.random.shuffle(neuron_indices)
        positive_connections = neuron_indices[:input_connection_range]
        negative_connections = neuron_indices[-input_connection_range:]

        input_weights_partition[input_neuron, positive_connections] = input_weight_scale
        input_weights_partition[input_neuron, negative_connections] = -input_weight_scale

    # Calculate statistics
    active_connections = input_weights_partition > 0
    input_fanout = np.sum(active_connections, axis=1) / input_weight_scale
    reservoir_fanin = np.sum(active_connections, axis=0) / input_weight_scale
    print(f"Input-to-reservoir fanout: {np.mean(input_fanout):.2f}, shape: {input_fanout.shape}")
    print(f"Reservoir fanin from input: {np.mean(reservoir_fanin):.2f}, shape: {reservoir_fanin.shape}")

    # Assign input weights to different partitions
    for partition in range(num_partitions):
        partition_input_weights = np.zeros((input_size, total_neurons))
        partition_input_weights[:, partition * partition_neurons : (partition + 1) * partition_neurons] = (
            input_weights_partition
        )
        input_weights.append(partition_input_weights.T)

    # Initialize reservoir weights (local connections) and cross-partition inhibitory weights
    neuron_indices = np.arange(partition_neurons)
    np.random.shuffle(neuron_indices)
    inhibitory_count = np.int32(inhibitory_fraction * partition_neurons)

    if initialize_reservoir_weights:
        local_reservoir_weights = np.zeros((total_neurons, total_neurons))
        cross_partition_inhibitory_weights = np.zeros((total_neurons, total_neurons))

        for i in range(partition_neurons):
            post_neuron = neuron_indices[i]
            z_post = post_neuron // (grid_x * grid_y)
            y_post = (post_neuron - z_post * grid_x * grid_y) // grid_x
            x_post = post_neuron % grid_x

            for j in range(partition_neurons):
                pre_neuron = neuron_indices[j]
                z_pre = pre_neuron // (grid_x * grid_y)
                y_pre = (pre_neuron - z_pre * grid_x * grid_y) // grid_x
                x_pre = pre_neuron % grid_x

                distance_squared = (x_post - x_pre) ** 2 + (y_post - y_pre) ** 2 + (z_post - z_pre) ** 2
                connection_probability = np.exp(-distance_squared / decay_factor)

                # Determine connection type and assign weights
                if i < inhibitory_count and j < inhibitory_count:
                    # Inhibitory-to-inhibitory (II) connections
                    if np.random.uniform() < 0.3 * connection_probability:
                        local_reservoir_weights[pre_neuron, post_neuron] = -local_reservoir_weight_scale
                elif i < inhibitory_count and j >= inhibitory_count:
                    # Excitatory-to-inhibitory (EI) connections
                    if np.random.uniform() < 0.1 * connection_probability:
                        local_reservoir_weights[pre_neuron, post_neuron] = local_reservoir_weight_scale
                elif i >= inhibitory_count and j < inhibitory_count:
                    # Inhibitory-to-excitatory (IE) connections
                    if np.random.uniform() < 0.05 * connection_probability:
                        local_reservoir_weights[pre_neuron, post_neuron] = -local_reservoir_weight_scale
                else:
                    # Excitatory-to-excitatory (EE) connections
                    if np.random.uniform() < 0.2 * connection_probability:
                        local_reservoir_weights[pre_neuron, post_neuron] = local_reservoir_weight_scale

        # Ensure no self-connections in the local reservoir
        np.fill_diagonal(local_reservoir_weights, 0)

        # Create cross-partition inhibitory connections
        for i in range(total_neurons):
            cross_partition_inhibitory_weights[
                i, (i + partition_neurons) % total_neurons
            ] = -long_distance_inhibitory_weight_scale

        # Assign local weights to each partition
        for partition in range(num_partitions):
            local_reservoir_weights[
                partition * partition_neurons : (partition + 1) * partition_neurons,
                partition * partition_neurons : (partition + 1) * partition_neurons,
            ] = local_reservoir_weights[:partition_neurons, :partition_neurons]

    # Convert to float32
    for i in range(len(input_weights)):
        input_weights[i] = input_weights[i].astype(np.float32)

    return input_weights, local_reservoir_weights.T, cross_partition_inhibitory_weights.T


def initialize_conv_weights(
    input_weight_scale,
    reservoir_weight_scale,
    input_channels,
    input_height,
    input_width,
    kernel_size,
    decay_factor=9,
    inhibitory_fraction=0.2,
    grid_x=4,
    grid_y=4,
    grid_z=4,
    initialize_reservoir_weights=True,
    reservoir_weights=None,
):
    """
    Initializes input-to-reservoir and reservoir-to-reservoir convolutional weights for an LSM.

    Parameters
    ---------
    input_weight_scale : float
        Scaling factor for input-to-reservoir connections.
    reservoir_weight_scale : float
        Scaling factor for reservoir-to-reservoir connections.
    input_channels : int
        Number of input channels (depth of the input image).
    input_height : int
        Height of the input image.
    input_width : int
        Width of the input image.
    kernel_size : int
        Size of the convolutional kernel.
    decay_factor : float
        Spatial decay factor for reservoir connections.
    inhibitory_fraction : float
        Fraction of neurons in the reservoir that are inhibitory.
    grid_x : int
        Size of the reservoir grid in the x-dimension.
    grid_y : int
        Size of the reservoir grid in the y-dimension.
    grid_z : int
        Size of the reservoir grid in the z-dimension.
    initialize_reservoir_weights : bool
        Whether to initialize reservoir weights.
    reservoir_weights : np.ndarray or None
        Pre-initialized reservoir weights, if available.

    Returns
    -------
    Tuple
    - input_weights (np.ndarray): Input-to-reservoir convolutional weight tensor.
    - reservoir_weights (np.ndarray): Reservoir-to-reservoir convolutional weight tensor.
    """
    # Calculate the total number of reservoir neurons
    num_reservoir_neurons = grid_x * grid_y * grid_z

    # Total number of input neurons based on input size and channels
    total_input_neurons = input_channels * input_height * input_width

    # Initialize input-to-reservoir weights (total_input_neurons x num_reservoir_neurons)
    input_weights = np.zeros((num_reservoir_neurons, input_channels, kernel_size, kernel_size))  # Match reservoir size
    for res_neuron in range(num_reservoir_neurons):
        for c in range(input_channels):
            input_weights[res_neuron, c, :, :] = np.random.randn(kernel_size, kernel_size) * input_weight_scale

    # Reservoir neuron inhibition/excitation setup
    inhibitory_count = np.int32(inhibitory_fraction * num_reservoir_neurons)

    # Initialize reservoir-to-reservoir convolutional weights (if needed)
    if initialize_reservoir_weights:
        reservoir_weights = np.zeros((num_reservoir_neurons, num_reservoir_neurons, kernel_size, kernel_size))

        iterator = tqdm(
            range(num_reservoir_neurons),
            desc="Initializing reservoir conv weights",
            unit="neuron",
            position=0,
            leave=True,
            dynamic_ncols=True,
            total=num_reservoir_neurons,
            initial=0,
            ascii=True,
        )
        for post_idx in iterator:
            for pre_idx in range(num_reservoir_neurons):
                if pre_idx < inhibitory_count and post_idx < inhibitory_count:
                    # Inhibitory-to-inhibitory (II)
                    connection_prob = 0.3 * np.exp(-((post_idx - pre_idx) ** 2) / decay_factor)
                    if np.random.uniform() < connection_prob:
                        reservoir_weights[pre_idx, post_idx] = (
                            -np.random.randn(kernel_size, kernel_size) * reservoir_weight_scale
                        )
                elif pre_idx >= inhibitory_count and post_idx < inhibitory_count:
                    # Excitatory-to-inhibitory (EI)
                    connection_prob = 0.1 * np.exp(-((post_idx - pre_idx) ** 2) / decay_factor)
                    if np.random.uniform() < connection_prob:
                        reservoir_weights[pre_idx, post_idx] = (
                            np.random.randn(kernel_size, kernel_size) * reservoir_weight_scale
                        )
                elif pre_idx < inhibitory_count and post_idx >= inhibitory_count:
                    # Inhibitory-to-excitatory (IE)
                    connection_prob = 0.05 * np.exp(-((post_idx - pre_idx) ** 2) / decay_factor)
                    if np.random.uniform() < connection_prob:
                        reservoir_weights[pre_idx, post_idx] = (
                            -np.random.randn(kernel_size, kernel_size) * reservoir_weight_scale
                        )
                else:
                    # Excitatory-to-excitatory (EE)
                    connection_prob = 0.2 * np.exp(-((post_idx - pre_idx) ** 2) / decay_factor)
                    if np.random.uniform() < connection_prob:
                        reservoir_weights[pre_idx, post_idx] = (
                            np.random.randn(kernel_size, kernel_size) * reservoir_weight_scale
                        )

    return input_weights, reservoir_weights
