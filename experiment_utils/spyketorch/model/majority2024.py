################################################################################
# Title:            majority2024.py                                            #
# Description:      Implementation of a deep SNN with STDP and anti-STDP using #
#                   a majority vote decision rule                              #
# Author:           Aidin Attar                                                #
# Date:             2024-07-01                                                 #
# Version:          0.1                                                        #
# Usage:            None                                                       #
# Notes:            None                                                       #
# Python version:   3.11.7                                                     #
################################################################################

import numpy as np
import SpykeTorch.functional as sf
import torch
import torch.nn as nn
from network_trainer import NetworkTrainer
from SpykeTorch import snn
from torch.nn.parameter import Parameter


class MajoritySNN(NetworkTrainer):
    """Implementation of the Mozafari et al. 2018 paper"""

    def __init__(self, in_channels=6, num_classes=10, device="cuda", tensorboard=False):
        super(MajoritySNN, self).__init__(num_classes=num_classes, device=device, tensorboard=tensorboard)

        self.block1 = nn.ModuleDict(
            {
                "conv": snn.Convolution(
                    in_channels=in_channels, out_channels=30, kernel_size=5, weight_mean=0.8, weight_std=0.05
                ),
                "stdp": None,
                "anti_stdp": None,
            }
        )
        self.block2 = nn.ModuleDict(
            {
                "conv": snn.Convolution(
                    in_channels=30, out_channels=250, kernel_size=3, weight_mean=0.8, weight_std=0.05
                ),
                "stdp": None,
                "anti_stdp": None,
            }
        )
        self.block3_1 = nn.ModuleDict(
            {
                "conv": snn.Convolution(
                    in_channels=250, out_channels=200, kernel_size=3, weight_mean=0.8, weight_std=0.05
                ),
                "stdp": None,
                "anti_stdp": None,
            }
        )
        self.block3_2 = nn.ModuleDict(
            {
                "conv": snn.Convolution(
                    in_channels=250, out_channels=200, kernel_size=5, weight_mean=0.8, weight_std=0.05
                ),
                "stdp": None,
                "anti_stdp": None,
            }
        )
        self.block3_3 = nn.ModuleDict(
            {
                "conv": snn.Convolution(
                    in_channels=250, out_channels=200, kernel_size=7, weight_mean=0.8, weight_std=0.05
                ),
                "stdp": None,
                "anti_stdp": None,
            }
        )

        self.block1_params = {
            "threshold": 15,
            "n_winners": 5,
            "inhibition_radius": 3,
        }
        self.block2_params = {
            "threshold": 10,
            "n_winners": 8,
            "inhibition_radius": 1,
        }
        self.block1["stdp"] = snn.STDP(
            layer=self.block1["conv"], learning_rate=(0.004, -0.003), use_stabilizer=True, lower_bound=0, upper_bound=1
        )
        self.block2["stdp"] = snn.STDP(
            layer=self.block2["conv"], learning_rate=(0.004, -0.003), use_stabilizer=True, lower_bound=0, upper_bound=1
        )
        self.block3_1["stdp"] = snn.STDP(
            layer=self.block3_1["conv"],
            learning_rate=(0.004, -0.003),
            use_stabilizer=False,
            lower_bound=0.2,
            upper_bound=0.8,
        )
        self.block3_2["stdp"] = snn.STDP(
            layer=self.block3_2["conv"],
            learning_rate=(0.004, -0.003),
            use_stabilizer=False,
            lower_bound=0.2,
            upper_bound=0.8,
        )
        self.block3_3["stdp"] = snn.STDP(
            layer=self.block3_3["conv"],
            learning_rate=(0.004, -0.003),
            use_stabilizer=False,
            lower_bound=0.2,
            upper_bound=0.8,
        )
        self.block3_1["anti_stdp"] = snn.STDP(
            layer=self.block3_1["conv"],
            learning_rate=(-0.004, 0.0005),
            use_stabilizer=False,
            lower_bound=0.2,
            upper_bound=0.8,
        )
        self.block3_2["anti_stdp"] = snn.STDP(
            layer=self.block3_2["conv"],
            learning_rate=(-0.004, 0.0005),
            use_stabilizer=False,
            lower_bound=0.2,
            upper_bound=0.8,
        )
        self.block3_3["anti_stdp"] = snn.STDP(
            layer=self.block3_3["conv"],
            learning_rate=(-0.004, 0.0005),
            use_stabilizer=False,
            lower_bound=0.2,
            upper_bound=0.8,
        )

        self.spk_cnt1 = 0
        self.spk_cnt2 = 0

        self.max_ap = Parameter(torch.Tensor([0.15]))
        self.to(device)
        # self.file = open("log_new.txt", "w")

    def forward(self, input, max_layer=3):
        """
        Forward pass of the network

        Parameters
        ----------
        input : torch.Tensor
            Input data
        max_layer : int
            Maximum layer to go through

        Returns
        -------
        torch.Tensor
            Output of the network
        """
        # Pad the input to avoid edge effects
        input = sf.pad(input.float(), (2, 2, 2, 2), 0)
        if self.training:
            pot = self.block1["conv"](input)
            spk, pot = sf.fire(pot, self.block1_params["threshold"], True)
            if max_layer == 1:
                self.update_layer1(input, pot)
                return spk, pot

            spk_in = sf.pad(sf.pooling(spk, 2, 2), (1, 1, 1, 1))
            pot = self.block2["conv"](spk_in)
            spk, pot = sf.fire(pot, self.block2_params["threshold"], True)
            if max_layer == 2:
                self.update_layer2(spk_in, pot)
                return spk, pot

            spk_in = sf.pad(sf.pooling(spk, 3, 3), (2, 2, 2, 2))
            pot_a = self.block3_1["conv"](spk_in)
            spk_a = sf.fire(pot_a)
            winners_a = sf.get_k_winners(pot_a, 1, 0, spk_a)

            pot_b = self.block3_2["conv"](spk_in)
            spk_b = sf.fire(pot_b)
            winners_b = sf.get_k_winners(pot_b, 1, 0, spk_b)

            pot_c = self.block3_3["conv"](spk_in)
            spk_c = sf.fire(pot_c)
            winners_c = sf.get_k_winners(pot_c, 1, 0, spk_c)

            winners = [winners_a, winners_b, winners_c]
            votes = np.zeros(self.num_classes)
            branch_votes = []

            for win in winners:
                branch_vote = -1
                if len(win) != 0:
                    branch_vote = self.decision_map[win[0][0]]
                    votes[branch_vote] += 1
                branch_votes.append(branch_vote)

            self.update_ctx(spk_in, [pot_a, pot_b, pot_c], [spk_a, spk_b, spk_c], winners)
            return self.get_output(winners, votes, branch_votes)
        else:
            return self.forward_eval(input, max_layer)

    def forward_eval(self, input, max_layer):
        """
        Forward pass of the network during evaluation

        Parameters
        ----------
        input : torch.Tensor
            Input data
        max_layer : int
            Maximum layer to go through

        Returns
        -------
        torch.Tensor
            Output of the network
        """
        pot = self.block1["conv"](input)
        spk, pot = sf.fire(pot, self.block1_params["threshold"], True)
        if max_layer == 1:
            return spk, pot

        spk_in = sf.pad(sf.pooling(spk, 2, 2), (1, 1, 1, 1))
        pot = self.block2["conv"](spk_in)
        spk, pot = sf.fire(pot, self.block2_params["threshold"], True)
        if max_layer == 2:
            return spk, pot

        spk_in = sf.pad(sf.pooling(spk, 3, 3), (2, 2, 2, 2))
        pot_a = self.block3_1["conv"](spk_in)
        spk_a = sf.fire(pot_a)
        winners_a = sf.get_k_winners(pot_a, 1, 0, spk_a)

        pot_b = self.block3_2["conv"](spk_in)
        spk_b = sf.fire(pot_b)
        winners_b = sf.get_k_winners(pot_b, 1, 0, spk_b)

        pot_c = self.block3_3["conv"](spk_in)
        spk_c = sf.fire(pot_c)
        winners_c = sf.get_k_winners(pot_c, 1, 0, spk_c)

        winners = [winners_a, winners_b, winners_c]
        votes = np.zeros(self.num_classes)
        branch_votes = []

        for win in winners:
            branch_vote = -1
            if len(win) != 0:
                branch_vote = self.decision_map[win[0][0]]
                votes[branch_vote] += 1
            branch_votes.append(branch_vote)

        self.update_ctx(spk_in, [pot_a, pot_b, pot_c], [spk_a, spk_b, spk_c], winners)
        return self.get_output(winners, votes, branch_votes)

    def update_layer1(self, input, pot):
        """Update the first layer of the network"""
        self.spk_cnt1 += 1
        if self.spk_cnt1 >= 500:
            self.spk_cnt1 = 0
            self.update_learning_rate(self.block1["stdp"])
        pot = sf.pointwise_inhibition(pot)
        spk = pot.sign()
        winners = sf.get_k_winners(pot, self.block1_params["n_winners"], self.block1_params["inhibition_radius"], spk)
        self.update_ctx(input, pot, spk, winners)

    def update_layer2(self, input, pot):
        """Update the second layer of the network"""
        self.spk_cnt2 += 1
        if self.spk_cnt2 >= 500:
            self.spk_cnt2 = 0
            self.update_learning_rate(self.block2["stdp"])
        pot = sf.pointwise_inhibition(pot)
        spk = pot.sign()
        winners = sf.get_k_winners(pot, self.block2_params["n_winners"], self.block2_params["inhibition_radius"], spk)
        self.update_ctx(input, pot, spk, winners)

    def update_ctx(self, input_spikes, potentials, output_spikes, winners):
        """
        Update the context of the network

        Parameters
        ----------
        input_spikes : torch.Tensor
            Input spikes
        potentials : torch.Tensor
            Neuron potentials
        output_spikes : torch.Tensor
            Output spikes
        winners : torch.Tensor
            Winners
        """
        self.ctx = {
            "input_spikes": input_spikes,
            "potentials": potentials,
            "output_spikes": output_spikes,
            "winners": winners,
        }

    def get_output(self, winners, votes, branch_votes):
        """Get the output of the network"""
        if len(winners) != 0:
            if np.sum(winners) >= 0:
                if np.max(votes) < 2:
                    return torch.tensor(branch_votes[1], device=self.device), branch_votes
                else:
                    return torch.tensor(np.argmax(votes), device=self.device), branch_votes
        return torch.tensor(-1, device=self.device)

    def update_learning_rate(self, stdp_layer):
        """Update the learning rate of the STDP layer"""
        ap = torch.tensor(stdp_layer.learning_rate[0][0].item(), device=stdp_layer.learning_rate[0][0].device) * 2
        ap = torch.min(ap, self.max_ap)
        an = ap * -0.75
        stdp_layer.update_all_learning_rate(ap.item(), an.item())

    def stdp(self, layer_idx):
        """
        Apply STDP to the network

        Parameters
        ----------
        layer_idx : int
            Index of the layer to apply STDP to
        """
        if layer_idx == 1:
            self.block1["stdp"](
                self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"]
            )
        elif layer_idx == 2:
            self.block2["stdp"](
                self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"]
            )
        else:
            raise ValueError("Invalid layer index")

    def update_learning_rates(self, stdp_ap, stdp_an, anti_stdp_ap, anti_stdp_an, layer_idx, branch_idx):
        """
        Update learning rates for STDP

        Parameters
        ----------
        stdp_ap : float
            Positive learning rate for STDP
        stdp_an : float
            Negative learning rate for STDP
        anti_stdp_ap : float
            Positive learning rate for anti-STDP
        anti_stdp_an : float
            Negative learning rate for anti-STDP
        layer_idx : int
            Index of the layer to update learning rates for
        """
        if branch_idx == 0:
            self.block3_1["stdp"].update_all_learning_rate(stdp_ap, stdp_an)
            self.block3_1["anti_stdp"].update_all_learning_rate(anti_stdp_an, anti_stdp_ap)
        elif branch_idx == 1:
            self.block3_2["stdp"].update_all_learning_rate(stdp_ap, stdp_an)
            self.block3_2["anti_stdp"].update_all_learning_rate(anti_stdp_an, anti_stdp_ap)
        elif branch_idx == 2:
            self.block3_3["stdp"].update_all_learning_rate(stdp_ap, stdp_an)
            self.block3_3["anti_stdp"].update_all_learning_rate(anti_stdp_an, anti_stdp_ap)
        else:
            raise ValueError("Invalid branch index")

    def reward(self, branch_idx):
        """Reward the network"""
        if branch_idx == 0:
            self.block3_1["stdp"](
                self.ctx["input_spikes"],
                self.ctx["potentials"][0],
                self.ctx["output_spikes"][0],
                self.ctx["winners"][0],
            )
        elif branch_idx == 1:
            self.block3_2["stdp"](
                self.ctx["input_spikes"],
                self.ctx["potentials"][1],
                self.ctx["output_spikes"][1],
                self.ctx["winners"][1],
            )
        elif branch_idx == 2:
            self.block3_3["stdp"](
                self.ctx["input_spikes"],
                self.ctx["potentials"][2],
                self.ctx["output_spikes"][2],
                self.ctx["winners"][2],
            )
        else:
            raise ValueError("Invalid branch index")

    def punish(self, branch_idx):
        """Punish the network"""
        if branch_idx == 0:
            self.block3_1["anti_stdp"](
                self.ctx["input_spikes"],
                self.ctx["potentials"][0],
                self.ctx["output_spikes"][0],
                self.ctx["winners"][0],
            )
        elif branch_idx == 1:
            self.block3_2["anti_stdp"](
                self.ctx["input_spikes"],
                self.ctx["potentials"][1],
                self.ctx["output_spikes"][1],
                self.ctx["winners"][1],
            )
        elif branch_idx == 2:
            self.block3_3["anti_stdp"](
                self.ctx["input_spikes"],
                self.ctx["potentials"][2],
                self.ctx["output_spikes"][2],
                self.ctx["winners"][2],
            )
        else:
            raise ValueError("Invalid branch index")

    def train_rl(self, data, target, layer_idx=3):
        """
        Train the network with reinforcement learning (R-STDP)

        Parameters
        ----------
        data : torch.Tensor
            Input data
        target : torch.Tensor
            Target data
        layer_idx : int
            Index of the layer to train

        Returns
        -------
        perf : np.array
            Performance of the network
        """
        self.train()
        perf = np.array([0, 0, 0])  # correct, wrong, silence

        for data_in, target_in in zip(data, target):
            data_in = data_in.to(self.device)
            target_in = target_in.to(self.device)
            d, branch_votes = self(data_in, layer_idx)

            for branch_idx, vote in enumerate(branch_votes):
                if vote != -1:
                    if vote == target_in:
                        self.reward(branch_idx)
                    else:
                        self.punish(branch_idx)

            if d != -1:
                if d == target_in:
                    perf[0] += 1
                else:
                    perf[1] += 1
            else:
                perf[2] += 1

        avg_loss = perf[1] / (perf[0] + perf[1] + perf[2])
        accuracy = perf[0] / (perf[0] + perf[1] + perf[2])

        # self.history['train_loss'].append(avg_loss)
        # self.history['train_acc'].append(accuracy)

        # Logging to TensorBoard
        if self.tensorboard:
            self.writer.add_scalar("Train/Loss_Iteration", avg_loss, self.iteration)
            self.writer.add_scalar("Train/Accuracy_Iteration", accuracy, self.iteration)

        self.iteration += 1

        return perf / len(data)

    def test(self, data, target, layer_idx=3):
        """
        Test the network

        Parameters
        ----------
        data : torch.Tensor
            Input data
        target : torch.Tensor
            Target data
        layer_idx : int
            Index of the layer to test

        Returns
        -------
        perf : np.array
            Performance of the network
        """
        self.eval()
        perf = np.array([0, 0, 0])  # correct, wrong, silence
        for data_in, target_in in zip(data, target):
            data_in = data_in.to(self.device)
            target_in = target_in.to(self.device)
            d, _ = self(data_in, layer_idx)
            if d != -1:
                if d == target_in:
                    perf[0] += 1
                else:
                    perf[1] += 1
            else:
                perf[2] += 1
        return perf / len(data)

    def compute_preds(self, data, target, layer_idx=3):
        """
        Compute evaluation metrics for the network

        Parameters
        ----------
        data : torch.Loader
            Input data
        epoch : int
            Current epoch number
        layer_idx : int
            Index of the layer to compute metrics

        Returns
        -------
        metrics : dict
            Dictionary containing evaluation metrics
        """
        self.eval()
        for data_in, target_in in zip(data, target):
            data_in = data_in.to(self.device)
            target_in = target_in.to(self.device)
            d, _ = self(data_in, layer_idx)
            if d != -1:
                self.all_preds.append(d.cpu().item())
                self.all_targets.append(target_in.cpu().item())

        self.all_preds = np.array(self.all_preds)
        self.all_targets = np.array(self.all_targets)
