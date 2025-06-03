################################################################################
# Title:            mozafari2018.py                                            #
# Description:      Implementation of the Mozafari et al. 2018 paper           #
# Author:           Aidin Attar                                                #
# Date:             2024-07-01                                                 #
# Version:          0.1                                                        #
# Usage:            None                                                       #
# Notes:            None                                                       #
# Python version:   3.11.7                                                     #
################################################################################

import SpykeTorch.functional as sf
import torch
import torch.nn as nn
from network_trainer import NetworkTrainer
from SpykeTorch import snn
from torch.nn.parameter import Parameter


class MozafariMNIST2018(NetworkTrainer):
    """Implementation of the Mozafari et al. 2018 paper"""

    def __init__(self, in_channels=6, num_classes=10, device="cuda", tensorboard=False):
        super(MozafariMNIST2018, self).__init__(num_classes=num_classes, device=device, tensorboard=tensorboard)

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
        self.block3 = nn.ModuleDict(
            {
                "conv": snn.Convolution(
                    in_channels=250, out_channels=200, kernel_size=5, weight_mean=0.8, weight_std=0.05
                ),
                "stdp": None,
                "anti_stdp": None,
            }
        )

        self.block1_params = {
            "threshold": 15,
            # 'threshold': 12,
            "n_winners": 5,
            "inhibition_radius": 3,
        }
        self.block2_params = {
            "threshold": 10,
            # 'threshold': 8,
            "n_winners": 8,
            "inhibition_radius": 1,
        }

        self.block1["stdp"] = snn.STDP(
            layer=self.block1["conv"], learning_rate=(0.004, -0.003), use_stabilizer=True, lower_bound=0, upper_bound=1
        )
        self.block2["stdp"] = snn.STDP(
            layer=self.block2["conv"], learning_rate=(0.004, -0.003), use_stabilizer=True, lower_bound=0, upper_bound=1
        )
        self.block3["stdp"] = snn.STDP(
            layer=self.block3["conv"],
            learning_rate=(0.004, -0.003),
            use_stabilizer=False,
            lower_bound=0.2,
            upper_bound=0.8,
        )
        self.block3["anti_stdp"] = snn.STDP(
            layer=self.block3["conv"],
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
        # input += torch.bernoulli(torch.full_like(input, 0.2))
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
            pot = self.block3["conv"](spk_in)
            spk = sf.fire(pot)
            winners = sf.get_k_winners(pot, 1, 0, spk)
            self.update_ctx(spk_in, pot, spk, winners)
            return self.get_output(winners)
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

        pot = self.block2["conv"](sf.pad(sf.pooling(spk, 2, 2), (1, 1, 1, 1)))
        spk, pot = sf.fire(pot, self.block2_params["threshold"], True)
        if max_layer == 2:
            return spk, pot

        pot = self.block3["conv"](sf.pad(sf.pooling(spk, 3, 3), (2, 2, 2, 2)))
        spk = sf.fire(pot)
        winners = sf.get_k_winners(pot, 1, 0, spk)
        return self.get_output(winners)

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

    def get_output(self, winners):
        """Get the output of the network"""
        if len(winners) != 0:
            # self.file.write(str(self.decision_map[winners[0][0]]) + "\n")
            return torch.tensor(self.decision_map[winners[0][0]], device=self.device)
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

    def update_learning_rates(self, stdp_ap, stdp_an, anti_stdp_ap, anti_stdp_an, layer_idx):
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
        self.block3["stdp"].update_all_learning_rate(stdp_ap, stdp_an)
        self.block3["anti_stdp"].update_all_learning_rate(anti_stdp_an, anti_stdp_ap)

    def reward(self):
        """Reward the network"""
        self.block3["stdp"](
            self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"]
        )

    def punish(self):
        """Punish the network"""
        self.block3["anti_stdp"](
            self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"]
        )
