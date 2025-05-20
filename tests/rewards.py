import unittest
import torch

from lfprop.rewards import rewards
from lfprop.rewards import reward_functions

from .helpers import t

customCELoss_obj = rewards.CustomCrossEntropyLoss()
sigBCELoss_obj = rewards.SigmoidBCELossWrapper()

class TestRewards(unittest.TestCase):
    
    def test_customcrossentropyloss(self):
        # test tensor_backward_hook TODO

        # test forward
        # inp needs to have at least two dimensions
        inp = t([[3., 3., 3.], [3., 2., 3.]])
        target = t([2, 1])
        softmax = torch.nn.functional.softmax(inp, dim=1)
        log_softmax = torch.nn.functional.log_softmax(inp, dim=1)
        regularized_log_softmax = log_softmax
        regularized_log_softmax = torch.where(softmax > customCELoss_obj.higher_bound, 0.0, regularized_log_softmax)
        regularized_log_softmax = torch.where(
            softmax < customCELoss_obj.lower_bound, -99999.99999, regularized_log_softmax
        )
        nll = torch.nn.functional.nll_loss(regularized_log_softmax, target)
        res = customCELoss_obj.forward(inp, target)
        self.assertTrue(torch.all(inp.eq(customCELoss_obj.stored_input)))
        self.assertTrue(torch.all(target.eq(customCELoss_obj.stored_target)))
        self.assertTrue(torch.all(softmax.eq(customCELoss_obj.stored_softmax)))
        self.assertTrue(torch.all(res.eq(nll)))
    
    def test_sigmoidbcelosswrapper(self):
        # test forward
        return
    
    def test_maximizesingleneuron(self):
        # test forward
        return

    def test_minimizesingleneuron(self):
        # test forward
        return
    
    def test_get_reward(self):
        # test get_reward
        return
    
class TestRewardFunctions(unittest.TestCase):
    
    def test_maximizesingleneuron(self):

        return
    
    def test_minimizesingleneuron(self):

        return
    
    def test_maximizesingleneuron(self):

        return
    
    