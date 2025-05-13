import unittest

import numpy as np
import torch
import torch.nn as tnn
import torchvision.datasets as tvisiondata
import torchvision.transforms as T

from lfprop.model import activations
from lfprop.model import custom_resnet
from lfprop.model import spiking_networks

step_object = activations.Step()
sum_object = custom_resnet.Sum()
#cbb = custom_resnet.CustomBasicBlock()

def t(x) -> torch.Tensor:
    try:
        return torch.tensor(x)
    except:
        return torch.from_numpy(x)

class TestModel(unittest.TestCase):

    def test_step(self):
        
        self.assertEqual(step_object.forward(t(1)), 1)
        self.assertEqual(step_object.forward(t(-1)), 0)
        self.assertEqual(step_object.forward(t(0)), 0)
        self.assertEqual(step_object.forward(t(1e-9)), 1)
        self.assertEqual(step_object.forward(t(-1e-9)), 0)
        '''
        self.assertEqual(step_object.forward(t(np.zeros((3, 3)))), np.zeros((3, 3)))
        self.assertEqual(step_object.forward(t(np.ones((2, 2)))), np.ones((2, 2)))
        '''
        #TODO does this also look at arrays/multi-element tensors?

    def test_sum(self):
        self.assertEqual(sum_object.forward(t(1), t(1)), 2)
        self.assertEqual(sum_object.forward(t(0), t(0)), 0)
        self.assertEqual(sum_object.forward(t(-1), t(-1)), -2)

    def test_custombasicsblock(self):
        return
    
    def test_custombottleneck(self):
        return
    
    def test_customresnet(self):
        return
    
    def test_customleaky(self):
        return
    
    def test_custommaxpool2d(self):
        return
    
    def test_lifmlp(self):
        return
    
    def test_smalllifmlp(self):
        return
    
    def test_lifcnn(self):
        return
    
    def test_gradlifmlp(self):
        return
    
    def test_gradlifcnn(self):
        return