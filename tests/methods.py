import unittest

import torch
import torch.nn as tnn
import torchvision.datasets as tvisiondata
import torchvision.transforms as T

from lfprop.model.activations import Step
from lfprop.model.custom_resnet import Sum, CustomBasicBlock, CustomBottleneck, CustomResNet
from lfprop.model.spiking_networks import CustomLeaky, CustomMaxPool2d, LifMLP, SmallLifMLP, LifCNN, GradLifMLP, GradLifCNN

'''
# define preliminary setup
transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
training_data = tvisiondata.MNIST(
    root=savepath,
    transform=transform,
    download=True,
    train=True,
)

validation_data = tvisiondata.MNIST(
    root=savepath,
    transform=transform,
    download=True,
    train=False,
)

training_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=False)
'''

step_object = Step()
sum_object = Sum()

class TestMethods(unittest.TestCase):

    def test_step(self):
        
        self.assertEqual(step_object.forward(torch.tensor(1)), 1)
        self.assertEqual(step_object.forward(torch.tensor(-1)), 0)
        self.assertEqual(step_object.forward(torch.tensor(0)), 0)
        self.assertEqual(step_object.forward(torch.tensor(1e-9)), 1)
        self.assertEqual(step_object.forward(torch.tensor(-1e-9)), 0)

        #TODO does this also look at arrays/multi-element tensors?

    def test_sum(self):
        return
    
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

if __name__ == '__main__':
    unittest.main()