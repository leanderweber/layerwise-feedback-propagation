from collections import OrderedDict
import unittest

import numpy as np
import torch
import torch.nn as tnn

from zennit import types as ztypes
from zennit.rules import ZBox, Epsilon, ZPlus, Norm, Pass

from lfprop.propagation import propagator_zennit as prop_z
from .helpers import t

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

hook_object = prop_z.LFPHook(True, input_modifiers=[lambda inp: inp])
epsilon_object = prop_z.LFPEpsilon(True)

# to test composites
layer_map = [
    (ztypes.Activation, Pass()),  # ignore activations
    (ztypes.AvgPool, Norm()),  # normalize relevance for any AvgPool
    (ztypes.Convolution, ZPlus()),  # any convolutional layer
    (ztypes.Linear, Epsilon(epsilon=1e-6))  # this is the dense Linear, not any
]
name_map = [
    (['linear0'], ZBox(-3., 3.)),
    (['conv0'], ZPlus()),
    (['linear1', 'linear2'], Epsilon(epsilon=1e-6)),
]
first_map = [
    (ztypes.Linear, ZBox(-3., 3.))
]
composite_special_first_layer = prop_z.SpecialFirstNamedLayerMapComposite(
    layer_map=layer_map, name_map=name_map, first_map=first_map
)
epsiloncomposite = prop_z.LFPEpsilonComposite()


class TestZennitPropagation(unittest.TestCase):

    def test_collect_leaves(self):
        # tests if childless module is selected as own leaf
        childless = tnn.Linear(4, 2)
        leaves = prop_z.collect_leaves(childless)
        self.assertIn(childless, [x for x in leaves])

        # TODO write more test cases with actual leaves
        # layers are children and are built recursively
    
    def test_save_input_hook(self):
        # tests 'saved_input' attribute of module
        model = tnn.Linear(4, 2)
        inp = t(np.zeros(4)).uniform_()[None, ...]
        prop_z.save_input_hook(model, inp)
        self.assertTrue(torch.all(inp.eq(model.saved_input)))

    def test_mod_param(self):
        # tests mod_params and corresponding context
        model = tnn.Linear(4, 2)
        with prop_z.mod_param_storage_context(model) as model:
            # stored params must be an attribute within context
            self.assertTrue(hasattr(model, "stored_modparams"))

            # store positive params
            params = model.weight
            model.stored_modparams = params[params > 0]
            self.assertNotIn(-1, torch.sign(torch.flatten(model.stored_modparams)))

        # fstored_params must be unavailable outside of context
        self.assertFalse(hasattr(model, "stored_modparams"))
    
    def test_specialfirstnamedlayercomposite(self):
        # tests mapping function of class SpecialFirstNameLayerMapComposite
        empty_ctx = {
        }
        #TODO anything in between?
        full_ctx = {
            composite_special_first_layer.composites[0]: name_map,
            composite_special_first_layer.composites[1]: layer_map
        }
        #TODO anything beyond?
        
        model = tnn.Sequential(OrderedDict([
            ('linear0', tnn.Linear(5000, 1024)),
            ('relu0', tnn.ReLU()),
            ('conv0', tnn.Conv2d(8, 16, 3, padding=1)),
            ('relu1', tnn.ReLU()),
            ('flatten', tnn.Flatten()),
            ('linear1', tnn.Linear(16 * 32 * 32, 1024)),
            ('relu2', tnn.ReLU()),
            ('linear2', tnn.Linear(1024, 10)),
        ]))
        # test if correct hooks are returned with empty ctx
        self.assertIsInstance(composite_special_first_layer.mapping(empty_ctx, 'linear0', model), ZBox)
        self.assertIsInstance(composite_special_first_layer.mapping(empty_ctx, 'linear1', model), Epsilon)

        # test if correct hooks are returned with ctx equivalent to composite
        self.assertIsInstance(composite_special_first_layer.mapping(full_ctx, 'conv0', model), ZPlus)
        self.assertIsInstance(composite_special_first_layer.mapping(full_ctx, 'linear2', model), Epsilon)

        # test if hooks are returned for non-existent layers
        self.assertIsNone(composite_special_first_layer.mapping(empty_ctx, 'gibberish', model))
        self.assertIsNone(composite_special_first_layer.mapping(full_ctx, 'conv', model))
    
    def test_lfphook(self):
        # tests forward and backward pass, copy and static methods of LFPHook
        model = tnn.Sequential(OrderedDict([
            ('linear0', tnn.Linear(500, 1024)),
            ('relu0', tnn.ReLU()),
            ('conv0', tnn.Conv2d(3, 3, 3, padding=1)),
            ('relu1', tnn.ReLU()),
            ('flatten', tnn.Flatten()),
            ('linear1', tnn.Linear(100 * 1024, 1024)),
            ('relu2', tnn.ReLU()),
            ('linear2', tnn.Linear(1024, 10)),
        ]))

        inp = torch.Tensor.uniform_(torch.zeros(3, 100, 500))
        output = model(inp)

        # test if input hook is correctly stored
        hook_object.forward(model, [inp], output) # inp is list to represent hook
        self.assertTrue(torch.all(inp.eq(hook_object.stored_tensors["input"][0])))

        model.stored_modparams = {'input': inp} # TODO this should work with context, but does not
        target = torch.ones(10)
        target[9] = 1
        #print(hook_object.backward(model, [inp], target)[0])
        #print(torch.sum(hook_object.backward(model, [inp], target)[0], dim=None)) # TODO what does reducer do?

        # TODO test copy?
    
    def test_lfpepsilon(self):
        # tests forward and backward pass, copy and static methods of LFPEpsilon
        model = tnn.Sequential(OrderedDict([
            ('linear0', tnn.Linear(500, 1024)),
            ('relu0', tnn.ReLU()),
            ('conv0', tnn.Conv2d(3, 3, 3, padding=1)),
            ('relu1', tnn.ReLU()),
            ('flatten', tnn.Flatten()),
            ('linear1', tnn.Linear(100 * 1024, 1024)),
            ('relu2', tnn.ReLU()),
            ('linear2', tnn.Linear(1024, 10)),
        ]))

        inp = torch.Tensor.uniform_(torch.zeros(3, 100, 500))
        output = model(inp)
        epsilon_object.register(model)

        # test if input hook is correctly stored
        epsilon_object.forward(model, [inp], output) # inp is list to represent hook
        self.assertTrue(torch.all(inp.eq(epsilon_object.stored_tensors["input"][0])))

        model.stored_modparams = {'input': inp} # TODO this should work with context, but does not
        target = torch.zeros(10)
        target[5] = 1
        # TODO How do you calculate this result in an independent way?
        #print(epsilon_object.backward(model, [inp], target)[0]) 
        #print(torch.sum(epsilon_object.backward(model, [inp], target)[0], dim=None)) 
        
        return
    
    def test_lfpepsiloncomposite(self):
        # tests constructor of LFPEpsilon
        # see test_specialfirstname
        return

class TestLXTPropagation(unittest.TestCase):
    
    def test_save_input_hook(self):
        # tests 'saved_input' attribute of module
        model = tnn.Linear(4, 2)
        inp = t(np.zeros(4)).uniform_()[None, ...]
        prop_z.save_input_hook(model, inp)
        self.assertTrue(torch.all(inp.eq(model.saved_input)))

    def test_mod_param(self):
        # tests mod_params and corresponding context
        model = tnn.Linear(4, 2)
        with prop_z.mod_param_storage_context(model) as model:
            # stored params must be an attribute within context
            self.assertTrue(hasattr(model, "stored_modparams"))

            # store positive params
            params = model.weight
            model.stored_modparams = params[params > 0]
            self.assertNotIn(-1, torch.sign(torch.flatten(model.stored_modparams)))

        # fstored_params must be unavailable outside of context
        self.assertFalse(hasattr(model, "stored_modparams"))
    
    def test_specialfirstnamedlayercomposite(self):
        # tests mapping function of class SpecialFirstNameLayerMapComposite
        empty_ctx = {
        }
        #TODO anything in between?
        full_ctx = {
            composite_special_first_layer.composites[0]: name_map,
            composite_special_first_layer.composites[1]: layer_map
        }
        #TODO anything beyond?
        
        model = tnn.Sequential(OrderedDict([
            ('linear0', tnn.Linear(5000, 1024)),
            ('relu0', tnn.ReLU()),
            ('conv0', tnn.Conv2d(8, 16, 3, padding=1)),
            ('relu1', tnn.ReLU()),
            ('flatten', tnn.Flatten()),
            ('linear1', tnn.Linear(16 * 32 * 32, 1024)),
            ('relu2', tnn.ReLU()),
            ('linear2', tnn.Linear(1024, 10)),
        ]))
        # test if correct hooks are returned with empty ctx
        self.assertIsInstance(composite_special_first_layer.mapping(empty_ctx, 'linear0', model), ZBox)
        self.assertIsInstance(composite_special_first_layer.mapping(empty_ctx, 'linear1', model), Epsilon)

        # test if correct hooks are returned with ctx equivalent to composite
        self.assertIsInstance(composite_special_first_layer.mapping(full_ctx, 'conv0', model), ZPlus)
        self.assertIsInstance(composite_special_first_layer.mapping(full_ctx, 'linear2', model), Epsilon)

        # test if hooks are returned for non-existent layers
        self.assertIsNone(composite_special_first_layer.mapping(empty_ctx, 'gibberish', model))
        self.assertIsNone(composite_special_first_layer.mapping(full_ctx, 'conv', model))
    
    def test_lfphook(self):
        # tests forward and backward pass, copy and static methods of LFPHook
        model = tnn.Sequential(OrderedDict([
            ('linear0', tnn.Linear(500, 1024)),
            ('relu0', tnn.ReLU()),
            ('conv0', tnn.Conv2d(3, 3, 3, padding=1)),
            ('relu1', tnn.ReLU()),
            ('flatten', tnn.Flatten()),
            ('linear1', tnn.Linear(100 * 1024, 1024)),
            ('relu2', tnn.ReLU()),
            ('linear2', tnn.Linear(1024, 10)),
        ]))

        inp = torch.Tensor.uniform_(torch.zeros(3, 100, 500))
        output = model(inp)

        # test if input hook is correctly stored
        hook_object.forward(model, [inp], output) # inp is list to represent hook
        self.assertTrue(torch.all(inp.eq(hook_object.stored_tensors["input"][0])))

        model.stored_modparams = {'input': inp} # TODO this should work with context, but does not
        target = torch.ones(10)
        target[9] = 1
        #print(hook_object.backward(model, [inp], target)[0])
        #print(torch.sum(hook_object.backward(model, [inp], target)[0], dim=None)) # TODO what does reducer do?

        # TODO test copy?
    
    def test_lfpepsilon(self):
        # tests forward and backward pass, copy and static methods of LFPEpsilon
        model = tnn.Sequential(OrderedDict([
            ('linear0', tnn.Linear(500, 1024)),
            ('relu0', tnn.ReLU()),
            ('conv0', tnn.Conv2d(3, 3, 3, padding=1)),
            ('relu1', tnn.ReLU()),
            ('flatten', tnn.Flatten()),
            ('linear1', tnn.Linear(100 * 1024, 1024)),
            ('relu2', tnn.ReLU()),
            ('linear2', tnn.Linear(1024, 10)),
        ]))

        inp = torch.Tensor.uniform_(torch.zeros(3, 100, 500))
        output = model(inp)
        epsilon_object.register(model)

        # test if input hook is correctly stored
        epsilon_object.forward(model, [inp], output) # inp is list to represent hook
        self.assertTrue(torch.all(inp.eq(epsilon_object.stored_tensors["input"][0])))

        model.stored_modparams = {'input': inp} # TODO this should work with context, but does not
        target = torch.zeros(10)
        target[5] = 1
        # TODO How do you calculate this result in an independent way?
        #print(epsilon_object.backward(model, [inp], target)[0]) 
        #print(torch.sum(epsilon_object.backward(model, [inp], target)[0], dim=None)) 
        
        return
    
    def test_lfpepsiloncomposite(self):
        # tests constructor of LFPEpsilon
        # see test_specialfirstname
        return

# test class methods and attributes
if __name__ == '__main__':
    unittest.main()
