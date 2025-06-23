<div align="center">
  <img src="docs/src/lfp_logo_v8.png" width="400"/>
  <p>Gradient-free Neural Network Training based on Layer-wise Relevance Propagation (LRP)</p>
</div>

![Python version](https://img.shields.io/badge/python-3.11-blue.svg)
[![PyPI version](https://badge.fury.io/py/lfprop.svg)](https://badge.fury.io/py/lfprop)
![Github License](https://img.shields.io/badge/license-BSD_3-lightgrey.svg)
[![Code style: ruff](https://img.shields.io/badge/code%20style-Ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![LXT](https://img.shields.io/badge/LXT-purple.svg?labelColor)](https://github.com/rachtibat/LRP-eXplains-Transformers)
[![Zennit](https://img.shields.io/badge/Zennit-darkred.svg)](https://github.com/chr5tphr/zennit)

### :octopus: Flexibility
LFP is highly flexible w.r.t. the models and objective functions it can be used with, as it does not require differentiability.
Consequently, it can be applied in non-differentiable architectures (e.g., Spiking Neural Networks) without requiring further adaptations,
and naturally handles discrete objectives, such as feedback directly obtained from humans.

### :gear: Efficiency
LFP applies an implicit weight-scaling of updates and only propagates feedback through nonzero connections and activations. This leads to sparsity of updates and the final model, while not sacrificing performance or convergence speed meaningfully compared to gradient descent. The obtained models can be pruned more easily since they represent information more efficiently.

### :page_with_curl: Paper
For more details, refer to our [Paper](https://arxiv.org/abs/2308.12053).

If you use this package in your research, please cite
```bibtex
@article{weber2025efficient,
  title={Efficient and Flexible Neural Network Training through Layer-wise Feedback Propagation},
  author={Leander Weber and Jim Berend and Moritz Weckbecker and Alexander Binder and Thomas Wiegand and Wojciech Samek and Sebastian Lapuschkin},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=9oToxYVOSW},
}
```

### :scroll: License
This project is licensed under the BSD-3 Clause License, since LRP (which LFP is based on) is a patented technology that can only be used free of charge for personal and scientific purposes.

## :rocket: Getting Started


### <a name="installation"></a> :fire: Installation

#### Using PyPI (Recommended)

LFP is available from PyPI, and we recommend this installation if you simply want to use LFP or run any of the notebooks or experiments in this repository.

```shell
pip install lfprop
```

If you would like to check out the ```minimal example.ipynb``` notebook, first clone the repository, and then install the necessary dependencies:
```shell
git clone https://github.com/leanderweber/layerwise-feedback-propagation
cd layerwise-feedback-propagation
pip install lfprop[quickstart]
```

Similarly, if you would like to run the scripts and notebooks for reproducing the paper experiments, you can run
```shell
git clone https://github.com/leanderweber/layerwise-feedback-propagation
cd layerwise-feedback-propagation
pip install lfprop[full]
```
instead to install the full dependencies.

#### <a name="poetryinstall"></a> Using [Poetry](https://python-poetry.org/)

If you would like to contribute to the repository, or extend the code in some way, we recommend the installation via Poetry:

```shell
git clone https://github.com/leanderweber/layerwise-feedback-propagation
cd layerwise-feedback-propagation
poetry install
```

This requires ```poetry-core>=2.0.0,<3.0.0```.

### :brain: How it works

Our implementation of LFP is based the LRP-implementation of the [LXT](https://github.com/rachtibat/LRP-eXplains-Transformers) library which is based on [PyTorch](https://pytorch.org/) and modifies the backward pass to return relevances instead of gradients.
*Note: We deprecated the implementation based on [zennit](https://github.com/chr5tphr/zennit) in LFP version 1.0.0 due to LXT being much faster and extensible to more architectures. Some experiments in the paper were based on the zennit implementation (cf. Figure 9), however, and can be found under commit https://github.com/leanderweber/layerwise-feedback-propagation/commit/8dcf5131fc66aa20480917b4cfe08e54b2945aa8*

```lfprop``` extends LXT to return relevances not only w.r.t. activations, but also w.r.t. parameters. Similar to LXT, this requires registering a composite to modify the backward pass.

*Standard*
```python
from lfprop.propagation import propagator_lxt
propagation_composite = propagator_lxt.LFPEpsilonComposite()
```

Some architectures may use special layer types that require the definition of dedicated rules for LFP. Similarly, non-classification tasks may require redefinition of the default LFP rules. Several examples are already implemented under ```lfprop/propagation```:

*LIF SNNs*
```python
from lfprop.propagation import propagator_snn
propagation_composite = propagator_snn.LFPSNNEpsilonComposite()
```

*ViT*
```python
from lfprop.propagation import propagator_vit
propagation_composite = propagator_vit.LFPEpsilonComposite()
```

*Simple Regression*
```python
from lfprop.propagation import propagator_regression
propagation_composite = propagator_regression.LFPEpsilonRegressionComposite()
```

Instead of an initial relevance, LFP requires an initial reward at the output, to be decomposed throughout the model. We implement several reward functions, with a similar signature to ```torch.nn.Loss``` functions.

```python
from lfprop.rewards import reward_functions as rewards
reward_func = rewards.SoftmaxLossReward(device)
```

To apply the modified backward pass, the composite simply needs to be registered.

After the backward pass is finished, the computed LFP-feedback can then be accessed via the (newly added) ```.feedback``` attribute of each parameter.

The model can simply be optimized using any ```torch.nn.Optimizer```, by first overwriting the ```.grad``` attribute by the corresponding (negative) feedback.

This results in the following training step:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
optimizer.zero_grad()

with propagation_composite.context(model) as modified:
    inputs = inputs.detach().requires_grad_(True)
    outputs = modified(inputs)

    # Calculate reward
    reward = torch.from_numpy(
        reward_func(outputs, labels).detach().cpu().numpy()
    ).to(device)

    # Calculate LFP
    input_reward = torch.autograd.grad(
        (outputs,), (inputs,), grad_outputs=(reward,), retain_graph=False
    )[0]

    # Write LFP Values into .grad attributes.
    for name, param in model.named_parameters():
        param.grad = -param.feedback

    # Optimization step
    optimizer.step()
```

### :mag: Examples

A simple, full example of how to train a LeNet model on MNIST can be found under ```minimal_example.ipynb```. An example using SNNs can be found under ```minimal_example_spiking_nets.ipynb```.Note that to run these notebooks, you need to install the necessary dependencies using ```lfprop[quickstart]```, as described under [Installation](#installation).

### :test_tube: Reproducing Experiments

To reproduce experiments from the paper, you first need to install the necessary dependencies with ```lfprop[full]```, as described under [Installation](#installation).

Most toy data experiments can then be reproduced by simply running the corresponding notebooks under ```nbs/```. You can find the used hyperparameters for the notebooks within the first two cells.

For reproducing the experiments that require training on more complex data and models (LFP for Non-ReLU, Pruning experiments, SNN experiments, ViT training), the training script is implemented in ```run_experiment.py```.
Hyperparameters for these experiments can be generated using the scripts under ```configs/```.

For reproducing these experiments first run
```bash
# 1. generate the config files
python configs/<experimentname>/config_generator<somesuffix>.py
# 2. run training script
python run_experiment.py --config_file=configs/<experimentname>/cluster/<selected-config-name>
```

For the pruning experiments, you can then run the ```nbs/*eval-clusterresults-pruning.ipynb``` notebooks using the obtained models.

## :bell: Roadmap

This is a first release of LFP, which does not work with all types of data or models, but we are actively working on extending the package. You can check this Roadmap to get an overview over features planned to the future.

- [x] LFP for CNNs and Fully-Connected Models
- [x] LFP for SNNs
- [x] LFP for ViT
- [x] LFP for Classification Tasks
- [ ] LFP for LLMs and larger Transformer-based Architectures

## :pencil2: Contributing
Feel free to contribute to the code, experiment with different models and datasets, and raise any suggestions or encountered problems as [Issues](https://github.com/leanderweber/layerwise-feedback-propagation/issues) or create a [Pull Request](https://github.com/leanderweber/layerwise-feedback-propagation/pulls).

For contributing, we recommend the [Installation via Poetry](#poetryinstall).

Note that we use [Ruff](https://github.com/astral-sh/ruff) for formatting and linting.
