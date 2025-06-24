import torch
import torchvision
import transformers

from lfprop.model import activations, custom_resnet, spiking_networks

from . import model_definitions

TORCHMODEL_MAP = {
    "vgg16": torchvision.models.vgg16,
    "resnet18": custom_resnet.custom_resnet18,
    "resnet34": custom_resnet.custom_resnet34,
    "customresnet18": custom_resnet.custom_resnet18,
    "vgg16bn": torchvision.models.vgg16_bn,
}

HUGGINGFACE_MODEL_MAP = {
    "vit": transformers.ViTForImageClassification,
}

SPIKING_MODEL_MAP = {
    "lifmlp": spiking_networks.LifMLP,
    "smalllifmlp": spiking_networks.SmallLifMLP,
    "lifcnn": spiking_networks.LifCNN,
    "deepersnn": spiking_networks.DeeperSNN,
    "lifresnetlike": spiking_networks.ResNet,
}

MODEL_MAP = {
    "lenet": model_definitions.LeNet,
    "cifar-vgglike": model_definitions.CifarVGGLike,
    "cifar-vgglike-bn": model_definitions.CifarVGGLikeBN,
    "dense-only": model_definitions.DenseOnly,
    "toydata-dense": model_definitions.ToyDataDense,
}

ACTIVATION_MAP = {
    "relu": torch.nn.ReLU,
    "sigmoid": torch.nn.Sigmoid,
    "silu": torch.nn.SiLU,
    "leakyrelu": torch.nn.LeakyReLU,
    "tanh": torch.nn.Tanh,
    "elu": torch.nn.ELU,
    "step": activations.Step,
    "negtanh": activations.NegTanh,
    "negrelu": activations.NegReLU,
    "neginnerrelu": activations.NegInnerReLU,
    "negstep": activations.NegStep,
}

EXCLUDED_MODULE_TYPES = [
    torchvision.models.VGG,
    model_definitions.LeNet,
    model_definitions.CifarVGGLike,
    model_definitions.DenseOnly,
]


def normal_pos(tensor, *args, **kwargs):
    """
    Initializes the tensor with only positive values (absolute value).
    Args:
        tensor (torch.Tensor): The tensor to initialize.
    """
    tensor.data = tensor.data.abs()


def normal_neg(tensor, *args, **kwargs):
    """
    Initializes the tensor with only negative values (negative absolute value).
    Args:
        tensor (torch.Tensor): The tensor to initialize.
    """
    tensor.data = -tensor.data.abs()


INIT_FUNCS = {"positive": normal_pos, "negative": normal_neg}


def replace_torchvision_last_layer(model, n_outputs):
    """
    Replaces the last classification layer of a torchvision model with a new Linear layer.
    Args:
        model (torch.nn.Module): The model whose last layer will be replaced.
        n_outputs (int): Number of output classes.
    """
    if isinstance(model, torchvision.models.VGG) or isinstance(model, torchvision.models.efficientnet.EfficientNet):
        # For VGG and EfficientNet, replace the last layer in the classifier
        classifier = model.classifier
        modules = [m for m in classifier.modules() if not isinstance(m, torch.nn.Sequential)]
        modules[-1] = torch.nn.Linear(modules[-1].in_features, n_outputs)
        model.classifier = torch.nn.Sequential(*modules)
    elif isinstance(model, torchvision.models.ResNet) or isinstance(model, torchvision.models.Inception3):
        # For ResNet and Inception, replace the fc layer
        classifier = model.fc
        model.fc = torch.nn.Linear(classifier.in_features, n_outputs)
    else:
        # For other models, try to replace the last layer in classifier
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, n_outputs)


def replace_torchvision_activations(model, activation):
    """
    Replaces all ReLU activations in a torchvision model with the specified activation.
    Args:
        model (torch.nn.Module): The model whose activations will be replaced.
        activation (callable): The activation function to use.
    """
    if isinstance(model, torchvision.models.VGG):
        # Replace ReLU in VGG classifier
        for module in model.modules():
            if isinstance(module, torch.nn.Sequential):
                seq_modules = [m for m in module.modules() if not isinstance(m, torch.nn.Sequential)]
                for i, mod in enumerate(seq_modules):
                    if isinstance(mod, torch.nn.ReLU):
                        module[i] = activation()
        # Check if any ReLU remains
        if len([m for m in model.modules() if isinstance(m, torch.nn.ReLU)]) > 0:
            raise ValueError("There are still ReLUs left after replacement!")
    elif isinstance(model, torchvision.models.ResNet):
        # Replace relu attribute in ResNet blocks
        for module in model.modules():
            if hasattr(module, "relu"):
                module.relu = activation()
        if len([m for m in model.modules() if isinstance(m, torch.nn.ReLU)]) > 0:
            raise ValueError("There are still ReLUs left after replacement!")
    else:
        print("Model type not supported, not changing activations")


def init_model_weights(model, init_func):
    """
    Initializes the weights of Conv2d, Conv3d, and Linear layers using the provided function.
    Args:
        model (torch.nn.Module): The model to initialize.
        init_func (callable): The initialization function.
    """

    def param_init(m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Linear):
            init_func(m.weight)
            if m.bias is not None:
                init_func(m.bias)

    model.apply(param_init)


def forward_fn_default(batch, model, device, lfp_step=True, **kwargs):
    """
    Standard forward pass for Torchvision models.
    Args:
        batch (tuple): Tuple of (inputs, labels).
        model (torch.nn.Module): The model to use.
        device (torch.device): Device to run on.
        lfp_step (bool): Whether to require gradients for inputs.
        **kwargs: Additional arguments.
    Returns:
        inputs (torch.Tensor): Input tensor.
        labels (torch.Tensor): Label tensor.
        outputs (torch.Tensor): Model outputs.
    """
    inputs, labels = batch
    inputs = inputs.to(device)
    labels = torch.tensor(labels).to(device)

    if lfp_step:
        inputs = inputs.detach().requires_grad_(True)

    outputs = model(inputs)

    return inputs, labels, outputs


def forward_fn_vit(batch, model, device, lfp_step=True, **kwargs):
    """
    Forward function for Huggingface ViT implementation.
    Args:
        batch (dict): Batch containing 'pixel_values' and 'labels'.
        model (torch.nn.Module): The model to use.
        device (torch.device): Device to run on.
        lfp_step (bool): Whether to require gradients for inputs.
        **kwargs: Additional arguments.
    Returns:
        inputs (torch.Tensor): Input tensor.
        labels (torch.Tensor): Label tensor.
        outputs (torch.Tensor): Model outputs.
    """
    labels = batch.get("labels", None).to(device)
    inputs = batch.get("pixel_values", None).to(device)

    if lfp_step:
        inputs = inputs.detach().requires_grad_(True)

    outputs = model(pixel_values=inputs)["logits"]

    return inputs, labels, outputs


def forward_fn_spiking(batch, model, device, lfp_step=True, n_steps=15, is_huggingface_data=False, **kwargs):
    """
    Forward function for spiking neural networks.
    Args:
        batch (tuple or dict): Batch of data.
        model (torch.nn.Module): The model to use.
        device (torch.device): Device to run on.
        lfp_step (bool): Whether to require gradients for inputs.
        n_steps (int): Number of time steps for spiking models.
        is_huggingface_data (bool): Whether the batch is in Huggingface format.
        **kwargs: Additional arguments.
    Returns:
        inputs (torch.Tensor): Input tensor.
        labels (torch.Tensor): Label tensor.
        spikes (torch.Tensor): Recorded spikes over time.
    """
    if not is_huggingface_data:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = torch.tensor(labels).to(device)
    else:
        inputs = batch.get("pixel_values", None).to(device)
        labels = batch.get("labels", None).to(device)

    if lfp_step:
        inputs = inputs.detach().requires_grad_(True)

    spk_rec = []
    for step in range(n_steps):
        outputs = model(inputs)
        spk_out, _ = outputs
        spk_rec.append(spk_out)
    spikes = torch.stack(spk_rec, dim=0)

    return inputs, labels, spikes


def get_model(model_name, device, **kwargs):
    """
    Gets the correct model and initializes it with the given parameters.
    Also sets some identifying attributes for the model.
    Args:
        model_name (str): Name of the model to be used.
        device (torch.device): Device to which the model should be moved.
        **kwargs: Additional keyword arguments for model initialization.
    Returns:
        model (torch.nn.Module): The initialized model.
    """
    replace_last_layer = kwargs.get("replace_last_layer", True)

    # Check if model_name is supported
    if (
        model_name not in MODEL_MAP
        and model_name not in TORCHMODEL_MAP
        and model_name not in HUGGINGFACE_MODEL_MAP
        and model_name not in SPIKING_MODEL_MAP
    ):
        raise ValueError("Model '{}' is not supported.".format(model_name))

    # Build model
    activation = kwargs.get("activation", "relu")
    if model_name in MODEL_MAP:
        # Custom model definitions
        model = MODEL_MAP[model_name](
            n_channels=kwargs.get("n_channels", 3),
            n_outputs=kwargs.get("n_outputs", 1000),
            activation=ACTIVATION_MAP[activation],
        )
        model.forward_fn = forward_fn_default
        model.is_huggingface = False

    elif model_name in TORCHMODEL_MAP:
        # Torchvision models
        model = TORCHMODEL_MAP[model_name](pretrained=kwargs.get("pretrained_model", True))
        if replace_last_layer:
            replace_torchvision_last_layer(model, kwargs.get("n_outputs", 1000))
        if activation != "relu":
            replace_torchvision_activations(model, ACTIVATION_MAP[activation])

        model.forward_fn = forward_fn_default
        model.is_huggingface = False

    elif model_name in HUGGINGFACE_MODEL_MAP:
        # Huggingface models (e.g., ViT)
        class_labels = kwargs.get("class_labels", [])
        model = HUGGINGFACE_MODEL_MAP[model_name].from_pretrained(
            kwargs.get("model_checkpoint", None),
            num_labels=len(class_labels),
            id2label={str(i): c for i, c in enumerate(class_labels)},
            label2id={c: str(i) for i, c in enumerate(class_labels)},
            _attn_implementation="eager",
        )
        model.forward_fn = forward_fn_vit
        model.is_huggingface = True

    elif model_name in SPIKING_MODEL_MAP:
        # Spiking neural network models
        model = SPIKING_MODEL_MAP[model_name](
            n_channels=kwargs.get("n_channels", 3),
            n_outputs=kwargs.get("n_outputs", 1000),
            beta=kwargs.get("beta", 0.9),
            reset_mechanism=kwargs.get("reset_mechanism", "subtract"),
            surrogate_disable=kwargs.get("surrogate_disable", False),
            spike_grad=kwargs.get("spike_grad", "step"),
            apply_noise=kwargs.get("apply_noise", False),
            noise_size=kwargs.get("noise_size", 1e-6),
            n_linear_inputs=kwargs.get("n_linear_inputs", 1),
        )
        model.forward_fn = forward_fn_spiking
        model.is_huggingface = False

    # Optionally initialize model weights with a custom function
    if "init_func" in kwargs.keys() and kwargs.get("init_func") != "default":
        init_func = INIT_FUNCS[kwargs.get("init_func")]
        init_model_weights(model, init_func)

    # Return model on correct device
    return model.to(device)


def list_layers(model):
    """
    Lists all layers (modules) in the model, excluding Sequential and certain custom types.
    Args:
        model (torch.nn.Module): The model to inspect.
    Returns:
        layers (list): List of module layers.
    """
    # Exclude specific types of modules
    layers = [module for module in model.modules() if type(module) not in [torch.nn.Sequential] + EXCLUDED_MODULE_TYPES]
    return layers
