import os

import yaml

config_dir = "configs/activationfuncs"

# os.makedirs(f"{config_dir}/local", exist_ok=True)
os.makedirs(f"{config_dir}/cluster", exist_ok=True)

base_config = {
    "savepath": "/mnt/output",
    "batch_size": 32,
    "n_channels": 3,
    "momentum": 0.9,
    "clip_update_threshold": 2.0,
    "reward_name": "softmaxlossreward",
    "reward_kwargs": {},
    "loss_name": "ce-loss",
    "transfer_epochs": 100,
    "batch_log": False,
    "wandb_key": "<wandb-key>",
    "disable_wandb": False,
    "verbose": False,
    "param_sparsity_log": False,
    "default_model_checkpoint": "google/vit-base-patch16-224-in21k",
    "snn_n_steps": 1,
    "snn_beta": 0.9,
    "snn_reset_mechanism": "step",
    "snn_surrogate_disable": False,
    "snn_spike_grad": "sigmoid",
    "snn_apply_noise": False,
    "snn_noise_size": 0.0,
}

DATASET_NAMES = ["isic", "cub", "food11"]
MODEL_NAMES = ["vgg16", "resnet18"]
ACTIVATIONS = ["relu", "tanh", "elu", "silu", "step", "sigmoid"]
PROPAGATOR_NAMES = ["vanilla-gradient", "lfp-epsilon"]
CLIP_UPDATES = [False]
WEIGHT_DECAYS = [0.0001]
SCHEDULER_NAMES = ["onecyclelr"]
SEEDS = [7240, 5110, 5628]

for dataset_name in DATASET_NAMES:
    for model_name in MODEL_NAMES:
        for activation in ACTIVATIONS:
            for propagator_name in PROPAGATOR_NAMES:
                for clip_updates in CLIP_UPDATES:
                    for weight_decay in WEIGHT_DECAYS:
                        for scheduler_name in SCHEDULER_NAMES:
                            for seed in SEEDS:
                                if model_name == "vgg16" and dataset_name == "food11":
                                    lr = 0.0001
                                else:
                                    lr = 0.001
                                base_config["lr"] = lr
                                base_config["propagator_name"] = propagator_name
                                base_config["seed"] = seed

                                base_config["data_path"] = f"/mnt/data/{dataset_name}"
                                base_config["dataset_name"] = dataset_name
                                base_config["model_name"] = model_name
                                base_config["activation"] = activation
                                base_config["wandb_project_name"] = (
                                    f"activationfuncs-{dataset_name}-transfer-{model_name}"
                                )

                                base_config["clip_updates"] = clip_updates
                                base_config["weight_decay"] = weight_decay
                                base_config["scheduler_name"] = scheduler_name

                                config_name = f"{base_config['dataset_name']}_{base_config['model_name']}"
                                config_name += f"_{base_config['activation']}_{base_config['lr']}"
                                config_name += f"_{base_config['propagator_name']}"
                                config_name += f"_{base_config['clip_updates']}_{base_config['weight_decay']}"
                                config_name += f"_{base_config['scheduler_name']}_{base_config['seed']}"

                                with open(
                                    f"{config_dir}/cluster/{config_name}_transfermodel.yaml",
                                    "w",
                                ) as outfile:
                                    yaml.dump(
                                        base_config,
                                        outfile,
                                        default_flow_style=False,
                                    )
