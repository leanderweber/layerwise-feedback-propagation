import os

import yaml


def create_directory(path):
    """
    Ensure the directory exists. If it doesn't, create it.

    Args:
        path (str): The directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def generate_config_name(base_config):
    """
    Generate a unique configuration name based on the parameters in the base configuration.

    Args:
        base_config (dict): The base configuration dictionary containing parameters.

    Returns:
        str: A unique configuration name.
    """
    return (
        f"{base_config['dataset_name']}_{base_config['model_name']}"
        f"_{base_config['optimizer_name']}_{base_config['lr']}"
        f"_{base_config['propagator_name']}_{base_config['weight_decay']}"
        f"_{base_config['scheduler_name']}_{base_config['seed']}"
    )


def save_config(config, directory, filename):
    """
    Save the configuration as a YAML file in the specified directory.

    Args:
        config (dict): The configuration dictionary to save.
        directory (str): The directory where the file will be saved.
        filename (str): The name of the file (without extension).
    """
    filepath = os.path.join(directory, f"{filename}.yaml")
    with open(filepath, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def main():
    """
    Main function to generate configuration files for ViT training.

    This function creates directories for storing configurations, defines a base configuration,
    iterates over parameter grids to generate multiple configurations, and saves them as YAML files.
    The first configuration for each dataset in `DATASET_NAMES` is saved to both `testing_dir` and `cluster_dir`,
    while the rest are saved only to `cluster_dir`.
    """
    # Define directories for saving configurations
    config_dir = "configs/vit-training"
    cluster_dir = os.path.join(config_dir, "cluster")
    testing_dir = os.path.join(config_dir, "testing")
    create_directory(cluster_dir)
    create_directory(testing_dir)

    # Define the base configuration with default parameters
    base_config = {
        "savepath": "/mnt/output/",
        "batch_size": 32,
        "n_channels": 3,
        "n_outputs": 3,
        "momentum": 0.9,
        "clip_update_threshold": 2.0,
        "reward_name": "softmaxlossreward",
        "reward_kwargs": {},
        "loss_name": "ce-loss",
        "activation": "relu",
        "epochs": 100,
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

    # Define parameter grids for generating configurations
    DATASET_NAMES = ["beans", "oxford-flowers"]
    MODEL_NAMES = ["vit"]
    LRS = [2e-3, 2e-4, 2e-5, 1e-5, 2e-6]
    PROPAGATOR_NAMES = ["lfp-epsilon", "vanilla-gradient"]
    CLIP_UPDATES = [False]
    WEIGHT_DECAYS = [0.0, 0.0001]
    SCHEDULER_NAMES = ["onecyclelr"]
    OPTIMIZER_NAMES = ["adam", "adamw", "sgd"]
    SEEDS = [7240, 5110, 5628]

    counter = 0  # Counter to track the number of configurations generated
    saved_datasets = set()  # Track datasets that have had a config saved to testing_dir

    # Iterate over all combinations of parameter grids
    for dataset_name in DATASET_NAMES:
        for model_name in MODEL_NAMES:
            for optimizer_name in OPTIMIZER_NAMES:
                for lr in LRS:
                    for propagator_name in PROPAGATOR_NAMES:
                        for clip_updates in CLIP_UPDATES:
                            # Adjust weight_decay based on optimizer_name
                            weight_decay_values = WEIGHT_DECAYS if optimizer_name == "sgd" else [0.0]
                            for weight_decay in weight_decay_values:
                                for scheduler_name in SCHEDULER_NAMES:
                                    for seed in SEEDS:
                                        # Update base_config with current parameters
                                        datapath = f"/mnt/data/{dataset_name}"
                                        base_config.update(
                                            {
                                                "lr": lr,
                                                "optimizer_name": optimizer_name,
                                                "propagator_name": propagator_name,
                                                "seed": seed,
                                                "data_path": datapath,
                                                "dataset_name": dataset_name,
                                                "model_name": model_name,
                                                "wandb_project_name": f"lfp-{dataset_name}-{model_name}",
                                                "clip_updates": clip_updates,
                                                "weight_decay": weight_decay,
                                                "scheduler_name": scheduler_name,
                                            }
                                        )

                                        # Generate config name and save the file
                                        config_name = generate_config_name(base_config)
                                        if dataset_name not in saved_datasets:
                                            # Save the first configuration for this dataset to both testing_dir
                                            # and cluster_dir
                                            save_config(base_config, testing_dir, config_name)
                                            saved_datasets.add(dataset_name)
                                        save_config(base_config, cluster_dir, config_name)
                                        counter += 1

    print(f"Created {counter} files!")  # Print the total number of configurations created


if __name__ == "__main__":
    main()
