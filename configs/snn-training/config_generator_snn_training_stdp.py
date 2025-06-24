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
        f"{base_config['dataset_name']}_{base_config['model_name']}_{base_config['epochs'][0]}_"
        f"_{base_config['propagator_name']}"
        f"_{base_config['seed']}"
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
    Main function to generate configuration files for SNN training.

    This function creates directories for storing configurations, defines a base configuration,
    iterates over parameter grids to generate multiple configurations, and saves them as YAML files.
    """
    # Define directories for saving configurations
    config_dir = "configs/snn-training"
    cluster_dir = os.path.join(config_dir, "cluster")
    local_dir = os.path.join(config_dir, "local")
    testing_dir = os.path.join(config_dir, "testing")
    create_directory(cluster_dir)
    create_directory(local_dir)
    create_directory(testing_dir)

    # Define the base configuration with default parameters
    base_config = {
        "savepath": "/mnt/output/",
        "wandb_key": "<wandb-key>",
        "disable_wandb": False,
        "epochs": [50, 50, 50, 50, 50],
        "propagator_name": "rstdp",
    }

    DATASET_NAMES = ["cifar10"]
    MODEL_NAMES = {
        "cifar10": ["deepersnn", "lifresnetlike"],
    }
    SEEDS = [7240, 5110, 5628]
    EPOCHS = [
        [50, 50, 50, 50, 50],
        [40, 40, 40, 40, 40],
        [30, 30, 30, 30, 30],
        [20, 20, 20, 20, 20],
        [15, 15, 15, 15, 15],
        [10, 10, 10, 10, 10],
        [5, 5, 5, 5, 5],
        [1, 1, 1, 1, 1],
    ]

    counter = 0
    saved_datasets = set()
    saved_local = set()

    for dataset_name in DATASET_NAMES:
        model_names = MODEL_NAMES["cifar10"]

        for epochs in EPOCHS:
            for model_name in model_names:
                for seed in SEEDS:
                    if dataset_name == "mnist":
                        batch_size = 128
                        datapath = f"/mnt/data/{dataset_name}"
                        n_channels = 1
                    elif dataset_name in ["cifar10", "cifar100"]:
                        batch_size = 128
                        datapath = f"/mnt/data/{dataset_name}"
                        n_channels = 3
                    else:
                        batch_size = 16
                        datapath = f"/mnt/data/{dataset_name}"
                        n_channels = 3

                    config = {
                        "savepath": base_config["savepath"],
                        "data_path": datapath,
                        "dataset_name": dataset_name,
                        "batch_size": batch_size,
                        "n_channels": n_channels,
                        "epochs": epochs,
                        "model_name": model_name,
                        "seed": seed,
                        "wandb_key": base_config["wandb_key"],
                        "disable_wandb": base_config["disable_wandb"],
                        "propagator_name": base_config["propagator_name"],
                        "wandb_project_name": f"rstdp-{dataset_name}-{model_name}",
                    }

                    config_name = generate_config_name(config)

                    if dataset_name not in saved_datasets:
                        save_config(config, testing_dir, config_name)
                        saved_datasets.add(dataset_name)
                    local_key = (dataset_name, model_name)
                    if local_key not in saved_local:
                        save_config(config, local_dir, config_name)
                        saved_local.add(local_key)
                    save_config(config, cluster_dir, config_name)
                    counter += 1

    print(f"Created {counter} files!")


if __name__ == "__main__":
    main()
