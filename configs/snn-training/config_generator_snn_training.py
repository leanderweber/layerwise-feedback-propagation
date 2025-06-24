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
        f"_beta{base_config['snn_beta']}"
        f"_reset{base_config['snn_reset_mechanism']}"
        f"_surdis{base_config['snn_surrogate_disable']}"
        f"_spkgrad{base_config['snn_spike_grad']}"
        f"_nsteps{base_config['snn_n_steps']}"
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
    The first configuration for each dataset in `DATASET_NAMES` is saved to both `testing_dir` and `cluster_dir`,
    while the rest are saved only to `cluster_dir`.
    Additionally, the first configuration for each (dataset_name, propagator_name) pair is saved to `local_dir`.
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
        "batch_size": 32,
        "momentum": 0.9,
        "clip_update_threshold": 2.0,
        "reward_name": "snnratecodedreward",
        "reward_kwargs": {},
        "loss_name": "snnratecodedloss",
        "activation": "relu",
        "epochs": 100,
        "clip_updates": False,
        "weight_decay": 0.0,
        "batch_log": False,
        "wandb_key": "<wandb-key>",
        "disable_wandb": False,
        "verbose": False,
        "param_sparsity_log": False,
        "default_model_checkpoint": "google/vit-base-patch16-224-in21k",
    }

    # Define parameter grids for generating configurations
    DATASET_NAMES = ["cifar10", "oxford-flowers"]
    # MODEL_NAMES = ["lifcnn", "lifvgg16"]
    LRS = [10e-5, 5e-4, 10e-4]
    PROPAGATOR_NAMES = ["lfp-epsilon", "vanilla-gradient"]
    SCHEDULER_NAMES = ["onecyclelr"]
    OPTIMIZER_NAMES = ["adam"]
    SEEDS = [7240, 5110, 5628]

    # New SNN parameter grids
    SNN_BETAS = [0.9]
    SNN_RESET_MECHANISMS = ["subtract"]
    SNN_SURROGATE_DISABLES = [False]
    SNN_SPIKE_GRADS = ["step", "atan"]
    SNN_N_STEPS = [25]
    SNN_APPLY_NOISE = [False]
    SNN_NOISE_SIZES = [1e-6, 1e-3, 1e-1]

    # Reduced parameter grids for oxford-flowers
    OXFORD_SNN_BETAS = [0.9]
    OXFORD_SNN_N_STEPS = [25]
    OXFORD_SNN_SPIKE_GRADS = ["step", "atan"]

    counter = 0  # Counter to track the number of configurations generated
    saved_datasets = set()  # Track datasets that have had a config saved to testing_dir
    saved_local = set()  # Track (dataset_name, propagator_name) pairs saved to local_dir

    # Iterate over all combinations of parameter grids
    for dataset_name in DATASET_NAMES:
        # Restrict model_names for oxford-flowers
        if dataset_name in ["oxford-flowers", "beans"]:
            model_names = ["deepersnn", "lifresnetlike"]
            snn_betas = OXFORD_SNN_BETAS
            snn_n_steps = OXFORD_SNN_N_STEPS
            snn_spike_grads_all = OXFORD_SNN_SPIKE_GRADS
            epochs = 100
        else:
            model_names = ["lifcnn", "deepersnn", "lifresnetlike"]
            snn_betas = SNN_BETAS
            snn_n_steps = SNN_N_STEPS
            snn_spike_grads_all = SNN_SPIKE_GRADS
            # Set epochs to 50 for mnist, cifar10, cifar100
            if dataset_name in ["mnist", "cifar10", "cifar100"]:
                epochs = 50
            else:
                epochs = 100

        for model_name in model_names:
            for optimizer_name in OPTIMIZER_NAMES:
                for lr in LRS:
                    for propagator_name in PROPAGATOR_NAMES:
                        for scheduler_name in SCHEDULER_NAMES:
                            for seed in SEEDS:
                                for snn_beta in snn_betas:
                                    for snn_reset_mechanism in SNN_RESET_MECHANISMS:
                                        for snn_surrogate_dis in SNN_SURROGATE_DISABLES:
                                            # Only vary snn_spike_grad if snn_surrogate_disable is False
                                            if not snn_surrogate_dis:
                                                spike_grads = snn_spike_grads_all
                                            else:
                                                spike_grads = ["step"]
                                            for snn_spike_grad in spike_grads:
                                                for snn_n_steps_val in snn_n_steps:
                                                    # Set batch_size and n_channels based on dataset_name
                                                    if dataset_name == "mnist":
                                                        batch_size = 128
                                                        datapath = "/mnt/data/"
                                                        n_channels = 1
                                                    elif dataset_name in [
                                                        "cifar10",
                                                        "cifar100",
                                                    ]:
                                                        batch_size = 128
                                                        datapath = "/mnt/data/"
                                                        n_channels = 3
                                                    else:
                                                        batch_size = 16
                                                        datapath = f"/mnt/data/{dataset_name}"
                                                        n_channels = 3

                                                    # Only vary snn_apply_noise and snn_noise_size
                                                    #  if snn_spike_grad is "step"
                                                    if snn_spike_grad == "step":
                                                        for snn_apply_noise in SNN_APPLY_NOISE:
                                                            if snn_apply_noise:
                                                                noise_sizes = SNN_NOISE_SIZES
                                                            else:
                                                                noise_sizes = [None]
                                                            for snn_noise_size in noise_sizes:
                                                                # Update base_config with current parameters
                                                                wandb_project_name = f"lfp-{dataset_name}-{model_name}"
                                                                base_config.update(
                                                                    {
                                                                        "lr": lr,
                                                                        "optimizer_name": optimizer_name,
                                                                        "propagator_name": propagator_name,
                                                                        "seed": seed,
                                                                        "data_path": datapath,
                                                                        "dataset_name": dataset_name,
                                                                        "model_name": model_name,
                                                                        "wandb_project_name": wandb_project_name,
                                                                        "scheduler_name": scheduler_name,
                                                                        "snn_beta": snn_beta,
                                                                        "snn_reset_mechanism": snn_reset_mechanism,
                                                                        "snn_surrogate_disable": snn_surrogate_dis,
                                                                        "snn_spike_grad": snn_spike_grad,
                                                                        "snn_n_steps": snn_n_steps_val,
                                                                        "batch_size": batch_size,
                                                                        "epochs": epochs,
                                                                        "n_channels": n_channels,
                                                                        "snn_apply_noise": snn_apply_noise,
                                                                        "snn_noise_size": snn_noise_size,
                                                                    }
                                                                )

                                                                # Generate config name and save the file
                                                                config_name = generate_config_name(base_config)
                                                                # Optionally, append noise info to
                                                                # config name for uniqueness
                                                                if snn_apply_noise and snn_noise_size is not None:
                                                                    config_name += f"_noise{snn_noise_size}"
                                                                elif not snn_apply_noise:
                                                                    config_name += "_nonoise"

                                                                if dataset_name not in saved_datasets:
                                                                    # Save the first configuration for
                                                                    # this dataset to both testing_dir
                                                                    # and cluster_dir
                                                                    save_config(
                                                                        base_config,
                                                                        testing_dir,
                                                                        config_name,
                                                                    )
                                                                    saved_datasets.add(dataset_name)
                                                                # Save the first config for each
                                                                # (dataset_name, propagator_name)
                                                                # to local_dir
                                                                local_key = (
                                                                    dataset_name,
                                                                    propagator_name,
                                                                )
                                                                if local_key not in saved_local:
                                                                    save_config(
                                                                        base_config,
                                                                        local_dir,
                                                                        config_name,
                                                                    )
                                                                    saved_local.add(local_key)
                                                                save_config(
                                                                    base_config,
                                                                    cluster_dir,
                                                                    config_name,
                                                                )
                                                                counter += 1
                                                    else:
                                                        # For non-"step" spike grads,
                                                        # do not vary snn_apply_noise or
                                                        #  snn_noise_size
                                                        base_config.update(
                                                            {
                                                                "lr": lr,
                                                                "optimizer_name": optimizer_name,
                                                                "propagator_name": propagator_name,
                                                                "seed": seed,
                                                                "data_path": datapath,
                                                                "dataset_name": dataset_name,
                                                                "model_name": model_name,
                                                                "wandb_project_name": wandb_project_name,
                                                                "scheduler_name": scheduler_name,
                                                                "snn_beta": snn_beta,
                                                                "snn_reset_mechanism": snn_reset_mechanism,
                                                                "snn_surrogate_disable": snn_surrogate_dis,
                                                                "snn_spike_grad": snn_spike_grad,
                                                                "snn_n_steps": snn_n_steps_val,
                                                                "batch_size": batch_size,
                                                                "epochs": epochs,
                                                                "n_channels": n_channels,
                                                                "snn_apply_noise": False,
                                                                "snn_noise_size": None,
                                                            }
                                                        )

                                                        config_name = generate_config_name(base_config)
                                                        config_name += "_nonoise"

                                                        if dataset_name not in saved_datasets:
                                                            save_config(
                                                                base_config,
                                                                testing_dir,
                                                                config_name,
                                                            )
                                                            saved_datasets.add(dataset_name)
                                                        local_key = (
                                                            dataset_name,
                                                            propagator_name,
                                                        )
                                                        if local_key not in saved_local:
                                                            save_config(
                                                                base_config,
                                                                local_dir,
                                                                config_name,
                                                            )
                                                            saved_local.add(local_key)
                                                        save_config(
                                                            base_config,
                                                            cluster_dir,
                                                            config_name,
                                                        )
                                                        counter += 1

    print(f"Created {counter} files!")  # Print the total number of configurations created


if __name__ == "__main__":
    main()
