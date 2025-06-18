################################################################################
# Description: This is based on the train.py from
# https://github.com/aidinattar/snn                                       #
################################################################################

import logging
import os
import time
from argparse import ArgumentParser
from types import SimpleNamespace

import joblib
import numpy as np
import torch
import wandb
import yaml
from tqdm import tqdm

import experiment_utils.spyketorch.utils as utils
from experiment_utils.spyketorch.model.deeper2024 import DeeperSNN
from experiment_utils.spyketorch.model.resnet2024 import ResSNN
from experiment_utils.utils.utils import set_random_seeds


def run_training_stdp(
    savepath,
    data_path,
    dataset_name,
    batch_size=128,
    n_channels=3,
    reward_kwargs={},
    epochs=[50, 50, 50, 50, 50],
    model_name="deepersnn",
    seed=None,
    wandb_key=None,
    disable_wandb=True,
    wandb_project_name="defaultproject",
):
    ratio = 1
    augment = False

    os.environ["WANDB_API_KEY"] = wandb_key
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if seed is None:
        str_seed = "0"
    else:
        str_seed = str(seed)
    savepath = os.path.join(savepath, str_seed)
    os.makedirs(savepath, exist_ok=True)
    os.makedirs(os.path.join(savepath, "models"), exist_ok=True)
    print("RUN:", savepath, seed)

    print("Building Paths...")

    # Wandb Path
    wandbpath = os.path.join(savepath, "wandb")
    os.makedirs(wandbpath, exist_ok=True)

    # Checkpoint Path
    ckpt_path = os.path.join(savepath, "ckpts")
    os.makedirs(ckpt_path, exist_ok=True)

    # Performance Metrics Path
    performancepath = os.path.join(savepath, "performance-metrics")
    os.makedirs(performancepath, exist_ok=True)

    # Wandb Stuff
    logdict = {
        "data_path": data_path,
        "dataset_name": dataset_name,
        "batch_size": batch_size,
        "n_channels": n_channels,
        "propagator_name": "rstdp",
        "epochs": epochs,
        "model_name": model_name,
        "seed": seed,
    }

    print("Intializing wandb")
    w_id = wandb.util.generate_id()
    wandb.init(
        id=w_id,
        project=wandb_project_name,
        dir=wandbpath,
        mode="disabled" if disable_wandb else "online",
        config=logdict,
    )
    joblib.dump(w_id, os.path.join(savepath, "wandb_id.joblib"))

    # Set seeds for reproducability
    if seed is not None:
        logging.info("Setting seeds...")
        set_random_seeds(seed)

    # Data
    print("Loading Initial State...")
    # with nostdout(verbose=verbose):
    train_loader, test_loader, metrics_loader, num_classes, in_channels = utils.prepare_data(
        dataset_name, data_path, batch_size, augment=False
    )

    # Initialize the model
    if model_name == "lifresnetlike":
        model = ResSNN(
            in_channels=in_channels,
            num_classes=num_classes,
            device=device,
        )
        final_epochs = epochs[3]
        max_layers = 4
    elif model_name == "deepersnn":
        model = DeeperSNN(
            in_channels=in_channels,
            num_classes=num_classes,
            device=device,
        )
        final_epochs = epochs[4]
        max_layers = 5
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Move the model to the appropriate device
    model.to(device)

    # Register hooks for activation maps
    model.register_hooks()

    #############################
    # Train the model           #
    #############################
    training_time = 0.0
    if os.path.exists(os.path.join(savepath, "elapsed.joblib")):
        training_time = joblib.load(os.path.join(savepath, "elapsed.joblib"))
        print(f"Loaded previous training time: {training_time:.2f} seconds")

    # First layer
    first_layer_name = f"{savepath}/models/{model_name}_{dataset_name}{'_augmented' if not augment else ''}"
    first_layer_name += f"{'_ratio_' + str(ratio) if ratio < 1.0 else ''}_first_layer.pth"
    if os.path.isfile(first_layer_name):
        print("Loading first layer model")
        model.load_state_dict(torch.load(first_layer_name), strict=False)
    else:
        print("Training First Layer")
        iterator = tqdm(range(epochs[0]), desc="Training First Layer", disable=True)
        for epoch in iterator:
            i = 0
            iterator_epoch = tqdm(
                train_loader,
                desc=f"Epoch {epoch}",
                position=1,
                leave=False,
                disable=True,
            )
            for i, (data, _) in enumerate(iterator_epoch):
                data = data.to(device)  # noqa: PLW2901
                start_time = time.time()
                model.train_unsupervised(data, layer_idx=1)
                elapsed = time.time() - start_time
                training_time += elapsed
                iterator.set_postfix({"Iteration": i + 1})
                i += 1  # noqa: PLW2901
            logdict = {"total_training_time": training_time}
            wandb.log(logdict)
        torch.save(model.state_dict(), first_layer_name)
    joblib.dump(training_time, os.path.join(savepath, "elapsed.joblib"))

    # Second layer
    second_layer_name = f"{savepath}/models/{model_name}_{dataset_name}{'_augmented' if not augment else ''}"
    second_layer_name += f"{'_ratio_' + str(ratio) if ratio < 1.0 else ''}_second_layer.pth"
    if os.path.isfile(second_layer_name):
        print("Loading second layer model")
        model.load_state_dict(torch.load(second_layer_name), strict=False)
    else:
        print("Training Second Layer")
        iterator = tqdm(range(epochs[1]), desc="Training Second Layer", disable=True)
        for epoch in iterator:
            i = 0
            iterator_epoch = tqdm(
                train_loader,
                desc=f"Epoch {epoch}",
                position=1,
                leave=False,
                disable=True,
            )
            for data, _ in iterator_epoch:
                data = data.to(device)  # noqa: PLW2901
                start_time = time.time()
                model.train_unsupervised(data, layer_idx=2)
                elapsed = time.time() - start_time
                training_time += elapsed
                iterator.set_postfix({"Iteration": i + 1})
                i += 1
            logdict = {"total_training_time": training_time}
            wandb.log(logdict)
        torch.save(model.state_dict(), second_layer_name)
    joblib.dump(training_time, os.path.join(savepath, "elapsed.joblib"))

    # Third layer
    third_layer_name = f"{savepath}/models/{model_name}_{dataset_name}{'_augmented' if not augment else ''}"
    third_layer_name += f"{'_ratio_' + str(ratio) if ratio < 1.0 else ''}_third_layer.pth"
    if os.path.isfile(third_layer_name):
        print("Loading third layer model")
        model.load_state_dict(torch.load(third_layer_name), strict=False)
    else:
        print("Training Third Layer")
        iterator = tqdm(range(epochs[2]), desc="Training Third Layer", disable=True)
        for epoch in iterator:
            i = 0
            iterator_epoch = tqdm(
                train_loader,
                desc=f"Epoch {epoch}",
                position=1,
                leave=False,
                disable=True,
            )
            for data, _ in iterator_epoch:
                data = data.to(device)  # noqa: PLW2901
                start_time = time.time()
                model.train_unsupervised(data, layer_idx=3)
                elapsed = time.time() - start_time
                training_time += elapsed
                iterator.set_postfix({"Iteration": i + 1})
                i += 1
            logdict = {"total_training_time": training_time}
            wandb.log(logdict)
        torch.save(model.state_dict(), third_layer_name)
    joblib.dump(training_time, os.path.join(savepath, "elapsed.joblib"))

    if model_name == "deepersnn":
        # Fourth layer
        fourth_layer_name = f"{savepath}/models/{model_name}_{dataset_name}{'_augmented' if not augment else ''}"
        fourth_layer_name += f"{'_ratio_' + str(ratio) if ratio < 1.0 else ''}_fourth_layer.pth"
        if os.path.isfile(fourth_layer_name):
            print("Loading fourth layer model")
            model.load_state_dict(torch.load(fourth_layer_name), strict=False)
        else:
            print("Training Fourth Layer")
            iterator = tqdm(range(epochs[3]), desc="Training Fourth Layer", disable=True)
            for epoch in iterator:
                i = 0
                iterator_epoch = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch}",
                    position=1,
                    leave=False,
                    disable=True,
                )
                for data, _ in iterator_epoch:
                    data = data.to(device)  # noqa: PLW2901
                    start_time = time.time()
                    model.train_unsupervised(data, layer_idx=4)
                    elapsed = time.time() - start_time
                    training_time += elapsed
                    iterator.set_postfix({"Iteration": i + 1})
                    i += 1
            logdict = {"total_training_time": training_time}
            wandb.log(logdict)
            torch.save(model.state_dict(), fourth_layer_name)
    joblib.dump(training_time, os.path.join(savepath, "elapsed.joblib"))

    # Train the R-STDP layer
    # Set learning rates
    if model_name == "deepersnn":
        apr5 = model.block5["stdp"].learning_rate[0][0].item()
        anr5 = model.block5["stdp"].learning_rate[0][1].item()
        app5 = model.block5["anti_stdp"].learning_rate[0][1].item()
        anp5 = model.block5["anti_stdp"].learning_rate[0][0].item()
    if model_name == "lifresnetlike":
        apr4 = model.block4["stdp"].learning_rate[0][0].item()
        anr4 = model.block4["stdp"].learning_rate[0][1].item()
        app4 = model.block4["anti_stdp"].learning_rate[0][1].item()
        anp4 = model.block4["anti_stdp"].learning_rate[0][0].item()

    adaptive_min, adaptive_int = (0, 1)

    # performance
    best_train = np.array([0.0, 0.0, 0.0, 0.0])  # correct, total, loss, epoch
    best_test = np.array([0.0, 0.0, 0.0, 0.0])  # correct, total, loss, epoch

    iterator = tqdm(range(final_epochs), desc="Training R STDP Layer", disable=False)
    print("Training R-STDP Layer")
    for epoch in iterator:
        model.epoch = epoch
        perf_train = np.array([0.0, 0.0, 0.0])
        total_correct_train = 0
        total_loss_train = 0
        total_samples_train = 0
        i = 0
        iterator_epoch = tqdm(
            train_loader,
            desc=f"Training epoch {epoch}",
            position=1,
            leave=False,
            disable=False,
        )
        for k, (data, targets) in enumerate(iterator_epoch):
            start_time = time.time()
            perf_train_batch = model.train_rl(data, targets, layer_idx=max_layers)
            iterator_epoch.set_postfix({"Performance": perf_train_batch})

            if model_name == "lifresnetlike":
                apr_adapt4 = apr4 * (perf_train_batch[1] * adaptive_int + adaptive_min)
                anr_adapt4 = anr4 * (perf_train_batch[1] * adaptive_int + adaptive_min)
                app_adapt4 = app4 * (perf_train_batch[0] * adaptive_int + adaptive_min)
                anp_adapt4 = anp4 * (perf_train_batch[0] * adaptive_int + adaptive_min)

                model.update_learning_rates(
                    stdp_ap=apr_adapt4,
                    stdp_an=anr_adapt4,
                    anti_stdp_ap=app_adapt4,
                    anti_stdp_an=anp_adapt4,
                    layer_idx=4,
                )

            if model_name == "deepersnn":
                apr_adapt5 = apr5 * (perf_train_batch[1] * adaptive_int + adaptive_min)
                anr_adapt5 = anr5 * (perf_train_batch[1] * adaptive_int + adaptive_min)
                app_adapt5 = app5 * (perf_train_batch[0] * adaptive_int + adaptive_min)
                anp_adapt5 = anp5 * (perf_train_batch[0] * adaptive_int + adaptive_min)

                model.update_learning_rates(
                    stdp_ap=apr_adapt5,
                    stdp_an=anr_adapt5,
                    anti_stdp_ap=app_adapt5,
                    anti_stdp_an=anp_adapt5,
                    layer_idx=5,
                )
            elapsed = time.time() - start_time
            training_time += elapsed

            perf_train += perf_train_batch

            total_correct_train += perf_train_batch[0]
            total_loss_train += perf_train_batch[1]
            total_samples_train += np.sum(perf_train_batch)
            i += 1

            iterator.set_postfix({"Performance": np.round(perf_train / (i + 1), 2)})

        perf_train /= len(train_loader)
        if best_train[0] <= perf_train[0]:
            best_train = np.append(perf_train, epoch)

        model.history["train_acc"].append(total_correct_train / total_samples_train)
        model.history["train_loss"].append(total_loss_train / total_samples_train)

        total_correct_test = 0
        total_loss_test = 0
        total_samples_test = 0
        for data, targets in test_loader:
            data = data.to(device)  # noqa: PLW2901
            targets = targets.to(device)  # noqa: PLW2901
            perf_test = model.test(data, targets, layer_idx=max_layers)
            if best_test[0] <= perf_test[0]:
                best_test = np.append(perf_test, epoch)
                torch.save(
                    model.state_dict(),
                    f"{savepath}/models/{model_name}_{dataset_name}_best.pth",
                )

            total_correct_test += perf_test[0]
            total_loss_test += perf_test[1]
            total_samples_test += np.sum(perf_test)

        logdict = {"epoch": epoch + 1}
        logdict.update({"train_micro_accuracy_top1": total_correct_train / total_samples_train})
        logdict.update({"train_criterion": total_loss_train / total_samples_train})
        logdict.update({"test_micro_accuracy_top1": total_correct_test / total_samples_test})
        logdict.update({"test_criterion": total_loss_test / total_samples_test})
        logdict.update({"total_training_time": training_time})
        wandb.log(logdict)

        # Log test performance to TensorBoard
        model.history["test_acc"].append(total_correct_test / total_samples_test)
        model.history["test_loss"].append(total_loss_test / total_samples_test)

    joblib.dump(training_time, os.path.join(savepath, "elapsed.joblib"))

    # Save training history
    model.save_history(file_path=f"{savepath}/models/{model_name}_{dataset_name}_history.csv")

    # Save activation maps
    model.save_activation_maps(file_path=f"{savepath}/models/{model_name}_{dataset_name}_activation_maps")


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config_file", default="None")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    print("Starting script...")

    args = get_args()
    print(f"CONFIG: {args.config_file}")
    with open(args.config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config["config_name"] = os.path.basename(args.config_file)[:-5]
        except yaml.YAMLError as exc:
            print(exc)
            config = {}

    config["config_file"] = args.config_file

    config = SimpleNamespace(**config)
    print(config)

    run_training_stdp(
        savepath=config.savepath,
        data_path=config.data_path,
        dataset_name=config.dataset_name,
        batch_size=config.batch_size,
        n_channels=config.n_channels,
        epochs=config.epochs,
        model_name=config.model_name,
        seed=config.seed,
        wandb_key=config.wandb_key,
        disable_wandb=config.disable_wandb,
        wandb_project_name=config.wandb_project_name,
    )
