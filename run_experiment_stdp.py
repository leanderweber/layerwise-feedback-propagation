################################################################################
# Description: This is based on the train.py from https://github.com/aidinattar/snn                                       #
################################################################################

from argparse import ArgumentParser
from types import SimpleNamespace
import os

import experiment_utils.spyketorch.utils as utils
import numpy as np
import torch
import yaml
import wandb
import joblib
import logging
from experiment_utils.data import dataloaders, datasets, transforms
from experiment_utils.spyketorch.model.deeper2024 import DeeperSNN
from experiment_utils.spyketorch.model.resnet2024 import ResSNN
from experiment_utils.utils.utils import set_random_seeds
from tqdm import tqdm

def run_training_stdp(
    savepath,
    data_path,
    dataset_name,
    lr,
    propagator_name,
    batch_size=128,
    pretrained_model=True,
    n_channels=3,
    momentum=0.9,
    weight_decay=0.0,
    scheduler_name="none",
    clip_updates=False,
    clip_update_threshold=2.0,
    reward_name="correct-class",
    reward_kwargs={},
    loss_name="ce-loss",
    epochs=5,
    snn_n_steps=15,
    model_name="cifar-vgglike",
    default_model_checkpoint="google/vit-base-patch16-224-in21k",
    snn_beta=0.9,
    snn_reset_mechanism="subtract",
    snn_surrogate_disable=False,
    snn_spike_grad="step",
    snn_apply_noise=False,
    snn_noise_size=1e-6,
    snn_ratio=1.0, #TODO
    snn_augment=False, #TODO
    optimizer_name="sgd",
    activation="relu",
    batch_log=False,
    param_sparsity_log=False,
    seed=None,
    wandb_key=None,
    disable_wandb=True,
    wandb_project_name="defaultproject",
    verbose=True,
):

    os.environ["WANDB_API_KEY"] = wandb_key
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if seed is None:
        str_seed = "0"
    else:
        str_seed = str(seed)
    savepath = os.path.join(savepath, str_seed)
    os.makedirs(savepath, exist_ok=True)
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
        "lr": lr,
        "propagator_name": propagator_name,
        "batch_size": batch_size,
        "n_channels": n_channels,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "scheduler_name": scheduler_name,
        "clip_updates": clip_updates,
        "clip_update_threshold": clip_update_threshold,
        "reward_name": reward_name,
        "loss_name": loss_name,
        "epochs": epochs,
        "snn_n_steps": snn_n_steps,
        "model_name": model_name,
        "default_model_checkpoint": default_model_checkpoint,
        "snn_beta": snn_beta,
        "snn_reset_mechanism": snn_reset_mechanism,
        "snn_surrogate_disable": snn_surrogate_disable,
        "snn_spike_grad": snn_spike_grad,
        "snn_apply_noise": snn_apply_noise,
        "snn_noise_size": snn_apply_noise,
        "snn_ratio": snn_ratio,
        "snn_augment": snn_augment,
        "optimizer_name": optimizer_name,
        "activation": activation,
        "batch_log": batch_log,
        "seed": seed,
    }
    logdict.update({f"reward_{k}": v for k, v in reward_kwargs.items()})
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
    train_dataset, collate_fn, dummy_input, class_labels = datasets.get_dataset(
        dataset_name,
        data_path,
        transforms.get_transforms(dataset_name, "train", model_path=default_model_checkpoint),
        mode="train",
        return_dummy_input=model_name == "vit",
    )
    test_dataset, _, _, _ = datasets.get_dataset(
        dataset_name,
        data_path,
        transforms.get_transforms(dataset_name, "test", model_path=default_model_checkpoint),
        mode="test",
        return_dummy_input=model_name == "vit",
    )

    train_loader = dataloaders.get_dataloader(
        dataset_name, train_dataset, batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = dataloaders.get_dataloader(
        dataset_name, test_dataset, batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Initialize the model
    if model_name == "resnet2024":
        model = ResSNN(
            n_channels=n_channels,
            num_classes=len(class_labels),
            device=device,
        )
        epochs = epochs[3]
        max_layers = 4
    elif model_name == "deeper2024":
        model = DeeperSNN(
            n_channels=n_channels,
            num_classes=len(class_labels),
            device=device,
        )
        epochs = epochs[4]
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
    # First layer
    first_layer_name = f"{savepath}/models/{model_name}_{dataset_name}{'_augmented' if config.get('augment', False) else ''}{'_ratio_' + str(ratio) if ratio < 1.0 else ''}_first_layer.pth"
    if os.path.isfile(first_layer_name):
        print("Loading first layer model")
        model.load_state_dict(torch.load(first_layer_name), strict=False)
    else:
        iterator = tqdm(range(epochs[0]), desc="Training First Layer")
        for epoch in iterator:
            i = 0
            iterator_epoch = tqdm(
                train_loader, desc=f"Epoch {epoch}", position=1, leave=False
            )
            for i, (data, _) in enumerate(iterator_epoch):
                data = data.to(device)
                model.train_unsupervised(data, layer_idx=1)
                iterator.set_postfix({"Iteration": i + 1})
                i += 1
        torch.save(model.state_dict(), first_layer_name)

    # Second layer
    second_layer_name = f"{savepath}/models/{model_name}_{dataset_name}{'_augmented' if config.get('augment', False) else ''}{'_ratio_' + str(ratio) if ratio < 1.0 else ''}_second_layer.pth"
    if os.path.isfile(second_layer_name):
        print("Loading second layer model")
        model.load_state_dict(torch.load(second_layer_name), strict=False)
    else:
        iterator = tqdm(range(epochs[1]), desc="Training Second Layer")
        for epoch in iterator:
            i = 0
            iterator_epoch = tqdm(
                train_loader, desc=f"Epoch {epoch}", position=1, leave=False
            )
            for data, _ in iterator_epoch:
                data = data.to(device)
                model.train_unsupervised(data, layer_idx=2)
                iterator.set_postfix({"Iteration": i + 1})
                i += 1
        torch.save(model.state_dict(), second_layer_name)

    # Third layer
    third_layer_name = f"{savepath}/models/{model_name}_{dataset_name}{'_augmented' if config.get('augment', False) else ''}{'_ratio_' + str(ratio) if ratio < 1.0 else ''}_third_layer.pth"
    if os.path.isfile(third_layer_name):
        print("Loading third layer model")
        model.load_state_dict(torch.load(third_layer_name), strict=False)
    else:
        iterator = tqdm(range(epochs[2]), desc="Training Third Layer")
        for epoch in iterator:
            i = 0
            iterator_epoch = tqdm(
                train_loader, desc=f"Epoch {epoch}", position=1, leave=False
            )
            for data, _ in iterator_epoch:
                data = data.to(device)
                model.train_unsupervised(data, layer_idx=3)
                iterator.set_postfix({"Iteration": i + 1})
                i += 1
        torch.save(model.state_dict(), third_layer_name)

    if model_name == "deeper2024":
        # Fourth layer
        fourth_layer_name = f"{savepath}/models/{model_name}_{dataset_name}{'_augmented' if config.get('augment', False) else ''}{'_ratio_' + str(ratio) if ratio < 1.0 else ''}_fourth_layer.pth"
        if os.path.isfile(fourth_layer_name):
            print("Loading fourth layer model")
            model.load_state_dict(torch.load(fourth_layer_name), strict=False)
        else:
            iterator = tqdm(range(epochs[3]), desc="Training Fourth Layer")
            for epoch in iterator:
                i = 0
                iterator_epoch = tqdm(
                    train_loader, desc=f"Epoch {epoch}", position=1, leave=False
                )
                for data, _ in iterator_epoch:
                    data = data.to(device)
                    model.train_unsupervised(data, layer_idx=4)
                    iterator.set_postfix({"Iteration": i + 1})
                    i += 1
            torch.save(model.state_dict(), fourth_layer_name)

    # Train the R-STDP layer
    # Set learning rates
    if model_name == "deeper2024":
        apr5 = model.block5["stdp"].learning_rate[0][0].item()
        anr5 = model.block5["stdp"].learning_rate[0][1].item()
        app5 = model.block5["anti_stdp"].learning_rate[0][1].item()
        anp5 = model.block5["anti_stdp"].learning_rate[0][0].item()
    if model_name == "resnet2024":
        apr4 = model.block4["stdp"].learning_rate[0][0].item()
        anr4 = model.block4["stdp"].learning_rate[0][1].item()
        app4 = model.block4["anti_stdp"].learning_rate[0][1].item()
        anp4 = model.block4["anti_stdp"].learning_rate[0][0].item()

    adaptive_min, adaptive_int = (0, 1)

    # performance
    best_train = np.array([0.0, 0.0, 0.0, 0.0])  # correct, total, loss, epoch
    best_test = np.array([0.0, 0.0, 0.0, 0.0])  # correct, total, loss, epoch

    try:
        iterator = tqdm(range(epochs), desc="Training R STDP Layer")
        for epoch in iterator:
            model.epoch = epoch
            perf_train = np.array([0.0, 0.0, 0.0])
            total_correct = 0
            total_loss = 0
            total_samples = 0
            i = 0
            iterator_epoch = tqdm(
                train_loader, desc=f"Training epoch {epoch}", position=1, leave=False
            )
            for k, (data, targets) in enumerate(iterator_epoch):
                perf_train_batch = model.train_rl(data, targets, layer_idx=max_layers)
                iterator_epoch.set_postfix({"Performance": perf_train_batch})

                if model_name == "resnet2024":
                    apr_adapt4 = apr4 * (
                        perf_train_batch[1] * adaptive_int + adaptive_min
                    )
                    anr_adapt4 = anr4 * (
                        perf_train_batch[1] * adaptive_int + adaptive_min
                    )
                    app_adapt4 = app4 * (
                        perf_train_batch[0] * adaptive_int + adaptive_min
                    )
                    anp_adapt4 = anp4 * (
                        perf_train_batch[0] * adaptive_int + adaptive_min
                    )

                    model.update_learning_rates(
                        stdp_ap=apr_adapt4,
                        stdp_an=anr_adapt4,
                        anti_stdp_ap=app_adapt4,
                        anti_stdp_an=anp_adapt4,
                        layer_idx=4,
                    )

                if model_name == "deeper2024":
                    apr_adapt5 = apr5 * (
                        perf_train_batch[1] * adaptive_int + adaptive_min
                    )
                    anr_adapt5 = anr5 * (
                        perf_train_batch[1] * adaptive_int + adaptive_min
                    )
                    app_adapt5 = app5 * (
                        perf_train_batch[0] * adaptive_int + adaptive_min
                    )
                    anp_adapt5 = anp5 * (
                        perf_train_batch[0] * adaptive_int + adaptive_min
                    )

                    model.update_learning_rates(
                        stdp_ap=apr_adapt5,
                        stdp_an=anr_adapt5,
                        anti_stdp_ap=app_adapt5,
                        anti_stdp_an=anp_adapt5,
                        layer_idx=5,
                    )

                perf_train += perf_train_batch

                total_correct += perf_train_batch[0]
                total_loss += perf_train_batch[1]
                total_samples += np.sum(perf_train_batch)
                i += 1

                iterator.set_postfix({"Performance": np.round(perf_train / (i + 1), 2)})

            perf_train /= len(train_loader)
            if best_train[0] <= perf_train[0]:
                best_train = np.append(perf_train, epoch)

            # Log training performance to TensorBoard
            if config.get("tensorboard", False):
                model.writer.add_scalar(
                    "Train/Accuracy", total_correct / total_samples, epoch
                )
                model.writer.add_scalar("Train/Loss", total_loss / total_samples, epoch)
            model.history["train_acc"].append(total_correct / total_samples)
            model.history["train_loss"].append(total_loss / total_samples)

            total_correct = 0
            total_loss = 0
            total_samples = 0
            for data, targets in test_loader:
                data = data.to(device)
                targets = targets.to(device)
                perf_test = model.test(data, targets, layer_idx=max_layers)
                if best_test[0] <= perf_test[0]:
                    best_test = np.append(perf_test, epoch)
                    torch.save(
                        model.state_dict(),
                        f"{savepath}/models/{model_name}_{dataset_name}_best.pth",
                    )

                total_correct += perf_test[0]
                total_loss += perf_test[1]
                total_samples += np.sum(perf_test)

            # Log test performance to TensorBoard
            if config.get("tensorboard", False):
                model.writer.add_scalar(
                    "Test/Accuracy", total_correct / total_samples, epoch
                )
                model.writer.add_scalar("Test/Loss", total_loss / total_samples, epoch)
            model.history["test_acc"].append(total_correct / total_samples)
            model.history["test_loss"].append(total_loss / total_samples)

            # Log additional metrics to TensorBoard
            model.all_preds = []
            model.all_targets = []
            for data, targets in metrics_loader:
                data = data.to(device)
                targets = targets.to(device)
                model.compute_preds(data, targets, layer_idx=max_layers)
            metrics = model.metrics()
            model.log_tensorboard(metrics, epoch)

            if epoch - best_test[3] > 5:
                break

    except KeyboardInterrupt:
        print("Training Interrupted")
        print("Best Train:", best_train)
        print("Best Test:", best_test)

    print("Best Train:", best_train)
    print("Best Test:", best_test)

    # Save training history
    model.save_history(file_path=f"{savepath}/models/{model_name}_{dataset_name}_history.csv")

    # Plot training history
    model.plot_history(file_path=f"{savepath}/models/{model_name}_{dataset_name}_history.png")

    # Save activation maps
    model.save_activation_maps(
        file_path=f"{savepath}/models/{model_name}_{dataset_name}_activation_maps"
    )

    # Close TensorBoard writer
    model.close_tensorboard()


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
        lr=config.lr,
        propagator_name=config.propagator_name,
        batch_size=config.batch_size,
        n_channels=config.n_channels,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        scheduler_name=config.scheduler_name,
        clip_updates=config.clip_updates,
        clip_update_threshold=config.clip_update_threshold,
        reward_name=config.reward_name,
        reward_kwargs=config.reward_kwargs,
        loss_name=config.loss_name,
        epochs=config.epochs,
        snn_n_steps=config.snn_n_steps,
        model_name=config.model_name,
        default_model_checkpoint=config.default_model_checkpoint,
        snn_beta=config.snn_beta,
        snn_reset_mechanism=config.snn_reset_mechanism,
        snn_surrogate_disable=config.snn_surrogate_disable,
        snn_spike_grad=config.snn_spike_grad,
        snn_apply_noise=config.snn_apply_noise,
        snn_noise_size=config.snn_noise_size,
        optimizer_name=config.optimizer_name,
        activation=config.activation,
        batch_log=config.batch_log,
        param_sparsity_log=config.param_sparsity_log,
        seed=config.seed,
        wandb_key=config.wandb_key,
        disable_wandb=config.disable_wandb,
        wandb_project_name=config.wandb_project_name,
        verbose=config.verbose,
    )
    )
