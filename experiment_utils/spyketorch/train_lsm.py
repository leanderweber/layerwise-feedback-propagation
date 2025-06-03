################################################################################
# Title:            train_lsm.py                                                     #
# Description:      Code to define and train a liquid state machine.           #
# Author:           Aidin Attar                                                #
# Date:             2024-10-02                                                 #
# Version:          0.2                                                        #
# Usage:            None                                                       #
# Notes:            None                                                       #
# Python version:   3.11.7                                                     #
################################################################################

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tonic
import torch
import torch.nn as nn
from model.lsm2024 import (
    LSM,
    ConvLSM,
    LSM_partition,
    initialize_conv_weights,
    initialize_partitioned_weights,
    initialize_weights,
)
from sklearn.metrics import accuracy_score, confusion_matrix
from tonic import DiskCachedDataset
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from utils import EarlyStopping, LabelEncoderTransform, caltech101_classes

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging


def main():
    parser = argparse.ArgumentParser(description="Train a Liquid State Machine (LSM) model.")
    parser.add_argument("--model", type=str, default="lsm", help="Model to train (lsm, lsm_partition).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer to use for training.")
    parser.add_argument("--loss", type=str, default="crossentropy", help="Loss function to use for training.")
    parser.add_argument("--dataset", type=str, default="nmnist", help="Dataset to use for training.")
    parser.add_argument("--reservoir_size", type=int, default=1000, help="Reservoir size for the LSM.")
    parser.add_argument("--output_size", type=int, default=10, help="Output size for the LSM.")
    parser.add_argument("--num_partitions", type=int, default=3, help="Number of partitions for the LSM.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--sparsity", type=float, default=0.1, help="Sparsity of the reservoir connections.")
    parser.add_argument("--alpha", type=float, default=0.9, help="Membrane potential decay rate.")
    parser.add_argument("--beta", type=float, default=0.9, help="Synaptic decay rate.")
    parser.add_argument("--threshold", type=float, default=1.0, help="Threshold for the LSM.")
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging.")
    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=f"runs/{args.model}/{args.dataset}")

    # Prepare data
    if args.dataset == "nmnist":
        sensor_size = tonic.datasets.NMNIST.sensor_size
        trans = tonic.transforms.Compose(
            [
                tonic.transforms.Denoise(filter_time=3000),
                # tonic.transforms.ToVoxelGrid(sensor_size=sensor_size, n_time_bins=100),
                tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=10000),
            ]
        )
        train_dataset = tonic.datasets.NMNIST(save_to="./data", train=True, transform=trans)
        test_dataset = tonic.datasets.NMNIST(save_to="./data", train=False, transform=trans)
    elif args.dataset == "cifar10-dvs":
        sensor_size = tonic.datasets.CIFAR10DVS.sensor_size
        trans = tonic.transforms.Compose(
            [
                # T.Resize(
                tonic.transforms.Denoise(filter_time=3000),
                tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=100000),
            ]
        )
        full_dataset = tonic.datasets.CIFAR10DVS(save_to="./data", transform=trans)
        # subset_indices = list(range(1000))  # Choose indices for the subset
        # subset_dataset = Subset(full_dataset, subset_indices)
        # train_dataset, test_dataset = torch.utils.data.random_split(subset_dataset, [0.8, 0.2])
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2], generator=generator)
        del full_dataset
    elif args.dataset == "ncaltech101":
        sensor_size = (1, 180, 240)
        trans = tonic.transforms.Compose(
            [
                T.Resize(sensor_size),
                T.CenterCrop(sensor_size),
                tonic.transforms.Denoise(filter_time=3000),
                tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=10000),
            ]
        )
        trans_target = LabelEncoderTransform(classes=caltech101_classes)
        full_dataset = tonic.datasets.NCALTECH101(save_to="./data", transform=trans, target_transform=trans_target)
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])
        del full_dataset
    elif args.dataset == "pokerdvs":
        sensor_size = tonic.datasets.POKERDVS.sensor_size
        trans = tonic.transforms.Compose(
            [
                tonic.transforms.Denoise(filter_time=3000),
                tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
            ]
        )
        train_dataset = tonic.datasets.POKERDVS(save_to="./data", train=True, transform=trans)
        test_dataset = tonic.datasets.POKERDVS(save_to="./data", train=False, transform=trans)
        # args.output_size = 4
        # print the unique labels
    elif args.dataset == "fashionmnist":
        sensor_size = (1, 28, 28)

    elif args.dataset == "mnist":
        raise NotImplementedError("MNIST dataset not supported yet.")
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    cached_train_dataset = DiskCachedDataset(train_dataset, cache_path=f"./data/cache/{args.dataset}/train/")
    cached_test_dataset = DiskCachedDataset(test_dataset, cache_path=f"./data/cache/{args.dataset}/test/")
    train_loader = DataLoader(
        cached_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
    )
    test_loader = DataLoader(
        cached_test_dataset,
        batch_size=len(cached_test_dataset),
        shuffle=False,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
    )

    del train_dataset, test_dataset

    # Set neuron parameters
    tauV = 16.0
    tauI = 16.0
    threshold = 20
    current_prefactor = np.float32(1 / tauI)
    alpha = np.float32(np.exp(-1 / tauI))
    beta = np.float32(1 - 1 / tauV)

    data, _ = next(iter(train_loader))
    print(f"Data shape: {data.shape}")
    print(f"Target counts: {np.unique(_, return_counts=True)}")

    if args.model != "lsm_conv":
        flat_data = torch.reshape(data, (data.shape[0], data.shape[1], -1))
        # Set the input size
        input_shape = flat_data.shape[-1]
    else:
        input_shape = data.shape[-3:]

    # data, _ = next(iter(test_loader))
    # print(f'Data shape: {data.shape}')
    # print(f'Target counts: {np.unique(_, return_counts=True)}')

    # Log the dataset in TensorBoard
    # if args.tensorboard:
    #     writer.add_embedding(flat_data, metadata=targets, label_img=data)

    # Initialize random sparse connectivity for the reservoir
    if args.model == "lsm":
        input_weights, reservoir_weights = initialize_weights(
            input_weight_scale=27,
            reservoir_weight_scale=2,
            input_connection_density=0.15,
            input_size=input_shape,
        )
        input_weights *= current_prefactor
    elif args.model == "lsm_partition":
        input_weights, reservoir_weights, inhibitory_weights = initialize_partitioned_weights(
            input_weight_scale=27,
            local_reservoir_weight_scale=2,
            long_distance_inhibitory_weight_scale=1,
            input_connection_density=0.15,
            input_size=input_shape,
            num_partitions=args.num_partitions,
        )
        for i in range(len(input_weights)):
            input_weights[i] *= current_prefactor
        inhibitory_weights *= current_prefactor
    elif args.model == "lsm_conv":
        input_height, input_width = data.shape[-2:]
        input_channels = data.shape[2]
        input_weights, reservoir_weights = initialize_conv_weights(
            input_weight_scale=27,
            reservoir_weight_scale=2,
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            kernel_size=5,
        )
        input_weights *= current_prefactor
    else:
        raise ValueError(f"Model {args.model} not supported.")

    reservoir_weights *= current_prefactor

    if args.model == "lsm":
        model = LSM(
            input_shape,
            args.reservoir_size,
            args.output_size,
            n_partitions=args.num_partitions,
            weight_in=input_weights,
            weight_res=reservoir_weights,
            sparsity=args.sparsity,
            alpha=alpha,
            beta=beta,
            threshold=threshold,
        ).to(args.device)
    elif args.model == "lsm_partition":
        model = LSM_partition(
            input_shape,
            args.reservoir_size,
            args.output_size,
            weight_in=input_weights,
            weight_lin=inhibitory_weights,
            weight_res=reservoir_weights,
            n_partitions=args.num_partitions,
            sparsity=args.sparsity,
            alpha=alpha,
            beta=beta,
            threshold=threshold,
        ).to(args.device)
    elif args.model == "lsm_conv":
        input_channels, input_height, input_width = data.shape[-3:]
        model = ConvLSM(
            input_channels,
            input_height,
            input_width,
            args.reservoir_size,
            args.output_size,
            weight_in=input_weights,
            weight_res=reservoir_weights,
            kernel_size=5,
            n_partitions=args.num_partitions,
            sparsity=args.sparsity,
            alpha=alpha,
            beta=beta,
            threshold=threshold,
        ).to(args.device)
    else:
        raise ValueError(f"Model {args.model} not supported.")

    # Log the model in TensorBoard
    # if args.tensorboard:
    #     writer.add_graph(model, data.to(args.device))

    # Define the loss function
    if args.loss == "crossentropy":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "bce":
        criterion = nn.BCELoss()
    else:
        raise ValueError(f"Loss function {args.loss} not supported.")

    # Define the optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters())
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters())
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported.")

    # Print model summary
    print(model)

    # Define scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Define early stopping
    early_stopping = EarlyStopping(patience=5, verbose=False, model=args.model)

    # Training loop
    iterator = tqdm(
        range(args.epochs),
        desc="Training",
        unit="epoch",
        position=0,
        leave=True,
        dynamic_ncols=True,
        total=args.epochs,
        initial=0,
        ascii=True,
    )
    for epoch in iterator:
        model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        iterator_epoch = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}",
            unit="batch",
            position=1,
            leave=False,
            dynamic_ncols=True,
            total=len(train_loader),
            initial=0,
            ascii=True,
        )
        for i, (data, target) in enumerate(iterator_epoch):
            data, target = data.to(args.device), target.to(args.device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output, _ = model(data)

            # Compute the loss
            loss = criterion(output, target)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += accuracy_score(target.cpu(), output.argmax(dim=1).cpu())

            # Log the loss and accuracy
            iterator_epoch.set_postfix(
                loss=loss.item(), accuracy=accuracy_score(target.cpu(), output.argmax(dim=1).cpu())
            )

            # Log the loss and accuracy in TensorBoard
            if args.tensorboard:
                writer.add_scalar("Train/Loss_Iteration", loss.item(), epoch * len(train_loader) + i)
                writer.add_scalar(
                    "Train/Accuracy_Iteration",
                    accuracy_score(target.cpu(), output.argmax(dim=1).cpu()),
                    epoch * len(train_loader) + i,
                )

        iterator.set_postfix(loss=epoch_loss, accuracy=epoch_accuracy / (i + 1))
        if args.tensorboard:
            writer.add_scalar("Train/Loss", epoch_loss, epoch)
            writer.add_scalar("Train/Accuracy", epoch_accuracy / (i + 1), epoch)

        # Update the learning rate and check for early stopping
        scheduler.step()
        early_stopping(epoch_loss, model)

        # Evaluation loop
        model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            iterator_test = tqdm(
                test_loader,
                desc="Evaluation",
                unit="batch",
                position=1,
                leave=False,
                dynamic_ncols=True,
                total=len(test_loader),
                initial=0,
                ascii=True,
            )
            for i, (data, target) in enumerate(iterator_test):
                data, target = data.to(args.device), target.to(args.device)

                # Forward pass
                output, _ = model(data)

                # Log the predictions
                y_true.extend(target.cpu().numpy())
                y_pred.extend(output.argmax(dim=1).cpu().numpy())

                iterator_test.set_postfix(accuracy=accuracy_score(y_true, y_pred))

            # print(y_true, y_pred)

            # Log the accuracy and confusion matrix in TensorBoard
            accuracy = accuracy_score(y_true, y_pred)
            iterator.set_postfix(loss=epoch_loss, accuracy=accuracy)
            cm = confusion_matrix(y_true, y_pred)
            if args.tensorboard:
                writer.add_scalar("Test/Accuracy", accuracy, epoch)
                fig, ax = plt.subplots()
                cax = ax.matshow(cm, cmap="coolwarm")
                fig.colorbar(cax)
                plt.xlabel("Predicted")
                plt.ylabel("True")
                writer.add_figure("Confusion Matrix", fig, epoch)

    # Compute the accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test accuracy: {accuracy}")

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    main()
