################################################################################
# Title:            train_lsm_radar.py                                         #
# Description:      Code to define and train a liquid state machine.           #
# Author:           Aidin Attar                                                #
# Date:             2024-10-28                                                 #
# Version:          0.1                                                        #
# Usage:            None                                                       #
# Notes:            None                                                       #
# Python version:   3.11.7                                                     #
################################################################################

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from model.lsm2024 import (
    LSM_radar,
    initialize_partitioned_weights,
)
from radar_datasets import DopNetH5Dataset, SoliH5Dataset, collate_fn
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import EarlyStopping

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
    if args.dataset == "gesture":
        train_file_path = "./data/gesture/train/preprocessed_gesture_data_train.h5"
        test_file_path = "./data/gesture/test/preprocessed_gesture_data_test.h5"
        person_list = ["A", "B", "C", "D", "E", "F"]
        gestures = [0, 1, 2, 3]  # Wave, Pinch, Swipe, Click
        train_dataset = DopNetH5Dataset(h5_file=train_file_path, person_list=person_list, gestures=gestures)
        test_dataset = DopNetH5Dataset(h5_file=test_file_path, person_list=person_list, gestures=gestures)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    elif args.dataset == "soli":
        file_path = "./data/Soli_Preprocessed/soli_Preprocessed.h5"
        soli_h5_dataset = SoliH5Dataset(h5_file=file_path, mode="unrolled")
        train_dataset, test_dataset = torch.utils.data.random_split(
            soli_h5_dataset, [int(len(soli_h5_dataset) * 0.8), len(soli_h5_dataset) - int(len(soli_h5_dataset) * 0.8)]
        )
        train_loader = DataLoader(soli_h5_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(soli_h5_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    # Set neuron parameters
    tauV = 16.0
    tauI = 16.0
    threshold = 20
    current_prefactor = np.float32(1 / tauI)
    alpha = np.float32(np.exp(-1 / tauI))
    beta = np.float32(1 - 1 / tauV)

    data, _, length = next(iter(train_loader))
    print(f"Data shape: {data.shape}")
    print(f"Target counts: {np.unique(_, return_counts=True)}")

    input_shape = 150  # 540 #data.shape[-1]

    # Initialize random sparse connectivity for the reservoir
    if args.model == "lsm_radar":
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
    else:
        raise ValueError(f"Model {args.model} not supported.")

    reservoir_weights *= current_prefactor

    if args.model == "lsm_radar":
        model = LSM_radar(
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
    else:
        raise ValueError(f"Model {args.model} not supported.")

    # class_weights = torch.tensor([
    #     466, 696, 479, 792
    # ], dtype=torch.float32, device=args.device)
    # class_weights = 1 / class_weights
    # class_weights /= class_weights.sum()
    class_weights = None

    # Define the loss function
    if args.loss == "crossentropy":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "bce":
        criterion = nn.BCELoss()
    elif args.loss == "nll":
        criterion = nn.NLLLoss(weight=class_weights)
    else:
        raise ValueError(f"Loss function {args.loss} not supported.")

    # Define the optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.002)
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
        for i, (data, target, length) in enumerate(iterator_epoch):
            data, target = data.to(args.device), target.to(args.device)

            target = target.long()

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output, _ = model(data, length)

            # Compute the loss
            loss = criterion(output, target)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # if i == 19:
            #     print(f'Output: {output}')
            #     print(f'Target: {target}')

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
            for i, (data, target, length) in enumerate(iterator_test):
                data, target = data.to(args.device), target.to(args.device)

                # Forward pass
                output, _ = model(data, length)

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
