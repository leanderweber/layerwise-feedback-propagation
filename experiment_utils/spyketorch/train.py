################################################################################
# Title:            train.py                                                   #
# Description:      This is the training script for the model                  #
# Author:           Aidin Attar                                                #
# Date:             2024-07-01                                                 #
# Version:          0.2                                                        #
# Usage:            python train.py                                            #
# Notes:            Added hooks, TensorBoard logging, and saving metrics       #
# Python version:   3.11.7                                                     #
################################################################################

import argparse
import os

import numpy as np
import torch
import utils
from model.deep2024 import DeepSNN
from model.deeper2024 import DeeperSNN
from model.deepr2024 import DeepRSNN, DeepRSNN2
from model.inception2024 import InceptionSNN
from model.majority2024 import MajoritySNN
from model.mozafari2018 import MozafariMNIST2018
from model.resnet2024 import ResSNN
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--model", type=str, default="mozafari2018", help="Model to use", choices=["mozafari2018", "deep2024", "deepr2024", "deepr2024_2", "inception2024", "majority2024", "resnet2024", "deeper2024"])
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset to use")
    parser.add_argument("--epochs", type=int, nargs='+', default=[2, 4, 100], help="Number of epochs to train each layer")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for the data loader")
    parser.add_argument("--augment", action="store_true", default=False, help="Enable data augmentation")
    parser.add_argument("--ratio", type=float, default=1.0, help="Ratio of training data to use")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    parser.add_argument("--tensorboard", action="store_true", default=False, help="Enable TensorBoard logging")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode")
    args = parser.parse_args()

    # Print the command line arguments
    print("############################################")
    print("Program to train the model")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")
    print("############################################")

    if not os.path.isdir("models"):
        os.mkdir("models")

    # Prepare the data
    train_loader, test_loader, metrics_loader, num_classes, in_channels = utils.prepare_data(args.dataset, args.batch_size, args.augment, args.ratio)

    # Initialize the model
    if args.model == "mozafari2018":
        model = MozafariMNIST2018(in_channels=in_channels, num_classes=num_classes, device=args.device, tensorboard=args.tensorboard)
        epochs = args.epochs[2]
        max_layers = 3
    elif args.model == "deep2024":
        model = DeepSNN(in_channels=in_channels, num_classes=num_classes, device=args.device, tensorboard=args.tensorboard)
        epochs = args.epochs[3]
        max_layers = 4
    elif args.model == "deepr2024":
        model = DeepRSNN(in_channels=in_channels, num_classes=num_classes, device=args.device, tensorboard=args.tensorboard)
        epochs = args.epochs[3]
        max_layers = 4
    elif args.model == "deepr2024_2":
        model = DeepRSNN2(in_channels=in_channels, num_classes=num_classes, device=args.device, tensorboard=args.tensorboard)
        epochs = args.epochs[3]
        max_layers = 4
    elif args.model == "inception2024":
        model = InceptionSNN(in_channels=in_channels, num_classes=num_classes, device=args.device, tensorboard=args.tensorboard)
        epochs = args.epochs[3]
        max_layers = 4
    elif args.model == "majority2024":
        model = MajoritySNN(in_channels=in_channels, num_classes=num_classes, device=args.device, tensorboard=args.tensorboard)
        epochs = args.epochs[2]
        max_layers = 3
    elif args.model == "resnet2024":
        model = ResSNN(in_channels=in_channels, num_classes=num_classes, device=args.device, tensorboard=args.tensorboard)
        epochs = args.epochs[3]
        max_layers = 4
    elif args.model == "deeper2024":
        model = DeeperSNN(in_channels=in_channels, num_classes=num_classes, device=args.device, tensorboard=args.tensorboard)
        epochs = args.epochs[4]
        max_layers = 5
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Move the model to the appropriate device
    model.to(args.device)

    if args.tensorboard:
        if args.augment:
            model.define_writer(log_dir=f"runs/{args.model}/{args.dataset}/augmented/ratio_{args.ratio}")
        elif args.ratio < 1.0:
            model.define_writer(log_dir=f"runs/{args.model}/{args.dataset}/ratio_{args.ratio}")
        else:
            model.define_writer(log_dir=f"runs/{args.model}/{args.dataset}")

    if args.debug:
        # Check if the model is on CUDA
        if utils.is_model_on_cuda(model):
            print("Model is on CUDA")
        else:
            print("Model is not on CUDA")

    # Register hooks for activation maps
    model.register_hooks()

    # Log model to TensorBoard
    if args.tensorboard:
        try:
            model.log_model(input_size=(15,in_channels,28,28))
        except RuntimeError:
            print("Model logging failed")

    #############################
    # Train the model           #
    #############################
    # First layer
    first_layer_name = f"models/{args.model}_{args.dataset}{'_augmented' if args.augment else ''}{'_ratio_' + str(args.ratio) if args.ratio < 1.0 else ''}_first_layer.pth"
    if os.path.isfile(first_layer_name):
        print("Loading first layer model")
        model.load_state_dict(
            torch.load(first_layer_name),
            strict = False
        )
    else:
        iterator = tqdm(range(args.epochs[0]), desc="Training First Layer")
        for epoch in iterator:
            i = 0
            iterator_epoch = tqdm(train_loader, desc=f"Epoch {epoch}", position=1, leave=False)
            for i, (data, _) in enumerate(iterator_epoch):
                data = data.to(args.device)  # Ensure data is on the correct device
                model.train_unsupervised(data, layer_idx=1)
                iterator.set_postfix({"Iteration": i+1})
                i += 1
        torch.save(model.state_dict(), first_layer_name)

    # second layer
    second_layer_name = f"models/{args.model}_{args.dataset}{'_augmented' if args.augment else ''}{'_ratio_' + str(args.ratio) if args.ratio < 1.0 else ''}_second_layer.pth"
    if os.path.isfile(second_layer_name):
        print("Loading second layer model")
        model.load_state_dict(
            torch.load(second_layer_name),
            strict = False
        )
    else:
        iterator = tqdm(range(args.epochs[1]), desc="Training Second Layer")
        for epoch in iterator:
            i = 0
            iterator_epoch = tqdm(train_loader, desc=f"Epoch {epoch}", position=1, leave=False)
            for data, _ in iterator_epoch:
                data = data.to(args.device)  # Ensure data is on the correct device
                model.train_unsupervised(data, layer_idx=2)
                iterator.set_postfix({"Iteration": i+1})
                i += 1
        # save model
        torch.save(model.state_dict(), second_layer_name)

    if args.model in ["deep2024", "deepr2024", "deepr2024_2", 'resnet2024', 'deeper2024']:
        # third layer
        third_layer_name = f"models/{args.model}_{args.dataset}{'_augmented' if args.augment else ''}{'_ratio_' + str(args.ratio) if args.ratio < 1.0 else ''}_third_layer.pth"
        if os.path.isfile(third_layer_name):
            print("Loading third layer model")
            model.load_state_dict(
                torch.load(third_layer_name),
                strict = False
            )
        else:
            iterator = tqdm(range(args.epochs[2]), desc="Training Third Layer")
            for epoch in iterator:
                i = 0
                iterator_epoch = tqdm(train_loader, desc=f"Epoch {epoch}", position=1, leave=False)
                for data, _ in iterator_epoch:
                    data = data.to(args.device)
                    model.train_unsupervised(data, layer_idx=3)
                    iterator.set_postfix({"Iteration": i+1})
                    i += 1
            # save model
            torch.save(model.state_dict(), third_layer_name)

    if args.model in ["inception2024"]:
        # third layer
        third_layer_name = f"models/{args.model}_{args.dataset}{'_augmented' if args.augment else ''}{'_ratio_' + str(args.ratio) if args.ratio < 1.0 else ''}_third_layer.pth"
        if os.path.isfile(third_layer_name):
            print("Loading third layer model")
            model.load_state_dict(
                torch.load(third_layer_name),
                strict = False
            )
        else:
            for branch in range(3):
                iterator = tqdm(range(args.epochs[2]), desc="Training Third Layer, Branch: " + str(branch))
                for epoch in iterator:
                    i = 0
                    iterator_epoch = tqdm(train_loader, desc=f"Epoch {epoch}", position=1, leave=False)
                    for data, _ in iterator_epoch:
                        data = data.to(args.device)
                        model.train_unsupervised(data, layer_idx=3, branch_idx = branch)
                        iterator.set_postfix({"Iteration": i+1})
                        i += 1
            # save model
            torch.save(model.state_dict(), third_layer_name)

    if args.model in ["deeper2024"]:
        # fourth layer
        fourth_layer_name = f"models/{args.model}_{args.dataset}{'_augmented' if args.augment else ''}{'_ratio_' + str(args.ratio) if args.ratio < 1.0 else ''}_fourth_layer.pth"
        if os.path.isfile(fourth_layer_name):
            print("Loading fourth layer model")
            model.load_state_dict(
                torch.load(fourth_layer_name),
                strict = False
            )
        else:
            iterator = tqdm(range(args.epochs[3]), desc="Training Fourth Layer")
            for epoch in iterator:
                i = 0
                iterator_epoch = tqdm(train_loader, desc=f"Epoch {epoch}", position=1, leave=False)
                for data, _ in iterator_epoch:
                    data = data.to(args.device)
                    model.train_unsupervised(data, layer_idx=4)
                    iterator.set_postfix({"Iteration": i+1})
                    i += 1
            # save model
            torch.save(model.state_dict(), fourth_layer_name)

    # # Log initial embeddings
    # embeddings, metadata, label_img = utils.get_embeddings_metadata(model, train_loader, args.device, max_layers)
    # model.writer.add_embedding(embeddings, metadata, label_img, global_step=0, tag='Embeddings')

    # Train the R-STDP layer
    # Set learning rates
    if args.model in ["deepr2024", "deepr2024_2"]:
        model.reset_learning_rates(3)

    adaptive_min, adaptive_int = (0, 1)
    if args.model in ["mozafari2018", "deepr2024", "deepr2024_2"]:
        apr3 = model.block3['stdp'].learning_rate[0][0].item()
        anr3 = model.block3['stdp'].learning_rate[0][1].item()
        app3 = model.block3['anti_stdp'].learning_rate[0][1].item()
        anp3 = model.block3['anti_stdp'].learning_rate[0][0].item()
    if args.model in ["deep2024", "deepr2024", "deepr2024_2", "inception2024", "resnet2024"]:
        apr4 = model.block4['stdp'].learning_rate[0][0].item()
        anr4 = model.block4['stdp'].learning_rate[0][1].item()
        app4 = model.block4['anti_stdp'].learning_rate[0][1].item()
        anp4 = model.block4['anti_stdp'].learning_rate[0][0].item()
    if args.model in ["majority2024"]:
        apr3_1 = model.block3_1['stdp'].learning_rate[0][0].item()
        anr3_1 = model.block3_1['stdp'].learning_rate[0][1].item()
        app3_1 = model.block3_1['anti_stdp'].learning_rate[0][1].item()
        anp3_1 = model.block3_1['anti_stdp'].learning_rate[0][0].item()
        apr3_2 = model.block3_2['stdp'].learning_rate[0][0].item()
        anr3_2 = model.block3_2['stdp'].learning_rate[0][1].item()
        app3_2 = model.block3_2['anti_stdp'].learning_rate[0][1].item()
        anp3_2 = model.block3_2['anti_stdp'].learning_rate[0][0].item()
        apr3_3 = model.block3_3['stdp'].learning_rate[0][0].item()
        anr3_3 = model.block3_3['stdp'].learning_rate[0][1].item()
        app3_3 = model.block3_3['anti_stdp'].learning_rate[0][1].item()
        anp3_3 = model.block3_3['anti_stdp'].learning_rate[0][0].item()
    if args.model in ["deeper2024"]:
        apr5 = model.block5['stdp'].learning_rate[0][0].item()
        anr5 = model.block5['stdp'].learning_rate[0][1].item()
        app5 = model.block5['anti_stdp'].learning_rate[0][1].item()
        anp5 = model.block5['anti_stdp'].learning_rate[0][0].item()

    # performance
    best_train = np.array([0., 0., 0., 0.]) # correct, total, loss, epoch
    best_test = np.array([0., 0., 0., 0.]) # correct, total, loss, epoch

    try:
        iterator = tqdm(range(epochs), desc="Training R STDP Layer")
        for epoch in iterator:
            model.epoch = epoch
            perf_train = np.array([0.0, 0.0, 0.0])
            total_correct = 0
            total_loss = 0
            total_samples = 0
            i = 0
            iterator_epoch = tqdm(train_loader, desc=f"Training epoch {epoch}", position=1, leave=False)
            for k, (data, targets) in enumerate(iterator_epoch):
                # if k == 0:
                #     model.log_inputs(data, epoch)

                perf_train_batch = model.train_rl(data, targets, layer_idx=max_layers)
                # utils.memory_usage()
                iterator_epoch.set_postfix({"Performance": perf_train_batch})

                if args.model in ["mozafari2018", "deepr2024", "deepr2024_2"]:
                    apr_adapt3 = apr3 * (perf_train_batch[1] * adaptive_int + adaptive_min)
                    anr_adapt3 = anr3 * (perf_train_batch[1] * adaptive_int + adaptive_min)
                    app_adapt3 = app3 * (perf_train_batch[0] * adaptive_int + adaptive_min)
                    anp_adapt3 = anp3 * (perf_train_batch[0] * adaptive_int + adaptive_min)

                    model.update_learning_rates(
                        stdp_ap = apr_adapt3,
                        stdp_an = anr_adapt3,
                        anti_stdp_ap = app_adapt3,
                        anti_stdp_an = anp_adapt3,
                        layer_idx = 3
                    )
                if args.model in ["deep2024", "deepr2024", "deepr2024_2", "inception2024", "resnet2024"]:
                    apr_adapt4 = apr4 * (perf_train_batch[1] * adaptive_int + adaptive_min)
                    anr_adapt4 = anr4 * (perf_train_batch[1] * adaptive_int + adaptive_min)
                    app_adapt4 = app4 * (perf_train_batch[0] * adaptive_int + adaptive_min)
                    anp_adapt4 = anp4 * (perf_train_batch[0] * adaptive_int + adaptive_min)

                    model.update_learning_rates(
                        stdp_ap = apr_adapt4,
                        stdp_an = anr_adapt4,
                        anti_stdp_ap = app_adapt4,
                        anti_stdp_an = anp_adapt4,
                        layer_idx = 4
                    )

                if args.model in ["deeper2024"]:
                    apr_adapt5 = apr5 * (perf_train_batch[1] * adaptive_int + adaptive_min)
                    anr_adapt5 = anr5 * (perf_train_batch[1] * adaptive_int + adaptive_min)
                    app_adapt5 = app5 * (perf_train_batch[0] * adaptive_int + adaptive_min)
                    anp_adapt5 = anp5 * (perf_train_batch[0] * adaptive_int + adaptive_min)

                    model.update_learning_rates(
                        stdp_ap = apr_adapt5,
                        stdp_an = anr_adapt5,
                        anti_stdp_ap = app_adapt5,
                        anti_stdp_an = anp_adapt5,
                        layer_idx = 5
                    )

                if args.model in ["majority2024"]:
                    apr_adapt3_1 = apr3_1 * (perf_train_batch[1] * adaptive_int + adaptive_min)
                    anr_adapt3_1 = anr3_1 * (perf_train_batch[1] * adaptive_int + adaptive_min)
                    app_adapt3_1 = app3_1 * (perf_train_batch[0] * adaptive_int + adaptive_min)
                    anp_adapt3_1 = anp3_1 * (perf_train_batch[0] * adaptive_int + adaptive_min)
                    apr_adapt3_2 = apr3_2 * (perf_train_batch[1] * adaptive_int + adaptive_min)
                    anr_adapt3_2 = anr3_2 * (perf_train_batch[1] * adaptive_int + adaptive_min)
                    app_adapt3_2 = app3_2 * (perf_train_batch[0] * adaptive_int + adaptive_min)
                    anp_adapt3_2 = anp3_2 * (perf_train_batch[0] * adaptive_int + adaptive_min)
                    apr_adapt3_3 = apr3_3 * (perf_train_batch[1] * adaptive_int + adaptive_min)
                    anr_adapt3_3 = anr3_3 * (perf_train_batch[1] * adaptive_int + adaptive_min)
                    app_adapt3_3 = app3_3 * (perf_train_batch[0] * adaptive_int + adaptive_min)
                    anp_adapt3_3 = anp3_3 * (perf_train_batch[0] * adaptive_int + adaptive_min)

                    model.update_learning_rates(
                        stdp_ap = apr_adapt3_1,
                        stdp_an = anr_adapt3_1,
                        anti_stdp_ap = app_adapt3_1,
                        anti_stdp_an = anp_adapt3_1,
                        layer_idx = 3,
                        branch_idx = 0
                    )
                    model.update_learning_rates(
                        stdp_ap = apr_adapt3_2,
                        stdp_an = anr_adapt3_2,
                        anti_stdp_ap = app_adapt3_2,
                        anti_stdp_an = anp_adapt3_2,
                        layer_idx = 3,
                        branch_idx = 1
                    )
                    model.update_learning_rates(
                        stdp_ap = apr_adapt3_3,
                        stdp_an = anr_adapt3_3,
                        anti_stdp_ap = app_adapt3_3,
                        anti_stdp_an = anp_adapt3_3,
                        layer_idx = 3,
                        branch_idx = 2
                    )

                perf_train += perf_train_batch

                total_correct += perf_train_batch[0]
                total_loss += perf_train_batch[1]
                total_samples += np.sum(perf_train_batch)
                i += 1

                iterator.set_postfix({"Performance": np.round(perf_train / (i + 1), 2)})

            # print(model.get_spike_counts())

            perf_train /= len(train_loader)
            if best_train[0] <= perf_train[0]:
                best_train = np.append(perf_train, epoch)

            # Log training performance to TensorBoard
            if args.tensorboard:
                model.writer.add_scalar('Train/Accuracy', total_correct / total_samples, epoch)
                model.writer.add_scalar('Train/Loss', total_loss / total_samples, epoch)
            model.history['train_acc'].append(total_correct / total_samples)
            model.history['train_loss'].append(total_loss / total_samples)

            total_correct = 0
            total_loss = 0
            total_samples = 0
            for data, targets in test_loader:
                data = data.to(args.device)  # Ensure data is on the correct device
                targets = targets.to(args.device)  # Ensure targets are on the correct device
                perf_test = model.test(data, targets, layer_idx=max_layers)
                if best_test[0] <= perf_test[0]:
                    best_test = np.append(perf_test, epoch)
                    torch.save(model.state_dict(), f"models/{args.model}_{args.dataset}_best.pth")

                total_correct += perf_test[0]
                total_loss += perf_test[1]
                total_samples += np.sum(perf_test)

            # Log test performance to TensorBoard
            if args.tensorboard:
                model.writer.add_scalar('Test/Accuracy', total_correct / total_samples, epoch)
                model.writer.add_scalar('Test/Loss', total_loss / total_samples, epoch)
            model.history['test_acc'].append(total_correct / total_samples)
            model.history['test_loss'].append(total_loss / total_samples)

            # Log additional metrics to TensorBoard
            model.all_preds = []
            model.all_targets = []
            for data, targets in metrics_loader:
                data = data.to(args.device)
                targets = targets.to(args.device)
                model.compute_preds(data, targets, layer_idx=max_layers)
            metrics = model.metrics()
            model.log_tensorboard(metrics, epoch)

            # # Log embeddings at the end of each epoch
            # embeddings, metadata, label_img = utils.get_embeddings_metadata(model, train_loader, args.device, max_layers)
            # model.writer.add_embedding(embeddings, metadata, label_img, global_step=epoch, tag='Embeddings')

            if epoch - best_test[3] > 5:
                break

    except KeyboardInterrupt:
        # model.file.close()
        print("Training Interrupted")
        print("Best Train:", best_train)
        print("Best Test:", best_test)

    print("Best Train:", best_train)
    print("Best Test:", best_test)

    # Save training history
    model.save_history(file_path=f"models/{args.model}_{args.dataset}_history.csv")

    # Plot training history
    model.plot_history(file_path=f"models/{args.model}_{args.dataset}_history.png")

    # Save activation maps
    model.save_activation_maps(file_path=f"models/{args.model}_{args.dataset}_activation_maps")

    # Close TensorBoard writer
    model.close_tensorboard()

if __name__ == "__main__":
    main()    main()
