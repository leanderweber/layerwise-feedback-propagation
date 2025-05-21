import contextlib
import logging
import os
import random
import sys
import time
import warnings
from argparse import ArgumentParser
from types import SimpleNamespace

import joblib
import numpy as np
import torch
import torchvision
import wandb
import yaml
from tqdm import tqdm

from experiment_utils.data import dataloaders, datasets, transforms
from experiment_utils.evaluation import evaluate
from experiment_utils.model import models
from experiment_utils.utils.utils import set_random_seeds
from lfprop.propagation import propagator_lxt, propagator_vit
from lfprop.rewards import rewards

warnings.filterwarnings("ignore")

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DummyFile(object):
    def write(self, x):
        pass


def gini_idx(param):
    param = param.detach().abs()
    sortedparam, indices = torch.sort(param.view(-1))

    sortedidx = torch.arange(0, sortedparam.numel(), 1).to(param.device)

    gini_numerator = 2 * (sortedparam * sortedidx).sum()
    gini_denominator = sortedparam.numel() * sortedparam.sum()
    gini_addendum = (sortedparam.numel() + 1) / sortedparam.numel()

    return gini_numerator / gini_denominator - gini_addendum


def cosine_similarity(a, b):
    numer = (a * b).sum()
    denom = (a**2).sum() ** 0.5 * (b**2).sum() ** 0.5
    cossim = numer / denom

    return cossim


@contextlib.contextmanager
def nostdout(verbose=True):
    if verbose:
        yield
    else:
        save_stdout = sys.stdout
        sys.stdout = DummyFile()
        yield
        sys.stdout = save_stdout


def get_head(model):
    if isinstance(model, torchvision.models.VGG) or isinstance(model, torchvision.models.efficientnet.EfficientNet):
        head = [m for m in model.classifier.modules() if not isinstance(m, torch.nn.Sequential)][-1]
    elif isinstance(model, torchvision.models.ResNet) or isinstance(model, torchvision.models.Inception3):
        head = model.fc
    else:
        head = model.classifier[-1]
    return head


class Trainer:
    def __init__(
        self,
        model_name,
        model,
        optimizer,
        criterion,
        device,
        batch_size,
        class_labels,
        scheduler=None,
        lfp_composite=None,
        schedule_lr_every_step=False,
        clip_updates=False,
        clip_update_threshold=2.0,
        dummy_input=None,
    ):
        self.model_name = model_name
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.batch_size = batch_size
        self.class_labels = class_labels
        self.lfp_composite = lfp_composite
        self.schedule_lr_every_step = schedule_lr_every_step
        self.clip_updates = clip_updates
        self.clip_update_threshold = clip_update_threshold
        self.global_epoch = 0
        self.global_step = 0
        self.acc_log = {"train": [], "base": [], "transfer": []}
        self.sparsity_log = {}
        self.clock_times = []
        self.last_param_updates = {}
        self.best_acc = 0
        self.sparsity_func = gini_idx
        self.dummy_input = dummy_input

    def grad_step(self, batch):
        self.model.train()
        with torch.enable_grad():
            self.optimizer.zero_grad()

            start1 = time.time()
            inputs, labels, outputs = self.model.forward_fn(batch, self.model, lfp_step=False)
            loss = self.criterion(outputs, labels)
            loss.backward()
            end1 = time.time()

            if self.clip_updates:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_update_threshold, 2.0)

            start2 = time.time()
            self.optimizer.step()
            end2 = time.time()

            clock_time = end2 - start2 + end1 - start1
            self.clock_times.append(clock_time)

        self.model.eval()

        self.global_step += 1

    def lfp_step(self, batch):
        self.model.train()

        with torch.enable_grad():
            self.optimizer.zero_grad()
            with self.lfp_composite.context(self.model, dummy_inputs=self.dummy_input) as modified:
                if self.global_step == 0:
                    print(modified)

                start1 = time.time()
                inputs, labels, outputs = self.model.forward_fn(batch, modified, lfp_step=True)

                # Calculate reward
                # Do like this to avoid tensors being kept in memory
                reward = torch.from_numpy(self.criterion(outputs, labels).detach().cpu().numpy()).to(device)

                # Write LFP Values into .grad attributes
                _ = torch.autograd.grad((outputs,), (inputs,), grad_outputs=(reward,), retain_graph=False)[0]
                end1 = time.time()

                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        param.grad = -param.feedback

                if self.clip_updates:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_update_threshold, 2.0)

                start2 = time.time()
                self.optimizer.step()
                end2 = time.time()

                clock_time = end2 - start2 + end1 - start1
                self.clock_times.append(clock_time)

        self.model.eval()

        self.global_step += 1

    def train(
        self,
        epochs,
        train_loader,
        test_loader,
        verbose=False,
        batch_log=False,
        param_sparsity_log=False,
        savepath=None,
        savename="ckpt",
        saveappendage="last",
        savefrequency=1,
        fromscratch=False,
    ):
        if not fromscratch and savepath:
            self.load(savepath, savename, saveappendage)

        eval_stats_train = self.eval(train_loader)
        eval_stats_test = self.eval(test_loader)

        print(
            "Train: Initial Eval: (Criterion) {:.2f}; (Accuracy) {:.2f}".format(
                float(eval_stats_train["criterion"]),
                (
                    float(eval_stats_train["accuracy_p050"])
                    if "accuracy_p050" in eval_stats_train.keys()
                    else float(eval_stats_train["micro_accuracy_top1"])
                ),
            )
        )

        print(
            "Test: Initial Eval: (Criterion) {:.2f}; (Accuracy) {:.2f}".format(
                float(np.mean(eval_stats_test["criterion"])),
                (
                    float(eval_stats_test["accuracy_p050"])
                    if "accuracy_p050" in eval_stats_test.keys()
                    else float(eval_stats_test["micro_accuracy_top1"])
                ),
            )
        )

        logdict = {"epoch": 0}
        logdict.update({"train_" + k: v for k, v in eval_stats_train.items()})
        logdict.update({"test_" + k: v for k, v in eval_stats_test.items()})
        logdict.update({"total_training_time": np.sum(self.clock_times)})
        wandb.log(logdict)

        # Store Initial State
        if savepath and epochs > 0:
            self.save(savepath, savename, "init")

        if param_sparsity_log:
            for name, param in self.model.named_parameters():
                if name not in self.sparsity_log.keys():
                    self.sparsity_log[name] = []
                self.sparsity_log[name].append(self.sparsity_func(param).detach().cpu().numpy())

        for epoch in range(epochs):
            with tqdm(total=len(train_loader), disable=not verbose) as pbar:
                for index, batch in enumerate(train_loader):
                    if self.lfp_composite is None:
                        # Grad Step
                        self.grad_step(batch)
                    else:
                        # LFP Step
                        self.lfp_step(batch)

                    if self.scheduler is not None and self.schedule_lr_every_step:
                        self.scheduler.step()

                    if batch_log and epoch == 0:
                        eval_stats_train = self.eval(train_loader)
                        eval_stats_test = self.eval(test_loader)
                        self.acc_log["train"].append(
                            float(eval_stats_train["accuracy_p050"])
                            if "accuracy_p050" in eval_stats_train.keys()
                            else float(eval_stats_train["micro_accuracy_top1"])
                        )
                        self.acc_log["test"].append(
                            float(eval_stats_test["accuracy_p050"])
                            if "accuracy_p050" in eval_stats_test.keys()
                            else float(eval_stats_test["micro_accuracy_top1"])
                        )

                        wandb.log(
                            {
                                "step": index + 1,
                                "acc_log_train": (
                                    float(eval_stats_train["accuracy_p050"])
                                    if "accuracy_p050" in eval_stats_train.keys()
                                    else float(eval_stats_train["micro_accuracy_top1"])
                                ),
                                "acc_log_test": (
                                    float(eval_stats_test["accuracy_p050"])
                                    if "accuracy_p050" in eval_stats_test.keys()
                                    else float(eval_stats_test["micro_accuracy_top1"])
                                ),
                            }
                        )

                    if param_sparsity_log:
                        for name, param in self.model.named_parameters():
                            if name not in self.sparsity_log.keys():
                                self.sparsity_log[name] = []
                            self.sparsity_log[name].append(self.sparsity_func(param).detach().cpu().numpy())

                    pbar.update(1)

            if self.scheduler is not None and not self.schedule_lr_every_step:
                self.scheduler.step()

            eval_stats_train = self.eval(train_loader)
            eval_stats_test = self.eval(test_loader)

            print(
                "Train: Epoch {}/{}: (Criterion) {:.2f}; (Accuracy) {:.2f}".format(
                    epoch + 1,
                    epochs,
                    float(eval_stats_train["criterion"]),
                    (
                        float(eval_stats_train["accuracy_p050"])
                        if "accuracy_p050" in eval_stats_train.keys()
                        else float(eval_stats_train["micro_accuracy_top1"])
                    ),
                )
            )

            print(
                "Test: Epoch {}/{}: (Criterion) {:.2f}; (Accuracy) {:.2f}".format(
                    epoch + 1,
                    epochs,
                    float(eval_stats_test["criterion"]),
                    (
                        float(eval_stats_test["accuracy_p050"])
                        if "accuracy_p050" in eval_stats_test.keys()
                        else float(eval_stats_test["micro_accuracy_top1"])
                    ),
                )
            )

            logdict = {"epoch": epoch + 1}
            logdict.update({"train_" + k: v for k, v in eval_stats_train.items()})
            logdict.update({"test_" + k: v for k, v in eval_stats_test.items()})
            logdict.update({"total_training_time": np.sum(self.clock_times)})
            wandb.log(logdict)

            self.global_epoch += 1

            if savepath:
                if epoch % savefrequency == 0:
                    self.save(savepath, savename, f"ep-{epoch + 1}")
                self.save(savepath, savename, "last")

                accuracy = (
                    float(eval_stats_test["accuracy_p050"])
                    if "accuracy_p050" in eval_stats_test.keys()
                    else float(eval_stats_test["micro_accuracy_top1"])
                )
                if accuracy > self.best_acc:
                    self.save(savepath, savename, "best")
                    self.best_acc = accuracy

    def eval(self, loader):
        print("Evaluating...")

        return_dict = evaluate.evaluate(self.model, loader, len(self.class_labels), self.criterion, device)

        return return_dict

    def save(self, savepath, savename, saveappendage):
        checkpoint = {
            "epoch": self.global_epoch,
            "step": self.global_step,
            "random_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(self.device),
                "numpy": np.random.get_state(),
                "random": random.getstate(),
            },
            "best_acc": self.best_acc,
        }
        if self.model.is_huggingface:
            self.model.save_pretrained(os.path.join(savepath, f"model_{savename}-{saveappendage}"))
        else:
            checkpoint["model"] = self.model.state_dict()
        if self.optimizer:
            checkpoint["optimizer"] = self.optimizer.state_dict()
        if self.scheduler:
            checkpoint["scheduler"] = self.scheduler.state_dict()
        if self.acc_log:
            checkpoint["acc_log"] = self.acc_log
        if self.sparsity_log:
            checkpoint["sparsity_log"] = self.sparsity_log
        if self.clock_times:
            checkpoint["clock_times"] = self.clock_times

        torch.save(checkpoint, os.path.join(savepath, f"{savename}-{saveappendage}.pt"))

    def load(self, savepath, savename, saveappendage):
        if os.path.exists(os.path.join(savepath, f"{savename}-{saveappendage}.pt")):
            checkpoint = torch.load(os.path.join(savepath, f"{savename}-{saveappendage}.pt"))
            if self.model:
                self.model.load_state_dict(checkpoint["model"])
            if self.optimizer:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            if "acc_log" in checkpoint:
                self.acc_log = checkpoint["acc_log"]
            if "sparsity_log" in checkpoint:
                self.sparsity_log = checkpoint["sparsity_log"]
            if "best_acc" in checkpoint:
                self.best_acc = checkpoint["best_acc"]
            if "clock_times" in checkpoint:
                self.clock_times = checkpoint["clock_times"]
            self.global_epoch = checkpoint["epoch"]
            self.global_step = checkpoint["step"]

            torch.set_rng_state(checkpoint["random_state"]["torch"])
            torch.cuda.set_rng_state(checkpoint["random_state"]["cuda"], device)
            np.random.set_state(checkpoint["random_state"]["numpy"])
            random.setstate(checkpoint["random_state"]["random"])

        if os.path.exists(os.path.join(savepath, f"model_{savename}-{saveappendage}")):
            model_checkpoint = os.path.join(savepath, f"model_{savename}-{saveappendage}")
            self.model = models.get_model(
                self.model_name,
                device,
                class_labels=self.class_labels,
                model_checkpoint=model_checkpoint,
            )

        else:
            print("No checkpoint found... not loading anything.")


def run_training_base(
    savepath,
    data_path,
    dataset_name,
    lr,
    propagator_name,
    batch_size=128,
    pretrained_model=True,
    n_channels=3,
    n_outputs=10,
    momentum=0.9,
    weight_decay=0.0,
    scheduler_name="none",
    clip_updates=False,
    clip_update_threshold=2.0,
    reward_name="correct-class",
    reward_kwargs={},
    loss_name="ce-loss",
    epochs=5,
    model_name="cifar-vgglike",
    default_model_checkpoint="google/vit-base-patch16-224-in21k",
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
        "model_name": model_name,
        "default_model_checkpoint": default_model_checkpoint,
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
    with nostdout(verbose=verbose):
        train_dataset, collate_fn, dummy_input, class_labels = datasets.get_dataset(
            dataset_name,
            data_path,
            transforms.get_transforms(dataset_name, "train", model_path=default_model_checkpoint),
            mode="train",
        )
        test_dataset, _, _, _ = datasets.get_dataset(
            dataset_name,
            data_path,
            transforms.get_transforms(dataset_name, "test", model_path=default_model_checkpoint),
            mode="test",
        )

    train_loader = dataloaders.get_dataloader(
        dataset_name, train_dataset, batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = dataloaders.get_dataloader(
        dataset_name, test_dataset, batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Propagation Composite
    propagator = propagator_vit if model_name == "vit" else propagator_lxt
    propagation_composites = {
        "lfp-epsilon": propagator.LFPEpsilonComposite(),
        "vanilla-gradient": None,
    }
    propagation_composite = propagation_composites[propagator_name]

    # Model
    model = models.get_model(
        model_name,
        device,
        n_channels=n_channels,
        n_outputs=n_outputs,
        replace_last_layer=True,
        activation=activation,
        pretrained_model=pretrained_model,
        model_checkpoint=default_model_checkpoint,
        class_labels=class_labels,
    )

    # Note: We aim to finetune the model here, so we set the embedding parameters to not require grad.
    if model_name == "vit":
        for name, param in model.vit.embeddings.named_parameters():
            param.requires_grad = False
    model.to(device)

    # Optimization
    optimizers = {
        "sgd": torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay),
        "adam": torch.optim.Adam(model.parameters(), lr=lr),
        "adamw": torch.optim.AdamW(model.parameters(), lr=lr),
    }
    optimizer = optimizers[optimizer_name]

    # LR Scheduling
    schedulers = {
        "none": (None, True),
        "onecyclelr": (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                lr,
                (1 if epochs == 0 else epochs * int(np.ceil(len(train_dataset) / batch_size))),
            ),
            True,
        ),
        "cycliclr": (
            torch.optim.lr_scheduler.CyclicLR(optimizer, lr * 0.001, lr),
            True,
        ),
        "steplr": (
            torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.94),
            False,
        ),
    }
    scheduler, schedule_lr_every_step = schedulers[scheduler_name]

    trainer = Trainer(
        model_name=model_name,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=(
            rewards.get_reward(reward_name, device, **reward_kwargs)
            if propagation_composite is not None
            else rewards.get_reward(loss_name, device, **reward_kwargs)
        ),
        device=device,
        batch_size=batch_size,
        class_labels=class_labels,
        lfp_composite=propagation_composite,
        schedule_lr_every_step=schedule_lr_every_step,
        clip_updates=clip_updates,
        clip_update_threshold=clip_update_threshold,
        dummy_input=dummy_input,
    )

    print("Training...")
    saveappendage = "last"
    savename = "model"
    trainer.train(
        epochs=epochs,
        train_loader=train_loader,
        test_loader=test_loader,
        verbose=verbose,
        batch_log=batch_log,
        param_sparsity_log=param_sparsity_log,
        savepath=ckpt_path,
        savename=savename,
        saveappendage=saveappendage,
        fromscratch=True,
    )

    # Eval base accuracy
    res_base = trainer.eval(test_loader)

    print(
        "Test Accuracies: {:.2f}".format(
            float(res_base["accuracy_p050"])
            if "accuracy_p050" in res_base.keys()
            else float(res_base["micro_accuracy_top1"])
        )
    )

    return trainer


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

    run_training_base(
        savepath=config.savepath,
        data_path=config.data_path,
        dataset_name=config.dataset_name,
        lr=config.lr,
        n_channels=config.n_channels,
        n_outputs=config.n_outputs,
        propagator_name=config.propagator_name,
        batch_size=config.batch_size,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        scheduler_name=config.scheduler_name,
        clip_updates=config.clip_updates,
        clip_update_threshold=config.clip_update_threshold,
        reward_name=config.reward_name,
        reward_kwargs=config.reward_kwargs,
        loss_name=config.loss_name,
        epochs=config.epochs,
        model_name=config.model_name,
        default_model_checkpoint=config.default_model_checkpoint,
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
