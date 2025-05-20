import unittest

# taken from nbs/investigate-simple-rewards.ipynb
import copy
import os
import random

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from tqdm import tqdm

from experiment_utils.data import dataloaders, datasets, transforms
from experiment_utils.evaluation import evaluate
from experiment_utils.model import models
from experiment_utils.model.model_definitions import ToyDataDense
from experiment_utils.utils.utils import register_backward_normhooks
from lfprop.propagation import propagator_lxt as propagator
from lfprop.rewards import rewards

dataset_name = "blobs"
n_channels = 2
model_name = "toydata-dense"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
lrs = np.sort(
    np.array(
        [np.arange(1, 2, 1, dtype=float) * 10**x for x in [-4]]
    ).flatten()
)
momentum = 0.95
reward_name = "misclassificationreward"
norm_backward = False
epochs = 10
n_datasets = 5

savepath = "rewards_results"
data_path = "rewards_data"

retrain_models = False
redraw_data = False

class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        test_dataset,
        val_dataset,
        optimizer,
        criterion,
        device,
        lfp_composite=None,
        norm_backward=False,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = None
        self.device = device
        self.lfp_composite = lfp_composite
        self.norm_backward = norm_backward
        self.global_epoch = 0
        self.global_step = 0

        self.train_loader = dataloaders.get_dataloader(
            dataset_name=dataset_name,
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        self.test_loader = dataloaders.get_dataloader(
            dataset_name=dataset_name,
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        self.val_loader = dataloaders.get_dataloader(
            dataset_name=dataset_name,
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        self.accuracy_log = {"train": [], "test": []}

    def grad_step(self, inputs, labels):
        # Backward norm
        if self.norm_backward:
            norm_handles = register_backward_normhooks(self.model)
        else:
            norm_handles = []

        self.model.train()
        with torch.enable_grad():
            self.optimizer.zero_grad()
            out = self.model(inputs)

            reward = self.criterion(out, labels)
            reward.backward()

            self.optimizer.step()

        self.model.eval()

        for handle in norm_handles:
            handle.remove()

        self.global_step += 1

    def lfp_step(self, inputs, labels):
        self.model.train()

        with torch.enable_grad():
            self.optimizer.zero_grad()
            with self.lfp_composite.context(self.model) as modified:
                inputs = inputs.detach().requires_grad_(True)
                outputs = modified(inputs)

                # Calculate reward
                # Do like this to avoid tensors being kept in memory
                reward = torch.from_numpy(self.criterion(outputs, labels).detach().cpu().numpy()).to(device)

                # Write LFP Values into .grad attributes
                torch.autograd.grad((outputs,), (inputs,), grad_outputs=(reward,), retain_graph=False)[0]

                for name, param in self.model.named_parameters():
                    param.grad = -param.feedback

                self.optimizer.step()

        self.model.eval()

        self.global_step += 1

    def train(
        self,
        epochs,
        verbose=False,
        savepath=None,
        savename="ckpt",
        saveappendage="last",
        fromscratch=False,
    ):
        if not fromscratch and savepath:
            self.load(savepath, savename, saveappendage)

        for epoch in range(epochs):
            with tqdm(total=len(self.train_loader), disable=not verbose) as pbar:
                for index, (inputs, labels) in enumerate(self.train_loader):
                    inputs = inputs.to(device)
                    labels = torch.tensor(labels).to(device)

                    if self.lfp_composite is None:
                        # Grad Step
                        self.grad_step(inputs, labels)
                    else:
                        # LFP Step
                        self.lfp_step(inputs, labels)

                    pbar.update(1)

                    if self.global_step % 5 == 0:
                        eval_stats = self.eval(datamodes=["train", "test"])
                        self.accuracy_log["train"].append(
                            (
                                float(eval_stats["train"]["accuracy_p050"])
                                if "accuracy_p050" in eval_stats["train"].keys()
                                else float(eval_stats["train"]["micro_accuracy_top1"])
                            ),
                        )
                        self.accuracy_log["test"].append(
                            (
                                float(eval_stats["test"]["accuracy_p050"])
                                if "accuracy_p050" in eval_stats["test"].keys()
                                else float(eval_stats["test"]["micro_accuracy_top1"])
                            ),
                        )

            if verbose:
                eval_stats = self.eval(datamodes=["train", "test"])

                print(
                    "Epoch {}/{}: (Train Criterion) {:.2f}; (Train Accuracy) {:.2f}; (Test Criterion) {:.2f};"
                    "(Test Accuracy) {:.2f}".format(
                        epoch + 1,
                        epochs,
                        float(np.mean(eval_stats["train"]["criterion"])),
                        (
                            float(eval_stats["train"]["accuracy_p050"])
                            if "accuracy_p050" in eval_stats["train"].keys()
                            else float(eval_stats["train"]["micro_accuracy_top1"])
                        ),
                        float(np.mean(eval_stats["test"]["criterion"])),
                        (
                            float(eval_stats["test"]["accuracy_p050"])
                            if "accuracy_p050" in eval_stats["test"].keys()
                            else float(eval_stats["test"]["micro_accuracy_top1"])
                        ),
                    )
                )

            self.global_epoch += 1

            if savepath:
                self.save(savepath, savename, "last")

    def eval(self, datamodes=["train", "test", "val"]):
        return_dict = {}

        if "train" in datamodes:
            return_data = evaluate.evaluate(self.model, self.train_loader, self.criterion, device)
            return_dict["train"] = return_data
        if "test" in datamodes:
            return_data = evaluate.evaluate(self.model, self.test_loader, self.criterion, device)
            return_dict["test"] = return_data
        if "val" in datamodes:
            return_data = evaluate.evaluate(self.model, self.val_loader, self.criterion, device)
            return_dict["val"] = return_data

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
        }
        if self.model:
            checkpoint["model"] = self.model.state_dict()
        if self.optimizer:
            checkpoint["optimizer"] = self.optimizer.state_dict()
        if self.scheduler:
            checkpoint["scheduler"] = self.scheduler.state_dict()
        if self.accuracy_log:
            checkpoint["accuracy_log"] = self.accuracy_log

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
            if "accuracy_log" in checkpoint:
                self.accuracy_log = checkpoint["accuracy_log"]
            self.global_epoch = checkpoint["epoch"]
            self.global_step = checkpoint["step"]

            torch.set_rng_state(checkpoint["random_state"]["torch"])
            torch.cuda.set_rng_state(checkpoint["random_state"]["cuda"], device)
            np.random.set_state(checkpoint["random_state"]["numpy"])
            random.setstate(checkpoint["random_state"]["random"])

        else:
            print("No checkpoint found... not loading anything.")

os.makedirs(data_path, exist_ok=True)
train_datasets = []
for d in range(n_datasets):
    train_datasets.append(
        datasets.get_dataset(
            dataset_name,
            os.path.join(data_path, f"{dataset_name}-train-{d}.json"),
            transforms.get_transforms(dataset_name, "train"),
            mode="train",
            redraw=redraw_data,
        )
    )
test_dataset = datasets.get_dataset(
    dataset_name,
    os.path.join(data_path, f"{dataset_name}-test.json"),
    transforms.get_transforms(dataset_name, "test"),
    mode="test",
    redraw=redraw_data,
)
val_dataset = datasets.get_dataset(
    dataset_name,
    os.path.join(data_path, f"{dataset_name}-val.json"),
    transforms.get_transforms(dataset_name, "test"),
    mode="test",
    redraw=redraw_data,
)

propagation_composites = {
    "lfp-epsilon": propagator.LFPEpsilonComposite(
        norm_backward=norm_backward,
    ),
}

if retrain_models or not os.path.exists(os.path.join(savepath, "init-ckpts", "initmodel-0.pt")):
    init_models = [models.get_model(model_name, n_channels, d.num_classes, device) for d in train_datasets]
    os.makedirs(os.path.join(savepath, "init-ckpts"), exist_ok=True)
    for m, model in enumerate(init_models):
        torch.save(
            model.state_dict(),
            os.path.join(savepath, "init-ckpts", f"initmodel-{m}.pt"),
        )
else:
    init_models = [models.get_model(model_name, n_channels, d.num_classes, device) for d in train_datasets]
    for m, model in enumerate(init_models):
        statedict = torch.load(os.path.join(savepath, "init-ckpts", f"initmodel-{m}.pt"))
        model.load_state_dict(statedict)

class TestRewardsIntegration(unittest.TestCase):

    def test_all(self):
        trainers = {}
        for lr in lrs:
            trainers[lr] = {}
            for name, prop_comp in propagation_composites.items():
                trainers[lr][name] = []
                for d in train_datasets:
                    model = copy.deepcopy(init_models[m])
                    self.assertIsNotNone(rewards.get_reward(reward_name, device))
                    trainers[lr][name].append(
                        Trainer(
                            model=model,
                            train_dataset=d,
                            test_dataset=test_dataset,
                            val_dataset=val_dataset,
                            optimizer=torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum),
                            criterion=rewards.get_reward(reward_name, device),
                            device=device,
                            lfp_composite=prop_comp,
                            norm_backward=norm_backward,
                        )
                    )


'''
# Plot Data
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

X = np.array([s[0] for s in train_datasets[0].samples])
y = [s[1] for s in train_datasets[0].samples]
ax.scatter(X[:, 0], X[:, 1], c=y, cmap="cool", edgecolors="k")


X = np.array([s[0] for s in test_dataset.samples])
y = [s[1] for s in test_dataset.samples]
ax.scatter(X[:, 0], X[:, 1], c=y, cmap="cool", edgecolors="k", alpha=0.6)

plt.show()


accuracies = {"val": {}, "test": {}, "train": {}}
for lr, trainers_lr in trainers.items():
    print(f"LR {lr}...")
    accuracies["val"][lr] = {}
    accuracies["test"][lr] = {}
    accuracies["train"][lr] = {}
    for name, trainer_list in trainers_lr.items():
        accuracies["val"][lr][name] = []
        accuracies["test"][lr][name] = []
        accuracies["train"][lr][name] = []

        for t, trainer in enumerate(trainer_list):
            ckpt_path = os.path.join(savepath, f"ckpts-lr-{lr}")
            os.makedirs(ckpt_path, exist_ok=True)
            savename = f"{name}-model-{t}"
            saveappendage = "last"
            if retrain_models or not os.path.exists(os.path.join(ckpt_path, f"{savename}-{saveappendage}.pt")):
                # print(f"Training {t+1}-th trainer for propagator {name}...")
                trainer.train(
                    epochs=epochs,
                    verbose=False,
                    savepath=ckpt_path,
                    savename=savename,
                    saveappendage=saveappendage,
                    fromscratch=True,
                )
            else:
                # print(f"Loading checkpoint {os.path.join(ckpt_path, f'{savename}-{saveappendage}.pt')}")
                trainer.load(savepath=ckpt_path, savename=savename, saveappendage=saveappendage)
            eval_stats = trainer.eval(datamodes=["test", "val", "train"])
            # print(f'(Test Accuracy) {res["test"]["accuracy"]}')
            accuracies["val"][lr][name].append(
                eval_stats["val"]["accuracy_p050"]
                if "accuracy_p050" in eval_stats["val"].keys()
                else eval_stats["val"]["micro_accuracy_top1"]
            )
            accuracies["test"][lr][name].append(
                eval_stats["test"]["accuracy_p050"]
                if "accuracy_p050" in eval_stats["test"].keys()
                else eval_stats["test"]["micro_accuracy_top1"]
            )
            accuracies["train"][lr][name].append(
                eval_stats["train"]["accuracy_p050"]
                if "accuracy_p050" in eval_stats["train"].keys()
                else eval_stats["train"]["micro_accuracy_top1"]
            )

# Plot Settings
# Set font properties.
import matplotlib.font_manager as font_manager

font_path = plt.matplotlib.get_data_path() + "/fonts/ttf/cmr10.ttf"
cmfont = font_manager.FontProperties(fname=font_path)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = cmfont.get_name()
plt.rcParams["mathtext.fontset"] = "cm"

# Set font size.
plt.rcParams["font.size"] = 15

# Disable unicode minus.
plt.rcParams["axes.unicode_minus"] = False

# Use mathtext for axes formatters.
plt.rcParams["axes.formatter.use_mathtext"] = True

plt.rcParams["axes.linewidth"] = 1.5

def plot_accuracies(accuracy_res, name, fname, colormap="Set1"):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))

    lrs = np.sort([k for k in accuracy_res.keys()])
    methods = [m for m in accuracy_res[lrs[0]].keys()]

    colors = np.linspace(0, 1, 9)
    palette = cm.get_cmap(colormap)(colors)
    pastel = 0.3
    palette = (1 - pastel) * palette + pastel * np.ones((9, 4))

    LABELS = {
        "lfp-epsilon": r"LFP-$\varepsilon$",
        "lfp-zplus-zminus": r"LFP-$z^+z^-$",
        "vanilla-gradient": r"Grad",
    }

    maxacclrs = []

    for m, method in enumerate(methods):
        plot_means = []
        plot_stds = []
        for lr in lrs:
            plot_means.append(np.mean(accuracy_res[lr][method]))
            plot_stds.append(np.std(accuracy_res[lr][method]))

        plot_means = np.array(plot_means)
        plot_stds = np.array(plot_stds)

        maxx = lrs[np.argmax(plot_means)]
        maxy = np.max(plot_means)
        print(f"MAX: {method} - {maxx}")
        maxacclrs.append((method, maxx, maxy))

        ax.plot(
            lrs,
            plot_means,
            color=palette[m],
            label=LABELS[method],
            linewidth=3,
            alpha=1,
        )
        ax.plot(
            maxx,
            maxy,
            color=palette[m],
            marker="d",
            markersize=10,
            markeredgecolor=(0.1, 0.1, 0.1, 1),
        )
        ax.fill_between(
            lrs,
            plot_means + plot_stds,
            plot_means - plot_stds,
            color=palette[m],
            alpha=0.2,
        )

    linelocs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.hlines(
        linelocs,
        xmin=-1,
        xmax=lrs[-1],
        color=(0.5, 0.5, 0.5, 1),
        linewidth=1.5,
        zorder=0,
    )

    ax.legend()
    ax.set_xscale("log")

    if name == "test":
        nam = "Test"
    if name == "train":
        nam = "Train"
    if name == "val":
        nam = "Val"

    ax.set_ylabel(f"{nam} Accuracy [%]")
    ax.set_xlabel("LR")
    ax.set_ylim([0.0, 1.01])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([0, 20, 40, 60, 80, 100])
    ax.set_xlim([lrs[0], lrs[-1]])
    ax.tick_params(length=6, width=2)
    # ax.set_xlim([0, 10e-2])
    plt.show()
    fig.savefig(fname)

    return maxacclrs


maxacclrs = {}
for name, val in accuracies.items():
    print(name)
    fname = os.path.join(savepath, f"accuracy-{name}.svg")
    maxacclrs[name] = plot_accuracies(val, name, fname)

def plot_accuracies_over_epochs(accuracy_logs, fname, colormap="Set1"):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    datasplit = [m for m in accuracy_logs[0].keys() if m == "train"]

    colors = np.linspace(0, 1, 9)
    palette = cm.get_cmap(colormap)(colors)
    pastel = 0.3
    palette = (1 - pastel) * palette + pastel * np.ones((9, 4))

    for s, split in enumerate(datasplit):
        plot_data = [accuracy_logs[i][split] for i in range(len(accuracy_logs))]
        plot_means = np.mean(plot_data, axis=0)
        plot_stds = np.std(plot_data, axis=0)

        xaxis = np.arange(0, len(plot_means)) * 5

        ax.plot(xaxis, plot_means, color=palette[s], label=split, linewidth=3.5, alpha=1)
        ax.fill_between(
            xaxis,
            plot_means + plot_stds,
            plot_means - plot_stds,
            color=palette[s],
            alpha=0.2,
        )

    linelocs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.hlines(
        linelocs,
        xmin=-1,
        xmax=xaxis[-1],
        color=(0.5, 0.5, 0.5, 1),
        linewidth=1.5,
        zorder=0,
    )

    ax.set_ylabel("Train Accuracy [%]")
    ax.set_xlabel("Iteration")
    ax.set_ylim([0.0, 1.01])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim([0.0, xaxis[-1]])
    ax.set_yticklabels([0, 20, 40, 60, 80, 100])
    ax.tick_params(length=6, width=2)
    # ax.set_xlim([0, 200])
    # ax.legend()
    plt.show()
    fig.savefig(fname)


toplot = maxacclrs["test"]
for name, lr, acc in toplot:
    print(name, lr, acc)
    plot_accuracies_over_epochs(
        [trainer.accuracy_log for trainer in trainers[lr][name]],
        fname=os.path.join(savepath, f"accuracy-lr-{lr}-{name}.svg"),
    )

def plot_classifier_boundaries(
    models,
    dataset,
    xlim=(0.25, 2.5),
    ylim=(0.25, 2.6),
    colormap="gist_rainbow",
    fname="",
):
    xgrid = np.arange(xlim[0], xlim[1], 0.01)
    ygrid = np.arange(ylim[0], ylim[1], 0.01)

    xx, yy = np.meshgrid(xgrid, ygrid)
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    grid = np.hstack((r1, r2))

    from matplotlib.colors import ListedColormap

    pastel = 0.4
    cmap = cm.get_cmap(colormap, 256)
    newcolors = [cmap(np.linspace(0.4, 0.92, 128))]
    newcolors = (1 - pastel) * np.repeat(newcolors, 2, axis=1)[0] + pastel * np.ones((256, 4))
    colormap = ListedColormap(newcolors.clip(max=1))

    edgecolor = np.ones(4) * 0.1

    # from matplotlib.colors import ListedColormap
    # colors = np.linspace(0, 1, 9)
    # palette = cm.get_cmap(colormap)(colors)
    # pastel = 0.0
    # palette = (1-pastel)*palette+pastel*np.ones((9, 4))
    # reduced_palette = np.array([palette[2], palette[8], palette[3]]) #green, grey, purple
    # colormap = ListedColormap(reduced_palette)

    yhats = []
    for model in models:
        yhat = (
            torch.nn.functional.softmax(model(torch.from_numpy(grid).float().to(device)), dim=1).detach().cpu().numpy()
        )
        # yhats.append(np.argmax(yhat, axis=1)/yhat.shape[1])
        yhats.append(yhat)

    yhat_mean = np.mean(yhats, axis=0)
    print(yhat_mean.shape)
    yhat_mean = np.argmax(yhat, axis=1)
    # yhat_std = np.std(yhats, axis=0)

    zz = yhat_mean.reshape(xx.shape)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    cnt = ax.contourf(xx, yy, zz, cmap=colormap, alpha=0.4)
    for c in cnt.collections:
        c.set_edgecolor("face")
        c.set_linewidth(0.000000000001)

    samples = copy.deepcopy(dataset.samples)
    np.random.shuffle(samples)
    # samples = samples[:200]

    X = np.array([s[0] for s in samples])
    y = [s[1] for s in samples]
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=colormap, edgecolors=edgecolor, alpha=1, marker="o")

    ax.set_xticks([])
    ax.set_yticks([])
    # plt.axis("off")

    plt.show()

    fig.savefig(fname)


toplot = maxacclrs["test"]
# for lr in pl_lrs:
#     print("LR", lr)
#     for name, trainer_list in trainers[lr].items():
for name, lr, acc in toplot:
    print(name, lr, acc)
    plot_classifier_boundaries(
        [trainer.model for trainer in trainers[lr][name]],
        test_dataset,
        fname=os.path.join(savepath, f"boundary-lr-{lr}-name-{name}.svg"),
    )

import networkx as nx


def visualize_weights(model, fname, colormap="PRGn"):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    G = nx.Graph()
    layers = [
        module
        for module in model.modules()
        if not isinstance(module, torch.nn.Sequential)
        and not isinstance(module, ToyDataDense)
        and not isinstance(module, torch.nn.ReLU)
        and not isinstance(module, torch.nn.LeakyReLU)
    ]

    colors = np.linspace(0.15, 0.85, 3)
    # the edges are colored purple or green depending on the output class
    palette = cm.get_cmap(colormap)(colors)
    pastel = 0.3
    palette = (1 - pastel) * palette + pastel * np.ones((3, 4))
    palette[1] *= (0.7, 0.7, 0.7, 0.5)

    nodes = []
    pos = {}
    edges = {"pos": [], "neg": [], "neutral": []}
    edge_colors = {"pos": [], "neg": [], "neutral": []}
    edge_alphas = {"pos": [], "neg": [], "neutral": []}
    for l, layer in enumerate(layers):
        # print([l for l in layer.named_parameters()])
        weights = layer.weight.data.detach().cpu().numpy()
        e_alphas = np.abs(weights)
        e_alphas /= np.max(e_alphas)
        # e_alphas *= 0.9
        # e_alphas += 0.1

        if l == 0:
            # Construct Input

            for i in range(weights.shape[1]):
                xcoords = np.linspace(-0.5, 0.5, weights.shape[1])
                nodename = f"I-{i}"
                nodes.append(nodename)
                pos[nodename] = (xcoords[i], len(layers) - 1)

        for i in range(weights.shape[0]):
            xcoords = (
                np.linspace(-1, 1, weights.shape[0])
                if l != len(layers) - 1
                else np.linspace(-0.5, 0.5, weights.shape[0])
            )
            nodename = f"L{l + 1}-{i}" if l != len(layers) - 1 else f"O-{i}"
            nodes.append(nodename)
            pos[nodename] = (xcoords[i], len(layers) - 1 - l - 1)

            for j in range(weights.shape[1]):
                pre = f"L{l}-{j}" if l > 0 else f"I-{j}"

                if weights[i][j] > 0 and e_alphas[i][j] > 0.33:
                    color = palette[2]
                    edges["pos"].append((pre, nodename))
                    edge_colors["pos"].append(color)
                    edge_alphas["pos"].append(e_alphas[i][j])
                elif weights[i][j] < 0 and e_alphas[i][j] > 0.33:
                    color = palette[0]
                    edges["neg"].append((pre, nodename))
                    edge_colors["neg"].append(color)
                    edge_alphas["neg"].append(e_alphas[i][j])
                else:
                    color = palette[1]
                    edges["neutral"].append((pre, nodename))
                    edge_colors["neutral"].append(color)
                    edge_alphas["neutral"].append(e_alphas[i][j])

    options = {"edgecolors": "tab:gray", "node_size": 50, "alpha": 0.9}
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=(0.2, 0.2, 0.2, 1.0), ax=ax, **options)
    collection_neutral = nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges["neutral"],
        width=3,
        # alpha=1.0, #edge_alphas,
        edge_color=edge_colors["neutral"],
    )
    collection_neutral.set_zorder(0)

    collection_pos = nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges["pos"],
        width=3,
        # alpha=1.0, #edge_alphas,
        edge_color=edge_colors["pos"],
    )
    collection_pos.set_zorder(1)

    collection_neg = nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges["neg"],
        width=3,
        # alpha=1.0, #edge_alphas,
        edge_color=edge_colors["neg"],
    )
    collection_neg.set_zorder(1)

    ax.text(0, len(layers) - 0.9, "Input", ha="center", va="center")
    ax.text(0, -1.1, "Output", ha="center", va="center")
    # nx.draw_networkx_labels(G, pos, {n: n for n in nodes}, font_size=15, font_color="black")
    #plt.show()
    fig.savefig(fname)


print("Init")

visualize_weights(init_models[0], fname=os.path.join(savepath, "weights-initmodel.svg"))
for name, lr, acc in toplot:
    print(name, lr, acc)
    visualize_weights(
        trainers[lr][name][0].model,
        fname=os.path.join(savepath, f"weights-name-{name}-lr-{lr}.svg"),
    )
'''