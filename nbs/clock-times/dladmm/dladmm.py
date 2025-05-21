import sys
import time

import numpy as np
import torch
import torch.nn as tnn

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

from . import common, input_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# initialize the neural network
def DladmmNet(images, label, num_of_neurons1, num_of_neurons2):
    seed_num = 13
    torch.random.manual_seed(seed=seed_num)
    W1 = torch.normal(size=(num_of_neurons1, 28 * 28), mean=0, std=0.1, device=device)
    torch.random.manual_seed(seed=seed_num)
    b1 = torch.normal(size=(num_of_neurons1, 1), mean=0, std=0.1, device=device)
    z1 = torch.matmul(W1, images) + b1
    a1 = common.relu(z1)
    torch.random.manual_seed(seed=seed_num)
    W2 = torch.normal(
        size=(num_of_neurons2, num_of_neurons1), mean=0, std=0.1, device=device
    )
    torch.random.manual_seed(seed=seed_num)
    b2 = torch.normal(size=(num_of_neurons2, 1), mean=0, std=0.1, device=device)
    z2 = torch.matmul(W2, a1) + b2
    a2 = common.relu(z2)
    torch.random.manual_seed(seed=seed_num)
    W3 = torch.normal(size=(10, num_of_neurons2), mean=0, std=0.1, device=device)
    torch.random.manual_seed(seed=seed_num)
    b3 = torch.normal(size=(10, 1), mean=0, std=0.1, device=device)
    imask = torch.eq(label, 0)
    z3 = torch.where(imask, -torch.ones_like(label), torch.ones_like(label))
    return W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3


# return the accuracy of the neural network model
def test_accuracy(W1, b1, W2, b2, W3, b3, images, labels):

    nums = labels.shape[1]
    z1 = torch.matmul(W1, images) + b1
    a1 = common.relu(z1)
    z2 = torch.matmul(W2, a1) + b2
    a2 = common.relu(z2)
    z3 = torch.matmul(W3, a2) + b3
    cost = common.cross_entropy_with_softmax(labels, z3) / nums
    actual = torch.argmax(labels, dim=0)
    pred = torch.argmax(z3, dim=0)

    eval_metrics = {
        "objective": cost.item(),
        "accuracy": (torch.sum(torch.eq(pred, actual)) / nums).item(),
    }

    return eval_metrics


# return the value of the augmented Lagrangian
def objective(
    x_train, y_train, W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3, u, v1, v2, rho
):
    r1 = torch.sum(
        (z1 - torch.matmul(W1, x_train) - b1) * (z1 - torch.matmul(W1, x_train) - b1)
    )
    r2 = torch.sum((z2 - torch.matmul(W2, a1) - b2) * (z2 - torch.matmul(W2, a1) - b2))
    r3 = torch.sum((z3 - torch.matmul(W3, a2) - b3) * (z3 - torch.matmul(W3, a2) - b3))
    loss = common.cross_entropy_with_softmax(y_train, z3)
    obj = loss + torch.trace(
        torch.matmul(z3 - torch.matmul(W3, a2) - b3, torch.transpose(u, 0, 1))
    )
    obj = obj + rho / 2 * r1 + rho / 2 * r2 + rho / 2 * r3
    obj = (
        obj
        + rho / 2 * torch.sum((a1 - common.relu(z1) + v1) * (a1 - common.relu(z1) + v1))
        + rho / 2 * torch.sum((a2 - common.relu(z2) + v2) * (a2 - common.relu(z2) + v2))
    )
    return obj


# mnist = input_data.mnist()
# W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3 = Net(x_train, y_train, num_of_neurons)
# initialization
# x_train = mnist.x_train
# y_train = mnist.y_train
# x_train = torch.transpose(x_train, 0, 1)
# y_train = torch.transpose(y_train, 0, 1)
# x_test = mnist.x_test
# y_test = mnist.y_test
# x_test = torch.transpose(x_test, 0, 1)
# y_test = torch.transpose(y_test, 0, 1)

# num_of_neurons_1 = 1000
# ITER = 200
# index = 0
# W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3 = Net(x_train, y_train, num_of_neurons)
# u = torch.zeros(z3.shape, device=device)
# t = 0
# train_acc = np.zeros(ITER)
# test_acc = np.zeros(ITER)
# linear_r = np.zeros(ITER)
# objective_value = np.zeros(ITER)
# train_cost = np.zeros(ITER)
# test_cost = np.zeros(ITER)
# rho = 1e-06
# tau = 1
# theta = 1
# t = 0
# flag = 0
# count = 0
# dlADMM


def train(model, x_train, y_train, x_test, y_test, epochs, **kwargs):

    evals = {
        "train_accuracy": [],
        "train_objective": [],
        "test_accuracy": [],
        "test_objective": [],
        "clock_time": [],
    }
    tr_losses = []

    x_train = torch.transpose(x_train, 0, 1)
    y_train = torch.transpose(y_train, 0, 1)
    x_test = torch.transpose(x_test, 0, 1)
    y_test = torch.transpose(y_test, 0, 1)

    W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3 = model

    u = torch.zeros(z3.shape, device=device)
    t = 0
    rho = 1e-06
    tau = 1
    theta = 1

    eval_stats_train = test_accuracy(W1, b1, W2, b2, W3, b3, x_train, y_train)
    eval_stats_test = test_accuracy(W1, b1, W2, b2, W3, b3, x_test, y_test)
    tr_obj = objective(
        x_train, y_train, W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3, u, 0, 0, rho
    ).item()
    test_obj = objective(
        x_train, y_train, W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3, u, 0, 0, rho
    ).item()
    print(
        "INIT: (Train Objective) {:.2f}; (Train Accuracy) {:.2f}; (Test Objective) {:.2f}; (Test Accuracy) {:.2f}".format(
            float(np.mean(eval_stats_train["objective"])),
            float(eval_stats_train["accuracy"]),
            float(np.mean(eval_stats_test["objective"])),
            float(eval_stats_test["accuracy"]),
        )
    )

    for epoch in range(epochs):
        pre = time.time()
        z3 = common.update_zl(a2, W3, b3, y_train, z3, u, rho)
        b3 = common.update_b(a2, W3, z3, b3, u, rho)
        W3 = common.update_W(a2, b3, z3, W3, u, rho, theta)
        a2 = common.update_a(W3, b3, z3, z2, a2, u, 0, rho, tau)
        z2 = common.update_z(a1, W2, b2, a2, 0, 0, rho)
        b2 = common.update_b(a1, W2, z2, b2, 0, rho)
        W2 = common.update_W(a1, b2, z2, W2, 0, rho, theta)
        a1 = common.update_a(W2, b2, z2, z1, a1, 0, 0, rho, tau)
        z1 = common.update_z(x_train, W1, b1, a1, 0, 0, rho)
        b1 = common.update_b(x_train, W1, z1, b1, 0, rho)
        W1 = common.update_W(x_train, b1, z1, W1, 0, rho, theta)
        W1 = common.update_W(x_train, b1, z1, W1, 0, rho, theta)
        b1 = common.update_b(x_train, W1, z1, b1, 0, rho)
        z1 = common.update_z(x_train, W1, b1, a1, 0, 0, rho)
        a1 = common.update_a(W2, b2, z2, z1, a1, 0, 0, rho, tau)
        W2 = common.update_W(a1, b2, z2, W2, 0, rho, theta)
        b2 = common.update_b(a1, W2, z2, b2, 0, rho)
        z2 = common.update_z(a1, W2, b2, a2, 0, 0, rho)
        a2 = common.update_a(W3, b3, z3, z2, a2, u, 0, rho, tau)
        W3 = common.update_W(a2, b3, z3, W3, u, rho, theta)
        b3 = common.update_b(a2, W3, z3, b3, u, rho)
        z3 = common.update_zl(a2, W3, b3, y_train, z3, u, rho)
        u = u + rho * (z3 - torch.matmul(W3, a2) - b3)
        post = time.time()

        tr_obj = objective(
            x_train, y_train, W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3, u, 0, 0, rho
        ).item()
        test_obj = objective(
            x_train, y_train, W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3, u, 0, 0, rho
        ).item()
        eval_stats_train = test_accuracy(W1, b1, W2, b2, W3, b3, x_train, y_train)
        eval_stats_test = test_accuracy(W1, b1, W2, b2, W3, b3, x_test, y_test)

        tr_losses.append(eval_stats_train["objective"])

        print(
            "Epoch {}/{}: (Train Objective) {:.2f}; (Train Accuracy) {:.2f}; (Test Objective) {:.2f}; (Test Accuracy) {:.2f}".format(
                epoch + 1,
                epochs,
                float(tr_obj),
                float(eval_stats_train["accuracy"]),
                float(test_obj),
                float(eval_stats_test["accuracy"]),
            )
        )

        evals["train_accuracy"].append(float(eval_stats_train["accuracy"]))
        evals["train_objective"].append(float(tr_obj))
        evals["test_accuracy"].append(float(eval_stats_test["accuracy"]))
        evals["test_objective"].append(float(test_obj))
        evals["clock_time"].append(post - pre)

        if (
            epoch > 2
            and tr_losses[epoch] > tr_losses[epoch - 1]
            and tr_losses[epoch - 1] > tr_losses[epoch - 2]
            and tr_losses[epoch - 2] > tr_losses[epoch - 3]
        ):
            rho = np.minimum(10 * rho, 0.1)
            # if num_of_neurons >= 100:
            tau = tau * 10
            theta = theta * 10

    return evals
