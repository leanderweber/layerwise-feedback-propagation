import logging
from typing import Optional, TypeVar

import torch
import torcheval
import torcheval.metrics
import torcheval.metrics.classification

from lfprop.model import spiking_networks

TRecall = TypeVar("TRecall")


# Recall patch: The multiclass recall of torcheval is currently bugged, see https://github.com/pytorch/torcheval/issues/150
# This patches the bug by masking out classes with no samples in both target and input.
def _recall_compute(
    num_tp: torch.Tensor,
    num_labels: torch.Tensor,
    num_predictions: torch.Tensor,
    average: Optional[str],
) -> torch.Tensor:
    """
    Compute recall for multiclass classification, with a patch to ignore classes
    that have no samples in both target and input.

    Args:
        num_tp (torch.Tensor): Number of true positives per class.
        num_labels (torch.Tensor): Number of ground-truth labels per class.
        num_predictions (torch.Tensor): Number of predicted labels per class.
        average (Optional[str]): Averaging method ('micro', 'macro', 'weighted', or None).

    Returns:
        torch.Tensor: Recall score(s) according to the averaging method.
    """
    if average in ("macro", "weighted"):
        # Ignore classes which have no samples in `target` and `input`
        mask = (num_labels != 0) | (num_predictions != 0)
        num_tp = num_tp[mask]
        num_labels = num_labels[mask]  # THIS IS THE PATCH

    recall = num_tp / num_labels

    # Handle NaNs that may arise from division by zero
    isnan_class = torch.isnan(recall)
    if isnan_class.any():
        nan_classes = isnan_class.nonzero(as_tuple=True)[0]
        logging.warning(
            f"One or more NaNs identified, as no ground-truth instances of "
            f"{nan_classes.tolist()} have been seen. These have been converted to zero."
        )
        recall = torch.nan_to_num(recall)

    if average == "micro":
        return recall
    elif average == "macro":
        return recall.mean()
    elif average == "weighted":
        # Weighted average recall, using the number of labels as weights
        weights = num_labels[mask] / num_labels.sum()
        return (recall * weights).sum()
    else:  # average is None
        return recall


@torch.inference_mode()
def compute(self: TRecall) -> torch.Tensor:
    """
    Return the recall score for the current state of the metric.

    NaN is returned if no calls to ``update()`` are made before ``compute()`` is called.

    Returns:
        torch.Tensor: Recall score(s).
    """
    return _recall_compute(self.num_tp, self.num_labels, self.num_predictions, self.average)


# Patch the compute method of MulticlassRecall with the fixed version
torcheval.metrics.classification.MulticlassRecall.compute = compute


def evaluate(
    model,
    loader,
    num_classes,
    criterion_func,
    device,
    n_steps=15,
    is_huggingface_data=False,
    binary=False,
):
    """
    Evaluates a model for one epoch over the provided data loader.

    Computes predictions, loss, and a set of classification metrics.
    Returns a dictionary with all computed metrics.

    Args:
        model: The model to evaluate.
        loader: DataLoader providing batches of data.
        num_classes (int): Number of classes in the classification task.
        criterion_func: Loss function or callable for computing the loss.
        device: Device to run evaluation on.
        n_steps (int, optional): Number of steps for temporal models. Defaults to 15.
        is_huggingface_data (bool, optional): Whether the data is from HuggingFace datasets. Defaults to False.

    Returns:
        dict: Dictionary mapping metric names to their computed values.
    """

    # Initialize metrics based on task type (binary or multiclass)
    if binary:
        metrics = {
            "criterion": torcheval.metrics.Mean(device=device),
            "accuracy_p040": torcheval.metrics.BinaryAccuracy(threshold=0.4, device=device),
            "accuracy_p050": torcheval.metrics.BinaryAccuracy(threshold=0.5, device=device),
            "accuracy_p060": torcheval.metrics.BinaryAccuracy(threshold=0.6, device=device),
            "precision_p040": torcheval.metrics.BinaryPrecision(threshold=0.4, device=device),
            "precision_p050": torcheval.metrics.BinaryPrecision(threshold=0.5, device=device),
            "precision_p060": torcheval.metrics.BinaryPrecision(threshold=0.6, device=device),
            "recall_p040": torcheval.metrics.BinaryRecall(threshold=0.4, device=device),
            "recall_p050": torcheval.metrics.BinaryRecall(threshold=0.5, device=device),
            "recall_p060": torcheval.metrics.BinaryRecall(threshold=0.6, device=device),
            "f1_p040": torcheval.metrics.BinaryF1Score(threshold=0.4, device=device),
            "f1_p050": torcheval.metrics.BinaryF1Score(threshold=0.5, device=device),
            "f1_p060": torcheval.metrics.BinaryF1Score(threshold=0.6, device=device),
        }
    else:
        metrics = {
            "criterion": torcheval.metrics.Mean(device=device),
            "micro_accuracy_top1": torcheval.metrics.MulticlassAccuracy(
                average="micro", num_classes=num_classes, k=1, device=device
            ),
            "micro_accuracy_top3": torcheval.metrics.MulticlassAccuracy(
                average="micro", num_classes=num_classes, k=3, device=device
            ),
            "micro_accuracy_top5": torcheval.metrics.MulticlassAccuracy(
                average="micro", num_classes=num_classes, k=5, device=device
            ),
            "micro_precision": torcheval.metrics.MulticlassPrecision(
                average="micro", num_classes=num_classes, device=device
            ),
            "micro_recall": torcheval.metrics.MulticlassRecall(average="micro", num_classes=num_classes, device=device),
            "micro_f1": torcheval.metrics.MulticlassF1Score(average="micro", num_classes=num_classes, device=device),
            "macro_accuracy_top1": torcheval.metrics.MulticlassAccuracy(
                average="macro", num_classes=num_classes, k=1, device=device
            ),
            "macro_accuracy_top3": torcheval.metrics.MulticlassAccuracy(
                average="macro", num_classes=num_classes, k=3, device=device
            ),
            "macro_accuracy_top5": torcheval.metrics.MulticlassAccuracy(
                average="macro", num_classes=num_classes, k=5, device=device
            ),
            "macro_precision": torcheval.metrics.MulticlassPrecision(
                average="macro", num_classes=num_classes, device=device
            ),
            "macro_recall": torcheval.metrics.MulticlassRecall(average="macro", num_classes=num_classes, device=device),
            "macro_f1": torcheval.metrics.MulticlassF1Score(average="macro", num_classes=num_classes, device=device),
        }

    # Set model to evaluation mode
    model.eval()

    # Iterate over data loader
    for i, batch in enumerate(loader):
        # Prepare inputs and labels using model's forward function
        with torch.no_grad():
            inputs, labels, outputs = model.forward_fn(
                batch,
                model,
                device,
                lfp_step=False,
                n_steps=n_steps,
                is_huggingface_data=is_huggingface_data,
            )

        # Compute loss/reward
        with torch.set_grad_enabled(True):
            if isinstance(criterion_func, torch.nn.modules.loss._Loss):
                # If criterion_func is a torch loss module, broadcast loss to match outputs shape
                crit = torch.ones_like(outputs) * criterion_func(outputs, labels)
            else:
                crit = criterion_func(outputs, labels)

        # For binary classification, apply sigmoid to outputs
        if binary:
            outputs = torch.nn.functional.sigmoid(outputs).squeeze()

        # For spiking models, sum spikes over time steps
        if isinstance(model, spiking_networks.LifMLP):
            outputs = outputs.sum(0)

        # Update all metrics with current batch
        for k, v in metrics.items():
            if k == "criterion":
                metrics[k].update(crit)
            else:
                metrics[k].update(outputs, labels)

    # Compute and collect all metrics into a dictionary
    return_dict = {m: metric.compute().detach().cpu().numpy() for m, metric in metrics.items()}

    # Return dictionary of metrics
    return return_dict
