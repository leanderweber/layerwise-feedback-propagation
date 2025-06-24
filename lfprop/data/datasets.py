import os

import datasets
import torch
import torchvision.datasets as tvisiondata

from . import custom_datasets

# Mapping of dataset names to their corresponding dataset classes
DATASET_MAPPING = {
    "food11": custom_datasets.Food11,
    "imagenet": custom_datasets.ImageNet,
    "cub": custom_datasets.CUB,
    "isic": custom_datasets.ISIC,
    "mnist": tvisiondata.MNIST,
    "cifar10": tvisiondata.CIFAR10,
    "cifar100": tvisiondata.CIFAR100,
    "circles": custom_datasets.SKLearnCircles,
    "blobs": custom_datasets.SKLearnBlobs,
    "swirls": custom_datasets.Swirls,
}

# Mapping of dataset names to HuggingFace dataset identifiers
HUGGINGFACE_DATASET_MAPPING = {
    "beans": "AI-Lab-Makerere/beans",
    "oxford-flowers": "nkirschi/oxford-flowers",
}


def collate_fn_vit(batch):
    """
    Collate function for Vision Transformer (ViT) style datasets.
    Stacks pixel values and labels from a batch of samples.

    Args:
        batch (list): List of samples, each a dict with 'pixel_values' and 'labels'.

    Returns:
        dict: Dictionary with stacked 'pixel_values' and 'labels' tensors.
    """
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


def get_dataset(dataset_name, root_path, transform, mode, **kwargs):
    """
    Loads and returns the specified dataset along with its collate function, dummy input, and class labels.

    Args:
        dataset_name (str): Name of the dataset to load.
        root_path (str): Root directory for dataset storage or loading.
        transform (callable): Transformations to apply to the dataset samples.
        mode (str): Mode of the dataset, either 'train' or 'test'.
        **kwargs: Additional keyword arguments for dataset constructors.

    Returns:
        tuple: (dataset, collate_fn, dummy_input, class_labels)
            - dataset: The loaded dataset object.
            - collate_fn: Collate function for DataLoader (if needed).
            - dummy_input: Example input for model shape inference (if available).
            - class_labels: List of class label names.

    Raises:
        ValueError: If mode or dataset_name is not supported.
    """
    # Check if mode is valid
    if mode not in ["train", "test"]:
        raise ValueError("Mode '{}' not supported. Mode needs to be one of 'train', 'test'".format(mode))

    # Map 'test' mode to 'val' for ImageNet compatibility
    if (dataset_name == "imagenet") and mode == "test":
        mode = "val"

    # Check if dataset_name is valid
    if dataset_name not in DATASET_MAPPING and dataset_name not in HUGGINGFACE_DATASET_MAPPING:
        raise ValueError("Dataset '{}' not supported.".format(dataset_name))

    if dataset_name in DATASET_MAPPING:
        # Adapt root_path for torchvision datasets
        if DATASET_MAPPING[dataset_name] not in [
            custom_datasets.ImageNet,
            custom_datasets.CUB,
            custom_datasets.ISIC,
            custom_datasets.SKLearnCircles,
            custom_datasets.SKLearnBlobs,
            custom_datasets.Swirls,
            custom_datasets.Food11,
        ]:
            root = os.path.join(root_path, dataset_name)
        else:
            root = root_path

        # Load the appropriate dataset
        if dataset_name not in ["mnist", "cifar10", "cifar100"]:
            # Custom datasets may require additional arguments
            dataset = DATASET_MAPPING[dataset_name](
                root=root,
                transform=transform,
                **{
                    **kwargs,
                    **{
                        "download": True,
                        "train": mode == "train",
                        "mode": mode,
                    },
                },
            )
        else:
            # Torchvision datasets use standard arguments
            dataset = DATASET_MAPPING[dataset_name](
                root=root,
                transform=transform,
                download=True,
                train=mode == "train",
            )
        class_labels = dataset.classes
        collate_fn = None
        dummy_input = None
    else:
        # Handle HuggingFace datasets
        root = root_path
        cache_dir = os.path.join(root, "cache")
        if not os.path.exists(root):
            print("DOWNLOAD HUGGINGFACE DATASET")
            orig_dataset = datasets.load_dataset(
                HUGGINGFACE_DATASET_MAPPING[dataset_name],
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
            orig_dataset.save_to_disk(root)
        else:
            print("USING EXISTING HUGGINGFACE DATASET")
            orig_dataset = datasets.load_from_disk(root)
        # Apply transform and select mode split
        dataset = orig_dataset.with_transform(transform)[mode]
        # Create dummy input for model shape inference if requested
        dummy_input = (
            {k: torch.randn(v.shape)[None, ...] for k, v in dataset[0].items() if not isinstance(v, int)}
            if kwargs.get("return_dummy_input", True)
            else None
        )
        # Extract class labels from dataset features
        class_labels = (
            dataset.features["labels"].names if "labels" in dataset.features.keys() else dataset.features["label"].names
        )
        collate_fn = collate_fn_vit

    # Return dataset, collate function, dummy input, and class labels
    return dataset, collate_fn, dummy_input, class_labels
