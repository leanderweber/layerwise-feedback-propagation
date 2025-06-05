################################################################################
# Title:            utils.py                                                   #
# Description:      Some utility functions                                     #
# Author:           Aidin Attar                                                #
# Date:             2024-07-01                                                 #
# Version:          0.4                                                        #
# Usage:            None                                                       #
# Notes:            None                                                       #
# Python version:   3.11.7                                                     #
################################################################################

import datetime
import os
import random

import numpy as np
import PIL
import psutil

# import tonic
import torch
import torchvision
import torchvision.transforms.v2

# from imageio import imwrite
from sklearn.preprocessing import LabelEncoder

from .s1c1 import S1C1
from .SpykeTorch import utils


def get_time():
    """Get the current time"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_time_stamp():
    """Get the current time stamp"""
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def get_time_stamp_ms():
    """Get the current time stamp with milliseconds"""
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")


def find_percentile(arr, value):
    """
    Find the percentile of a value in an array

    Parameters
    ----------
    arr : list
        The array of values
    value : float
        The value to find the percentile of

    Returns
    -------
    float
        The percentile of the value in the array
    """
    arr = np.array(arr)
    sorted_arr = np.sort(arr)
    rank = np.sum(sorted_arr <= value)
    percentile = (rank / len(arr)) * 100

    return percentile


def find_percentile_index(arr, percentile):
    """
    Find the index of a percentile in an array

    Parameters
    ----------
    arr : list
        The array of values
    percentile : float
        The percentile to find the index of

    Returns
    -------
    int
        The index of the percentile in the array
    """
    arr = np.array(arr)
    sorted_arr = np.sort(arr)
    index = int((percentile / 100) * len(arr))

    return index


def find_percentile_value(arr, percentile):
    """
    Find the value of a percentile in an array

    Parameters
    ----------
    arr : list
        The array of values
    percentile : float
        The percentile to find the value of

    Returns
    -------
    float
        The value of the percentile in the array
    """
    arr = np.array(arr)
    sorted_arr = np.sort(arr)
    index = int((percentile / 100) * len(arr))
    value = sorted_arr[index]

    return value


def find_percentile_range(arr, percentile):
    """
    Find the range of a percentile in an array

    Parameters
    ----------
    arr : list
        The array of values
    percentile : float
        The percentile to find the range of

    Returns
    -------
    tuple
        The range of the percentile in the array
    """
    arr = np.array(arr)
    sorted_arr = np.sort(arr)
    index = int((percentile / 100) * len(arr))
    value = sorted_arr[index]
    lower = sorted_arr[:index]
    upper = sorted_arr[index:]
    lower_range = (np.min(lower), np.max(lower))
    upper_range = (np.min(upper), np.max(upper))

    return lower_range, upper_range


def get_embeddings_metadata(model, dataloader, device, max_layer=None):
    """
    Get embeddings and metadata from a single batch in a dataloader

    Parameters
    ----------
    model : torch.nn.Module
        The model to get embeddings from
    dataloader : torch.utils.data.DataLoader
        The dataloader to get metadata from
    device : torch.device
        The device to use
    max_layer : int
        Maximum layer to get embeddings from

    Returns
    -------
    torch.Tensor
        The embeddings
    list
        The metadata
    torch.Tensor
        The label images
    """
    model.eval()
    embeddings = []
    metadata = []
    label_imgs = []

    with torch.no_grad():
        data, target = next(iter(dataloader))
        for data_in, target_in in zip(data, target):
            data_in, target_in = data_in.to(device), target_in.to(device)
            output = model.get_embeddings(data_in, max_layer=max_layer)
            embeddings.append(output)
            metadata.append(target_in.cpu().numpy())
            label_imgs.append(data_in.cpu())

    embeddings = [e.unsqueeze(0) if e.dim() == 0 else e for e in embeddings]
    embeddings = torch.cat(embeddings)
    label_imgs = torch.stack(label_imgs).squeeze(1)
    return embeddings, metadata, label_imgs


def prepare_data(dataset, root_path, batch_size, augment=False, ratio=1):
    """
    Prepare the data for training

    Parameters
    ----------
    dataset : str
        Name of the dataset to use
    batch_size : int
        Batch size for the data loader
    augment : bool
        Whether to augment the data or not

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        Data loader for the training set
    test_loader : torch.utils.data.DataLoader
        Data loader for the test set
    num_classes : int
        Number of classes in the datasetd
    """
    kernels = [
        utils.DoGKernel(3, 3 / 9, 6 / 9),
        utils.DoGKernel(3, 6 / 9, 3 / 9),
        utils.DoGKernel(7, 7 / 9, 14 / 9),
        utils.DoGKernel(7, 14 / 9, 7 / 9),
        utils.DoGKernel(13, 13 / 9, 26 / 9),
        utils.DoGKernel(13, 26 / 9, 13 / 9),
    ]

    # Load dataset
    if dataset == "mnist":
        filter = utils.Filter(kernels, padding=6, thresholds=50)
        s1c1 = S1C1(filter, timesteps=15)
        if not augment:
            data_root = root_path
            num_classes = 10
            in_channels = 6
            train_data = utils.CacheDataset(
                torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=s1c1)
            )
            test_data = utils.CacheDataset(
                torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=s1c1)
            )
        else:
            num_classes = 10
            in_channels = 6
            augmentations = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomRotation(30),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomVerticalFlip(),
                ]
            )
            transform_original = s1c1
            transofrm_augmented = torchvision.transforms.Compose([augmentations, s1c1])

            train_data_original = utils.CacheDataset(
                torchvision.datasets.MNIST(
                    root=root_path,
                    train=True,
                    download=True,
                    transform=transform_original,
                )
            )
            train_data_augmented = utils.CacheDataset(
                torchvision.datasets.MNIST(
                    root=root_path,
                    train=True,
                    download=True,
                    transform=transofrm_augmented,
                )
            )

            if ratio < 1:
                random_indices = torch.randperm(len(train_data_augmented))[: int(ratio * len(train_data_augmented))]
                train_data_augmented = torch.utils.data.Subset(train_data_augmented, random_indices)

            train_data = torch.utils.data.ConcatDataset([train_data_original, train_data_augmented])
            test_data = utils.CacheDataset(
                torchvision.datasets.MNIST(root=root_path, train=False, download=True, transform=s1c1)
            )

    elif dataset == "cifar10":
        # TODO: Changed default parameters for CIFAR-10 dataset from the original code.
        # kernels = [
        #     utils.DoGKernel(3,3/9,6/9),utils.DoGKernel(3,3/9,6/9),utils.DoGKernel(3,3/9,6/9),
        #     utils.DoGKernel(3,6/9,3/9),utils.DoGKernel(3,6/9,3/9),utils.DoGKernel(3,6/9,3/9),
        #     utils.DoGKernel(7,7/9,14/9),utils.DoGKernel(7,7/9,14/9),utils.DoGKernel(7,7/9,14/9),
        #     utils.DoGKernel(7,14/9,7/9),utils.DoGKernel(7,14/9,7/9),utils.DoGKernel(7,14/9,7/9),
        #     utils.DoGKernel(13,13/9,26/9),utils.DoGKernel(13,13/9,26/9),utils.DoGKernel(13,13/9,26/9),
        #     utils.DoGKernel(13,26/9,13/9),utils.DoGKernel(13,26/9,13/9),utils.DoGKernel(13,26/9,13/9),
        # ]

        # filter = utils.Filter(kernels, padding=6, thresholds=50, multy_channel=True)
        # s1c1 = S1C1(filter, timesteps=15)
        # Adjust the kernel sizes or add new ones if needed
        kernels = [
            utils.DoGKernel(5, 3 / 9, 6 / 9),
            utils.DoGKernel(5, 3 / 9, 6 / 9),
            utils.DoGKernel(5, 3 / 9, 6 / 9),
            utils.DoGKernel(5, 6 / 9, 3 / 9),
            utils.DoGKernel(5, 6 / 9, 3 / 9),
            utils.DoGKernel(5, 6 / 9, 3 / 9),
            utils.DoGKernel(9, 7 / 9, 14 / 9),
            utils.DoGKernel(9, 7 / 9, 14 / 9),
            utils.DoGKernel(9, 7 / 9, 14 / 9),
            utils.DoGKernel(9, 14 / 9, 7 / 9),
            utils.DoGKernel(9, 14 / 9, 7 / 9),
            utils.DoGKernel(9, 14 / 9, 7 / 9),
            utils.DoGKernel(15, 13 / 9, 26 / 9),
            utils.DoGKernel(15, 13 / 9, 26 / 9),
            utils.DoGKernel(15, 13 / 9, 26 / 9),
            utils.DoGKernel(15, 26 / 9, 13 / 9),
            utils.DoGKernel(15, 26 / 9, 13 / 9),
            utils.DoGKernel(15, 26 / 9, 13 / 9),
        ]

        # Adjust padding and thresholds based on new kernel sizes
        filter = utils.Filter(kernels, padding=8, thresholds=[40] * len(kernels), multy_channel=True)

        # Increase timesteps for temporal encoding
        s1c1 = S1C1(filter, timesteps=25)
        # # Adjust padding and thresholds based on new kernel sizes
        # filter = utils.Filter(kernels, padding=8, thresholds=[40] * len(kernels))
        # # Increase timesteps for temporal encoding
        # s1c1 = S1C1(filter, timesteps=20)

        num_classes = 10
        data_root = root_path
        in_channels = 18

        train_data = utils.CacheDataset(
            torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=s1c1)
        )
        print("Train data sample shape:", train_data[0][0].shape)
        test_data = utils.CacheDataset(
            torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=s1c1)
        )
    elif dataset == "emnist":
        filter = utils.Filter(kernels, padding=6, thresholds=50)
        s1c1 = S1C1(filter, timesteps=15)

        num_classes = 47
        data_root = root_path
        in_channels = 6
        train_data = utils.CacheDataset(
            torchvision.datasets.EMNIST(
                root=data_root,
                split="digits",
                train=True,
                download=True,
                transform=s1c1,
            )
        )
        test_data = utils.CacheDataset(
            torchvision.datasets.EMNIST(
                root=data_root,
                split="digits",
                train=False,
                download=True,
                transform=s1c1,
            )
        )
    # elif dataset == "nmnist":
    #     num_classes = 10
    #     data_root = root_path
    #     in_channels = 2
    #     trans = tonic.transforms.Compose(
    #         [
    #             tonic.transforms.Denoise(filter_time=3000),
    #             tonic.transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size, n_time_bins=15),
    #         ]
    #     )
    #     train_data = utils.CacheDataset(tonic.datasets.NMNIST(save_to=data_root, train=True, transform=trans))
    #     test_data = utils.CacheDataset(tonic.datasets.NMNIST(save_to=data_root, train=False, transform=trans))
    # elif dataset == "cifar10-dvs":
    #     num_classes = 10
    #     data_root = root_path
    #     in_channels = 2
    #     trans = tonic.transforms.Compose(
    #         [
    #             tonic.transforms.Denoise(filter_time=3000),
    #             tonic.transforms.ToFrame(sensor_size=tonic.datasets.CIFAR10DVS.sensor_size, n_time_bins=15),
    #         ]
    #     )
    #     full_dataset = tonic.datasets.CIFAR10DVS(save_to=data_root, transform=trans)
    #     generator = torch.Generator().manual_seed(42)
    #     train_data, test_data = torch.utils.data.random_split(full_dataset, [0.8, 0.2], generator=generator)
    #     train_data = utils.CacheDataset(train_data)
    #     test_data = utils.CacheDataset(test_data)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if ratio < 1 and not augment:
        train_size = int(ratio * len(train_data))
        random_indices = torch.randperm(len(train_data))[:train_size]
        train_data = torch.utils.data.Subset(train_data, random_indices)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=len(test_data),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    metrics_indices = torch.randperm(len(test_data))[:batch_size]
    metrics_data = torch.utils.data.Subset(test_data, metrics_indices)
    metrics_loader = torch.utils.data.DataLoader(
        metrics_data,
        batch_size=len(metrics_data),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, test_loader, metrics_loader, num_classes, in_channels


def is_model_on_cuda(model):
    """Check if the model is on CUDA"""
    return next(model.parameters()).is_cuda


# class SmallNORBDataset:
#     """
#     Code partially taken from https://github.com/ndrplz/small_norb.git

#     This script generates the NORB dataset from the raw data files. The NORB dataset
#     is a dataset of stereo images of 3D objects. The dataset is available at
#     https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/. The dataset is divided into
#     two parts: a small dataset and a large dataset. The small dataset contains 24300
#     training examples and 24300 test examples. The large dataset contains 24300
#     training examples and 24300 test examples. The small dataset is used in this
#     script.

#     The dataset is stored for each example as a 96x96 image. The images are stored
#     in jpegs, so they need to be decoded.

#     The dataset is stored in a binary format. The training set is stored in a file
#     called 'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat'. The test set is
#     stored in a file called 'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat'.
#     The labels for the training set are stored in a file called
#     'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat'. The labels for the test set
#     are stored in a file called 'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat'.
#     """

#     def __init__(self, dataset_root):
#         self.dataset_root = dataset_root
#         self.dataset_files = self._get_dataset_files()
#         self.data = self._load_data()

#     def _get_dataset_files(self):
#         files = ["cat", "info", "dat"]
#         prefixes = {
#             "train": "smallnorb-5x46789x9x18x6x2x96x96-training",
#             "test": "smallnorb-5x01235x9x18x6x2x96x96-testing",
#         }
#         dataset_files = {
#             split: {
#                 f: os.path.join(self.dataset_root, f"{prefixes[split]}-{f}.mat")
#                 for f in files
#             }
#             for split in ["train", "test"]
#         }
#         return dataset_files

#     def _load_data(self):
#         data = {
#             split: [
#                 self._load_example(i, split)
#                 for i in tqdm(range(24300), desc=f"Loading {split} data")
#             ]
#             for split in ["train", "test"]
#         }
#         return data

#     def _load_example(self, i, split):
#         example = {}
#         example["category"] = self._load_category(i, split)
#         example["info"] = self._load_info(i, split)
#         example["images"] = self._load_images(i, split)
#         return example

#     def _load_category(self, i, split):
#         with open(self.dataset_files[split]["cat"], "rb") as f:
#             f.seek(i * 4 + 20)
#             (category,) = struct.unpack("<i", f.read(4))
#         return category

#     def _load_info(self, i, split):
#         with open(self.dataset_files[split]["info"], "rb") as f:
#             f.seek(i * 16 + 20)
#             info = struct.unpack("<4i", f.read(16))
#         return info

#     def _load_images(self, i, split):
#         with open(self.dataset_files[split]["dat"], "rb") as f:
#             f.seek(i * 2 * 96 * 96 + 24)
#             images = np.fromfile(f, dtype=np.uint8, count=2 * 96 * 96).reshape(
#                 2, 96, 96
#             )
#         return images

#     def show_random_examples(self, split):
#         fig, axes = plt.subplots(nrows=1, ncols=2)
#         for example in np.random.choice(self.data[split], 5):
#             fig.suptitle(f"Category: {example['category']} Info: {example['info']}")
#             axes[0].imshow(example["images"][0], cmap="gray")
#             axes[1].imshow(example["images"][1], cmap="gray")
#             plt.waitforbuttonpress()
#             plt.cla()

#     def export_to_jpg(self, export_dir, train_size, test_size):
#         for split in ["train", "test"]:
#             split_dir = os.path.join(export_dir, split)
#             os.makedirs(split_dir, exist_ok=True)

#             # Delete everything in the split directory
#             for root, dirs, files in os.walk(split_dir):
#                 for file in files:
#                     os.remove(os.path.join(root, file))
#                 for dir in dirs:
#                     sub_dir = os.path.join(root, dir)
#                     for sub_root, sub_dirs, sub_files in os.walk(sub_dir):
#                         for sub_file in sub_files:
#                             os.remove(os.path.join(sub_root, sub_file))
#                         for sub_sub_dir in sub_dirs:
#                             os.rmdir(os.path.join(sub_root, sub_sub_dir))
#                     os.rmdir(sub_dir)

#             if split == "train":
#                 size = train_size
#             else:
#                 size = test_size
#             for i, example in enumerate(
#                 tqdm(
#                     self.data[split][:size],
#                     desc=f"Exporting {split} images to {export_dir}",
#                 )
#             ):
#                 for j, image in enumerate(example["images"]):
#                     if not os.path.exists(
#                         os.path.join(split_dir, str(example["category"]))
#                     ):
#                         os.makedirs(
#                             os.path.join(split_dir, str(example["category"])),
#                             exist_ok=True,
#                         )
#                     # imwrite(os.path.join(split_dir, f'{i:06d}_{example["category"]}_{example["info"][0]}_{j}.jpg'), image)
#                     imwrite(
#                         os.path.join(
#                             split_dir,
#                             str(example["category"]),
#                             f"{i:06d}_{example['category']}_{example['info'][0]}_{j}.jpg",
#                         ),
#                         image,
#                     )


# def generate_norb_dataset(train_size, test_size, dataset_root, export_dir):
#     dataset = SmallNORBDataset(dataset_root)
#     dataset.export_to_jpg(export_dir, train_size, test_size)


def memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024**2:.2f} MB")  # RSS = Resident Set Size (RAM usage)


class PoissonSpikeEncoding:
    def __init__(self, timesteps=100, max_rate=20):
        self.timesteps = timesteps
        self.max_rate = max_rate

    def __call__(self, image):
        image = image.float() / 255.0  # Normalize pixel values
        spike_prob = image * self.max_rate / 1000.0
        spikes = torch.rand((self.timesteps, *image.shape)) < spike_prob.unsqueeze(0)
        return spikes.float()


class LatencyEncoding:
    def __init__(self, timesteps=100):
        self.timesteps = timesteps

    def __call__(self, image):
        image = image.float() / 255.0
        spike_times = (1 - image) * (self.timesteps - 1)
        spikes = torch.zeros((self.timesteps, *image.shape))
        for t in range(self.timesteps):
            spikes[t][spike_times <= t] = 1
        return spikes.float()


# Define the early stopping class
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, model=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.model = model

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), os.path.join("models", "lsm.pt"))
        self.val_loss_min = val_loss


class LabelEncoderTransform:
    def __init__(self, classes):
        self.encoder = LabelEncoder()
        self.encoder.fit(classes)

    def __call__(self, label):
        return self.encoder.transform([label])[0]


caltech101_classes = [
    "BACKGROUND_Google",
    "Faces_easy",
    "Leopards",
    "Motorbikes",
    "accordion",
    "airplanes",
    "anchor",
    "ant",
    "barrel",
    "bass",
    "beaver",
    "binocular",
    "bonsai",
    "brain",
    "brontosaurus",
    "buddha",
    "butterfly",
    "camera",
    "cannon",
    "car_side",
    "ceiling_fan",
    "cellphone",
    "chair",
    "chandelier",
    "cougar_body",
    "cougar_face",
    "crab",
    "crayfish",
    "crocodile",
    "crocodile_head",
    "cup",
    "dalmatian",
    "dollar_bill",
    "dolphin",
    "dragonfly",
    "electric_guitar",
    "elephant",
    "emu",
    "euphonium",
    "ewer",
    "ferry",
    "flamingo",
    "flamingo_head",
    "garfield",
    "gerenuk",
    "gramophone",
    "grand_piano",
    "hawksbill",
    "headphone",
    "hedgehog",
    "helicopter",
    "ibis",
    "inline_skate",
    "joshua_tree",
    "kangaroo",
    "ketch",
    "lamp",
    "laptop",
    "llama",
    "lobster",
    "lotus",
    "mandolin",
    "mayfly",
    "menorah",
    "metronome",
    "minaret",
    "nautilus",
    "octopus",
    "okapi",
    "pagoda",
    "panda",
    "pigeon",
    "pizza",
    "platypus",
    "pyramid",
    "revolver",
    "rhino",
    "rooster",
    "saxophone",
    "schooner",
    "scissors",
    "scorpion",
    "sea_horse",
    "snoopy",
    "soccer_ball",
    "stapler",
    "starfish",
    "stegosaurus",
    "stop_sign",
    "strawberry",
    "sunflower",
    "tick",
    "trilobite",
    "umbrella",
    "watch",
    "water_lilly",
    "wheelchair",
    "wild_cat",
    "windsor_chair",
    "wrench",
    "yin_yang",
]


class AddSaltPepperNoise(object):
    def __init__(self, salt_prob=0.01, pepper_prob=0.01):
        """
        salt_prob: Probability of adding salt (white) noise
        pepper_prob: Probability of adding pepper (black) noise
        """
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob

    def __call__(self, tensor):
        if isinstance(tensor, PIL.Image.Image):
            tensor = torch.tensor(np.array(tensor))
        # Ensure the input tensor is in [C, H, W] format
        for c in range(tensor.size(0)):  # Iterate over channels
            # Salt (white noise)
            salt_mask = torch.rand(tensor.size(0), tensor.size(1)) < self.salt_prob
            tensor[c][salt_mask] = 1.0  # Set to white (max intensity)

            # Pepper (black noise)
            pepper_mask = torch.rand(tensor.size(0), tensor.size(1)) < self.pepper_prob
            tensor[c][pepper_mask] = 0.0  # Set to black (min intensity)

        return tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(salt_prob={self.salt_prob}, pepper_prob={self.pepper_prob})"


class RandomRowColumnMasking(object):
    def __init__(self, mask_type="row", p=0.5):
        """
        mask_type: 'row' for row masking, 'column' for column masking
        p: probability of masking a row/column
        """
        self.mask_type = mask_type
        self.p = p

    def __call__(self, tensor):
        if isinstance(tensor, PIL.Image.Image):
            tensor = torch.tensor(np.array(tensor))
        if self.mask_type == "row":
            # Randomly mask rows
            for i in range(tensor.size(0)):  # Assuming shape is [channels, height, width]
                if random.random() < self.p:
                    tensor[:, i, :] = 0  # Mask entire row
        elif self.mask_type == "column":
            # Randomly mask columns
            for i in range(tensor.size(1)):  # Assuming shape is [channels, height, width]
                if random.random() < self.p:
                    tensor[:, :, i] = 0  # Mask entire column
        return tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(mask_type={self.mask_type}, p={self.p})"


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)
