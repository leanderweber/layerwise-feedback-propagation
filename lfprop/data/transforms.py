import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import transformers


class AddGaussianNoise:
    """
    Transform that adds Gaussian noise to a tensor with a 50% probability.

    Args:
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.
    """

    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        # With 50% probability, add Gaussian noise to the tensor
        if np.random.choice([0, 1]):
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            return tensor

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)


class DoubleCompose(T.Compose):
    """
    Compose transforms that operate on (img, target) pairs.

    Args:
        transforms (list): List of transforms, each accepting (img, target).
    """

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class DoubleResize(torch.nn.Module):
    """
    Resize both image and target using separate resize transforms.

    Args:
        res1: Resize transform for the image.
        res2: Resize transform for the target.
    """

    def __init__(self, res1, res2):
        super().__init__()
        self.res1 = res1
        self.res2 = res2

    def forward(self, img, target):
        return self.res1(img), self.res2(target)


class DoubleToTensor(torch.nn.Module):
    """
    Convert both image and target to tensors using separate transforms.

    Args:
        t1: Transform for the image.
        t2: Transform for the target.
    """

    def __init__(self, t1, t2) -> None:
        super().__init__()
        self.t1 = t1
        self.t2 = t2

    def __call__(self, img, target):
        return self.t1(img), self.t2(target)


class DoubleNormalize(torch.nn.Module):
    """
    Normalize both image and target using separate normalization transforms.

    Args:
        norm1: Normalization transform for the image.
        norm2: Normalization transform for the target.
    """

    def __init__(self, norm1, norm2):
        super().__init__()
        self.norm1 = norm1
        self.norm2 = norm2

    def forward(self, img, target):
        return self.norm1(img), self.norm2(target)


class DoubleRandomHorizontalFlip(T.RandomHorizontalFlip):
    """
    Randomly horizontally flip both image and target with a given probability.

    Args:
        p (float): Probability of flipping.
    """

    def __init__(self, p=0.5):
        super().__init__(p)

    def forward(self, img, target):
        if torch.rand(1) < self.p:
            return F.hflip(img), F.hflip(target)
        return img, target


class DoubleRandomVerticalFlip(T.RandomVerticalFlip):
    """
    Randomly vertically flip both image and target with a given probability.

    Args:
        p (float): Probability of flipping.
    """

    def __init__(self, p=0.5):
        super().__init__(p)

    def forward(self, img, target):
        if torch.rand(1) < self.p:
            return F.vflip(img), F.vflip(target)
        return img, target


class DoubleRandomApply(T.RandomApply):
    """
    Randomly apply a list of transforms to both image and target with a given probability.

    Args:
        transforms (list): List of transforms to apply.
        p (float): Probability of applying the transforms.
    """

    def __init__(self, transforms, p=0.5):
        super().__init__(transforms, p)

    def forward(self, img, target):
        # Only apply the transforms with probability p
        if self.p < torch.rand(1):
            return img, target
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class DoubleRandomRotation(T.RandomRotation):
    """
    Randomly rotate both image and target by the same angle.

    Args:
        degrees (sequence or number): Range of degrees to select from.
        interpolation (InterpolationMode): Interpolation mode.
        expand (bool): Whether to expand the output image to fit the rotated image.
        center (sequence): Center of rotation.
        fill (sequence or number): Fill value for area outside the rotated image.
    """

    def __init__(
        self,
        degrees,
        interpolation=T.InterpolationMode.NEAREST,
        expand=False,
        center=None,
        fill=0,
    ):
        super().__init__(degrees, interpolation, expand, center, fill)

    def forward(self, img, target):
        """
        Rotate both img and target by the same random angle.

        Args:
            img (PIL Image or Tensor): Image to be rotated.
            target (PIL Image or Tensor): Target to be rotated.

        Returns:
            tuple: Rotated (img, target).
        """
        fill1 = self.fill
        fill2 = self.fill
        channels1, _, _ = F.get_dimensions(img)
        channels2, _, _ = F.get_dimensions(target)
        # Prepare fill values for each channel if needed
        if isinstance(img, torch.Tensor):
            if isinstance(fill1, (int, float)):
                fill1 = [float(fill1)] * channels1
            else:
                fill1 = [float(f) for f in fill1]
        if isinstance(target, torch.Tensor):
            if isinstance(fill2, (int, float)):
                fill2 = [float(fill2)] * channels2
            else:
                fill2 = [float(f) for f in fill2]
        angle = self.get_params(self.degrees)

        return F.rotate(img, angle, self.interpolation, self.expand, self.center, fill1), F.rotate(
            target, angle, self.interpolation, self.expand, self.center, fill2
        )


def replace_tensor_value_(tensor, a, b):
    """
    Replace all occurrences of value `a` in the tensor with value `b` (in-place).

    Args:
        tensor (Tensor): Input tensor.
        a: Value to replace.
        b: Replacement value.

    Returns:
        Tensor: The modified tensor.
    """
    tensor[tensor == a] = b
    return tensor


TRANSFORM_MAP = {
    "imagenet": {
        "train": T.Compose(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
        "test": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
        "val": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
    },
    "food11": {
        "train": T.Compose(
            [
                T.Resize((224, 224), interpolation=T.functional.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.25),
                T.RandomVerticalFlip(p=0.25),
                T.RandomApply(
                    transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))],
                    p=0.25,
                ),
                T.RandomApply(transforms=[T.RandomRotation(degrees=(0, 30))], p=0.25),
                T.RandomApply(
                    transforms=[T.ColorJitter(brightness=0.1, saturation=0.1, hue=0.1)],
                    p=0.25,
                ),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        "test": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        "val": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    },
    "cub": {
        "train": T.Compose(
            [
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                AddGaussianNoise(0, 0.05),
                T.RandomHorizontalFlip(),
                T.RandomAffine(10, (0.2, 0.2), (0.8, 1.2)),
                T.Normalize(
                    (0.47473491, 0.48834997, 0.41759949),
                    (0.22798773, 0.22288573, 0.25982403),
                ),
            ]
        ),
        "test": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    (0.47473491, 0.48834997, 0.41759949),
                    (0.22798773, 0.22288573, 0.25982403),
                ),
            ]
        ),
        "val": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    (0.47473491, 0.48834997, 0.41759949),
                    (0.22798773, 0.22288573, 0.25982403),
                ),
            ]
        ),
    },
    "isic": {
        "train": T.Compose(
            [
                T.Resize((224, 224), interpolation=T.functional.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.25),
                T.RandomVerticalFlip(p=0.25),
                T.RandomApply(
                    transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))],
                    p=0.25,
                ),
                T.RandomApply(transforms=[T.RandomRotation(degrees=(0, 30))], p=0.25),
                T.RandomApply(
                    transforms=[T.ColorJitter(brightness=0.1, saturation=0.1, hue=0.1)],
                    p=0.25,
                ),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        "test": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        "val": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    },
    "mnist": {
        "train": T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))]),
        "test": T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))]),
    },
    "splitmnist": {
        "train": T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))]),
        "test": T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))]),
    },
    "cifar10": {
        "train": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4911, 0.4820, 0.4467), (0.2022, 0.1993, 0.2009)),
            ]
        ),
        "test": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4911, 0.4820, 0.4467), (0.2022, 0.1993, 0.2009)),
            ]
        ),
    },
    "cifar100": {
        "train": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4911, 0.4820, 0.4467), (0.2022, 0.1993, 0.2009)),
            ]
        ),
        "test": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4911, 0.4820, 0.4467), (0.2022, 0.1993, 0.2009)),
            ]
        ),
    },
    "splitcifar100": {
        "train": T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "test": T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    },
    "circles": {
        "train": lambda x: torch.from_numpy(x).float(),
        "test": lambda x: torch.from_numpy(x).float(),
    },
    "blobs": {
        "train": lambda x: torch.from_numpy(x).float(),
        "test": lambda x: torch.from_numpy(x).float(),
    },
    "swirls": {
        "train": lambda x: torch.from_numpy(x).float(),
        "test": lambda x: torch.from_numpy(x).float(),
    },
}


def get_vit_transform_beans(model_path):
    """
    Returns a batch transform function for the 'beans' dataset using a ViT feature extractor.

    Args:
        model_path (str): Path or identifier of the pretrained ViT model.

    Returns:
        function: A function that processes a batch of images and labels for ViT input.
    """
    feature_extractor = transformers.ViTImageProcessor.from_pretrained(model_path)

    def vit_transform_beans(batch):
        """
        Processes a batch of images and labels for the 'beans' dataset.

        Args:
            batch (dict): A batch containing 'image' (list of PIL images) and 'labels'.

        Returns:
            dict: Dictionary with pixel values and labels, ready for ViT input.
        """
        # Convert list of PIL images to pixel values tensor
        inputs = feature_extractor([x for x in batch["image"]], return_tensors="pt")
        # Attach labels to the inputs
        inputs["labels"] = batch["labels"]
        return inputs

    return vit_transform_beans


def get_vit_transform_oxfordflowers(model_path):
    """
    Returns a batch transform function for the 'oxford-flowers' dataset using a ViT feature extractor.

    Args:
        model_path (str): Path or identifier of the pretrained ViT model.

    Returns:
        function: A function that processes a batch of images and labels for ViT input.
    """
    feature_extractor = transformers.ViTImageProcessor.from_pretrained(model_path)

    def vit_transform_oxfordflowers(batch):
        """
        Processes a batch of images and labels for the 'oxford-flowers' dataset.

        Args:
            batch (dict): A batch containing 'image' (list of PIL images) and 'label'.

        Returns:
            dict: Dictionary with pixel values and labels, ready for ViT input.
        """
        # Convert list of PIL images to pixel values tensor
        inputs = feature_extractor([x for x in batch["image"]], return_tensors="pt")
        # Attach labels to the inputs
        inputs["labels"] = batch["label"]
        return inputs

    return vit_transform_oxfordflowers


HUGGINGFACE_TRANSFORMS = {
    "beans": get_vit_transform_beans,
    "oxford-flowers": get_vit_transform_oxfordflowers,
}


def get_transforms(dataset_name, mode, model_path=None):
    """
    Retrieves the appropriate transform(s) for a given dataset and mode.

    Args:
        dataset_name (str): Name of the dataset.
        mode (str): Mode of operation, e.g., 'train', 'test', or 'val'.
        model_path (str, optional): Path or identifier of the pretrained model (for HuggingFace datasets).

    Returns:
        Callable: The transform or batch transform function for the dataset and mode.

    Raises:
        ValueError: If the dataset is not supported.
    """
    # Check if dataset_name is supported
    if dataset_name not in TRANSFORM_MAP and dataset_name not in HUGGINGFACE_TRANSFORMS:
        raise ValueError("Dataset '{}' not supported.".format(dataset_name))

    # Select the appropriate transform
    if dataset_name in TRANSFORM_MAP:
        transforms = TRANSFORM_MAP[dataset_name][mode]
    else:
        # For HuggingFace datasets, return the batch transform function
        transforms = HUGGINGFACE_TRANSFORMS[dataset_name](model_path)

    return transforms
