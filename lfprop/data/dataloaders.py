import torch.utils.data as tdata

DATALOADER_MAPPING = {
    "food11": tdata.DataLoader,
    "imagenet": tdata.DataLoader,
    "cub": tdata.DataLoader,
    "isic": tdata.DataLoader,
    "mnist": tdata.DataLoader,
    "cifar10": tdata.DataLoader,
    "cifar100": tdata.DataLoader,
    "circles": tdata.DataLoader,
    "blobs": tdata.DataLoader,
    "swirls": tdata.DataLoader,
    "beans": tdata.DataLoader,
    "oxford-flowers": tdata.DataLoader,
}


def get_dataloader(dataset_name, dataset, batch_size, shuffle, collate_fn=None):
    """
    Retrieves the appropriate DataLoader instance for a given dataset.

    This function selects and initializes the correct DataLoader class based on the provided
    `dataset_name`. It uses a mapping (`DATALOADER_MAPPING`) from dataset names to DataLoader
    constructors. If the dataset name is not supported, a ValueError is raised.

    Args:
        dataset_name (str): The name of the dataset, used to select the appropriate DataLoader.
        dataset (torch.utils.data.Dataset): The dataset object to be loaded.
        batch_size (int): Number of samples per batch to load.
        shuffle (bool): Whether to shuffle the data at every epoch.
        collate_fn (callable, optional): Function to merge a list of samples to form a mini-batch.
            Defaults to None.

    Returns:
        torch.utils.data.DataLoader: An instance of the DataLoader for the specified dataset.

    Raises:
        ValueError: If the provided `dataset_name` is not supported by `DATALOADER_MAPPING`.

    Example:
        dataloader = get_dataloader(
            dataset_name="cifar10",
            dataset=my_dataset,
            batch_size=32,
            shuffle=True
    """

    # Check if dataset_name is valid
    if dataset_name not in DATALOADER_MAPPING:
        raise ValueError("Dataloader for dataset '{}' not supported.".format(dataset_name))

    # Load correct dataloader
    dataloader = DATALOADER_MAPPING[dataset_name](
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )

    # Return dataset
    return dataloader
