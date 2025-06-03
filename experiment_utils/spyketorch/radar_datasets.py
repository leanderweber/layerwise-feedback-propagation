import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class DopNetH5Dataset(Dataset):
    def __init__(self, h5_file, person_list, gestures, transform=None, target_width=144):
        self.h5_file = h5_file
        self.person_list = person_list
        self.gestures = gestures
        self.transform = transform
        self.target_width = target_width  # Fixed width to pad to
        self.data_index = self._build_index()

    def _build_index(self):
        """Builds an index for easy access to each sample."""
        index = []
        with h5py.File(self.h5_file, "r") as h5f:
            for person in self.person_list:
                for gesture in self.gestures:
                    # Dynamically find all samples under each gesture
                    gesture_group = h5f[f"{person}/{gesture}"]
                    samples = list(gesture_group.keys())
                    for sample in samples:
                        index.append((person, gesture, sample))
        return index

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # Open the HDF5 file in read mode
        with h5py.File(self.h5_file, "r") as h5f:
            # Get the sample details
            person, gesture, sample = self.data_index[idx]
            data = h5f[f"{person}/{gesture}/{sample}"][:]

        # Pad or trim data to the target width
        padded_data = np.zeros((data.shape[0], self.target_width), dtype=np.float32)
        padded_data[:, : data.shape[1]] = data[:, : self.target_width]

        # Convert to tensor and apply transform if specified
        data = torch.tensor(data, dtype=torch.float32)
        label = gesture  # Use gesture as label
        if self.transform:
            data = self.transform(data)

        return data, label


def collate_fn(batch):
    # Find the maximum width in the batch
    # max_width = max(data.shape[1] for data, _ in batch)
    # max_width = 540
    max_width = 150

    # Pad each sample in the batch to the max width
    padded_batch = []
    lengths = []
    for data, label in batch:
        lengths.append(data.shape[1])
        padded_data = np.zeros((data.shape[0], max_width), dtype=np.float32)
        padded_data[:, : data.shape[1]] = data
        padded_batch.append((torch.tensor(padded_data, dtype=torch.float32), label))

    # Separate data and labels for the batch
    data, labels = zip(*padded_batch)
    return torch.stack(data), torch.tensor(labels), lengths


class SoliH5Dataset(Dataset):
    def __init__(self, h5_file, mode, transform=None):
        self.h5_file = h5_file
        self.transform = transform
        assert mode in ["mdoppler", "unrolled"]
        self.mode = mode
        self.data_index = self._build_index()

    def _build_index(self):
        """Builds an index for easy access to each sample."""
        index = []
        with h5py.File(self.h5_file, "r") as h5f:
            for key in h5f.keys():
                index.append(key)

        return index

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # Open the HDF5 file in read mode
        with h5py.File(self.h5_file, "r") as h5f:
            data = h5f[self.data_index[idx]][self.mode][:]
            label = h5f[self.data_index[idx]]["label"][0]

        # Convert to tensor and apply transform if specified
        data = torch.tensor(data, dtype=torch.float32)
        if self.transform:
            data = self.transform(data)

        return data.T, label


def main():
    dataset = "soli"

    if dataset == "DopNet":
        # Load the dataset and DataLoader
        h5_file = os.path.join("data", "gesture", "train", "preprocessed_gesture_data_train.h5")
        person_list = ["A", "B", "C", "D", "E", "F"]
        gestures = [0, 1, 2, 3]  # Wave, Pinch, Swipe, Click
        dopnet_h5_dataset = DopNetH5Dataset(h5_file=h5_file, person_list=person_list, gestures=gestures)
        dopnet_h5_dataloader = DataLoader(dopnet_h5_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

        for batch_x, batch_y in dopnet_h5_dataloader:
            print("Batch spectrograms shape:", batch_x.shape)
            print("Batch labels:", batch_y)
            break

        # Load the test dataset and DataLoader
        test_h5_file = os.path.join("data", "gesture", "test", "preprocessed_gesture_data_test.h5")
        dopnet_test_dataset = DopNetH5Dataset(h5_file=test_h5_file, person_list=person_list, gestures=gestures)
        dopnet_test_dataloader = DataLoader(dopnet_test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

        for test_batch_x, _ in dopnet_test_dataloader:
            print("Test batch spectrograms shape:", test_batch_x.shape)
            break

    elif dataset == "soli":
        # Load the dataset and DataLoader
        h5_file = os.path.join("data", "Soli_Preprocessed", "soli_Preprocessed.h5")
        soli_h5_dataset = SoliH5Dataset(h5_file=h5_file, mode="unrolled")
        # split the dataset into training and validation
        train_dataset, val_dataset = torch.utils.data.random_split(
            soli_h5_dataset, [int(len(soli_h5_dataset) * 0.8), len(soli_h5_dataset) - int(len(soli_h5_dataset) * 0.8)]
        )
        soli_h5_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

        for batch_x, batch_y, length in soli_h5_dataloader:
            print("Batch mdoppler shape:", batch_x.shape)
            print("Batch labels:", batch_y)
            break

        for test_batch_x, test_batch_y, length in val_dataloader:
            print("Validation batch mdoppler shape:", test_batch_x.shape)
            print("Validation batch labels:", test_batch_y)
            break


if __name__ == "__main__":
    main()
