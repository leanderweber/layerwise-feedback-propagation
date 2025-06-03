################################################################################
# Title:            preprocess_soli.py                                         #
# Description:      Perform preprocessing on Google's soli gesture data        #
# Author:           Aidin Attar                                                #
# Date:             2024-10-29                                                 #
# Version:          0.1                                                        #
# Usage:            None                                                       #
# Notes:            None                                                       #
# Python version:   3.11.7                                                     #
################################################################################

import os

import h5py
import numpy as np
from tqdm import tqdm


def generate_mdoppler(data):
    """
    Generate the mDoppler map from the input range-doppler data.

    Parameters
    ----------
    data : numpy.ndarray
        The input range-doppler data.

    Returns
    -------
    numpy.ndarray
        The mDoppler map.
    """
    data = data.reshape(data.shape[0], 32, 32)

    # Aggregate along the range dimension for each frame to get the velocity profile
    micro_doppler = np.mean(data, axis=1)

    # Threshold the spectrogram to remove noise and binarize the data
    micro_doppler[micro_doppler > 0] = 1.0

    return micro_doppler


def unroll_stack(data):
    """
    Unroll the range-doppler stack into a single array.

    Parameters
    ----------
    data : numpy.ndarray
        The input range-doppler stack.

    Returns
    -------
    numpy.ndarray
        The unrolled range-doppler stack.
    """
    data[data > 0] = 1.0

    # Cut off everything after 384 bins
    data = data[:, :384]

    return data


def preprocess_soli(data_dir, output_dir, channel=3):
    """
    Preprocess the Google Soli gesture data.

    Parameters
    ----------
    data_dir : str
        The directory containing the raw data.
    output_dir : str
        The directory to save the preprocessed data.
    channel : int
        The number of channel to use in the data.
    mode : str
        The preprocessing mode to use. Options are 'mdoppler' and 'unrolled'.
    """
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get a list of all the files in the data directory and consider only the HDF5 files
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith(".h5")]

    output_path = os.path.join(output_dir, "soli_Preprocessed.h5")

    with h5py.File(output_path, "w") as h5f:
        # Loop through each file in the data directory
        for file in tqdm(files):
            # Load the data from the file
            with h5py.File(os.path.join(data_dir, file), "r") as f:
                data = f[f"ch{channel}"][:]
                label = f["label"][0]

            # Preprocess the data
            mdoppler = generate_mdoppler(data)
            unrolled = unroll_stack(data)

            # Create a group for each file
            grp = h5f.require_group(file.split(".")[0])

            # Save the preprocessed data and label
            grp.create_dataset("mdoppler", data=mdoppler)
            grp.create_dataset("unrolled", data=unrolled)
            grp.create_dataset("label", data=label)


if __name__ == "__main__":
    data_dir = "data/SoliData"
    output_dir = "data/Soli_Preprocessed"

    preprocess_soli(data_dir, output_dir)
