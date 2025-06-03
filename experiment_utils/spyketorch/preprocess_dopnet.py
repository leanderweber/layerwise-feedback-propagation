################################################################################
# Title:            preprocess_dopnet.py                                       #
# Description:      Perform preprocessing on DopNet gesture data               #
# Author:           Aidin Attar                                                #
# Date:             2024-10-28                                                 #
# Version:          0.2                                                        #
# Usage:            None                                                       #
# Notes:            None                                                       #
# Python version:   3.11.7                                                     #
################################################################################

import os

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


def minimum_error_threshold(image):
    # Flatten image to 1D for histogram analysis
    pixel_values = image.ravel().reshape(-1, 1)

    # Fit a Gaussian Mixture Model with two components (foreground and background)
    gmm = GaussianMixture(n_components=2, covariance_type="full")
    gmm.fit(pixel_values)

    # Means of the two Gaussian components
    means = gmm.means_.flatten()

    # Compute the threshold as the midpoint between the two means
    threshold = np.mean(means)

    # Binarize image based on the threshold
    binary_image = (image >= threshold).astype(np.uint8)

    return binary_image, threshold


def multi_scale_gaussian_blur(x):
    # Apply Gaussian blur at multiple scales
    blur_small = cv2.GaussianBlur(x, (3, 3), sigmaX=0.5)
    blur_medium = cv2.GaussianBlur(x, (5, 5), sigmaX=1)
    blur_large = cv2.GaussianBlur(x, (11, 11), sigmaX=2)

    # Combine the blurred images with weights
    combined_blur = 0.5 * blur_small + 0.3 * blur_medium + 0.2 * blur_large
    return np.clip(combined_blur, 0, 255).astype(np.uint8)


def preprocess_data(x, thresholding_method="otsu"):
    # Unwrap MATLAB cell array structure to get to the numeric array
    while isinstance(x, np.ndarray) and x.dtype == "O":
        x = x[0]  # Unwrap one level

    # Convert to dB
    x = 20 * np.log10(abs(x) / np.amax(abs(x)))

    # Convert to grayscale (0–255)
    x = 255 * (x - np.min(x)) / (np.max(x) - np.min(x))

    # Thresholding with 95th percentile
    threshold = np.percentile(x, 94)
    x = (x >= threshold).astype(np.uint8)

    # Apply Gaussian Filter
    x = cv2.GaussianBlur(x, (5, 5), sigmaX=1)
    # x = multi_scale_gaussian_blur(x)

    if thresholding_method == "otsu":
        # Apply Otsu's thresholding
        _, x = cv2.threshold(x.astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif thresholding_method == "met":
        # Apply MET thresholding
        x, _ = minimum_error_threshold(x)
    elif thresholding_method == "adaptive":
        x = x.astype(np.uint8)
        x = cv2.adaptiveThreshold(x, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 49, 6)
    else:
        raise ValueError(f"Unknown thresholding method: {thresholding_method}")

    return x


def preprocess_data_with_kmeans(x, k=2):
    # Preprocess as usual up to the Gaussian blur step
    while isinstance(x, np.ndarray) and x.dtype == "O":
        x = x[0]

    x = 20 * np.log10(abs(x) / np.amax(abs(x)))
    x = 255 * (x - np.min(x)) / (np.max(x) - np.min(x))
    x = cv2.GaussianBlur(x, (5, 5), sigmaX=1)
    _, x = cv2.threshold(x.astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Flatten image for clustering
    x_flattened = x.reshape((-1, 1))

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x_flattened)
    clustered = kmeans.labels_.reshape(x.shape)

    return clustered


def preprocess_data_with_gmm(x, n_components=2):
    # Preprocess as usual up to the Gaussian blur step
    while isinstance(x, np.ndarray) and x.dtype == "O":
        x = x[0]

    x = 20 * np.log10(abs(x) / np.amax(abs(x)))
    x = 255 * (x - np.min(x)) / (np.max(x) - np.min(x))
    x = cv2.GaussianBlur(x, (5, 5), sigmaX=1)

    # Flatten image for clustering
    x_flattened = x.reshape((-1, 1))

    # Apply Gaussian Mixture Model clustering
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(x_flattened)
    clustered = gmm.predict(x_flattened).reshape(x.shape)

    return clustered


def create_hdf5_file(root_dir, person_list, gestures, thresholding_method="otsu"):
    # Create HDF5 file to save preprocessed training data
    with h5py.File(os.path.join(root_dir, "preprocessed_gesture_data_train.h5"), "w") as h5f:
        iter_person = tqdm(
            person_list,
            desc="Persons",
            unit="persons",
            leave=True,
            position=0,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
            dynamic_ncols=True,
        )
        for person in iter_person:
            # Load .mat file for each person
            mat_path = os.path.join(root_dir, f"Data_Per_PersonData_Training_Person_{person}.mat")
            mat_data = sio.loadmat(mat_path)

            iter_gestures = tqdm(
                gestures,
                desc=f"Person {person}",
                unit="gestures",
                unit_scale=True,
                leave=False,
                position=1,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
                dynamic_ncols=True,
            )
            for gesture in iter_gestures:
                iter_samples = tqdm(
                    range(len(mat_data["Data_Training"]["Doppler_Signals"][0][0][0][gesture])),
                    desc=f"Gesture {gesture}",
                    unit="samples",
                    unit_scale=True,
                    leave=False,
                    position=2,
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
                    dynamic_ncols=True,
                )
                for sample in iter_samples:
                    # Preprocess data
                    x = preprocess_data(
                        mat_data["Data_Training"]["Doppler_Signals"][0][0][0][gesture][sample], thresholding_method
                    )
                    # x = preprocess_data(mat_data["Data_Training"]["Doppler_Signals"][0][0][0][gesture][sample])

                    # Save to HDF5 file with structure /person/gesture/sample
                    grp = h5f.require_group(f"{person}/{gesture}")
                    grp.create_dataset(f"sample_{sample}", data=x, compression="gzip")


# Define the gesture-to-number mapping
gesture_mapping = {"Wave": 0, "Pinch": 1, "Swipe": 2, "Click": 3}


def parse_label(label_string):
    """Parse the label string to extract gesture (as number) and person information."""
    parts = label_string.split()
    gesture_name = parts[0]  # e.g., "Swipe"
    sample_num = parts[1]  # e.g., "16"
    person = parts[-1]  # e.g., "F"

    # Convert gesture to its corresponding number
    gesture_num = gesture_mapping.get(gesture_name, -1)  # Default to -1 if not found

    return gesture_num, sample_num, person


def create_hdf5_test_file(test_file_path, output_dir, thresholding_method="otsu"):
    # Load test data file
    test_data = sio.loadmat(test_file_path)

    # Create HDF5 file to save preprocessed test data
    with h5py.File(os.path.join(output_dir, "preprocessed_gesture_data_test.h5"), "w") as h5f:
        iter_samples = tqdm(
            range(len(test_data["Data_rand"])),
            desc="Test Samples",
            unit="samples",
            leave=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
            dynamic_ncols=True,
        )

        for sample_num in iter_samples:
            # Get each sample and preprocess it
            x = preprocess_data(test_data["Data_rand"][sample_num][0][0][0], thresholding_method)
            # x = preprocess_data(test_data["Data_rand"][sample_num][0][0][0])

            # Extract label information
            label_string = test_data["Data_rand"][sample_num][1][0]
            gesture_num, _, person = parse_label(label_string)

            # Create group structure using person and gesture number: /person/gesture_num/sample_num
            group_path = f"{person}/{gesture_num}"
            grp = h5f.require_group(group_path)
            grp.create_dataset(f"sample_{sample_num}", data=x, compression="gzip")


def plot_sample(test_file_path, sample_num):
    test_data = sio.loadmat(test_file_path)
    x = test_data["Data_rand"][int(sample_num)][0][0][0]

    # Apply dB conversion for visualization
    x = 20 * np.log10(abs(x) / np.amax(abs(x)))

    # Display spectrogram
    plt.imshow(x, vmin=-50, vmax=0, cmap="jet", aspect="auto")
    plt.colorbar()
    plt.title(f"Test Sample {sample_num}")
    plt.xlabel("Time")
    plt.ylabel("Doppler")
    plt.show()


if __name__ == "__main__":
    # Directory for training .mat files and output
    train_dir = "data/gesture/train"
    test_dir = "data/gesture/test"
    person_list = ["A", "B", "C", "D", "E", "F"]
    gestures = [0, 1, 2, 3]  # Wave, Pinch, Swipe, Click
    thresholding_method = "otsu"  # 'otsu' or 'met'

    # Create HDF5 for training data
    create_hdf5_file(train_dir, person_list, gestures, thresholding_method)

    # Path to test .mat file
    test_file_path = "data/gesture/test/Data_For_Test_Random.mat"
    create_hdf5_test_file(test_file_path, test_dir, thresholding_method)

    # Optional: Plot a specific test sample
    # sample_num = input("Please tell me the number of the Sample you want to display: ")
    # plot_sample(test_file_path, sample_num)
