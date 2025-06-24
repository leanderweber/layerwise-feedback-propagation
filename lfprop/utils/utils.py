import logging
import os
import random

import joblib
import numpy as np
import torch


def set_random_seeds(seed):
    """
    Set random seeds for reproducibility across torch, numpy, and random modules.

    Args:
        seed (int): The seed value to use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def save_rng_state(savepath, device):
    """
    Save the current RNG (random number generator) states for torch, torch.cuda, numpy, and random.

    Args:
        savepath (str): Directory path to save RNG state files.
        device (torch.device or int): CUDA device identifier.
    """
    print("SAVING RNG STATES")
    logging.info("SAVING RNG STATES")

    # Save torch RNG state
    torch_state = torch.get_rng_state()
    joblib.dump(torch_state, os.path.join(savepath, "torch-state.joblib"))

    # Save torch CUDA RNG state
    torch_cuda_state = torch.cuda.get_rng_state(device)
    joblib.dump(torch_cuda_state, os.path.join(savepath, "torch-cuda-state.joblib"))

    # Save numpy RNG state
    np_state = np.random.get_state()
    joblib.dump(np_state, os.path.join(savepath, "np-state.joblib"))

    # Save python random RNG state
    random_state = random.getstate()
    joblib.dump(random_state, os.path.join(savepath, "random-state.joblib"))


def load_rng_state(savepath, device):
    """
    Load and restore RNG (random number generator) states for torch, torch.cuda, numpy, and random.

    Args:
        savepath (str): Directory path to load RNG state files from.
        device (torch.device or int): CUDA device identifier.
    """
    print("LOADING RNG STATES")
    logging.info("LOADING RNG STATES")

    # Load torch RNG state
    torch_state = joblib.load(os.path.join(savepath, "torch-state.joblib"))
    torch.set_rng_state(torch_state)

    # Load torch CUDA RNG state
    torch_cuda_state = joblib.load(os.path.join(savepath, "torch-cuda-state.joblib"))
    torch.cuda.set_rng_state(torch_cuda_state, device)

    # Load numpy RNG state
    np_state = joblib.load(os.path.join(savepath, "np-state.joblib"))
    np.random.set_state(np_state)

    # Load python random RNG state
    random_state = joblib.load(os.path.join(savepath, "random-state.joblib"))
    random.setstate(random_state)
