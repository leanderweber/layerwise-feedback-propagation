import torch

def t(x) -> torch.Tensor:
    try:
        return torch.tensor(x)
    except:
        return torch.from_numpy(x)