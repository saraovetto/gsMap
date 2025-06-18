"""
Wrapper functions for pytorch.
"""

import torch


def torch_device(index=-1):
    if torch.cuda.is_available():
        if index >= 0:
            return torch.device(f"cuda:{index}")
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def torch_sync():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()
