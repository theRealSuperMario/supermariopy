import torch
import numpy as np


def to_numpy(x, permute=False):
    """automatically detach and move to cpu if necessary."""
    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            x = x.detach().cpu().numpy()
        else:
            x = x.detach().numpy()
    if permute:
        x = np.transpose(x, (0, 2, 3, 1))  # NCHW --> NHWC
    return x


def to_torch(x, permute=False):
    """automatically convert numpy array to torch and permute channels from NHWC to NCHW"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    x = x.to(device)

    if permute:
        x = x.permute((0, 3, 1, 2))  # NHWC --> NCHW
    return x

