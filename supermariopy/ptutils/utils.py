import numpy as np
import torch
from supermariopy.ptutils import compat as ptcompat


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
    """automatically convert numpy array to torch and permute
    channels from NHWC to NCHW"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        x = x.to(device)

    if permute:
        x = x.permute((0, 3, 1, 2))  # NHWC --> NCHW
    if x.dtype is torch.float64:
        x = x.type(torch.float32)
    return x


def split_stack(x, split_sizes, split_dim, stack_dim):
    """Split x along dimension split_dim and stack again at dimension stack_dim"""
    t = torch.stack(torch.split(x, split_sizes, dim=split_dim), dim=stack_dim)
    return t


def split_stack_reshape(x, split_sizes=3):
    """split x at dimension 1, stack them again and reshape into batch axis."""
    t = split_stack(x, split_sizes, split_dim=1, stack_dim=1)
    shape_ = list(x.shape)
    shape_[0] = -1
    shape_[1] = split_sizes
    return t.view(shape_)


def linear_variable(
    step, start, end, start_value, end_value, clip_min=0.0, clip_max=1.0
):
    """linear from (a, alpha) to (b, beta), i.e.
    (beta - alpha) / (b - a) * (x - a) + alpha """
    if not isinstance(step, torch.Tensor):
        step = torch.tensor(step)
    linear = (end_value - start_value) / (end - start) * (
        ptcompat.torch_astype(step, torch.float32) - start
    ) + start_value
    return linear.clamp(clip_min, clip_max)
