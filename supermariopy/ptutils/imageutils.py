import torch
from typing import *


def convert_range(
    array: torch.Tensor, input_range: Iterable[int], target_range: Iterable[int]
) -> torch.Tensor:
    """convert range of array from input range to target range

    Parameters
    ----------
    array: torch.Tensor
        array in any shape
    input_range: Iterable[int]
        range of array values in format [min, max]
    output_range: Iterable[int]
        range of rescaled array values in format [min, max]

    Returns
    -------
    torch.Tensor
        rescaled array
        
    Examples
    --------
        t = imageutils.convert_range(np.array([-1, 1]), [-1, 1], [0, 1])
        assert np.allclose(t, np.array([0, 1]))
        t = imageutils.convert_range(np.array([0, 1]), [0, 1], [-1, 1])
        assert np.allclose(t, np.array([-1, 1]))
    """
    if input_range[1] <= input_range[0]:
        raise ValueError
    if target_range[1] <= target_range[0]:
        raise ValueError

    a = input_range[0]
    b = input_range[1]
    c = target_range[0]
    d = target_range[1]
    return (array - a) / (b - a) * (d - c) + c
