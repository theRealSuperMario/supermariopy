from typing import *
import numpy as np


def one_hot(a, num_classes, axis=-1):
    """numpy equivalent of tf.one_hot
    
    Parameters
    ----------
    a : np.ndarray
        numpy array of any shape
    num_classes : int
        number of classes for one-hot representation
    axis : int, optional
        along which axis to make one_hot, by default -1
    
    Returns
    -------
    array
        one-hot array
    """

    a_onehot = np.eye(num_classes)[a.reshape(-1)]
    a_onehot = np.reshape(a_onehot, a.shape + (num_classes,))
    a_onehot = np.rollaxis(a_onehot, -1, start=axis)
    return a_onehot


def argmax_one_hot(a, axis=1):
    """ short for one_hot(np.argmax(a, axis=-1(), a.shape[-1]) """
    B, H, W, P = a.shape
    argmax_map = np.argmax(a, axis=-1)
    m_one_hot = one_hot(argmax_map, P, axis=-1)
    return m_one_hot


def np_map_fn(func: Callable, data: Tuple) -> Tuple:
    """map func along axis 0 of each item in data.

    # TODO: fails when tuple has length 1

    Similar to tf.map_fn
    
    Parameters
    ----------
    func : Callable
        function to map to the items in data
    data : Tuple[np.ndarray]
        
    Returns
    -------
    Tuple[np.ndarray]
        function `func` applied to each element in `data`

    Examples
    --------
        data = (np.arange(10).reshape(10, 1), np.arange(10)[::-1].reshape(10, 1))
        output = np_map_fn(lambda x: (x[0]**2, x[1]**2), data)
        output[0].squeeze()
        >>> array([ 0,  1,  4,  9, 16, 25, 36, 49, 64, 81])

        output = np_map_fn(lambda x: (x[0]**2, x[1]**2), data)
        output[0].shape
        >>> (10, 1)
    """
    generator = zip(*map(lambda x: [x[i, ...] for i in range(x.shape[0])], data))
    # (data[0][0], data[0][1], ...), (data[1][0], data[1][1], ...), ...
    outputs = map(func, generator)
    outputs = list(map(np.stack, zip(*outputs)))
    return outputs
