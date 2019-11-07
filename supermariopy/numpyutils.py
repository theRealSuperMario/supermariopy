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

