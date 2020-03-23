import pytest
import numpy as np
from supermariopy import numpyutils as npu
from supermariopy import metrics


def test_segmentation_accuracy():
    n_classes = 10
    H = 128
    target = np.random.randint(0, 10, (2, H, H))
    prediction = npu.one_hot(target, n_classes, axis=-1)

    accuracy, _ = metrics.segmentation_accuracy(
        prediction, target, n_classes, target_is_one_hot=False
    )
    assert np.allclose(accuracy, np.ones_like(accuracy))
    assert accuracy.shape == (2, n_classes)

    target_one_hot = npu.one_hot(target, n_classes, axis=-1)
    accuracy, _ = metrics.segmentation_accuracy(
        prediction, target_one_hot, n_classes, target_is_one_hot=True
    )
    assert np.allclose(accuracy, np.ones_like(accuracy))
    assert accuracy.shape == (2, n_classes)

