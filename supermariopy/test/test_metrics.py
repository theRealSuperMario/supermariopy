import numpy as np
from supermariopy import metrics
from supermariopy import numpyutils as npu


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


def test_segmentation_iou():
    """ Test 1: IOU of perfect match should be 1.0 """
    a = np.zeros((100, 100), dtype=np.int32)
    a[:50, :50] = 1
    b = a.copy()

    iou = metrics.segmentation_iou(a, b, n_classes=None)
    assert np.allclose(iou, np.ones_like(iou))

    iou = metrics.segmentation_iou(a, b, n_classes=2)
    assert np.allclose(iou, np.ones_like(iou))

    """ Test 2:
    a is 25% 1, 75% 0
    b is 100% 0

    IoU class 0 : 0.75 / 1 = 0.75
    IoU class 1 : 0 (no intersection)
    """
    a = np.zeros((100, 100), dtype=np.int32)
    a[:50, :50] = 1
    b = np.zeros((100, 100), dtype=np.int32)

    iou = metrics.segmentation_iou(b, a, n_classes=None)
    assert np.allclose(iou, np.array([0.75, 0]))

    iou = metrics.segmentation_iou(a, b, n_classes=2)
    assert np.allclose(iou, np.array([0.75, 0]))

    """ Test 3:
    a is 50% 0, 50% 1, horizontally split
    b is 50% 0, 50% 1, vertically split

    Overlap is 25%

    IoU class 0 = IoU class 1
    IoU = [0.5 / 0.75, 0.5 / 0.75]
    """
    a = np.zeros((100, 100), dtype=np.int32)
    a[:50, :] = 1
    b = np.zeros((100, 100), dtype=np.int32)
    b[:, :50] = 1

    iou = metrics.segmentation_iou(a, b, n_classes=None)
    assert np.allclose(iou, np.array([0.25 / 0.75, 0.25 / 0.75]))

    iou = metrics.segmentation_iou(a, b, n_classes=2)
    assert np.allclose(iou, np.array([0.25 / 0.75, 0.25 / 0.75]))


def test_segmentation_coverage():
    """ Test 1: covarge of perfect match should be 1.0 """
    a = np.zeros((100, 100), dtype=np.int32)
    a[:50, :50] = 1
    b = a.copy()

    coverage = metrics.segmentation_coverage(a, b, n_classes=None)
    assert np.allclose(coverage, np.ones_like(coverage))

    coverage = metrics.segmentation_coverage(a, b, n_classes=2)
    assert np.allclose(coverage, np.ones_like(coverage))

    """ Test 2:
    a is 25% 1, 75% 0
    b is 100% 0

    coverage class 0 : 0.75 / 1 = 0.75
    coverage class 1 : 0 (no intersection)
    """
    a = np.zeros((100, 100), dtype=np.int32)
    a[:50, :50] = 1
    b = np.zeros((100, 100), dtype=np.int32)

    coverage = metrics.segmentation_coverage(a, b, n_classes=2)
    assert np.allclose(coverage, np.array([0.75, 0]))

    """ Test 3:
    a is 50% 0, 50% 1, horizontally split
    b is 50% 0, 50% 1, vertically split

    Overlap is 25%

    coverage class 0 = IoU class 1
    coverage = [0.25 / 0.5, 0.25 / 0.5]
    """
    a = np.zeros((100, 100), dtype=np.int32)
    a[:50, :] = 1
    b = np.zeros((100, 100), dtype=np.int32)
    b[:, :50] = 1

    coverage = metrics.segmentation_coverage(a, b, n_classes=None)
    assert np.allclose(coverage, np.array([0.25 / 0.5, 0.25 / 0.5]))

    coverage = metrics.segmentation_coverage(a, b, n_classes=2)
    assert np.allclose(coverage, np.array([0.25 / 0.5, 0.25 / 0.5]))
