import deprecation
import numpy as np
import supermariopy
from supermariopy import denseposelib
from supermariopy import numpyutils as npu


def compute_best_iou_remapping(predicted_labels, true_labels):
    """Compute best possible remapping of predictions to labels in terms of IOU.

    Parameters
    ----------
    predicted_labels : np.ndarray of type np.int or np.uint8
        predicted segmentation labels. shaped [N, H, W]
    true_labels : np.ndarray of type np.int or np.uint8
        true segmentation labels. shaped [N, H, W]

    Returns
    -------
    dict
        remapping dict

    Examples
    --------

        remap_dict = compute_best_iou_remapping(predicted_labels, true_labels)
        remapped_labels = remap_parts(predicted_labels, remap_dict)

    See also
    --------
    .. supermariopy.denseposelib.remap_parts

    """
    unique_labels = np.unique(true_labels)
    # num_unique_labels = len(unique_labels)
    unique_pred_labels = np.unique(predicted_labels)
    best_remappings = {}

    for pred_val in unique_pred_labels:
        pred_i = predicted_labels == pred_val
        pred_i = np.expand_dims(pred_i, axis=-1)
        label_i = np.stack([true_labels == k for k in unique_labels], axis=-1)
        N = np.sum(
            np.sum(label_i, axis=(1, 2)) > 1.0, axis=0
        )  # when part not in GT, then do not count it for normalization
        N = np.reshape(N, (1, -1))
        all_I = np.sum(np.logical_and(label_i, pred_i), axis=(1, 2))
        all_U = np.sum(np.logical_or(label_i, pred_i), axis=(1, 2))

        # if union is 0, writes -1 to IOU to prevent it beeing the maximum
        iou = np.where(all_U > 0.0, all_I / all_U, np.ones_like(all_I) * -1.0)

        best_iou_idx = np.argmax(np.sum(iou, axis=0) / N, axis=-1)
        best_label = unique_labels[best_iou_idx]
        best_remappings[pred_val] = int(np.squeeze(best_label))
    return best_remappings


def segmentation_iou(predicted_labels, true_labels, n_classes=None):
    """Intersection over Union metric.

    If stack of true and predicted labels is provided,
    will calculate IoU over entire stack, and not return mean.

    Parameters
    ----------
    predicted_labels : np.ndarray
        single image labels shaped [H, W] of dtype int or
        stack of predicted labels shaped [N, H, W]
    true_labels : np.ndarray
        single image labels shaped [H, W] of dtype int or
        stack of true labels shaped [N, H, W]
    n_classes : int
        number of classes for segmentation problem.
        If None, will use unique values from true_labels.
        Provide n_classes if not all labels occur in `true_labels`.
    Returns
    -------
    np.ndarray
        array of IoU values
    """

    if n_classes is None:
        classes = np.unique(true_labels)
    else:
        classes = np.arange(n_classes)
    Intersection = np.stack(
        [(true_labels == t) & (predicted_labels == t) for t in classes], axis=-1,
    )
    Union = np.stack(
        [(true_labels == t) | (predicted_labels == t) for t in classes], axis=-1,
    )

    axis = tuple(list(range(len(Intersection.shape) - 1)))
    IoU = np.sum(Intersection, axis=axis) / np.sum(Union, axis=axis)
    return IoU


def segmentation_coverage(predicted_labels, true_labels, n_classes):
    """ coverage is intersection over true.
    It is sometimes used in unsupervised learning as a calibration step.

    Parameters
    ----------
    predicted_labels : np.ndarray
        single image labels shaped [H, W] of dtype int
        or stack of predicted labels shaped [N, H, W]
    true_labels : np.ndarray
        single image labels shaped [H, W] of dtype int
        or stack of true labels shaped [N, H, W]
    n_classes : int
        number of classes for segmentation problem.
        If None, will use unique values from true_labels.
        Provide n_classes if not all labels occur in `true_labels`.
    Returns
    -------
    np.ndarray
        array of coverage values

    References
    ----------
    [1] : Collins2018DeepFeatureFactorization
    """

    if n_classes is None:
        classes = np.unique(true_labels)
    else:
        classes = np.arange(n_classes)
    Intersection = np.stack(
        [(true_labels == t) & (predicted_labels == t) for t in classes], axis=-1
    )
    True_labels = np.stack([true_labels == t for t in classes], axis=-1)

    axis = tuple(list(range(len(Intersection.shape) - 1)))
    coverage = np.sum(Intersection, axis=axis) / np.sum(True_labels, axis=axis)
    coverage = np.nan_to_num(coverage)
    return coverage


def get_best_segmentation(groundtruth_segmentation, inferred_segmentation):
    """Remap inferred segmentation onto ground truth segmentation so that it
    matches the groundtruth_segmentation as good as possible

    Parameters
    ----------
    groundtruth_segmentation : np.ndarray
        Ground truth segmentation label map[N, H, W]

    inferred_segmentation : np.ndarray
        Inferred segmentation label map [N, H, W]
    Returns
    -------
    np.ndarray
        remapped inferred segmentation
    """

    best_remapping = denseposelib.compute_best_iou_remapping(
        inferred_segmentation, groundtruth_segmentation
    )
    remapped_inferred = denseposelib.remap_parts(inferred_segmentation, best_remapping)
    return remapped_inferred


@deprecation.deprecated(
    deprecated_in="0.2",
    removed_in="0.3",
    current_version=supermariopy.__version__,
    details="it is not clear what this metric does exactly.",
)
def segmentation_accuracy(prediction, target, num_classes, target_is_one_hot=False):
    """return per class accuracy

        target : , index map-style

    Parameters
    ----------
    prediction : np.ndarray
        shape [N, H, W, C]
    target : np.ndarray
        shape [N, H, W] or [N, H, W, C], depending on `target_is_one_hot`
    num_classes : int
        number of classes
    target_is_one_hot : bool, optional
        if False, will perform one-hot transformation internally, by default False

    Returns
    -------
    np.array
        array with accuracies shaped [N, num_classes]
    np.array
        mean accuracy across all classes, shaped [N, ]
    """

    if not target_is_one_hot:
        target_one_hot = npu.one_hot(target, num_classes, -1)
    else:
        target_one_hot = target
    prediction = np.argmax(prediction, axis=-1)
    prediction_one_hot = npu.one_hot(prediction, num_classes, -1)
    accuracies = np.mean(
        target_one_hot == prediction_one_hot, axis=(1, 2), keepdims=True
    )
    accuracies = np.reshape(accuracies, (-1, num_classes))
    mean_accuracy = np.mean(accuracies, axis=1)
    return accuracies, mean_accuracy
