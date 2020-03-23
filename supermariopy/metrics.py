import numpy as np
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
