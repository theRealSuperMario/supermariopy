import numpy as np


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
