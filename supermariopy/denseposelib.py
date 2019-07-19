from matplotlib import pylab as plt
import cv2
import numpy as np
from skimage import measure
from scipy.ndimage.measurements import center_of_mass
from scipy.misc import imresize


def calculate_centroids(labels, cca=False, background=0):
    """ 
    Calculate centroids from label map. 
    Maybe use connected components analysis (cca) before to get only connected labels.
    
    labels : ndarray
        an array of ints giving the corresponding part labels
    cca : bool
        if connected components analysis should be done to isolate connected label regions.
    background: int
        which label is considered background and therefore not calculated

    returns 
    centroids : list of array
        each array contains ints as centroid locations in pixel coordinates
    centroid_labels : list
        list of int label ids corresponding to centroids

    Examples

    from matplotlib import pylab as plt
    import cv2
    import numpy as np

    image_paths = [
        "front_IUV.png",
        "behind_IUV.png",
    ]

    IUV = list(map( cv2.imread, image_paths))
    I = list(map(lambda x: x[:, :, 0], IUV))

    centroids, centroid_labels = calculate_centroids(I[0], True)
    texts = list(map(str, centroid_labels))
    plt.close("all")
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(I[0])
    plot_centroids(ax, centroids, texts)
    """
    unique_labels = np.unique(labels)
    unique_labels = set(unique_labels) - set([background])
    label_masks = [labels == label_id for label_id in unique_labels]

    def _calc_centroid(x):
        """ the actual calc centroid function """
        return np.array(center_of_mass(x)).astype(np.int)

    centroids = []
    centroid_labels = []
    for label_mask, label_id in zip(label_masks, unique_labels):
        if cca:
            connected_labels = measure.label(label_mask)
            connected_ids = set(np.unique(connected_labels))
            connected_ids -= set([0])  # suppress background
            connected_masks = [connected_labels == cid for cid in connected_ids]
            current_centroids = list(map(_calc_centroid, connected_masks))
            centroids += current_centroids
            centroid_labels += [label_id] * len(current_centroids)
        else:
            current_centroids = _calc_centroid(label_mask)
            centroids.append(current_centroids)
            centroid_labels.append(label_id)
    return centroids, centroid_labels


def plot_centroids(ax, centroids, texts):
    """
    plot centroid points into image given axis and texts.
    Centroids can be obtained by @calculate_centroids.

    ax : axis handle
        axis handle where to put text
    centroids : list of arrays
        each array contains ints as centroid locations in pixel coordinates
    texts : `list` of `str`
        texts that should be displayed at centroid location

    Examples

    from matplotlib import pylab as plt
    import cv2
    import numpy as np

    image_paths = [
        "front_IUV.png",
        "behind_IUV.png",
    ]

    IUV = list(map( cv2.imread, image_paths))
    I = list(map(lambda x: x[:, :, 0], IUV))

    centroids, centroid_labels = calculate_centroids(I[0], True)
    texts = list(map(str, centroid_labels))
    plt.close("all")
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(I[0])
    plot_centroids(ax, centroids, texts)
    """
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    for p, t in zip(centroids, texts):
        if np.all(p > 0):
            ax.text(
                p[1], p[0], t, fontsize=10, bbox=props, horizontalalignment="center"
            )


def filter_parts(part_map, included_parts):
    """
    Filter out only included part labels from part_map.

    part_map : ndarray
        an array of part labels (int) for each pixel location
    included_parts : list or array
        array of ints specifying parts that should remain

    returns
    new_part_map : ndarray
        an array where only included parts are present.
    """
    new_part_map = np.zeros_like(part_map)
    for i in included_parts:
        mask = part_map == i
        new_part_map[mask] = part_map[mask]
    return new_part_map


def remap_parts(part_map, remap_dict):
    """
    remaps labels according to a remapping dictionary.
        
    part_map : ndarray
        an array of part labels (int) for each pixel location
    remap_dict : dict
        a dict where each key is an int giving the original part id and 
        each value is an int giving the new part id

    returns
    new_part_map : ndarray
        an array with new labels


    Example:

    from matplotlib import pylab as plt
    import cv2
    import numpy as np

    image_paths = [
        "front_IUV.png",
        "behind_IUV.png",
    ]

    IUV = list(map( cv2.imread, image_paths))
    I = list(map(lambda x: x[:, :, 0], IUV))
    I = I[0] # keep it simple
    semantic_remap_dict = {
        "arm" : ['left_upper_arm',
                'right_upper_arm',
                'left_upper_arm',
                'right_upper_arm',
                'left_lower_arm',
                'right_lower_arm',
                'left_lower_arm',
                'right_lower_arm'
                ],
        "leg" : [
            'back_upper_front_leg',
            'back_upper_left_leg',
            'right_upper_leg',
            'left_upper_leg',
            'back_right_lower_leg',
            'back_left_lower_leg',
            'right_lower_leg',
            'left_lower_leg'
        ],
        'head': ['left_head', 'right_head'],
        'hand': ['right_hand', 'left_hand'],
        'chest': ['chest'],
        'back' : ['back'],
        'foot': ['left_foot', 'right_foot'],
        'background' : ['background']
    }
    new_part_list = list(semantic_remap_dict.keys())

    remap_dict = {}
    for i, new_label in enumerate(new_part_list):
        old_keys = semantic_remap_dict[new_label]
        remap_dict.update({denseposelib.PART_LIST.index(o) : i for o in old_keys})

    print(remap_dict)

    new_I = denseposelib.remap_parts(I, remap_dict)
    plt.imshow(new_I)
    ax = plt.gca()
    centroids, centroid_labels = denseposelib.calculate_centroids(new_I, cca=True)
    texts = list(map(lambda x: new_part_list[x], centroid_labels))
    denseposelib.plot_centroids(ax, centroids, texts)
    """
    new_part_map = np.zeros_like(part_map)
    for old_id, new_id in remap_dict.items():
        mask = part_map == old_id
        new_part_map[mask] = new_id
    return new_part_map


def semantic_remap_dict2remap_dict(semantic_remap_dict, new_part_list):
    """
    Returns a dictionary of (new_label_id, old_label_id) pairs
    that can be used with @remap_parts to regroup the semantic
    annotation from densepose output.

    semantic_remap_dict: dict
        a dictionary where each key is the new part name (e.g. "arm")
        and each value is a list of old part names (e.g. ['left_upper_arm', 'right_upper_arm']).
        The list of old part names have to be from @PART_LIST.
    new_part_list : list
        the complete list of all new part names as `str`

    Example
        semantic_remap_dict = {
            "arm" : ['left_upper_arm',
                    'right_upper_arm',
                    'left_upper_arm',
                    'right_upper_arm',
                    'left_lower_arm',
                    'right_lower_arm',
                    'left_lower_arm',
                    'right_lower_arm'
                    ],
            "leg" : [
                'back_upper_front_leg',
                'back_upper_left_leg',
                'right_upper_leg',
                'left_upper_leg',
                'back_right_lower_leg',
                'back_left_lower_leg',
                'right_lower_leg',
                'left_lower_leg'
            ],
            'head': ['left_head', 'right_head'],
            'hand': ['right_hand', 'left_hand'],
            'chest': ['chest'],
            'back' : ['back'],
            'foot': ['left_foot', 'right_foot'],
            'background' : ['background']
        }
        new_part_list = list(semantic_remap_dict.keys())
        remap_dict = semantic_remap_dict2remap_dict(semantic_remap_dict, new_part_list)

    """
    remap_dict = {}
    for i, new_label in enumerate(new_part_list):
        old_keys = semantic_remap_dict[new_label]
        remap_dict.update({PART_LIST.index(o): i for o in old_keys})
    return remap_dict


def compute_iou(pred, label):
    """
    compoute iou between predicted labels and labels

    pred : ndarray of shape [H, W, N] and dtype int
        array with predicted labels
    label : ndarray of shape [H, W, N] and dtype int
        array with ground truth labels

    Returns
    IOU : ndarray of shape [N]
        array with IOUs
    unique_labels : ndarray of shape [n]
        array with unique labels in of GT label array
    """
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels)

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))

    return I / U, unique_labels


def compute_best_iou_remapping(pred, label):
    """
    given predicted labels, compute best possible remapping of predictions
    to labels so that it maximizes each labels iou.
    pred shape : ndarray of shape [batch, H, W] and dtype int where each item is a label map
    label shape : ndarray of shape [batch, H, W] and dtype int where each item is a label map

    returns : 
    best_remappings: dict
        a dictionary where each key is an int representing the old key and each value is an int representing
        the new key

    """
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels)
    unique_pred_labels = np.unique(pred)
    best_remappings = {}

    for index, pred_val in enumerate(unique_pred_labels):
        pred_i = pred == pred_val
        pred_i = np.expand_dims(pred_i, axis=-1)
        label_i = np.stack([label == k for k in unique_labels], axis=-1)
        N = np.sum(
            np.sum(label_i, axis=(1, 2)) > 1.0, axis=0
        )  # when part not in GT, then do not count it for normalization
        N = np.reshape(N, (1, -1))
        all_I = np.sum(np.logical_and(label_i, pred_i), axis=(1, 2))
        all_U = np.sum(np.logical_or(label_i, pred_i), axis=(1, 2))

        # if union is 0, writes -1 to IOU to prevent it beeing the maximum
        iou = np.where(all_U > 0.0, all_I / all_U, np.ones_like(all_I) * -1.0)

        best_iou_idx = np.argmax(np.sum(iou, axis=0) / N, axis=-1)
        best_iou = iou[best_iou_idx]
        best_label = unique_labels[best_iou_idx]
        best_remappings[pred_val] = int(np.squeeze(best_label))
    return best_remappings


def resize_labels(labels, size):
    label_list = np.split(labels, labels.shape[0], axis=0)
    label_list = list(
        map(lambda x: imresize(np.squeeze(x), size, interp="nearest"), label_list)
    )
    labels = np.stack(label_list, axis=0)
    return labels


PART_DICT_ID2STR = {
    0: "background",
    1: "back",
    2: "chest",
    3: "right_hand",
    4: "left_hand",
    5: "left_foot",
    6: "right_foot",
    7: "back_upper_front_leg",
    8: "back_upper_left_leg",
    9: "right_upper_leg",
    10: "left_upper_leg",
    11: "back_right_lower_leg",
    12: "back_left_lower_leg",
    13: "right_lower_leg",
    14: "left_lower_leg",
    15: "left_upper_arm",
    16: "right_upper_arm",
    17: "left_upper_arm",
    18: "right_upper_arm",
    19: "left_lower_arm",
    20: "right_lower_arm",
    21: "left_lower_arm",
    22: "right_lower_arm",
    23: "left_head",
    24: "right_head",
}

PART_LIST = list(PART_DICT_ID2STR.values())
