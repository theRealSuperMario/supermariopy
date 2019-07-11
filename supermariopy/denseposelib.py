from matplotlib import pylab as plt
import cv2
import numpy as np
from skimage import measure
from scipy.ndimage.measurements import center_of_mass

def calculate_centroids(labels, cca=False):
    ''' 
    Calculate centroids from label map. 
    Maybe use connected components analysis (cca) before to get only connected labels.
    
    labels : ndarray
        an array of ints giving the corresponding part labels
    cca : bool
        if connected components analysis should be done to isolate connected label regions.

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
    '''
    unique_labels = np.unique(labels)
    label_masks = [labels == label_id for label_id in unique_labels]
    
    def _calc_centroid(x):
        ''' the actual calc centroid function '''
        return np.array(center_of_mass(x)).astype(np.int)
    
    centroids = []
    centroid_labels = []
    for label_mask, label_id in zip(label_masks, unique_labels):
        if cca:
            connected_labels = measure.label(label_mask)
            connected_ids = set(np.unique(connected_labels))
            connected_ids -= set([0]) # suppress background
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
    '''
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
    '''
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for p, t in zip(centroids, texts):
        if np.all(p > 0):
            ax.text(p[1], p[0], t, fontsize=10, bbox=props)

def filter_parts(part_map, included_parts):
    '''
    Filter out only included part labels from part_map

    part_map : ndarray
        an array of part labels (int) for each pixel location
    included_parts : list or array
        array of ints specifying parts that should remain

    returns
    new_part_map : ndarray
        an array where only included parts are present.
    '''
    new_part_map = np.zeros_like(part_map)
    for i in included_parts:
        mask = part_map == i
        new_part_map[mask] = part_map[mask]
    return new_part_map
