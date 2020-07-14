# TODO: factor this out into separate class
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian
from abc import ABC, abstractmethod
import numpy as np
from typing import *
from supermariopy import imageutils

# class CRF:
# """inspired by https://gist.github.com/pesser/0ba227dd1a7b55e96b482a61cc74cad1"""


class SegmentationAlgorithm(ABC):
    @abstractmethod
    def __call__(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        pass


class SegmentationFromKeypoints(SegmentationAlgorithm):
    def __init__(self, var=0.1, n_steps=5):
        self.var = var
        self.n_steps = n_steps

    def __call__(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        if not imageutils.is_in_range(image, [0, 255]):
            raise imageutils.RangeError(image, "image", [0, 255])

        dynamic_range = image.max() / image.min()
        is_low_range = dynamic_range <= 2.0
        if is_low_range:
            import warnings

            warnings.warn(
                Warning("image has low dynamic range. Maybe convert to [0, 255]?")
            )

        h, w, c = image.shape
        keypoint_probs = imageutils.keypoints_to_heatmaps(
            (h, w), keypoints, var=self.var
        )
        keypoint_probs = np.rollaxis(keypoint_probs, 2, 0)
        bg_prob = 1.0 - np.amax(keypoint_probs, axis=0, keepdims=True)
        probs = np.concatenate([bg_prob, keypoint_probs], axis=0)
        n_labels = probs.shape[0]
        probs_flat = np.reshape(probs, (n_labels, -1))

        d = dcrf.DenseCRF(h * w, n_labels)

        # flatten everything for dense crf

        # Set unary according to keypoints
        U = -np.log(probs_flat + 1.0e-6).astype(np.float32)
        d.setUnaryEnergy(U.copy(order="C"))

        # This creates the color-independent features and then add them to the CRF
        feats = create_pairwise_gaussian(sdims=(3, 3), shape=image.shape[:2])
        d.addPairwiseEnergy(
            feats,
            compat=3,
            kernel=dcrf.DIAG_KERNEL,
            normalization=dcrf.NORMALIZE_SYMMETRIC,
        )

        # This creates the color-dependent features and then add them to the CRF
        feats = create_pairwise_bilateral(
            sdims=(80, 80), schan=(13, 13, 13), img=image, chdim=2
        )
        d.addPairwiseEnergy(
            feats,
            compat=10,
            kernel=dcrf.DIAG_KERNEL,
            normalization=dcrf.NORMALIZE_SYMMETRIC,
        )

        # Run five inference steps.
        Q = d.inference(self.n_steps)

        # Find out the most probable class for each pixel.
        MAP = np.argmax(Q, axis=0)
        MAP = MAP.reshape((h, w))
        return MAP


# TODO: add unary from Guided Filtering (Collins2018DeepFeatureFactorization)


def run_crf(
    image,
    unary,
    n_labels,
    sxy_bilateral=80,
    srgb_bilateral=13,
    sxy_pairwise=3,
    compat_bilateral=10,
    compat_pairwise=3,
    pairwise=True,
):
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], n_labels)
    d.setUnaryEnergy(unary)
    d.addPairwiseBilateral(
        sxy=(sxy_bilateral, sxy_bilateral),
        srgb=(srgb_bilateral, srgb_bilateral, srgb_bilateral),
        rgbim=image,
        compat=compat_pairwise,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )
    if pairwise:
        d.addPairwiseGaussian(sxy=sxy_pairwise, compat=compat_bilateral)
    Q = d.inference(10)
    map_ = np.argmax(Q, axis=0).reshape(image.shape[:2])
    return map_


def process_batches(
    t: Union[np.ndarray, np.ndarray], segmentation_algorithm: SegmentationAlgorithm
) -> Dict:
    """Load data from .npy file and infer segmentation
    """
    print("processing batches")
    img_batch, keypoints_batch = t
    func = functools.partial(
        batched_keypoints_to_segments,
        **{"segmentation_algorithm": segmentation_algorithm}
    )
    labels, labels_rgb, heatmaps, ims_with_keypoints = imageutils.np_map_fn(
        lambda x: func(x[0], x[1]), (img_batch, keypoints_batch)
    )
    heatmaps = np.squeeze(heatmaps).astype(np.float32)
    labels_rgb = labels_rgb.astype(np.float32)

    processed_data = {
        "labels": labels,
        "labels_rgb": labels_rgb,
        "heatmaps": heatmaps,
        "ims_with_keypoints": ims_with_keypoints,
    }
    return processed_data


def batched_keypoints_to_segments(
    img: np.ndarray,
    keypoints: np.ndarray,
    segmentation_algorithm: SegmentationAlgorithm,
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_keypoints = keypoints.shape[0]
    MAP = segmentation_algorithm(img, keypoints)
    MAP_colorized = imageutils.make_colors(
        n_keypoints + 1, with_background=True, background_id=0
    )[MAP]
    heatmaps = imageutils.keypoints_to_heatmaps(
        img.shape[:2], keypoints, segmentation_algorithm.var
    )
    heatmaps *= heatmaps > 0.8
    heatmaps_rgb = imageutils.colorize_heatmaps(
        heatmaps[np.newaxis, ...], imageutils.make_colors(n_keypoints)
    )

    img_resized = cv2.resize(img, (256, 256), cv2.INTER_LINEAR)
    img_resized = imageutils.convert_range(img_resized, [0, 255], [0, 1])
    im_with_keypoints = imageutils.draw_keypoint_markers(
        img_resized,
        keypoints,
        marker_list=[str(i) for i in range(10)] + ["x", "o", "v", "<", ">", "*"],
        font_scale=1,
        thickness=4,
    )
    im_with_keypoints = cv2.resize(
        im_with_keypoints, (img.shape[1], img.shape[1]), cv2.INTER_LINEAR
    )
    return MAP, MAP_colorized, heatmaps_rgb, im_with_keypoints
