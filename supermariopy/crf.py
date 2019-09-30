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

