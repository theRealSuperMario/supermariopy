# TODO: factor this out into separate class
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian
from abc import ABC, abstractmethod
import numpy as np
from typing import *
from supermariopy import imageutils


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
