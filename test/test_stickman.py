import pytest
from supermariopy import stickman
from supermariopy import imageutils
from supermariopy import plotting
from matplotlib import pyplot as plt
import numpy as np


class Test_example_joint_models:
    @pytest.mark.mpl_image_compare
    def test_joint_model(self):
        # kps = stickman.EXAMPLE_JOINT_MODELS["JointModel_15"]
        # joint_img = stickman.make_joint_img((128, 128), kps, stickman.JointModel_15)
        kps = stickman.VUNetStickman.get_example_valid_keypoints()
        joint_img = stickman.VUNetStickman.make_joint_img(
            (128, 128),
            stickman.VUNET_JOINT_ORDER,
            stickman.VUNetStickman.get_example_valid_keypoints() * 128,
        )

        plt.imshow(joint_img)
        return plt.gcf()

    def test_get_bounding_box(self):
        kps = stickman.VUNetStickman.get_example_valid_keypoints()
        box = stickman.get_bounding_boxes(kps, (128, 128), 32)
        joint_img = stickman.VUNetStickman.make_joint_img(
            (128, 128),
            stickman.VUNET_JOINT_ORDER,
            stickman.VUNetStickman.get_example_valid_keypoints() * 128,
        )

        box_image = plotting.overlay_boxes_without_labels(joint_img, box)
        plt.imshow(box_image)
        return plt.gcf()

    def test_invalid_stickman(self):
        kps = stickman.VUNetStickman.get_example_invalid_keypoints()
        box = stickman.get_bounding_boxes(kps, (128, 128), 32)
        joint_img = stickman.VUNetStickman.make_joint_img(
            (128, 128),
            stickman.VUNET_JOINT_ORDER,
            stickman.VUNetStickman.get_example_invalid_keypoints() * 128,
        )

        box_image = plotting.overlay_boxes_without_labels(joint_img, box)
        plt.imshow(box_image)
        return plt.gcf()
