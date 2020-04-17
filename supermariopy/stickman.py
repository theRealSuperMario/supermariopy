import cv2
import numpy as np
import torch
from torchvision.utils import make_grid
import os.path as osp
import random
from collections import namedtuple
from skimage import io
from skimage.draw import circle, line, line_aa
import numpy as np
import cv2

# works for coco at least
VUNET_JOINT_ORDER = [
    "rankle",
    "rknee",
    "rhip",
    "lhip",
    "lknee",
    "lankle",
    "rwrist",
    "relbow",
    "rshoulder",
    "lshoulder",
    "lelbow",
    "lwrist",
    "cnose",
    "leye",
    "reye",
]


class VUNetStickman:
    """Stickman image generator according to https://github.com/CompVis/vunet/blob/master/batches.py"""

    @staticmethod
    def get_example_valid_keypoints():
        example_keypoints = np.array(
            [
                [0.48106802, 0.8998802],
                [0.44145063, 0.5546431],
                [0.5546431, 0.45276988],
                [0.65085673, 0.43579102],
                [0.5489835, 0.5999201],
                [0.6112394, 0.9111994],
                [0.3395774, 0.32825816],
                [0.44711027, 0.2886408],
                [0.5037065, 0.1924272],
                [0.6225586, 0.16412908],
                [0.5999201, 0.29430044],
                [0.4301314, 0.35089666],
                [0.51502573, 0.10187323],
                [0.526345, 0.07923473],
                [0.49238726, 0.07923473],
            ],
        )
        return example_keypoints

    @staticmethod
    def get_example_invalid_keypoints():
        example_keypoints = np.array(
            [
                [0.38960117, 0.9117471],
                [0.66272366, 0.6988722],
                [0.32533708, 0.5100964],
                [0.1164787, 0.4618983],
                [-1.0, -1.0],
                [-1.0, -1.0],
                [0.87961507, 0.5221459],
                [0.71895474, 0.3936177],
                [0.562311, 0.14861076],
                [0.26910597, 0.08033014],
                [-1.0, -1.0],
                [-1.0, -1.0],
                [-1.0, -1.0],
                [-1.0, -1.0],
                [-1.0, -1.0],
            ]
        )
        return example_keypoints

    @staticmethod
    def make_joint_img(img_shape, jo, joints):
        # three channels: left, right, center
        scale_factor = img_shape[1] / 128
        thickness = int(3 * scale_factor)
        imgs = list()
        for i in range(3):
            imgs.append(np.zeros(img_shape[:2], dtype="uint8"))

        body = ["lhip", "lshoulder", "rshoulder", "rhip"]
        body_pts = np.array([[joints[jo.index(part), :] for part in body]])
        if np.min(body_pts) >= 0:
            body_pts = np.int_(body_pts)
            cv2.fillPoly(imgs[2], body_pts, 255)

        right_lines = [
            ("rankle", "rknee"),
            ("rknee", "rhip"),
            ("rhip", "rshoulder"),
            ("rshoulder", "relbow"),
            ("relbow", "rwrist"),
        ]
        for line in right_lines:
            l = [jo.index(line[0]), jo.index(line[1])]
            if np.min(joints[l]) >= 0:
                a = tuple(np.int_(joints[l[0]]))
                b = tuple(np.int_(joints[l[1]]))
                cv2.line(imgs[0], a, b, color=255, thickness=thickness)

        left_lines = [
            ("lankle", "lknee"),
            ("lknee", "lhip"),
            ("lhip", "lshoulder"),
            ("lshoulder", "lelbow"),
            ("lelbow", "lwrist"),
        ]
        for line in left_lines:
            l = [jo.index(line[0]), jo.index(line[1])]
            if np.min(joints[l]) >= 0:
                a = tuple(np.int_(joints[l[0]]))
                b = tuple(np.int_(joints[l[1]]))
                cv2.line(imgs[1], a, b, color=255, thickness=thickness)

        rs = joints[jo.index("rshoulder")]
        ls = joints[jo.index("lshoulder")]
        cn = joints[jo.index("cnose")]
        neck = 0.5 * (rs + ls)
        a = tuple(np.int_(neck))
        b = tuple(np.int_(cn))
        if np.min(a) >= 0 and np.min(b) >= 0:
            cv2.line(imgs[0], a, b, color=127, thickness=thickness)
            cv2.line(imgs[1], a, b, color=127, thickness=thickness)

        cn = tuple(np.int_(cn))
        leye = tuple(np.int_(joints[jo.index("leye")]))
        reye = tuple(np.int_(joints[jo.index("reye")]))
        if np.min(reye) >= 0 and np.min(leye) >= 0 and np.min(cn) >= 0:
            cv2.line(imgs[0], cn, reye, color=255, thickness=thickness)
            cv2.line(imgs[1], cn, leye, color=255, thickness=thickness)

        img = np.stack(imgs, axis=-1)
        if img_shape[-1] == 1:
            img = np.mean(img, axis=-1)[:, :, None]
        return img

    @staticmethod
    def valid_joints(*joints):
        """ list of [N, 2] keypoints """
        j = np.stack(joints)
        return (j >= 0).all()

    @staticmethod
    def normalize(imgs, coords, stickmen, jo, box_factor):
        out_imgs = list()
        out_stickmen = list()

        bs = len(imgs)
        for i in range(bs):
            img = imgs[i]
            joints = coords[i]
            stickman = stickmen[i]

            h, w = img.shape[:2]
            o_h = h
            o_w = w
            h = h // 2 ** box_factor
            w = w // 2 ** box_factor
            wh = np.array([w, h])
            wh = np.expand_dims(wh, 0)

            bparts = [
                ["lshoulder", "lhip", "rhip", "rshoulder"],
                ["lshoulder", "rshoulder", "cnose"],
                ["lshoulder", "lelbow"],
                ["lelbow", "lwrist"],
                ["rshoulder", "relbow"],
                ["relbow", "rwrist"],
                ["lhip", "lknee"],
                ["rhip", "rknee"],
            ]
            ar = 0.5

            part_imgs = list()
            part_stickmen = list()
            for bpart in bparts:
                part_img = np.zeros((h, w, 3))
                part_stickman = np.zeros((h, w, 3))
                M = VUNetStickman.get_crop(bpart, joints, jo, wh, o_w, o_h, ar)

                if M is not None:
                    part_img = cv2.warpPerspective(
                        img, M, (h, w), borderMode=cv2.BORDER_REPLICATE
                    )
                    part_stickman = cv2.warpPerspective(
                        stickman, M, (h, w), borderMode=cv2.BORDER_REPLICATE
                    )

                part_imgs.append(part_img)
                part_stickmen.append(part_stickman)
            img = np.concatenate(part_imgs, axis=2)
            stickman = np.concatenate(part_stickmen, axis=2)

            out_imgs.append(img)
            out_stickmen.append(stickman)
        out_imgs = np.stack(out_imgs)
        out_stickmen = np.stack(out_stickmen)
        return out_imgs, out_stickmen

    @staticmethod
    def get_crop(bpart, joints, jo, wh, o_w, o_h, ar=1.0):
        bpart_indices = [jo.index(b) for b in bpart]
        part_src = np.float32(joints[bpart_indices])

        # fall backs
        if not valid_joints(part_src):
            if bpart[0] == "lhip" and bpart[1] == "lknee":
                bpart = ["lhip"]
                bpart_indices = [jo.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices])
            elif bpart[0] == "rhip" and bpart[1] == "rknee":
                bpart = ["rhip"]
                bpart_indices = [jo.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices])
            elif (
                bpart[0] == "lshoulder"
                and bpart[1] == "rshoulder"
                and bpart[2] == "cnose"
            ):
                bpart = ["lshoulder", "rshoulder", "rshoulder"]
                bpart_indices = [jo.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices])

        if not valid_joints(part_src):
            return None

        if part_src.shape[0] == 1:
            # leg fallback
            a = part_src[0]
            b = np.float32([a[0], o_h - 1])
            part_src = np.float32([a, b])

        if part_src.shape[0] == 4:
            pass
        elif part_src.shape[0] == 3:
            # lshoulder, rshoulder, cnose
            if bpart == ["lshoulder", "rshoulder", "rshoulder"]:
                segment = part_src[1] - part_src[0]
                normal = np.array([-segment[1], segment[0]])
                if normal[1] > 0.0:
                    normal = -normal

                a = part_src[0] + normal
                b = part_src[0]
                c = part_src[1]
                d = part_src[1] + normal
                part_src = np.float32([a, b, c, d])
            else:
                assert bpart == ["lshoulder", "rshoulder", "cnose"]
                neck = 0.5 * (part_src[0] + part_src[1])
                neck_to_nose = part_src[2] - neck
                part_src = np.float32([neck + 2 * neck_to_nose, neck])

                # segment box
                segment = part_src[1] - part_src[0]
                normal = np.array([-segment[1], segment[0]])
                alpha = 1.0 / 2.0
                a = part_src[0] + alpha * normal
                b = part_src[0] - alpha * normal
                c = part_src[1] - alpha * normal
                d = part_src[1] + alpha * normal
                # part_src = np.float32([a,b,c,d])
                part_src = np.float32([b, c, d, a])
        else:
            assert part_src.shape[0] == 2

            segment = part_src[1] - part_src[0]
            normal = np.array([-segment[1], segment[0]])
            alpha = ar / 2.0
            a = part_src[0] + alpha * normal
            b = part_src[0] - alpha * normal
            c = part_src[1] - alpha * normal
            d = part_src[1] + alpha * normal
            part_src = np.float32([a, b, c, d])

        dst = np.float32([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
        part_dst = np.float32(wh * dst)

        M = cv2.getPerspectiveTransform(part_src, part_dst)
        return M


def n_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def valid_joints(*joints):
    j = np.stack(joints)
    return (j >= 0).all()


def add_joints_to_img(img, kps, joints, color=[0, 0, 255]):
    # params
    border_safety = 25
    h, w = img.shape[0:2]
    r_1 = int(h / 250)

    # draw keypoints
    for kp in kps:
        x = np.min([w - border_safety, kp[0]])  # x
        y = np.min([h - border_safety, kp[1]])  # y
        rr, cc = circle(y, x, r_1)
        img[rr, cc, 0] = color[0]
        img[rr, cc, 1] = color[1]
        img[rr, cc, 2] = color[2]

    # draw joints
    for jo in joints:
        rr, cc, val = line_aa(
            int(kps[jo[0], 1]),
            int(kps[jo[0], 0]),
            int(kps[jo[1], 1]),
            int(kps[jo[1], 0]),
        )  # [jo_0_y, jo_0_x, jo_1_y, jo_1_x]

        img[rr, cc, 0] = color[0] * val
        img[rr, cc, 1] = color[1]
        img[rr, cc, 2] = color[2]

    return img


def get_bounding_boxes(kps, img_size, box_size):
    """ Return bounding box coordinates around keypoints in format XYXY.
        Simply add and subtract a fixed box_size from the keypoints.
        Keypoint format is [N, 2] and X, Y in range [0, 1].

        Note that bounding box coordinates are not clipped to the image size, yet
    """
    kps *= np.array(img_size).reshape((-1, 2))
    half_width = box_size // 2
    offset = np.array([-half_width, -half_width, half_width, half_width])
    box_coordinates = np.concatenate([kps, kps], -1) + offset.reshape((-1, 4))
    box_list = np.split(box_coordinates, box_coordinates.shape[0], axis=0)
    box_list = [np.squeeze(b) for b in box_list]
    return box_list

