from collections import namedtuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from skimage.draw import circle, line_aa

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
    """Stickman image generator according to
    https://github.com/CompVis/vunet/blob/master/batches.py"""

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
            joint_idx = [jo.index(line[0]), jo.index(line[1])]
            if np.min(joints[joint_idx]) >= 0:
                a = tuple(np.int_(joints[joint_idx[0]]))
                b = tuple(np.int_(joints[joint_idx[1]]))
                cv2.line(imgs[0], a, b, color=255, thickness=thickness)

        left_lines = [
            ("lankle", "lknee"),
            ("lknee", "lhip"),
            ("lhip", "lshoulder"),
            ("lshoulder", "lelbow"),
            ("lelbow", "lwrist"),
        ]
        for line in left_lines:
            joint_idx = [jo.index(line[0]), jo.index(line[1])]
            if np.min(joints[joint_idx]) >= 0:
                a = tuple(np.int_(joints[joint_idx[0]]))
                b = tuple(np.int_(joints[joint_idx[1]]))
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


JOINT_ORDER_2 = [
    "r_hip",
    "r_knee",
    "r_foot_rear",
    "r_foot_mid",
    "r_foot_front",
    "l_hip",
    "l_knee",
    "l_foot_rear",
    "l_foot_mid",
    "l_foot_front",
    "neck",
    "nose",
    "head",
    "l_shoulder",
    "l_elbow",
    "l_wirst",
    "l_hand",
    "r_shoulder",
    "r_elbow",
    "r_wrist",
    "r_hand",
]


JointModel = namedtuple(
    "JointModel",
    # "body right_lines left_lines head_lines face rshoulder lshoulder
    # headup kps_to_use right_hand left_hand head_part
    # total_relative_joints kp_to_joint kps_to_change kps_to_change_rel norm_T",
    (
        "body right_lines left_lines head_lines face rshoulder lshoulder "
        "headup keypoints_to_plot right_hand left_hand head_part "
        "joints_to_plot keypoint_idx_to_joint joint_to_keypoint_idx"
    ),
)


def reversed(forward):
    _reversed = {v: k for k, v in forward.items()}
    return _reversed


# TODO overload namedtuple
_keypoint_idx_to_joint = {
    1: "r_hip",
    2: "r_knee",
    3: "r_foot_rear",
    4: "r_foot_mid",
    5: "r_foot_front",
    6: "l_hip",
    7: "l_knee",
    8: "l_foot_rear",
    9: "l_foot_mid",
    10: "l_foot_front",
    13: "neck",
    14: "nose",
    15: "head",
    17: "l_shoulder",
    18: "l_elbow",
    19: "l_wrist",
    22: "l_hand",
    25: "r_shoulder",
    26: "r_elbow",
    27: "r_wrist",
    30: "r_hand",
}

_joint_to_keypoint_idx = reversed(_keypoint_idx_to_joint)

JointModelHuman36 = JointModel(
    body=[1, 25, 13, 17, 6],
    right_lines=[
        (5, 4),
        (4, 3),
        (3, 2),
        (2, 1),
        (1, 25),
        (25, 26),
        (26, 27),
        (27, 30),
    ],
    left_lines=[
        (10, 9),
        (9, 8),
        (8, 7),
        (7, 6),
        (6, 17),
        (17, 18),
        (18, 19),
        (19, 22),
    ],
    head_lines=[(13, 14), (14, 15)],
    face=[],
    rshoulder=25,
    lshoulder=17,
    headup=15,
    keypoints_to_plot=[
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        13,
        14,
        15,
        17,
        18,
        19,
        22,
        25,
        26,
        27,
        30,
    ],
    joints_to_plot=[
        ["r_elbow", "r_hand"],
        ["r_shoulder", "r_elbow"],
        ["l_shoulder", "r_shoulder"],
        ["l_shoulder", "l_elbow"],
        ["l_elbow", "l_hand"],
        ["r_hip", "r_shoulder"],
        ["l_hip", "l_shoulder"],
        ["r_hip", "l_hip"],
        ["r_hip", "r_knee"],
        ["r_knee", "r_foot_mid"],
        ["l_hip", "l_knee"],
        ["l_knee", "l_foot_mid"],
        ["neck", "nose"],
        ["nose", "head"],
    ],
    right_hand=[19, 20],
    left_hand=[15, 16],
    head_part=[17, 13, 12],
    # kp_to_joint=[
    #     "r_hip",
    #     "r_knee",
    #     "r_foot_rear",
    #     "r_foot_mid",
    #     "r_foot_front",
    #     "l_hip",
    #     "l_knee",
    #     "l_foot_rear",
    #     "l_foot_mid",
    #     "l_foot_front",
    #     "neck",
    #     "nose",
    #     "head",
    #     "l_shoulder",
    #     "l_elbow",
    #     "l_wrist",
    #     "l_hand",
    #     "r_shoulder",
    #     "r_elbow",
    #     "r_wrist",
    #     "r_hand",
    # ],
    keypoint_idx_to_joint=_keypoint_idx_to_joint,
    joint_to_keypoint_idx=_joint_to_keypoint_idx
    # kps_to_change=[1, 2, 4, 6, 7, 9, 15, 17, 18, 22, 25, 26, 30],
    # kps_to_change_rel=[
    #     0,
    #     1,
    #     3,
    #     5,
    #     6,
    #     8,
    #     12,
    #     13,
    #     14,
    #     16,
    #     17,
    #     18,
    #     20,
    # ],
    # norm_T=[
    #     t3p,  # head
    #     t5p,  # body
    #     partial(t2p, ids=[25, 26]),  # right upper arm
    #     partial(t2p, ids=[26, 30]),  # right lower arm
    #     partial(t2p, ids=[17, 18]),  # left upper arm
    #     partial(t2p, ids=[18, 22]),  # left lower arm
    #     partial(t2p, ids=[1, 2]),  # right upper leg
    #     partial(t2p, ids=[2, 3]),  # right lower leg
    #     partial(t2p, ids=[6, 7]),  # left upper leg
    #     partial(t2p, ids=[7, 8]),  # left lower leg
    # ],
)


class Stickman3D:
    # @staticmethod
    # def get_example_pose3d():
    #     example_keypoints_3D = np.array(
    #         [
    #             [-91.679, 154.404, 907.261],
    #             [-223.23566, 163.80551, 890.5342],
    #             [-188.4703, 14.077106, 475.1688],
    #             [-261.84055, 186.55286, 61.438915],
    #             [-264.62787, 28.95641, 20.834599],
    #             [-266.93124, -45.763702, 26.877338],
    #             [39.877888, 145.00247, 923.98785],
    #             [-11.675994, 160.89919, 484.39148],
    #             [-51.550297, 220.14624, 35.834396],
    #             [-40.52279, 58.267826, 22.911175],
    #             [-33.55925, -16.026846, 30.447956],
    #             [-91.69202, 154.39796, 907.36],
    #             [-132.34781, 215.73018, 1128.8396],
    #             [-97.1674, 202.34435, 1383.1466],
    #             [-112.97073, 127.96946, 1477.4457],
    #             [-120.03289, 190.96477, 1573.4],
    #             [-97.1674, 202.34435, 1383.1466],
    #             [25.895456, 192.35947, 1296.1571],
    #             [107.10581, 116.050285, 1040.5062],
    #             [129.8381, -48.024918, 850.94806],
    #             [129.8381, -48.024918, 850.94806],
    #             [56.46485, -112.51781, 872.32465],
    #             [162.02069, -108.723694, 778.2846],
    #             [162.02069, -108.723694, 778.2846],
    #             [-97.1674, 202.34435, 1383.1466],
    #             [-230.36955, 203.17923, 1311.9639],
    #             [-315.40536, 164.55284, 1049.1747],
    #             [-350.77136, 43.442127, 831.3473],
    #             [-350.77136, 43.442127, 831.3473],
    #             [-301.10486, -37.945614, 861.5011],
    #             [-379.28616, -18.244892, 711.8155],
    #             [-379.28616, -18.244892, 711.8155],
    #         ],
    #         dtype=np.float32,
    #     )
    #     return example_keypoints_3D
    @staticmethod
    def get_example_pose3d():
        example_keypoints_3D = np.array(
            [
                [-176.73076784, -321.04861816, 5203.88206303],
                [-52.96191118, -309.7044902, 5251.08279046],
                [-155.64155821, 73.07175588, 5448.80702584],
                [-29.83157265, 506.78445233, 5400.13833872],
                [-91.64919684, 518.0979873, 5550.28395776],
                [-119.40835667, 498.56626101, 5617.16337351],
                [-300.49984317, -332.39276616, 5156.68125222],
                [-258.24048857, 99.60905053, 5244.68147203],
                [-209.48436389, 548.83382497, 5290.76367197],
                [-284.95536665, 532.89748031, 5434.0933464],
                [-320.98769068, 512.45266715, 5496.61273112],
                [-176.71873095, -321.14758597, 5203.87428583],
                [-109.15762285, -529.7281668, 5123.8906273],
                [-140.19117593, -780.12137495, 5074.60478778],
                [-153.18189183, -886.97613704, 5130.16533353],
                [-118.93483197, -970.22827344, 5058.5990502],
                [-140.19117593, -780.12137495, 5074.60478778],
                [-259.08997397, -690.13358895, 5050.59205089],
                [-370.67089216, -448.59930095, 5134.17726128],
                [-462.28663516, -290.82947286, 5307.6274481],
                [-462.28663516, -290.82947286, 5307.6274481],
                [-420.48697021, -327.24163077, 5390.85525678],
                [-518.00553708, -228.74555358, 5362.7729695],
                [-518.00553708, -228.74555358, 5362.7729695],
                [-140.19117593, -780.12137495, 5074.60478778],
                [-19.76034113, -716.91810277, 5140.27255285],
                [35.79161007, -470.14491849, 5257.73845884],
                [13.89246482, -279.85293245, 5421.06854165],
                [13.89246482, -279.85293245, 5421.06854165],
                [-63.49624559, -321.59137006, 5468.7011777],
                [12.12980397, -175.03984307, 5510.04796613],
                [12.12980397, -175.03984307, 5510.04796613],
            ],
            dtype=np.float32,
        )
        return example_keypoints_3D

    @staticmethod
    def get_example_pose3d_world():
        example_keypoints_3D = np.array(
            [
                [-91.679, 154.404, 907.261],
                [-223.23566, 163.80551, 890.5342],
                [-188.4703, 14.077106, 475.1688],
                [-261.84055, 186.55286, 61.438915],
                [-264.62787, 28.95641, 20.834599],
                [-266.93124, -45.763702, 26.877338],
                [39.877888, 145.00247, 923.98785],
                [-11.675994, 160.89919, 484.39148],
                [-51.550297, 220.14624, 35.834396],
                [-40.52279, 58.267826, 22.911175],
                [-33.55925, -16.026846, 30.447956],
                [-91.69202, 154.39796, 907.36],
                [-132.34781, 215.73018, 1128.8396],
                [-97.1674, 202.34435, 1383.1466],
                [-112.97073, 127.96946, 1477.4457],
                [-120.03289, 190.96477, 1573.4],
                [-97.1674, 202.34435, 1383.1466],
                [25.895456, 192.35947, 1296.1571],
                [107.10581, 116.050285, 1040.5062],
                [129.8381, -48.024918, 850.94806],
                [129.8381, -48.024918, 850.94806],
                [56.46485, -112.51781, 872.32465],
                [162.02069, -108.723694, 778.2846],
                [162.02069, -108.723694, 778.2846],
                [-97.1674, 202.34435, 1383.1466],
                [-230.36955, 203.17923, 1311.9639],
                [-315.40536, 164.55284, 1049.1747],
                [-350.77136, 43.442127, 831.3473],
                [-350.77136, 43.442127, 831.3473],
                [-301.10486, -37.945614, 861.5011],
                [-379.28616, -18.244892, 711.8155],
                [-379.28616, -18.244892, 711.8155],
            ],
            dtype=np.float32,
        )
        return example_keypoints_3D

    @staticmethod
    def get_example_pose3d_univ():
        example_keypoints_3D = np.array(
            [
                [-176.73076784, -321.04861816, 5203.88206303],
                [-52.96191118, -309.7044902, 5251.08279046],
                [-155.64155821, 73.07175588, 5448.80702584],
                [-29.83157265, 506.78445233, 5400.13833872],
                [-91.64919684, 518.0979873, 5550.28395776],
                [-119.40835667, 498.56626101, 5617.16337351],
                [-300.49984317, -332.39276616, 5156.68125222],
                [-258.24048857, 99.60905053, 5244.68147203],
                [-209.48436389, 548.83382497, 5290.76367197],
                [-284.95536665, 532.89748031, 5434.0933464],
                [-320.98769068, 512.45266715, 5496.61273112],
                [-176.71873095, -321.14758597, 5203.87428583],
                [-109.15762285, -529.7281668, 5123.8906273],
                [-140.19117593, -780.12137495, 5074.60478778],
                [-153.18189183, -886.97613704, 5130.16533353],
                [-118.93483197, -970.22827344, 5058.5990502],
                [-140.19117593, -780.12137495, 5074.60478778],
                [-259.08997397, -690.13358895, 5050.59205089],
                [-370.67089216, -448.59930095, 5134.17726128],
                [-462.28663516, -290.82947286, 5307.6274481],
                [-462.28663516, -290.82947286, 5307.6274481],
                [-420.48697021, -327.24163077, 5390.85525678],
                [-518.00553708, -228.74555358, 5362.7729695],
                [-518.00553708, -228.74555358, 5362.7729695],
                [-140.19117593, -780.12137495, 5074.60478778],
                [-19.76034113, -716.91810277, 5140.27255285],
                [35.79161007, -470.14491849, 5257.73845884],
                [13.89246482, -279.85293245, 5421.06854165],
                [13.89246482, -279.85293245, 5421.06854165],
                [-63.49624559, -321.59137006, 5468.7011777],
                [12.12980397, -175.03984307, 5510.04796613],
                [12.12980397, -175.03984307, 5510.04796613],
            ],
            dtype=np.float32,
        )
        return example_keypoints_3D

    @staticmethod
    def plot3d(keypoints: np.array, joint_model: JointModel):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        xs, ys, zs = np.split(keypoints, 3, axis=-1)
        ax.scatter(xs, ys, zs)
        for i, jo in enumerate(joint_model.joints_to_plot):
            idx_0 = joint_model.joint_to_keypoint_idx[jo[0]]
            idx_1 = joint_model.joint_to_keypoint_idx[jo[1]]
            x0, y0, z0 = np.split(keypoints[idx_0, :], 3, axis=-1)
            x1, y1, z1 = np.split(keypoints[idx_1, :], 3, axis=-1)
            ax.plot(
                np.concatenate([x0, x1], axis=0),
                np.concatenate([y0, y1], axis=0),
                np.concatenate([z0, z1], axis=0),
            )
        return fig, ax
