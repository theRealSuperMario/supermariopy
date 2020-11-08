#!/usr/bin/env python3
import os

import click
import cv2
from matplotlib import pyplot as plt
from supermariopy import denseposelib, imageutils


@click.command()
@click.argument("iuv-path")
def main(iuv_path):
    I, u, v = denseposelib.load_iuv(iuv_path)
    base_dir = os.path.dirname(iuv_path)
    out_name = "{}_IRGB.png".format(os.path.splitext(iuv_path)[0])
    colors = imageutils.make_colors(
        len(denseposelib.PART_LIST),
        cmap=plt.cm.coolwarm,
        with_background=True,
        background_id=0,
    )
    I_colors = imageutils.convert_range(colors[I], [0, 1], [0, 255])
    cv2.imwrite(os.path.join(base_dir, out_name), I_colors)


if __name__ == "__main__":
    main()
