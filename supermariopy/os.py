import os


def append_fname(x, suffix):
    pardir = os.path.dirname(x)
    basename = os.path.basename(x)
    fname, ext = os.path.splitext(basename)
    return os.path.join(pardir, fname + suffix + ext)


def prepend_fname(x, prefix):
    pardir = os.path.dirname(x)
    basename = os.path.basename(x)
    fname, ext = os.path.splitext(basename)
    return os.path.join(pardir, prefix + fname + ext)
