import numpy as np
import torch
from supermariopy.ptutils import compat as ptcompat


def test_torch_gather():
    # idx_chunked = idx.chunk(2, 2)
    # masked = img[
    #     torch.arange(N).view(N, 1),
    #     :,
    #     idx_chunked[0].squeeze(),
    #     idx_chunked[1].squeeze(),
    # ]
    # final = masked.expand(1, *masked.shape).permute(1, 3, 2, 0).view(*img.shape)

    idx = torch.zeros((10), dtype=torch.int64)
    arr = torch.linspace(-1, 1, 30, dtype=torch.float32).view((-1, 3))
    gathered = ptcompat.torch_gather(arr, idx)
    assert np.allclose(gathered, np.stack([gathered[0, :]] * 10))


def test_torch_gather_nd():
    import skimage

    bs = 4
    nk = 16
    image_t = torch.from_numpy(skimage.data.astronaut())
    params = ptcompat.torch_tile_nd(
        image_t.view((1, 1, 512, 512, 3)), [bs, nk, 1, 1, 1]
    )  # batch of stack of images
    indices = torch.stack(
        torch.meshgrid(
            torch.arange(bs),
            torch.arange(nk),
            torch.arange(128),
            torch.arange(128),
            torch.arange(3),
        ),
        dim=-1,
    )  # get 128 x 128 image slice from each item
    out = ptcompat.torch_gather_nd(params, indices)
    assert out.shape == (4, 16, 128, 128, 3)
