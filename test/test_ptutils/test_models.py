import pytest
import torch
from torch.autograd import Variable


def test_hourglass():
    from torchviz import make_dot

    num_stacks = 1
    num_blocks = 1  # number of residual blocks
    num_classes = 25
    from supermariopy.ptutils.models import hg

    model = hg(
        num_stacks=num_stacks, num_blocks=num_blocks, num_classes=num_classes, depth=1
    )
    x = Variable(torch.rand(1, 3, 256, 256))
    (y,) = model(x)
    make_dot(y.mean())
    assert True


def test_bottleneck():
    from torchviz import make_dot
    from supermariopy.ptutils.models import Bottleneck, summary

    block = Bottleneck(128, 64, use_batch_norm=False)
    x = torch.rand(1, 128, 128, 128)
    y = block(x)
    # summary((128, 128, 128), block)
    assert y.shape == y.shape


def test_encoder_model():
    from torchviz import make_dot
    from supermariopy.ptutils.models import EncoderModel, summary

    config = [512, 256]
    encoder = EncoderModel(3, 0, 64, config)
    x = torch.rand(1, 3, 128, 128)
    y = encoder(x)

    summary(x.shape[1:], encoder)
    assert y.shape == (1, 64, 1, 1)


def test_decoder_model():
    from torchviz import make_dot
    from supermariopy.ptutils.models import DecoderModel, summary

    config = [512, 256, 128]
    decoder = DecoderModel(64, 25, config)
    x = torch.rand(1, 64, 1, 1)
    y = decoder(x)

    summary(x.shape[1:], decoder)
    y = decoder(x)
    assert y.shape == (1, 25, 16, 16)
