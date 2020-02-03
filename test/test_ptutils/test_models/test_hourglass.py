import pytest
import torch
from torch.autograd import Variable


def test_hourglassNet():
    num_stacks = 1
    num_blocks = 1  # number of residual blocks
    num_classes = 25
    from supermariopy.ptutils.models.hourglass import hg

    model = hg(
        num_stacks=num_stacks, num_blocks=num_blocks, num_classes=num_classes, depth=1
    )
    x = Variable(torch.rand(2, 3, 256, 256))
    (y,) = model(x)
    assert y.shape == (2, num_classes, 64, 64)


def test_hourglass():
    """Test Hourglass block. 
    Hourglass block is supposed to map the same image dimensions back to the original image dimensions,
    but with more features.
    """

    from supermariopy.ptutils.models.hourglass import Hourglass, Bottleneck

    num_channels = 128
    model = Hourglass(Bottleneck, 3, 64, 1)
    x = Variable(torch.rand(2, 3, 256, 256))

    # Have to do initial conv, otherwise output channels will be 3
    x = torch.nn.Conv2d(3, num_channels, 3, 1, 1)(x)
    y = model(x)
    assert y.shape == (2, num_channels, 256, 256)


def test_bottleneck():
    from torchviz import make_dot
    from supermariopy.ptutils.models.hourglass import Bottleneck
    from supermariopy.ptutils.models import summary

    block = Bottleneck(128, 64)
    x = torch.rand(1, 128, 128, 128)
    y = block(x)
    # summary((128, 128, 128), block)
    assert y.shape == y.shape

    # def test_encoder_model():
    #     from torchviz import make_dot
    #     from supermariopy.ptutils.models import EncoderModel, summary

    #     config = [512, 256]
    #     encoder = EncoderModel(3, 0, 64, config)
    #     x = torch.rand(1, 3, 128, 128)
    #     y = encoder(x)

    #     summary(x.shape[1:], encoder)
    #     assert y.shape == (1, 64, 1, 1)

    # def test_decoder_model():
    #     from torchviz import make_dot
    #     from supermariopy.ptutils.models import DecoderModel, summary

    #     config = [512, 256, 128]
    #     decoder = DecoderModel(64, 25, config)
    #     x = torch.rand(1, 64, 1, 1)
    #     y = decoder(x)

    #     summary(x.shape[1:], decoder)
    #     y = decoder(x)
    #     assert y.shape == (1, 25, 16, 16)

    # def test_discriminator_model():
    # from supermariopy.ptutils.models import DiscriminatorModel, summary

    # x1 = torch.rand(1, 64, 1, 1)
    # x2 = torch.rand(1, 128, 1, 1)
    # disc = DiscriminatorModel(2, input_channels=[64, 128], d_size=512)
    # y = disc(x1, x2)
    # assert y.shape == (1, 1)
