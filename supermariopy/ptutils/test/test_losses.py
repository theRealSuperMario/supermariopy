import pytest
import torch
from supermariopy.ptutils import losses as smptlosses
import numpy as np


def test_vgg19_loss():
    x = torch.zeros((1, 3, 224, 224))
    y = torch.zeros((1, 3, 224, 224))

    criterion = smptlosses.VGGLoss(None)
    l = criterion(x, y)
    assert np.allclose(l.numpy(), np.array([0]))


def test_vgg19_with_l1_loss():
    x = torch.zeros((1, 3, 224, 224))
    y = torch.zeros((1, 3, 224, 224))

    criterion = smptlosses.VGGLossWithL1(None)
    l = criterion(x, y)
    assert np.allclose(l.numpy(), np.array([0]))


class Test_PerceptualVGG:
    def test_encode_features(self):
        x = torch.zeros((1, 3, 224, 224))

        vgg1 = smptlosses.VGG19()
        out_1 = vgg1(x)

        import torchvision

        vgg2 = smptlosses.PerceptualVGG()
        out_2 = vgg2(x)

        h1_1 = out_1[1]
        h1_2 = out_2["relu2_2"]

        assert np.allclose(h1_1.detach().numpy(), h1_2.detach().numpy())

    def test_vgg_loss(self):
        x = torch.zeros((1, 3, 224, 224))
        vgg2 = smptlosses.PerceptualVGG()

        losses = vgg2.loss(x, x)
        assert len(losses) == 6

        vgg2 = smptlosses.PerceptualVGG(use_gram=True)
        losses = vgg2.loss(x, x)
        assert len(losses) == 12
