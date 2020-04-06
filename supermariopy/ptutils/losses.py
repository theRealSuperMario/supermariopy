import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional
from torch.nn import functional as F
import numpy as np
import torchvision

## TORCHVISION
import torchvision.models as models
from torchvision import transforms, utils, datasets
from supermariopy.ptutils import nn as nn


# VGG architecture, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        """VGG19 perceptual loss.

        Uses the following feature layers:
        [
            "input_1",
            "block1_conv2",
            "block2_conv2",
            "block3_conv2",
            "block4_conv2",
            "block5_conv2"
        ]
        Parameters
        ----------
        torch : [type]
            [description]
        requires_grad : bool, optional
            if True, will also train VGG layers, by default False

        References
        ----------
        [1] : https://github.com/NVlabs/SPADE/blob/master/models/networks/architecture.py

        See Also
        --------
        tfutils.losses.VGG19Features
        """
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            # disable gradient on VGG layers
            for param in self.parameters():
                param.requires_grad = False

    def _normalize(self, x):
        """normalize with imagenet mean and standard deviations"""
        # TODO: imagenet normalization

        return x

    def forward(self, X):
        """assumes X to be in range [0, 1].
        
        Parameters
        ----------
        X : [type]
            [description]
        
        Returns
        -------
        list
            list of features for perceptual loss
        """
        X = self._normalize(X)
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(torch.nn.Module):
    def __init__(self, gpu_ids):
        """
        
        Parameters
        ----------
        torch : [type]
            [description]
        gpu_ids : [type]
            [description]

        References
        ----------
        ..[1] https://github.com/NVlabs/SPADE/blob/master/models/networks/loss.py
        """
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        self.criterion = torch.nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class VGGLossWithL1(VGGLoss):
    def __init__(self, gpu_ids, l1_alpha=1.0, vgg_alpha=1.0):
        self.l1_alpha = l1_alpha
        self.vgg_alpha = vgg_alpha
        super(VGGLossWithL1, self).__init__(gpu_ids)

    def forward(self, x, y):
        vgg_loss = super(VGGLossWithL1, self).forward(x, y)
        loss = self.criterion(x, y) * self.l1_alpha + vgg_loss * self.vgg_alpha
        return loss


# TODO: add VGG19+L1 loss
