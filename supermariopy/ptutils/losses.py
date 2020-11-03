from typing import List

import torch
import torchvision
from supermariopy.ptutils import compat as ptcompat


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
        [1] : https://github.com/NVlabs/SPADE/blob/master/models/networks/architecture.py # noqa

        See Also
        --------
        tfutils.losses.VGG19Features
        """
        super().__init__()
        vgg_pretrained_features = (
            torchvision.models.vgg19(pretrained=True).eval().features
        )
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
        from supermariopy.dl.constants import IMAGENET_MEAN, IMAGENET_STD

        mean = torch.from_numpy(IMAGENET_MEAN).view((1, 3, 1, 1))
        mean = mean.to(x.device)
        std = torch.from_numpy(IMAGENET_STD).view((1, 3, 1, 1))
        std = std.to(x.device)

        y = (x - mean) / std

        return y

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
    def __init__(self, gpu_ids, weights=[1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]):
        """
        input is assumed to be in range [0, 1]
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
        self.weights = weights

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class VGGLossWithL1(VGGLoss):
    def __init__(
        self,
        gpu_ids,
        l1_alpha=1.0,
        vgg_alpha=1.0,
        weights=[1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0],
    ):
        """input is assumed to be in range [0, 1]"""
        self.l1_alpha = l1_alpha
        self.vgg_alpha = vgg_alpha
        super(VGGLossWithL1, self).__init__(gpu_ids, weights=weights)

    def forward(self, x, y):
        vgg_loss = super(VGGLossWithL1, self).forward(x, y)
        loss = self.criterion(x, y) * self.l1_alpha + vgg_loss * self.vgg_alpha
        return loss


class PerceptualVGG(torch.nn.Module):
    VGG_OUTPUT = ["input", "relu1_2", "relu2_2", "relu3_2", "relu4_2", "relu5_2"]

    def __init__(
        self,
        vgg=torchvision.models.vgg19(pretrained=True).eval(),
        feature_weights=[1.0] * 6,
        use_gram=False,
        gram_weights=[0.1] * 6,
    ):
        """ this implementation seems to be different than the one above in `VGGLoss`
        this implementation is based on the vunet paper
        """
        super().__init__()
        if isinstance(vgg, torch.nn.DataParallel):
            self.vgg_layers = vgg.module.features
        else:
            self.vgg_layers = vgg.features

        self.feature_weights = feature_weights
        self.gram_weights = gram_weights
        self.use_gram = use_gram

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float)
            .unsqueeze(dim=0)
            .unsqueeze(dim=-1)
            .unsqueeze(dim=-1),
        )

        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float)
            .unsqueeze(dim=0)
            .unsqueeze(dim=-1)
            .unsqueeze(dim=-1),
        )
        self.target_layers = {
            "3": "relu1_2",
            "8": "relu2_2",
            "13": "relu3_2",
            "22": "relu4_2",
            "31": "relu5_2",
        }

    def forward(self, x):
        """ x in range [0, 1] """
        x = (x - self.mean) / self.std

        out = {"input": x}

        for name, submodule in self.vgg_layers._modules.items():
            if name in self.target_layers:
                x = submodule(x)
                out[self.target_layers[name]] = x
            else:
                x = submodule(x)
        return out

    def grams(self, fs):
        gs = list()
        for f in fs:
            bs, c, h, w = list(f.shape)
            f = ptcompat.torch_reshape(f, [bs, c, h * w])
            ft = f.permute([0, 2, 1])
            g = torch.matmul(f, ft)
            g = g / (4.0 * h * w)
            gs.append(g)
        return gs

    def loss(self, target: torch.Tensor, pred: torch.Tensor) -> List[torch.Tensor]:
        VGGOutput = self.VGG_OUTPUT
        target_feats = self(target)
        target_feats = [target_feats[k] for k in VGGOutput]
        pred_feats = self(pred)
        pred_feats = [pred_feats[k] for k in VGGOutput]

        criterion = torch.nn.L1Loss()

        losses = [
            criterion(xf, yf).unsqueeze(dim=-1).to(torch.float)
            for xf, yf in zip(target_feats, pred_feats)
        ]

        if self.use_gram:
            target_grams = self.grams(target_feats)
            pred_grams = self.grams(pred_feats)
            gram_losses = [
                criterion(xf, yf).unsqueeze(dim=-1).to(torch.float)
                for xf, yf in zip(target_grams, pred_grams)
            ]
            losses = losses + gram_losses

        return losses
