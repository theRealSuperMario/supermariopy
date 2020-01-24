import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from collections import OrderedDict


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, activation=nn.ReLU(inplace=True)):
        """Implements a simple residual block.

        y = x + conv(a(x))
        
        Parameters
        ----------
        in_filters : int
            input and output channel dimension
        activation : callable, optional
            activation function to use, short `a`, by default nn.ReLU(inplace=True)
        """
        super(ResidualBlock, self).__init__()
        self.activation = activation
        self.conv = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        residual = x
        residual = self.activation(residual)
        residual = self.conv(residual)
        return x + residual


class Bottleneck(nn.Module):
    expansion = 2  # factor to expand incoming features due to skip in connections.

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        use_batch_norm=True,
        activation=nn.ReLU(inplace=True),
    ):
        """Implements a Res-Net Style feature bottleneck.

        The bottleneck is implemented as follows:
            input : x

            - optional: bach_norm(x)
            - conv(x), with num_features = planes
            - relu(x)

            - optional: bach_norm(x)
            - conv(x), with num_features = planes
            - relu(x)

            - optional: bach_norm(x)
            - conv(x), with num_features = 2 * planes
            - relu(x)

        Examples
        --------
            1. simple usage
            ```python
                from torchviz import make_dot
                block = Bottleneck(128, 64, use_batch_norm=False)
                x = torch.rand(1, 128, 128, 128)
                y = block(x)
                print(x.shape, y.shape)
                >>> torch.Size([1, 128, 128, 128]) torch.Size([1, 128, 128, 128])
                make_dot(y.mean())
            ```

        Parameters
        ----------
        inplanes : int
            Number of features of input tensor.
        planes : int
            number of feature in the bottleneck
        stride : int, optional
            [description], by default 1
        downsample : callable, optional
            a downsampling function to apply before merging the residual, by default None
        use_batch_norm : bool, optional
            [description], by default True
        activation : callable
            activation function, by default nn.ReLU(inplace=True)
        """
        super(Bottleneck, self).__init__()

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        if use_batch_norm:
            self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True
        )
        if use_batch_norm:
            self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.activation = activation
        # self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = x
        if self.use_batch_norm:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv1(out)

        if self.use_batch_norm:
            out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)

        if self.use_batch_norm:
            out = self.bn3(out)
        out = self.activation(out)
        out = self.conv3(out)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(
        self, block, num_blocks_per_level, planes=128, depth=1, use_batch_norm=True
    ):
        """Make a single hourglass module.

        The hourglass is assumed to be `depth`-levels deep, i.e. have that many downsampling stages.
        At each downsampling stage, there are `num_blocks_per_level` functional `blocks`, i.e. residual blocks.
        Each block returns feature blocks with `planes` - depth.

        Parameters
        ----------
        nn : [type]
            [description]
        block : callable
            Block function, i.e. residual block.
        num_blocks : [type]
            number of blocks per downsampling stage.
        planes : int
            number of features within each block.
        depth : int
            how many times to go down in the hourglass network.
        """
        super(Hourglass, self).__init__()

        self.use_batch_norm = use_batch_norm
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks_per_level, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(
                block(
                    planes * block.expansion, planes, use_batch_norm=self.use_batch_norm
                )
            )
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hour_glass = []
        for current_depth in range(depth):
            residual_blocks = []
            for i_block in range(3):
                residual_blocks.append(self._make_residual(block, num_blocks, planes))
            if current_depth == 0:
                residual_blocks.append(self._make_residual(block, num_blocks, planes))
            hour_glass.append(nn.ModuleList(residual_blocks))
        return nn.ModuleList(hour_glass)

    def _hour_glass_forward(self, n, x):
        """recursively merge each level of the hourglass network"""
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    """Stacked-Hourglass Network from Newell et al ECCV 2016

    References
    ----------
    [1]: https://arxiv.org/pdf/1603.06937.pdf
    """

    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=16, depth=4):
        super(HourglassNet, self).__init__()

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats * block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, depth=4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            if i < num_stacks - 1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
        self.hour_glass = nn.ModuleList(hg)
        self.residuals = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=True,
                ),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(conv, bn, self.relu,)

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            y = self.hour_glass[i](x)
            y = self.residuals[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_
        return out


# TODO: coord conv


class EncoderModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_extra_resnets=0,
        out_features=128,
        config=[256, 128],
        block=ResidualBlock,
    ):
        super(EncoderModel, self).__init__()
        self.in_channels = in_channels
        self.num_extra_blocks = 0
        self.out_features = out_features
        self.depth = len(config)

        n_features_out = config[0]

        self.conv0 = torch.nn.Conv2d(self.in_channels, n_features_out, 3, padding=1)
        self.block0 = block(n_features_out)

        self.res_blocks = []
        self.downsampling_layers = []
        layers = []

        layers.append(self.conv0)
        layers.append(self.block0)

        # Definition of module
        for n_features_in, n_features_out in zip(config[0:], config[1:]):
            d = nn.Conv2d(n_features_in, n_features_out, 3, stride=2, padding=2)
            r = block(n_features_out)
            layers.append(d)
            layers.append(r)
            self.downsampling_layers.append(d)
            self.res_blocks.append(r)

        n_features_out = config[-1]
        for i_extra_resnet in range(num_extra_resnets):
            r = block(n_features_out)
            layers.append(r)

        layers.append(torch.nn.ReLU(inplace=True))

        self._encoder = torch.nn.Sequential(*layers)
        self.final_conv = torch.nn.Conv2d(n_features_out, out_features, 1, padding=0)

    def forward(self, x):
        """Encode image x.
        
        Parameters
        ----------
        x : [type]
            x.shape = [C, 128, 128]
        """
        y = self._encoder(x)
        y = torch.mean(y, dim=[2, 3], keepdim=True)
        y = self.final_conv(y)
        return y


from functools import partial


class DecoderModel(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        config,
        activation=torch.nn.ReLU(inplace=True),
        block=ResidualBlock,
    ):
        super(DecoderModel, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_channels
        self.config = config
        self.reverse_config = config[::-1]
        self.depth = len(config)

        n_features = config[-1]
        self.initial_conv = torch.nn.Conv2d(
            in_channels, n_features * 4 * 4, 1, stride=1
        )

        layers = []
        d = nn.Conv2d(n_features, n_features, kernel_size=3, padding=1)
        r = block(n_features)
        layers.append(d)
        layers.append(r)

        for n_features_in, n_features_out in zip(
            self.reverse_config[:-1], self.reverse_config[1:]
        ):
            r = block(n_features_in)
            u = torch.nn.modules.Upsample(scale_factor=2, mode="bilinear")
            c = torch.nn.Conv2d(n_features_in, n_features_out, kernel_size=3, padding=1)
            a = activation
            layers.append(r)
            layers.append(u)
            layers.append(c)
            layers.append(a)

        n_features = config[0]
        r = block(n_features)
        c = torch.nn.Conv2d(n_features, out_channels, kernel_size=3, padding=1)
        layers.append(r)
        layers.append(c)

        self._decoder = torch.nn.Sequential(*layers)

    def forward(self, x):
        y = self.initial_conv(x)
        y = y.view(-1, self.config[-1], 4, 4)
        y = self._decoder(y)
        return y


def hg(**kwargs):
    model = HourglassNet(
        Bottleneck,
        num_stacks=kwargs["num_stacks"],
        num_blocks=kwargs["num_blocks"],
        num_classes=kwargs["num_classes"],
        depth=kwargs["depth"],
    )
    return model


def summary(input_size, model):
    """print Keras-style summary for model given input_size

    References
    ----------
    https://gist.github.com/HTLife/b6640af9d6e7d765411f8aa9aa94b837
    """

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = -1
            summary[m_key]["output_shape"] = list(output.size())
            summary[m_key]["output_shape"][0] = -1

            params = 0
            if hasattr(module, "weight"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                if module.weight.requires_grad:
                    summary[m_key]["trainable"] = True
                else:
                    summary[m_key]["trainable"] = False
            if hasattr(module, "bias"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(1, *in_size)).type(dtype) for in_size in input_size]
    else:
        x = Variable(torch.rand(1, *input_size)).type(dtype)

    print(x.shape)
    print(type(x[0]))
    # create properties
    summary = OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    model(x)
    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    trainable_params = 0
    for layer in summary:
        ## input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            str(layer),
            str(summary[layer]["output_shape"]),
            str(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)
    print("================================================================")
    print("Total params: " + str(total_params))
    print("Trainable params: " + str(trainable_params))
    print("Non-trainable params: " + str(total_params - trainable_params))
    print("----------------------------------------------------------------")
    return summary
