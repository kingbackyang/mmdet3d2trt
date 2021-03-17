from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import load_checkpoint
from torch import nn as nn

from mmdet.models import BACKBONES


class InvertResidualBlock(nn.Module):

    def __init__(self, conv_cfg, norm_cfg, inp, oup):
        super(InvertResidualBlock, self).__init__()

        self.pw1 = build_conv_layer(conv_cfg, inp, inp, 1, stride=1, padding=0)
        self.norm1 = build_norm_layer(norm_cfg, inp)[1]
        self.activate6 = nn.ReLU6(inplace=True)
        self.dw2 = build_conv_layer(conv_cfg, inp, inp, 3, stride=1, groups=inp, padding=1)
        self.norm2 = build_norm_layer(norm_cfg, inp)[1]
        self.pw3 = build_conv_layer(conv_cfg, inp, oup, 1, stride=1, padding=0)
        self.norm3 = build_norm_layer(norm_cfg, oup)[1]

        self.skip_layer = inp == oup
        self.activation = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(inp, oup, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        identity_x = x
        x = self.pw1(x)
        x = self.norm1(x)
        x = self.activate6(x)
        x = self.dw2(x)
        x = self.norm2(x)
        x = self.activate6(x)
        x = self.pw3(x)
        x = self.norm3(x)
        if self.skip_layer:
            x = identity_x + x
        else:
            x = self.norm3(self.conv(identity_x)) + x
        return x


@BACKBONES.register_module()
class SECOND_INVMB2RES(nn.Module):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(self,
                 in_channels=128,
                 out_channels=[128, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg=dict(type='Conv2d', bias=False)
                 ):
        super(SECOND_INVMB2RES, self).__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                build_conv_layer(conv_cfg, in_filters[i], in_filters[i], 3, stride=layer_strides[i], groups=in_filters[i], padding=1),
                build_norm_layer(norm_cfg, in_filters[i])[1],
                build_conv_layer(conv_cfg, in_filters[i], out_channels[i], 1, stride=1, padding=0),
                build_norm_layer(norm_cfg, out_channels[i])[1],
                nn.ReLU(inplace=True),
            ]
            for j in range(layer_num):
                block.append(InvertResidualBlock(conv_cfg, norm_cfg, out_channels[i], out_channels[i]))
                block.append(nn.ReLU(inplace=True))

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

    def init_weights(self, pretrained=None):
        """Initialize weights of the 2D backbone."""
        # Do not initialize the conv layers
        # to follow the original implementation
        if isinstance(pretrained, str):
            from mmdet3d.utils import get_root_logger
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        return tuple(outs)
