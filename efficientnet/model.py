import os
from typing import List, Tuple, NoReturn

import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientnet.conv2d import Conv2dSame
from efficientnet.params import BlockArgs, GlobalParams
from efficientnet.params import get_efficientnet_params, get_blocks_args, round_filters, round_repeats

SPATIAL_DIMS = (2, 3) # PyTorch expects image data [batch, channel, height, width]

NUM_CLASSES = 196

def drop_connect(x: torch.Tensor, training: bool, drop_connect_rate: float) -> torch.Tensor:
    if not training or drop_connect_rate == 0:
        return x
    keep_prob = 1.0 - drop_connect_rate
    batch_size = x.size()[0]
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype)
    binary_tensor = torch.floor(random_tensor).to("cuda")
    x = torch.div(x, keep_prob) * binary_tensor
    return x

def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def global_average_pooling2d(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x, SPATIAL_DIMS, keepdim=False)

class SeperableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False):
       super(SeperableConv2d, self).__init__()
       self.conv1 = Conv2dSame(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
       self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class MBConvBlock(nn.Module):
    """A class of MBConv: Mobile Inverted Residual Bottleneck.
    """

    def __init__(self, block_args: BlockArgs, global_params: GlobalParams):
        super(MBConvBlock, self).__init__()
        self._block_args = block_args
        self._activation = swish
        self.has_se = (self._block_args.se_ratio is not None) and (
            self._block_args.se_ratio > 0) and (self._block_args.se_ratio <= 1)

        # Builds the block accordings to arguments.
        self._build()

    def _build(self) -> NoReturn:
        """Builds block according to the arguments."""
        filters = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            # Expansion phase
            self._expand_conv = nn.Conv2d(
                in_channels=self._block_args.input_filters,
                out_channels=filters,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False)
            self._bn0 = nn.BatchNorm2d(filters)

        kernel_size = self._block_args.kernel_size
        padding = kernel_size // 2 # simple padding
        # Depth-wise convolution phase:
        self._depthwise_conv = SeperableConv2d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=(kernel_size, kernel_size),
            stride=self._block_args.strides,
            padding=padding,
            bias=False)
        self._bn1 = nn.BatchNorm2d(filters)

        if self.has_se:
            num_reduced_filters = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            # Squeeze and Excitation layer
            self._se_reduce = nn.Conv2d(
                in_channels=filters,
                out_channels=num_reduced_filters,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=True)
            self._se_expand = nn.Conv2d(
                in_channels=num_reduced_filters,
                out_channels=filters,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=True)

        # Output phase:
        self._project_conv = nn.Conv2d(
            in_channels=filters,
            out_channels=self._block_args.output_filters,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=False)
        self._bn2 = nn.BatchNorm2d(self._block_args.output_filters)

    def _forward_se(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Squeeze and Excitation layer.
        """
        old_x = x
        x = torch.mean(x, SPATIAL_DIMS, keepdim=True)
        x = self._se_expand(self._activation(self._se_reduce(x)))
        return torch.sigmoid(x) * old_x

    def forward(self, x: torch.Tensor, drop_connect_rate: float=None) -> torch.Tensor:
        old_x = x

        if self._block_args.expand_ratio != 1:
            x = self._activation(self._bn0(self._expand_conv(x)))

        x = self._activation(self._bn1(self._depthwise_conv(x)))

        if self.has_se:
            x = self._forward_se(x)

        x = self._bn2(self._project_conv(x))

        if self._block_args.id_skip:
            if all(
                s == 1 for s in self._block_args.strides
            ) and self._block_args.input_filters == self._block_args.output_filters:
                # only apply drop_connect if skip presents.
                # if drop_connect_rate:
                #     x = drop_connect(x, self.training, drop_connect_rate)
                x = torch.add(x, old_x)
        
        return x

class EfficientNet(nn.Module):
    """A class of EfficientNet
    """
    
    def __init__(self, blocks_args: List[BlockArgs]=None, global_params: GlobalParams=None):
        super(EfficientNet, self).__init__()
        self._global_params = global_params
        self._blocks_args = blocks_args
        self._activation = swish
        self._build()

    def _build(self) -> NoReturn:
        # Stem part
        in_channels = 3
        out_channels = round_filters(32, self._global_params)
        self._conv_stem = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            bias=False)
        self._bn0 = nn.BatchNorm2d(out_channels)

        # Builds blocks
        self._blocks = []
        for block_args in self._blocks_args:
            # Update block input and output filters based on depth multiplier
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params))

            # The first block needs to take care of stride and filter size increase
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, strides=(1, 1))
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        for idx, block in enumerate(self._blocks):
            setattr(self, "_block%02d" % idx, block)

        # Head part
        in_channels = block_args.output_filters
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=False)
        self._bn1 = nn.BatchNorm2d(out_channels)

        self._avg_pooling = global_average_pooling2d

        # self._dropout = None
        # if self._global_params.dropout_rate > 0:
        #     self._dropout = nn.Dropout2d(self._global_params.dropout_rate)

        self._fc = nn.Linear(out_channels, self._global_params.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forwards Stem layers
        x = self._activation(self._bn0(self._conv_stem(x)))

        # Forwards blocks
        for idx, block in enumerate(self._blocks):
            drop_rate = self._global_params.drop_connect_rate
            if drop_rate:
                drop_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_rate)

        # Forwards final layers and returns logits
        x = self._activation(self._bn1(self._conv_head(x)))
        x = self._avg_pooling(x)
        # if self._dropout:
        #     x = self._dropout(x)
        x = self._fc(x)

        return x

def get_model_params(model_name: str, override_params: dict) -> Tuple[List[BlockArgs], GlobalParams]:
    """Get the blocks args and global params for a given model."""
    blocks_args = get_blocks_args()
    width_coefficient, depth_coefficient, _, dropout_rate = get_efficientnet_params(model_name)
    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=0.2,
        num_classes=NUM_CLASSES,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None)
    if override_params:
        global_params = global_params._replace(**override_params)

    return blocks_args, global_params

def build_efficientnet(model_name: str, override_params: dict=None):
    """A helper function to creates a model

    Args:
        model_name: string, the predefined model name.
        override_params: dict, a dictionary of params for overriding.

    Returns:
        The model.
    """
    blocks_args, global_params = get_model_params(model_name, override_params)
    model = EfficientNet(blocks_args, global_params)
    return model
