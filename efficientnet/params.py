from collections import namedtuple
import math
from typing import List, Tuple

GlobalParams = namedtuple("GlobalParams", [
    "batch_norm_momentum", "batch_norm_epsilon", "dropout_rate",
    "num_classes", "width_coefficient", "depth_coefficient",
    "depth_divisor", "min_depth", "drop_connect_rate",
])

EfficientNetParams = namedtuple("EfficientNetParams", [
    "width_coefficient", "depth_coefficient", "resolution", "dropout_rate",
])

_efficient_net_params = {
    "efficientnet-b0": EfficientNetParams(width_coefficient=1.0, depth_coefficient=1.0, resolution=224, dropout_rate=0.2),
    "efficientnet-b1": EfficientNetParams(width_coefficient=1.0, depth_coefficient=1.1, resolution=240, dropout_rate=0.2),
    "efficientnet-b2": EfficientNetParams(width_coefficient=1.1, depth_coefficient=1.2, resolution=260, dropout_rate=0.3),
    "efficientnet-b3": EfficientNetParams(width_coefficient=1.2, depth_coefficient=1.4, resolution=300, dropout_rate=0.3),
    "efficientnet-b4": EfficientNetParams(width_coefficient=1.4, depth_coefficient=1.8, resolution=380, dropout_rate=0.4),
    "efficientnet-b5": EfficientNetParams(width_coefficient=1.6, depth_coefficient=2.2, resolution=456, dropout_rate=0.4),
    "efficientnet-b6": EfficientNetParams(width_coefficient=1.8, depth_coefficient=2.6, resolution=528, dropout_rate=0.5),
    "efficientnet-b7": EfficientNetParams(width_coefficient=2.0, depth_coefficient=3.1, resolution=600, dropout_rate=0.5),
}

def get_efficientnet_params(model_name: str) -> EfficientNetParams:
    if model_name not in _efficient_net_params:
        raise NotImplementedError("model name is not pre-defined: %s" % model_name)
    return _efficient_net_params[model_name]

BlockArgs = namedtuple("BlockArgs", [
    "kernel_size", "num_repeat", "input_filters", "output_filters",
    "expand_ratio", "id_skip", "strides", "se_ratio",
])

_blocks_args = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16, expand_ratio=1, id_skip=True, strides=(1, 1), se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24, expand_ratio=6, id_skip=True, strides=(2, 2), se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40, expand_ratio=6, id_skip=True, strides=(2, 2), se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80, expand_ratio=6, id_skip=True, strides=(2, 2), se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112, expand_ratio=6, id_skip=True, strides=(1, 1), se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192, expand_ratio=6, id_skip=True, strides=(2, 2), se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320, expand_ratio=6, id_skip=True, strides=(1, 1), se_ratio=0.25),
]

def get_blocks_args() -> List[BlockArgs]:
    return _blocks_args


def round_filters(filters: int, global_params: GlobalParams) -> int:
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    if not multiplier:
        return filters

    filters *= multiplier
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats: int, global_params: GlobalParams) -> int:
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))
