import torch.nn as nn
from functools import reduce

def _count_convNd(module, output):
    kernel_size = module.kernel_size
    in_channels = module.in_channels
    out_channels = module.out_channels

    filters_per_channel = out_channels // module.groups
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    conv_per_position_flops = reduce(lambda x, y: x * y, kernel_size) * in_channels * filters_per_channel

    active_elements_count = output.shape[0] * reduce(lambda x, y: x * y, output.shape[2:])

    overall_conv_flops = conv_per_position_flops * active_elements_count
    return overall_conv_flops

def _count_linear(module, output):
    total_ops = module.in_features * module.out_features
    return total_ops