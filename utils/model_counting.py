"""
Counting code adapted from counting example code from Micronet Challenge 2019
https://github.com/google-research/google-research/blob/master/micronet_challenge/counting.py
"""

import torch
import torch.nn as nn

import numpy as np

def get_conv_output_size(image_size, filter_size, padding, stride):
    """Calculates the output size of convolution.
    The input, filter and the strides are assumed to be square.
    Arguments:
    image_size: int, Dimensions of the input image (square assumed).
    filter_size: int, Dimensions of the kernel (square assumed).
    padding: str, padding added to the input image. 'same' or 'valid'
    stride: int, stride with which the kernel is applied (square assumed).
    Returns:
    int, output size.
    """
    if padding == 'same':
        pad = filter_size // 2
    elif padding == 'valid':
        pad = 0
    else:
        raise NotImplementedError('Padding: %s should be `same` or `valid`.'
                              % padding)
    out_size = np.ceil((image_size - filter_size + 1. + 2 * pad) / stride)
    return int(out_size)

def count_DepthwiseConv2d(conv_layer, x):
    """
    Calculates the number of mults, adds
    for a depthwise Conv2d pytorch  module, given input x
    Assume no sparsity and same padding.
    """
    out_shape = conv_layer(x).shape
    input_size = x.shape[2]
    k_size, stride = conv_layer.kernel_size[0], conv_layer.stride[0]
    c_in, c_out = conv_layer.in_channels, conv_layer.out_channels
    padding = 'same'
    
    flop_mults = flop_adds = 0
    
    # assert that the group size is equal to the number of input channels
    assert conv_layer.groups == c_in
    
    # Each application of the kernel can be thought as a dot product between
    # the flattened kernel and patches of the image.
    vector_length = (k_size * k_size)

    # Number of elements in the output is OUT_SIZE * OUT_SIZE * OUT_CHANNEL
    n_output_elements = get_conv_output_size(input_size, k_size, padding,
                                             stride) ** 2 * c_in
    # Each output is the product of one dot product. Dot product of two
    # vectors of size n needs n multiplications and n - 1 additions.
    flop_mults += vector_length * n_output_elements
    flop_adds += (vector_length - 1) * n_output_elements
    
    try:
        # if bias has a shape, continue
        conv_layer.bias.shape
        # If we have bias we need one more addition per dot product.
        flop_adds += n_output_elements
    except:
        pass
    
    # make sure the calculated number of output elements equals the actual
    assert np.prod(out_shape) == n_output_elements

    return flop_mults, flop_adds


def count_Conv2d(conv_layer, x):
    """
    Calculates the number of mults, adds
    for a Conv2d pytorch module, given an input x.
    Assume no sparsity and same padding.
    """
    out_shape = conv_layer(x).shape
    input_size = x.shape[2]
    k_size, stride = conv_layer.kernel_size[0], conv_layer.stride[0]
    c_in, c_out = conv_layer.in_channels, conv_layer.out_channels
    padding = 'same'
    
    flop_mults = flop_adds = 0
    
    # Each application of the kernel can be thought as a dot product between
    # the flattened kernel and patches of the image.
    vector_length = (k_size * k_size * c_in)

    # Number of elements in the output is OUT_SIZE * OUT_SIZE * OUT_CHANNEL
    n_output_elements = get_conv_output_size(input_size, k_size, padding,
                                             stride) ** 2 * c_out
    # Each output is the product of one dot product. Dot product of two
    # vectors of size n needs n multiplications and n - 1 additions.
    flop_mults += vector_length * n_output_elements
    flop_adds += (vector_length - 1) * n_output_elements
    
    try:
        # if bias has a shape, continue
        conv_layer.bias.shape
        # If we have bias we need one more addition per dot product.
        flop_adds += n_output_elements
    except:
        pass
    
    # make sure the calculated number of output elements equals the actual
    assert np.prod(out_shape) == n_output_elements

    return flop_mults, flop_adds

def count_FullyConnected(fc_layer, x):
    """
    Calculates the number of mults, adds
    for a Linear pytorch module, given an input x.
    Assume no sparsity.
    """
    input_size = x.shape[1]
    c_in, c_out = fc_layer.in_features, fc_layer.out_features

    flop_mults = flop_adds = 0    
    flop_mults += c_in * c_out
    # We have one less addition than the number of multiplications per output
    # channel.
    flop_adds += (c_in - 1) * c_out

    try:
        # if bias has a shape, continue
        fc_layer.bias.shape
        # If we have bias we need one more addition per dot product.
        flop_adds += c_out
    except:
        pass
    
    return flop_mults, flop_adds

def countReLU(x):
    """
    For the purposes of the "freebie" quantization scoring, ReLUs can be
    assumed to be performed on 16-bit inputs. Thus, we track them as
    multiplications in our accounting, which can also be assumed to be
    performed on reduced precision inputs.
    """
    flop_adds = 0
    flop_mults = np.prod(list(x.shape))
    return flop_mults, flop_adds

def count_AvgPool2d(pool_layer, x):
    """
    Calculates the number of mults, adds
    for an AvgPool2d pytorch module, given an input x.
    """
    n_channels = x.shape[1]
    y = pool_layer(x)
    output_size = y.shape[2]
    kernel_size = pool_layer.kernel_size
    stride = pool_layer.stride
    padding = pool_layer.padding
    
    flop_mults = flop_adds = 0
    # we perform averages of size kernel_size * kernel_size
    flop_adds += output_size * output_size * (kernel_size * kernel_size - 1) * n_channels
    # For each output channel we will make a division.
    flop_mults += output_size * output_size * n_channels
    
    return flop_mults, flop_adds

def count_AdaptiveAvgPool2d(pool_layer, x):
    """
    Calculates the number of mults, adds
    for an AdaptiveAvgPool2d pytorch module, given an input x.
    Only implemented for output_size = 1
    """
    n_channels = x.shape[1]
    input_size = x.shape[2]
    y = pool_layer(x)
    output_size = y.shape[2]
    # only implemented for output size = 1
    assert output_size == 1
    stencil_size = (input_size+output_size-1) // output_size
    
    flop_mults = flop_adds = 0
    # we perform averages of size input_size * input_size for each channel for output_size = 1
    flop_adds += (input_size * input_size - 1) * n_channels
    # For each output channel we will make a division.
    flop_mults += output_size * output_size * n_channels
    
    return flop_mults, flop_adds

def count_dim_reduce(reduce_layer, x):
    total_mults = total_adds = 0
    # (dim_reduce)
    # (0): ReLU
    relu_layer = reduce_layer[0]
    flop_mults, flop_adds = countReLU(x)
    total_mults += flop_mults; total_adds += flop_adds
    x = relu_layer(x)
    # (1) Conv2d
    conv_layer = reduce_layer[1]
    flop_mults, flop_adds = count_Conv2d(conv_layer, x)
    total_mults += flop_mults; total_adds += flop_adds
    x = conv_layer(x)
    # (2) Batch Norm
    norm_layer = reduce_layer[2]
    x = norm_layer(x)

    return total_mults, total_adds

def count_Fit(fit_block, x, prev):
    total_mults = total_adds = 0
    if prev is None:
        return total_mults, total_adds
    
    elif x.size(2) != prev.size(2):
        # (relu): ReLU
        flop_mults, flop_adds = countReLU(prev)
        total_mults += flop_mults; total_adds += flop_adds
        prev = fit_block.relu(prev)
        
        # (p1) Sequential
        # (0) AvgPool2d
        pool_layer = fit_block.p1[0]
        flop_mults, flop_adds = count_AvgPool2d(pool_layer, prev)
        total_mults += flop_mults; total_adds += flop_adds
        p1 = pool_layer(prev)
        # (1) Conv2d
        conv_layer = fit_block.p1[1]
        flop_mults, flop_adds = count_Conv2d(conv_layer, p1)
        total_mults += flop_mults; total_adds += flop_adds
        p1 = conv_layer(p1)
        
        # (p2) Sequential
        # (0) ConstantPad2d
        # (1) ConstantPad2d
        pad_layer = fit_block.p2[0]
        p2 = pad_layer(prev)
        pad_layer = fit_block.p2[1]
        p2 = pad_layer(p2)
        # (2) AvgPool2d
        pool_layer = fit_block.p2[2]
        flop_mults, flop_adds = count_AvgPool2d(pool_layer, p2)
        total_mults += flop_mults; total_adds += flop_adds
        p2 = pool_layer(p2)
        # (3) Conv2d
        conv_layer = fit_block.p2[3]
        flop_mults, flop_adds = count_Conv2d(conv_layer, p2)
        total_mults += flop_mults; total_adds += flop_adds
        p2 = conv_layer(p2)
        # new prev is concatenated. No operations
        prev = torch.cat([p1, p2], 1)
        
        # (bn) Batch Norm
        norm_layer = fit_block.bn
        prev = norm_layer(prev)
        
        return total_mults, total_adds
        
    else:
        return count_dim_reduce(fit_block.dim_reduce, prev)
    
def count_SeparableConv2d(sep_conv2d_layer, x):
    total_mults = total_adds = 0
    # depthwise
    depth_layer = sep_conv2d_layer.depthwise
    flop_mults, flop_adds = count_DepthwiseConv2d(depth_layer, x)
    total_mults += flop_mults; total_adds += flop_adds
    x = depth_layer(x)
    # pointwise
    point_layer = sep_conv2d_layer.pointwise
    flop_mults, flop_adds = count_Conv2d(point_layer, x)
    total_mults += flop_mults; total_adds += flop_adds
    x = point_layer(x)
    
    return total_mults, total_adds

def count_SeparableBranch(separable_branch, x):
    total_mults = total_adds = 0
    
    # (block1)
    block1 = separable_branch.block1
    # (0): ReLU
    flop_mults, flop_adds = countReLU(x)
    total_mults += flop_mults; total_adds += flop_adds
    x = block1[0](x)
    
    # (1) Separable Conv2d
    sep_conv2d_layer = block1[1]
    flop_mults, flop_adds = count_SeparableConv2d(sep_conv2d_layer, x)
    total_mults += flop_mults; total_adds += flop_adds
    x = sep_conv2d_layer(x)
    
    # (2) Batch Norm
    norm_layer = block1[2]
    x = norm_layer(x)
    
    # block2
    block2 = separable_branch.block2
    # (0): ReLU
    flop_mults, flop_adds = countReLU(x)
    total_mults += flop_mults; total_adds += flop_adds
    x = block2[0](x)
    
    # (1) Separable Conv2d
    sep_conv2d_layer = block2[1]
    flop_mults, flop_adds = count_SeparableConv2d(sep_conv2d_layer, x)
    total_mults += flop_mults; total_adds += flop_adds
    x = sep_conv2d_layer(x)
    
    # (2) Batch Norm
    norm_layer = block2[2]
    x = norm_layer(x)
    
    return total_mults, total_adds

def count_NormalCell(normalcell, x, prev):
    total_mults = total_adds = 0
    # run fit
    fit_block = normalcell.fit
    flop_mults, flop_adds = count_Fit(fit_block, x, prev)
    total_mults += flop_mults; total_adds += flop_adds
    prev = fit_block((x, prev))

    # run dim_reduce
    reduce_block  = normalcell.dem_reduce
    flop_mults, flop_adds = count_dim_reduce(reduce_block, x)
    total_mults += flop_mults; total_adds += flop_adds
    h = reduce_block(x)

    # get x1
    block1_left = normalcell.block1_left
    flop_mults, flop_adds = count_SeparableBranch(block1_left, h)
    total_mults += flop_mults; total_adds += flop_adds
    # block1_right is empty


    # get x2
    block2_left = normalcell.block2_left
    flop_mults, flop_adds = count_SeparableBranch(block2_left, prev)
    total_mults += flop_mults; total_adds += flop_adds

    block2_right = normalcell.block2_right
    flop_mults, flop_adds = count_SeparableBranch(block2_right, h)
    total_mults += flop_mults; total_adds += flop_adds

    # get x3
    block3_left = normalcell.block3_left
    flop_mults, flop_adds = count_AvgPool2d(block3_left, h)
    total_mults += flop_mults; total_adds += flop_adds
    # block3_right is empty


    # get x4
    block4_left = normalcell.block4_left
    flop_mults, flop_adds = count_AvgPool2d(block4_left, prev)
    total_mults += flop_mults; total_adds += flop_adds

    block4_right = normalcell.block4_left
    flop_mults, flop_adds = count_AvgPool2d(block4_left, prev)
    total_mults += flop_mults; total_adds += flop_adds

    # get x5
    block5_left = normalcell.block5_left
    flop_mults, flop_adds = count_SeparableBranch(block5_left, prev)
    total_mults += flop_mults; total_adds += flop_adds

    block5_right = normalcell.block5_right
    flop_mults, flop_adds = count_SeparableBranch(block5_right, h)
    total_mults += flop_mults; total_adds += flop_adds
    
    return total_mults, total_adds

def count_ReductionCell(reductioncell, x, prev):
    total_mults = total_adds = 0
    # run fit
    fit_block = reductioncell.fit
    flop_mults, flop_adds = count_Fit(fit_block, x, prev)
    total_mults += flop_mults; total_adds += flop_adds
    prev = fit_block((x, prev))

    # run dim_reduce
    reduce_block  = reductioncell.dim_reduce
    flop_mults, flop_adds = count_dim_reduce(reduce_block, x)
    total_mults += flop_mults; total_adds += flop_adds
    h = reduce_block(x)

    # get layer1block1
    layer1block1_left = reductioncell.layer1block1_left
    flop_mults, flop_adds = count_SeparableBranch(layer1block1_left, prev)
    total_mults += flop_mults; total_adds += flop_adds

    layer1block1_right = reductioncell.layer1block1_right
    flop_mults, flop_adds = count_SeparableBranch(layer1block1_right, h)
    total_mults += flop_mults; total_adds += flop_adds

    layer1block1 = reductioncell.layer1block1_left(prev) + reductioncell.layer1block1_right(h)

    # get layer1block2
    # left is maxpool, so no flop
    layer1block2_left = reductioncell.layer1block2_left

    layer1block2_right = reductioncell.layer1block2_right
    flop_mults, flop_adds = count_SeparableBranch(layer1block2_right, prev)
    total_mults += flop_mults; total_adds += flop_adds

    layer1block2 = reductioncell.layer1block2_left(h) + reductioncell.layer1block2_right(prev)

    # get layer1block3
    layer1block3_left = reductioncell.layer1block3_left
    flop_mults, flop_adds = count_AvgPool2d(layer1block3_left, h)
    total_mults += flop_mults; total_adds += flop_adds

    layer1block3_right = reductioncell.layer1block3_right
    flop_mults, flop_adds = count_SeparableBranch(layer1block3_right, prev)
    total_mults += flop_mults; total_adds += flop_adds

    # get layer2block1
    # left is maxpool, so no flop
    layer1block3_right = reductioncell.layer1block3_right
    flop_mults, flop_adds = count_SeparableBranch(layer1block3_right, layer1block1)
    total_mults += flop_mults; total_adds += flop_adds

    # get layer2block2
    layer2block2_left = reductioncell.layer2block2_left
    flop_mults, flop_adds = count_AvgPool2d(layer2block2_left, layer1block1)
    total_mults += flop_mults; total_adds += flop_adds
    # layer2block2_right is just Sequential() so no FLOP
    return total_mults, total_adds