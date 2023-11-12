from typing import Union

import numpy as np
import torch
from torch import tensor

from caveat import current_device


def hot_argmax(batch: tensor, axis: int = -1) -> tensor:
    """Encoded given axis as one-hot based on argmax for that axis.

    Args:
        batch (tensor): Input tensor.
        axis (int, optional): Axis index to encode. Defaults to -1.

    Returns:
        tensor: One hot encoded tensor.
    """
    batch = batch.swapaxes(axis, -1)
    argmax = batch.argmax(axis=-1)
    eye = torch.eye(batch.shape[-1])
    eye = eye.to(current_device())
    batch = eye[argmax]
    return batch.swapaxes(axis, -1)


def conv_size(
    size: Union[tuple[int, int], int],
    kernel_size: Union[tuple[int, int], int] = 3,
    stride: Union[tuple[int, int], int] = 2,
    padding: Union[tuple[int, int], int] = 1,
    dilation: Union[tuple[int, int], int] = 1,
) -> np.array:
    """Calculate output dimensions for 2d convolution.

    Args:
        size (Union[tuple[int, int], int]): Input size, may be integer if symetric.
        kernel_size (Union[tuple[int, int], int], optional): Kernel_size. Defaults to 3.
        stride (Union[tuple[int, int], int], optional): Stride. Defaults to 2.
        padding (Union[tuple[int, int], int], optional): Input padding. Defaults to 1.
        dilation (Union[tuple[int, int], int], optional): Dilation. Defaults to 1.

    Returns:
        np.array: Output size.
    """
    if isinstance(size, int):
        size = (size, size)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    return (
        np.array(size)
        + 2 * np.array(padding)
        - np.array(dilation) * (np.array(kernel_size) - 1)
        - 1
    ) // np.array(stride) + 1


def transconv_size(
    size: Union[tuple[int, int], int],
    kernel_size: Union[tuple[int, int], int] = 3,
    stride: Union[tuple[int, int], int] = 2,
    padding: Union[tuple[int, int], int] = 1,
    dilation: Union[tuple[int, int], int] = 1,
    output_padding: Union[tuple[int, int], int] = 1,
) -> np.array:
    """Calculate output dimension for 2d transpose convolution.

    Args:
        size (Union[tuple[int, int], int]): Input size, may be integer if symetric.
        kernel_size (Union[tuple[int, int], int], optional): Kernel size. Defaults to 3.
        stride (Union[tuple[int, int], int], optional): Stride. Defaults to 2.
        padding (Union[tuple[int, int], int], optional): Input padding. Defaults to 1.
        dilation (Union[tuple[int, int], int], optional): Dilation. Defaults to 1.
        output_padding (Union[tuple[int, int], int], optional): Output padding. Defaults to 1.

    Returns:
        np.array: Output size.
    """
    if isinstance(size, int):
        size = (size, size)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding)
    return (
        (np.array(size) - 1) * np.array(stride)
        - 2 * np.array(padding)
        + np.array(dilation) * (np.array(kernel_size) - 1)
        + np.array(output_padding)
        + 1
    )


def calc_output_padding(size: Union[tuple[int, int], int]) -> np.array:
    """Calculate output padding for a transposed convolution such that output dims will
    match dimensions of inputs to a convolution of given size.
    For each dimension, padding is set to 1 if even size, otherwise 0.

    Args:
        size (Union[tuple[int, int], int]): input size (h, w)

    Returns:
        np.array: required padding
    """
    if isinstance(size, int):
        size = (size, size)
    h, w = size
    return (int(h % 2 == 0), int(w % 2 == 0))
