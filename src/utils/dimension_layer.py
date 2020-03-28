"""
This file is not used in the main code.
It is only helpful to compute the dimension of the output of a layer.
"""
import numpy as np


def compute_dimension(input_size, kernel_size, stride, padding, dilation):
    """
    Computes the output size of an image that goes through a layer.

    Args:
        input_size (int): the size of the input.

        kernel_size (int): the size of the layer's kernel.

        stride (int): the stride of the layer.

        padding (int): the size of the layer's padding.

        dilation (int): the dilation of the layer.

    Returns:
        output_size (int): the size of the output.
    """
    return np.floor((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


if __name__ == "__main__":
    print(compute_dimension(4, kernel_size=3, stride=1, padding=1, dilation=1))
