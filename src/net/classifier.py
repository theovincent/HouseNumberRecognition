"""
This module is the classifier. It is make to classify 11 classes.

Classes:
    Net
"""
import torch.nn as nn
import numpy as np


def compute_dimension(input_size, kernel_size, stride, padding, dilatation):
    """
    Computes the output size of an image that goes through a layer.

    Args:
        input_size (int): the size of the input.

        kernel_size (int): the size of the layer's kernel.

        stride (int): the stride of the layer.

        padding (int): the size of the layer's padding.

        dilatation (int): the dilatation of the layer.

    Returns:
        output_size (int): the size of the output.
    """
    return np.floor((input_size + 2 * padding - dilatation * (kernel_size - 1) - 1) / stride + 1)


class Net(nn.Module):
    """
    A CNN to classify 11 classes.
    """
    def __init__(self):
        """
        Defines the layers and functions that will be used by the CNN.
        """
        super(Net, self).__init__()

        # Input size : (32 x 32 x 3)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        # Output size : (30 x 30 x 32)

        # Input size : (15 x 15 x 32)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        # Output size : (13 x 13 x 64)

        # Input size : (6 x 6 x 64)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=2, dilation=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        # Output size : (4 x 4 x 128)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)

        # Input size : (2 x 2 x 128)
        self.gap = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
        # Output size : (1 x 1 x 128)

        # Input size : (1 x 128)
        self.linear11 = nn.Linear(128, 11)
        # Output size : (1 x 11)

        self.soft_max = nn.Softmax(dim=1)

    def forward(self, batch):
        """
        Construct the CNN.

        Args:
            batch (tensor, size = (N x 32 x 32 x 3), dtype=float64):
                the batch that will go through the CNN (with N the size of the batch).

        Returns:
            batch (tensor, size = (N x 32 x 32 x 3), dtype=long):
                the batch that has gone through the CNN (with N the size of the batch).
        """
        # x is a tensor with dtype=float64, we need to have dtype=float
        # Size : (32 x 32 x 3)
        batch = self.layer1(batch.float())
        # Size : (30 x 30 x 32)
        batch = self.max_pool(batch)
        # Size : (15 x 15 x 32)

        batch = self.layer2(batch)
        # Size : (13 x 13 x 64)
        batch = self.max_pool(batch)
        # Size : (6 x 6 x 64)

        batch = self.layer3(batch)
        # Size : (4 x 4 x 128)
        batch = self.max_pool(batch)
        # Size : (2 x 2 x 128)

        batch = self.gap(batch)
        # Size : (1 x 1 x 128)
        batch = batch.view(-1, 128)
        # Size : (1 x 128)
        batch = self.linear11(batch)
        # Size : (1 x 11)

        return batch


if __name__ == "__main__":
    NET = Net()
