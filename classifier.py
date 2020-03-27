import torch
import torch.nn as nn
import numpy as np
from datagenerator import HouseNumberDataset
from torch.utils.data import DataLoader


def compute_dimension(x, k, s, p, d):
    return np.floor((x + 2 * p - d * (k - 1) - 1) / s + 1)


class Net(nn.Module):
    def __init__(self):
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

        self.max_pool = nn.MaxPool2d(kernel_size =2, stride=2, padding=0, dilation=1)

        # Input size : (2 x 2 x 128)
        self.gap = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
        # Output size : (1 x 1 x 128)

        # Input size : (1 x 128)
        self.linear11 = nn.Linear(128, 11)
        # Output size : (1 x 11)

        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x):
        # x is a tensor with dtype=float64, we need to have dtype=float
        # Size : (32 x 32 x 3)
        x = self.layer1(x.float())
        # Size : (30 x 30 x 32)
        x = self.max_pool(x)
        # Size : (15 x 15 x 32)

        x = self.layer2(x)
        # Size : (13 x 13 x 64)
        x = self.max_pool(x)
        # Size : (6 x 6 x 64)

        x = self.layer3(x)
        # Size : (4 x 4 x 128)
        x = self.max_pool(x)
        # Size : (2 x 2 x 128)

        x = self.gap(x)
        # Size : (1 x 1 x 128)
        x = x.view(-1, 128)
        # Size : (1 x 128)
        x = self.linear11(x)
        # Size : (1 x 11)

        return x


if __name__ == "__main__":
    DATA_ROOT = 'data/train_32x32.mat'
    DATASET = HouseNumberDataset(DATA_ROOT, True)
    net = Net()

    DATALOADER = DataLoader(DATASET, batch_size=100, shuffle=True, num_workers=4)

    for i, data in enumerate(DATALOADER, 0):
        (images, labels) = data
        out = net(images.float())
        print(out)
        print(labels)
