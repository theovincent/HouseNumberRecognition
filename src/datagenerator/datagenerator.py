"""
This module is a data generator adapted to the SVHN dataset.

Classes:
    HouseNumberDataset
"""
from pathlib import Path
from torch.utils.data import Dataset
from scipy.io import loadmat
import numpy as np


class HouseNumberDataset(Dataset):
    """
    The data generator class.
    """
    def __init__(self, data_root, for_dataloader=False):
        """
        Construct the data generator from the data_root.
        In mode for_dataloader, the images are transformed
        to fit the dataloader expectations.

        Args:
            data_root (string): the root where the data is.

            for_dataloader (boolean): indicates
                if the data will be used to go through the CNN
                or if it will be use to visualise the image.
        """
        data_dictionary = loadmat(data_root)
        self.samples = data_dictionary["X"]  # [:, :, :, : 500]
        self.labels = data_dictionary["y"]  # [: 500]
        self.training = for_dataloader

    def __len__(self):
        """
        Returns:
            the length of the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Select an item. Transforms the image if self.training is true.

        Args:
            idx (int): the index of the selected item.

        Returns:
            if self.training:     sample (array, shape=(3, 32, 32): a transformed image

                                  label (int): the label of the image

            if not self.training: sample (array, shape=(32, 32, 3)): an image

                                  label (int): the label of the image
        """
        if self.training:
            return self.transform(idx), self.labels[idx]
        return self.samples[:, :, :, idx], self.labels[idx]

    def transform(self, idx):
        """
        Transform the image that has the index 'idx'.
        It separes the color by changing the shape:
        from (32, 32, 3) to (3, 32, 32) and normalises each color between -1 to 1.

        Arg:
            idx (int): the image index.

        Returns:
            image (array, shape=(3, 32, 32)): the transformed image.
        """
        (height, width, nb_colors) = self.samples.shape[:3]
        image = np.ones((nb_colors, height, width), dtype=np.double)

        for color in range(nb_colors):
            image[color, :, :] = normalize(self.samples[:, :, color, idx])

        return image


def normalize(img):
    """
    Normalize the img between -1 to 1.

    Args:
        img (array, any shape): the image to normalize.

    Return:
        (array, same shape as img): the normalized image.
    """
    min_image = np.min(img)
    return 2 * (img - min_image) / (np.max(img) - min_image) - 1


if __name__ == '__main__':
    DATA_ROOT = Path("../../data/train_32x32.mat")
    DATASET = HouseNumberDataset(DATA_ROOT, for_dataloader=True)
    print(len(DATASET))
