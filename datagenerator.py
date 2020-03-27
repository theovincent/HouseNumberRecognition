from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.io import loadmat
import numpy as np


class HouseNumberDataset(Dataset):
    def __init__(self, data_root, training=False):
        data_dictionary = loadmat(data_root)
        self.samples = data_dictionary["X"]  # [:, :, :, : 500]
        self.labels = data_dictionary["y"]  # [: 500]
        self.training = training

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.training:
            return self.transform(idx), self.labels[idx]
        return self.samples[:, :, :, idx], self.labels[idx]

    def transform(self, idx):
        (h, w, c) = self.samples.shape[:3]
        image = np.ones((c, h, w), dtype=np.double)

        for i in range(c):
            image[i, :, :] = normalize(self.samples[:, :, i, idx])

        return image


def normalize(img):
    min_image = np.min(img)
    return 2 * (img - min_image) / (np.max(img) - min_image) - 1


if __name__ == '__main__':
    DATA_ROOT = 'data/train_32x32.mat'
    DATASET = HouseNumberDataset(DATA_ROOT, training=True)
    list_int = np.zeros(len(DATASET))
    DATALOADER = DataLoader(DATASET, batch_size=1, shuffle=True, num_workers=1)

    for i, batch in enumerate(DATALOADER):
        list_int[i] = int(batch[1][0])

    print(max(list_int))
