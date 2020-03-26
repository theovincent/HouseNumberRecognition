from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.io import loadmat


class HouseNumberDataset(Dataset):
    def __init__(self, data_root):
        data_dictionary = loadmat(data_root)
        self.samples = data_dictionary["X"][: 2]
        self.labels = data_dictionary["y"][: 2]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[:, :, :, idx], self.labels[idx]


if __name__ == '__main__':
    DATA_ROOT = 'data/train_32x32.mat'
    DATASET = HouseNumberDataset(DATA_ROOT)
    """print(type(DATASET[0]))
    print(DATASET[100][0].shape)
    print(DATASET[122:131][0].shape)"""

    DATALOADER = DataLoader(DATASET, batch_size=1, shuffle=True, num_workers=2)

    for i, batch in enumerate(DATALOADER):
        print(i, batch)
