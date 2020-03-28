"""
This script allows the user to observe on which kind of images the classifier is mistaken.
"""
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from src.datagenerator.datagenerator import HouseNumberDataset
from src.net.classifier import Net
from src.utils.show_image import show_image


# Parameters
CURRENT_TRAINING = 0
# If there is a BrokenPipe Error on windows, put NB_WORKERS = 0
NB_WORKERS = 0
NB_EXAMPLES = 10

# Paths
LOAD_ROOT = Path("../net_data/") / "training_{}.pth".format(CURRENT_TRAINING)
TEST_ROOT = Path("../data/test_32x32.mat")

# Create the net and load it
NET = Net()
NET.load_state_dict(torch.load(LOAD_ROOT))
NET.eval()

# Use GPU, if it is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NET = NET.to(DEVICE)
if DEVICE == 'cuda':
    NET = nn.DataParallel(NET)

# Get the data
TESTSET = HouseNumberDataset(TEST_ROOT, for_dataloader=True)
TESTLOADER = DataLoader(TESTSET, batch_size=1, shuffle=False, num_workers=NB_WORKERS)

# Examples
INDEX_EXAMPLES = np.zeros((NB_EXAMPLES, 12))

with torch.no_grad():
    # The number of images on which the generator is mistaken
    nb_current_ex = 0
    step = 0
    nb_steps = len(TESTSET)

    # The iterator on the data loader
    iter_loader = iter(TESTLOADER)

    while step < nb_steps and nb_current_ex < NB_EXAMPLES:
        # Get the inputs and the labels
        (inputs, labels) = next(iter_loader)
        (inputs, labels) = inputs.to(DEVICE), labels.to(DEVICE)

        # Format the targets
        targets = labels.view(labels.size()[0]).long()

        # Forward + Backward + Optimize
        outputs = NET.soft_max(NET(inputs))

        # If the net is mistaken, w
        if int(labels[0]) != int(torch.argmax(outputs)):
            INDEX_EXAMPLES[nb_current_ex, 0] = step
            INDEX_EXAMPLES[nb_current_ex, 1:] = outputs.numpy()
            nb_current_ex += 1

        step += 1

TESTSET_VISUALISATION = HouseNumberDataset(TEST_ROOT, for_dataloader=False)

for example in INDEX_EXAMPLES:
    index_image = int(example[0])
    (img, label) = TESTSET_VISUALISATION[index_image]
    show_image(img, index_image, np.argmax(example[1:]), label[0])
