"""
This script allows the user to observe on which kind of images the classifier is mistaken.
It shows the first, NB_EXAMPLES images that are wrongly estimated by the nets.
"""
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from src.datagenerator.datagenerator import HouseNumberDataset
from src.nets.three_layer_net import Net3Layers
from src.utils.show_image import show_image


# Parameters
CURRENT_TRAINING = 0
# If there is a BrokenPipe Error on windows, put NB_WORKERS = 0
NB_WORKERS = 0
# The number of example where the nets is mistaken
NB_EXAMPLES = 10

# Paths
LOAD_ROOT = Path("../net_data/") / "training_{}.pth".format(CURRENT_TRAINING)
TEST_ROOT = Path("../data/test_32x32.mat")

# Create the nets
NET3LAYERS = Net3Layers()

# Use GPU, if it is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NET3LAYERS = NET3LAYERS.to(DEVICE)
if torch.cuda.is_available():
    NET3LAYERS = nn.DataParallel(NET3LAYERS)

# Load the nets
NET3LAYERS.load_state_dict(torch.load(LOAD_ROOT))
NET3LAYERS.eval()

# Get the data
TESTSET = HouseNumberDataset(TEST_ROOT, for_dataloader=True)
TESTLOADER = DataLoader(TESTSET, batch_size=1, shuffle=False, num_workers=NB_WORKERS)

# Stores the index of the images that are wrongly estimated
INDEX_EXAMPLES = np.zeros((NB_EXAMPLES, 12))

with torch.no_grad():
    # The current number of images on which the generator is mistaken
    NB_CURRENT_EX = 0
    STEP = 0
    NB_STEPS = len(TESTSET)

    # The iterator on the data loader
    ITER_LOADER = iter(TESTLOADER)

    # Continue until the number of example asked is reached
    while STEP < NB_STEPS and NB_CURRENT_EX < NB_EXAMPLES:
        # Get the input and the label
        (INPUTS, LABELS) = next(ITER_LOADER)
        (INPUTS, LABELS) = INPUTS.to(DEVICE), LABELS.to(DEVICE)

        # Format the targets
        TARGETS = LABELS.view(LABELS.size()[0]).long()

        # Get the output
        SOFT_MAX = nn.Softmax(dim=1)
        OUTPUTS = SOFT_MAX(NET3LAYERS(INPUTS))

        # If the nets is mistaken, we register the index of the image
        if int(LABELS[0]) != int(torch.argmax(OUTPUTS)):
            INDEX_EXAMPLES[NB_CURRENT_EX, 0] = STEP
            # We need to transfer OUTPUTS to the cpu to convert it in an array
            OUTPUTS = OUTPUTS.cpu()
            INDEX_EXAMPLES[NB_CURRENT_EX, 1:] = OUTPUTS.numpy()
            NB_CURRENT_EX += 1

        STEP += 1

# We get the images without transformation
TESTSET_VISUALISATION = HouseNumberDataset(TEST_ROOT, for_dataloader=False)

# Plot every image in INDEX_EXAMPLES
for example in INDEX_EXAMPLES:
    index_image = int(example[0])
    (img, label) = TESTSET_VISUALISATION[index_image]
    show_image(img, index_image, np.argmax(example[1:]), label[0])
