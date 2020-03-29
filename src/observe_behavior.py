"""
This module allows the user to observe the behavior of the nets on the data :

On which kind of images the classifier is mistaken ?
It shows the first, NB_EXAMPLES images that are wrongly estimated by the nets.

On which label does the net make the more errors ?
"""
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from src.datagenerator.datagenerator import HouseNumberDataset
from src.nets.three_layer_net import Net3Layers
from src.utils.show_image import show_image


def show_mistakes(net, set_root, nb_example):
    """
    Show the first nb_example images on which the net is mistaken.

    Args:
        net (Dataset): the net that we want to try.

        set_root (string): the path where the data is.

        nb_example (int): the number of examples on which the net is mistaken.
    """
    # Get the data
    dataset = HouseNumberDataset(set_root, for_dataloader=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=NB_WORKERS)

    # Stores the index of the images that are wrongly estimated
    # 11 because : INDEX_EXAMPLES[i] = [index_image, probability of each class]
    index_examples = np.zeros((nb_example, 11))

    with torch.no_grad():
        # The current number of images on which the generator is mistaken
        nb_current_ex = 0
        step = 0
        nb_steps = len(dataset)

        # The iterator on the data loader
        iter_loader = iter(loader)

        # Continue until the number of example asked is reached
        while step < nb_steps and nb_current_ex < nb_example:
            # Get the input and the label
            (inputs, labels) = next(iter_loader)
            (inputs, labels) = inputs.to(DEVICE), labels.to(DEVICE)

            # Get the output
            soft_max = nn.Softmax(dim=1)
            outputs = soft_max(net(inputs))

            # If the nets is mistaken, we register the index of the image
            if int(labels[0]) != int(torch.argmax(outputs)):
                index_examples[nb_current_ex, 0] = step
                # We need to transfer OUTPUTS to the cpu to convert it in an array
                outputs = outputs.cpu()
                index_examples[nb_current_ex, 1:] = outputs.numpy()
                nb_current_ex += 1

            step += 1

    # We get the images without transformation
    dataset_visualisation = HouseNumberDataset(set_root, for_dataloader=False)

    # Plot every image in INDEX_EXAMPLES
    for example in index_examples:
        index_image = int(example[0])
        (img, label) = dataset_visualisation[index_image]
        show_image(img, index_image, np.argmax(example[1:]), label[0])


def print_mistakes(net, set_root):
    """
    Print the proportion of the errors for each label

    Args:
        net (Dataset): the net that we want to try.

        set_root (string): the path where the data is.
    """
    # Get the data
    dataset = HouseNumberDataset(set_root, for_dataloader=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=NB_WORKERS)

    # Stores errors for each label
    errors = np.zeros(10)

    for data in loader:
        # Get the input and the labels
        (inputs, labels) = data
        (inputs, labels) = inputs.to(DEVICE), labels.to(DEVICE)

        # Get the output
        outputs = net(inputs)

        # If the nets is mistaken, we add one to the label
        if int(labels[0]) != int(torch.argmax(outputs)):
            errors[int(labels[0])] += 1

    print(errors / sum(errors))


if __name__ == "__main__":
    # Parameters
    NB_LAYERS = 3  # !!! The net has to be changed when created underneath !!!
    CURRENT_TRAINING = 0
    # If there is a BrokenPipe Error on windows, put NB_WORKERS = 0
    NB_WORKERS = 0
    # The number of example where the nets is mistaken
    NB_EXAMPLE = 10

    # Paths
    LOAD_ROOT = Path("../net_data/") / "lay{}training_{}.pth".format(NB_LAYERS, CURRENT_TRAINING)
    TEST_ROOT = Path("../data/test_32x32.mat")

    # Create the nets. !!! To change if we want to change the number of layers !!!
    NET = Net3Layers()

    # Use GPU, if it is available
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NET = NET.to(DEVICE)
    if torch.cuda.is_available():
        NET = nn.DataParallel(NET)

    # Load the nets
    NET.load_state_dict(torch.load(LOAD_ROOT))
    NET.eval()

    # Show the mistakes
    # show_mistakes(NET, TEST_ROOT, NB_EXAMPLE)

    # Print the number of error for each label
    print_mistakes(NET, TEST_ROOT)
