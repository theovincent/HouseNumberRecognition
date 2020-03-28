"""
This script trains the model. It also computes the training and test accuracy.
Two models can be chosen : a three layer classifier or a six layer classifier.
"""
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
from src.nets.three_layer_net import Net3Layers
from src.nets.six_layer_net import Net6Layers
from src.datagenerator.datagenerator import HouseNumberDataset
from src.utils.store_metrics import store_metrics

# Parameters
NB_LAYERS = 6  # !!! The net has to be changed when created underneath !!!
NB_TRAINING = 0  # The number of training that has been done
NB_EPOCHS = 20
SIZE_BATCHES = 15
# If there is a BrokenPipe Error on windows, put NB_WORKERS = 0
NB_WORKERS = 0

# Paths
TRAIN_ROOT = Path("../data/train_32x32.mat")
TEST_ROOT = Path("../data/test_32x32.mat")
SAVE_ROOT = Path("../net_data/") / "lay{}training_{}.pth".format(NB_LAYERS, NB_TRAINING)
LOAD_ROOT = Path("../net_data/") / "lay{}training_{}.pth".format(NB_LAYERS, NB_TRAINING - 1)
RESULTS_ROOT = Path("../net_data/results/")
TRAIN_ACCURACY_ROOT = RESULTS_ROOT / "lay{}training_accuracy_{}.txt".format(NB_LAYERS, NB_TRAINING)
TRAIN_LOSS_ROOT = RESULTS_ROOT / "lay{}training_loss_{}.txt".format(NB_LAYERS, NB_TRAINING)
TEST_ACCURACY_ROOT = RESULTS_ROOT / "lay{}test_accuracy_{}.txt".format(NB_LAYERS, NB_TRAINING)
TEST_LOSS_ROOT = RESULTS_ROOT / "lay{}test_loss_{}.txt".format(NB_LAYERS, NB_TRAINING)


# Take the trainset and the testset
TRAINSET = HouseNumberDataset(TRAIN_ROOT, for_dataloader=True)
TESTSET = HouseNumberDataset(TEST_ROOT, for_dataloader=True)

# Load the trainset and the testset
print("Loading data ...")
TRAINLOADER = DataLoader(TRAINSET, batch_size=SIZE_BATCHES, shuffle=True, num_workers=NB_WORKERS)
TESTLOADER = DataLoader(TESTSET, batch_size=SIZE_BATCHES, shuffle=True, num_workers=NB_WORKERS)

# Create the nets
NET3LAYERS = Net6Layers()

# Use GPU, if it is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NET3LAYERS = NET3LAYERS.to(DEVICE)
if torch.cuda.is_available():
    NET3LAYERS = nn.DataParallel(NET3LAYERS)

# Load former trainings
if NB_TRAINING > 0:
    NET3LAYERS.load_state_dict(torch.load(LOAD_ROOT))

# Define the loss and the optimizer
CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = optim.SGD(NET3LAYERS.parameters(), lr=0.001, momentum=0.9)

# Statistics
TRAIN_LOSS = np.ones(NB_EPOCHS)
TRAIN_ACCURACY = np.zeros(NB_EPOCHS)
TEST_LOSS = np.ones(NB_EPOCHS)
TEST_ACCURACY = np.zeros(NB_EPOCHS)
NB_TRAIN_IMAGES = len(TRAINSET)
NB_TEST_IMAGES = len(TESTSET)

print("Start training ...")
for epoch in range(NB_EPOCHS):
    # Set the loss and the accuracy to 0 for each epoch
    running_loss = 0.0
    running_accuracy = 0.0

    # --- The training --- #
    print("Training : epoch {}".format(epoch + 1))
    NET3LAYERS.train()
    for steps, data in enumerate(TRAINLOADER):
        # Get the inputs and the labels
        (inputs, labels) = data
        (inputs, labels) = inputs.to(DEVICE), labels.to(DEVICE)

        # Format the targets
        targets = labels.view(labels.size()[0]).long()

        # Zero the parameter gradients
        OPTIMIZER.zero_grad()

        # Forward + Backward + Optimize
        outputs = NET3LAYERS(inputs)
        loss = CRITERION(outputs, targets)
        loss.backward()
        OPTIMIZER.step()

        # Register statistics
        with torch.no_grad():
            running_loss += loss.item()
            for index_image in range(labels.size()[0]):
                if int(labels[index_image]) != int(torch.argmax(outputs[index_image])):
                    running_accuracy += 1

    # Update the loss and the accuracy
    TRAIN_LOSS[epoch] = running_loss / NB_TRAIN_IMAGES
    TRAIN_ACCURACY[epoch] = 1 - running_accuracy / NB_TRAIN_IMAGES
    running_loss = 0.0
    running_accuracy = 0.0

    # --- Test --- #
    print("Test : epoch {}".format(epoch + 1))
    NET3LAYERS.eval()
    with torch.no_grad():
        for steps, data in enumerate(TESTLOADER):
            # Get the inputs and the labels
            (inputs, labels) = data
            (inputs, labels) = inputs.to(DEVICE), labels.to(DEVICE)

            # Format the targets
            targets = labels.view(labels.size()[0]).long()

            # Forward + Backward + Optimize
            outputs = NET3LAYERS(inputs)
            loss = CRITERION(outputs, targets)

            # Register statistics
            running_loss += loss.item()
            for index_image in range(labels.size()[0]):
                if int(labels[index_image]) != int(torch.argmax(outputs[index_image])):
                    running_accuracy += 1

    # Update the loss and the accuracy
    TEST_LOSS[epoch] = running_loss / NB_TEST_IMAGES
    TEST_ACCURACY[epoch] = 1 - running_accuracy / NB_TEST_IMAGES


print('Finished Training')

# Save the weights, the loss and the accuracy
torch.save(NET3LAYERS.state_dict(), SAVE_ROOT)
store_metrics(TRAIN_LOSS, TRAIN_LOSS_ROOT)
store_metrics(TRAIN_ACCURACY, TRAIN_ACCURACY_ROOT)
store_metrics(TEST_LOSS, TEST_LOSS_ROOT)
store_metrics(TEST_ACCURACY, TEST_ACCURACY_ROOT)
