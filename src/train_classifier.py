"""
This script trains the model. It also computes the training and test accuracy.
"""
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
from src.net.classifier import Net
from src.datagenerator.datagenerator import HouseNumberDataset

# Parameters
NUMBER_TRAINING = 0
NB_EPOCHS = 2
SIZE_BATCHES = 15
# If there is a BrokenPipe Error on windows, put NB_WORKERS = 0
NB_WORKERS = 0
PRINT_EVERY = 500

# Paths
TRAIN_ROOT = Path("../data/train_32x32.mat")
TEST_ROOT = Path("../data/test_32x32.mat")
SAVE_ROOT = Path("../net_data/") / "training_{}.pth".format(NUMBER_TRAINING)
LOAD_ROOT = Path("../net_data/") / "training_{}.pth".format(NUMBER_TRAINING - 1)

# Take the trainset and the testset
TRAINSET = HouseNumberDataset(TRAIN_ROOT, for_dataloader=True)
TESTSET = HouseNumberDataset(TEST_ROOT, for_dataloader=True)

# Load the trainset and the testset
print("Loading data ...")
TRAINLOADER = DataLoader(TRAINSET, batch_size=SIZE_BATCHES, shuffle=True, num_workers=NB_WORKERS)
TESTLOADER = DataLoader(TESTSET, batch_size=SIZE_BATCHES, shuffle=True, num_workers=NB_WORKERS)

# Create the net
NET = Net()

# Load former trainings
if NUMBER_TRAINING > 0:
    NET.load_state_dict(torch.load(LOAD_ROOT))

# Use GPU, if it is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NET = NET.to(DEVICE)
if DEVICE == 'cuda':
    NET = nn.DataParallel(NET)

# Define the loss and the optimizer
CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = optim.SGD(NET.parameters(), lr=0.001, momentum=0.9)

# Statistics
TRAIN_ACCURACY = np.zeros(NB_EPOCHS)
TRAIN_LOSS = np.ones(NB_EPOCHS)
TEST_ACCURACY = np.zeros(NB_EPOCHS)
TEST_LOSS = np.ones(NB_EPOCHS)

print("Start training ...")
for epoch in range(NB_EPOCHS):
    # Set the loss and the accuracy to 0 for each epoch
    running_loss = 0.0

    # --- The training --- #
    NET.train()
    for steps, data in enumerate(TRAINLOADER):
        # Get the inputs and the labels
        (inputs, labels) = data
        (inputs, labels) = inputs.to(DEVICE), labels.to(DEVICE)

        # Format the targets
        targets = labels.view(labels.size()[0]).long()

        # Zero the parameter gradients
        OPTIMIZER.zero_grad()

        # Forward + Backward + Optimize
        outputs = NET(inputs)
        loss = CRITERION(outputs, targets)
        loss.backward()
        OPTIMIZER.step()

        # Register statistics
        running_loss += loss.item()
        if steps % PRINT_EVERY == 0 and steps != 0:
            print("Epoch : {}. The running loss for training set is {}.".format(epoch + 1, running_loss / PRINT_EVERY))
            running_loss = 0.0

    # Set the loss and the accuracy to 0 for each epoch
    running_loss = 0.0
    running_accuracy = 0.0

    # --- The training --- #
    NET.eval()
    with torch.no_grad():
        for steps, data in enumerate(TESTLOADER):
            # Get the inputs and the labels
            (inputs, labels) = data
            (inputs, labels) = inputs.to(DEVICE), labels.to(DEVICE)

            # Format the targets
            targets = labels.view(labels.size()[0]).long()

            # Forward + Backward + Optimize
            outputs = NET(inputs)
            loss = CRITERION(outputs, targets)

            if steps % PRINT_EVERY == 0:
                print("Epoch : {}. The running loss for the test set is {}.".format(epoch + 1, running_loss / PRINT_EVERY))
                running_loss = 0.0


print('Finished Training')

# Save the weights
torch.save(NET.state_dict(), SAVE_ROOT)
