"""
This script trains the model. It also computes the training and test accuracy.
Two models can be chosen : a three layer classifier or a six layer classifier.
"""
from pathlib import Path
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from src.nets.three_layer_net import Net3Layers
from src.nets.six_layer_net import Net6Layers
from src.datagenerator.datagenerator import HouseNumberDataset
from src.utils.store_results import store_results
from src.utils.metrics import Metrics

# Parameters
NB_LAYERS = 3  # !!! The net has to be changed when created underneath !!!
NB_TRAINING = 0  # The number of training that has been done
NB_EPOCHS = 10
SIZE_BATCHES = 15
# If there is a BrokenPipe Error on windows, put NB_WORKERS = 0
NB_WORKERS = 0

# Paths
TRAIN_ROOT = Path("../data/train_32x32.mat")
TEST_ROOT = Path("../data/test_32x32.mat")
SAVE_ROOT = Path("../net_data/") / "lay{}training_{}.pth".format(NB_LAYERS, NB_TRAINING)
LOAD_ROOT = Path("../net_data/") / "lay{}training_{}.pth".format(NB_LAYERS, NB_TRAINING - 1)


# Take the trainset and the testset
TRAINSET = HouseNumberDataset(TRAIN_ROOT, for_dataloader=True)
TESTSET = HouseNumberDataset(TEST_ROOT, for_dataloader=True)

# Load the trainset and the testset
print("Loading data ...")
TRAINLOADER = DataLoader(TRAINSET, batch_size=SIZE_BATCHES, shuffle=True, num_workers=NB_WORKERS)
TESTLOADER = DataLoader(TESTSET, batch_size=SIZE_BATCHES, shuffle=True, num_workers=NB_WORKERS)

# Create the nets. !!! To change if we want to change the number of layers !!!
NET = Net3Layers()

# Use GPU, if it is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NET = NET.to(DEVICE)
if torch.cuda.is_available():
    NET = nn.DataParallel(NET)

# Load former trainings
if NB_TRAINING > 0:
    NET.load_state_dict(torch.load(LOAD_ROOT))

# Define the loss and the optimizer
CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = optim.SGD(NET.parameters(), lr=0.001, momentum=0.9)

# Statistics
TRAIN_METRICS = Metrics(NB_EPOCHS)
TEST_METRICS = Metrics(NB_EPOCHS)

print("Start training ...")
for epoch in range(NB_EPOCHS):
    # --- The training --- #
    print("Training : epoch {}".format(epoch + 1))
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
        with torch.no_grad():
            for index_image in range(labels.size()[0]):
                TRAIN_METRICS.update(int(torch.argmax(outputs[index_image])), labels[index_image], loss.item())

    # Save the metrics
    TRAIN_METRICS.save(epoch)

    # --- Test --- #
    print("Test : epoch {}".format(epoch + 1))
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

            # Register statistics
            for index_image in range(labels.size()[0]):
                if int(labels[index_image]) != int(torch.argmax(outputs[index_image])):
                    TEST_METRICS.update(int(torch.argmax(outputs[index_image])), labels[index_image], loss.item())

    # Save the metrics
    TEST_METRICS.save(epoch)

print('Finished Training')

# Save the weights
torch.save(NET.state_dict(), SAVE_ROOT)

# Get the metrics
(TRAIN_AVG_ACC, TRAIN_F_BETA, TRAIN_ACC, TRAIN_LOSS) = TRAIN_METRICS.get()
(TEST_AVG_ACC, TEST_F_BETA, TEST_ACC, TEST_LOSS) = TEST_METRICS.get()

# Paths to the metrics
RESULTS_ROOT_TRAIN = Path("../net_data/results/train/")
RESULTS_ROOT_TEST = Path("../net_data/results/test/")

TRAIN_AVG_ACC_ROOT = RESULTS_ROOT_TRAIN / "lay{}training_avegrage_accuracy_{}.txt".format(NB_LAYERS, NB_TRAINING)
TRAIN_F_BETA_ROOT = RESULTS_ROOT_TRAIN / "lay{}training_F_beta_{}.txt".format(NB_LAYERS, NB_TRAINING)
TRAIN_ACC_ROOT = RESULTS_ROOT_TRAIN / "lay{}training_accuracy_{}.txt".format(NB_LAYERS, NB_TRAINING)
TRAIN_LOSS_ROOT = RESULTS_ROOT_TRAIN / "lay{}training_loss_{}.txt".format(NB_LAYERS, NB_TRAINING)

TEST_AVG_ACC_ROOT = RESULTS_ROOT_TEST / "lay{}test_average_accuracy_{}.txt".format(NB_LAYERS, NB_TRAINING)
TEST_F_BETA_ROOT = RESULTS_ROOT_TEST / "lay{}test_F_beta_{}.txt".format(NB_LAYERS, NB_TRAINING)
TEST_ACC_ROOT = RESULTS_ROOT_TEST / "lay{}test_accuracy_{}.txt".format(NB_LAYERS, NB_TRAINING)
TEST_LOSS_ROOT = RESULTS_ROOT_TEST / "lay{}test_loss_{}.txt".format(NB_LAYERS, NB_TRAINING)

# Store the results
store_results(TRAIN_AVG_ACC, TRAIN_AVG_ACC_ROOT)
store_results(TRAIN_F_BETA, TRAIN_F_BETA_ROOT)
store_results(TRAIN_ACC, TRAIN_ACC_ROOT)
store_results(TRAIN_LOSS, TRAIN_LOSS_ROOT)

store_results(TEST_AVG_ACC, TEST_AVG_ACC_ROOT)
store_results(TEST_F_BETA, TEST_F_BETA_ROOT)
store_results(TEST_ACC, TEST_AVG_ACC_ROOT)
store_results(TEST_LOSS, TEST_LOSS_ROOT)
