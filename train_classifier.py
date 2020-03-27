from classifier import Net
from datagenerator import HouseNumberDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from torch.autograd import Variable


# Parameter
NUMBER_TRAINING = 0
NB_EPOCHS = 2
SIZE_BATCHES = 10
NB_WORKERS = 0
print_every = 2000

# Paths
DATA_ROOT = 'data/train_32x32.mat'
SAVE_ROOT = 'net_data/training_{}'.format(NUMBER_TRAINING)
LOAD_ROOT = 'net_data/training_{}'.format(NUMBER_TRAINING - 1)

# Take the data
DATASET = HouseNumberDataset(DATA_ROOT, training=True)

# Load the data
DATALOADER = DataLoader(DATASET, batch_size=SIZE_BATCHES, shuffle=True, num_workers=NB_WORKERS)

# Create the net
NET = Net()

# Use of GPU, if it is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NET = NET.to(device)
if device == 'cuda':
    NET = nn.DataParallel(NET)

# Load former trainings
if NUMBER_TRAINING > 0:
    NET.load_state_dict(torch.load(LOAD_ROOT))

# Define the loss and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(NET.parameters(), lr=0.001, momentum=0.9)


for epoch in range(NB_EPOCHS):  # loop over the dataset multiple times
    # We set the loss to 0 for each epoch
    running_loss = 0.0
    for steps, data in enumerate(DATALOADER):
        # get the inputs
        (inputs, labels) = data
        (inputs, labels) = inputs.to(device), labels.to(device)

        targets = labels.view(labels.size()[0]).long()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = NET(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if steps % print_every == 0:
            print("Epoch : {}. The running loss is {}.".format(epoch + 1, running_loss / print_every))
            running_loss = 0.0

print('Finished Training')

# Save the weigth
torch.save(NET.state_dict(), SAVE_ROOT)


