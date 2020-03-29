"""
This script is made to observe the data sets :
Is there one data set unbalanced ?
"""
from pathlib import Path
import numpy as np
from src.datagenerator.datagenerator import HouseNumberDataset


# Parameters
# If there is a BrokenPipe Error on windows, put NB_WORKERS = 0
NB_WORKERS = 0


# Paths to the data sets
TRAIN_ROOT = Path("../data/train_32x32.mat")
EXTRA_ROOT = Path("../data/extra_32x32.mat")
TEST_ROOT = Path("../data/test_32x32.mat")

# Data generators
TRAINSET = HouseNumberDataset(TRAIN_ROOT, for_dataloader=False)
EXTRASET = HouseNumberDataset(EXTRA_ROOT, for_dataloader=False)
TESTSET = HouseNumberDataset(TEST_ROOT, for_dataloader=False)

# Counters
NB_OCCURENCES_TRAIN = np.zeros(10)
NB_OCCURENCES_EXTRA = np.zeros(10)
NB_OCCURENCES_TEST = np.zeros(10)

for element in TRAINSET:
    NB_OCCURENCES_TRAIN[int(element[1])] += 1

for element in EXTRASET:
    NB_OCCURENCES_EXTRA[int(element[1])] += 1

for element in TESTSET:
    NB_OCCURENCES_TEST[int(element[1])] += 1

# Show the proportion of each class in each data set
print("Proportion of each label in the train set")
print(NB_OCCURENCES_TRAIN / len(TRAINSET))
print("Proportion of each label in the extra set")
print(NB_OCCURENCES_EXTRA / len(EXTRASET))
print("Proportion of each label in the test set")
print(NB_OCCURENCES_TEST / len(TESTSET))
