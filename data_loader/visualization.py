import sys

from force_data_reader import ForceDataReader

import numpy as np
from os import path, makedirs
import datetime
import scipy
from matplotlib import pyplot as plt
from statistics import median

DATA_FOLDER = "../dataset/"

def load_dataset():
    print("****Loading dataset, this will take a while...")
    train_pickle=DATA_FOLDER+'train'
    validation_pickle=DATA_FOLDER+'eval'
    test_pickle=DATA_FOLDER+'test'

    train=ForceDataReader(train_pickle)
    valid=ForceDataReader(validation_pickle)
    test=ForceDataReader(test_pickle)
    return train, valid, test

if __name__=='__main__':
    NR_SAMPLES = 10
    train, valid, test = load_dataset()
    train_forces = train.get_full_data()[2]
    valid_forces = valid.get_full_data()[2]
    test_forces = test.get_full_data()[2]
    forces = np.concatenate((train_forces,valid_forces,test_forces))
    l2forces = [np.linalg.norm(force) for force in forces]

    l2forces.sort()
    median = median(l2forces)
    print("median: ", median) # ~3.34
    print("total number of samples: ", len(l2forces))

    plt.figure()
    plt.eventplot(l2forces)
    plt.show()

