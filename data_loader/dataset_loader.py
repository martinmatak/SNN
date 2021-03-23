import sys

from force_data_reader import ForceDataReader

import numpy as np
from os import path, makedirs
import datetime
import scipy
import matplotlib as mpl
import shutil

DATA_FOLDER = "../dataset/"

def load_dataset():
    print '****Loading dataset, this will take a while...'
    train_pickle=DATA_FOLDER+'train'
    #validation_pickle=DATA_FOLDER+'eval'
    test_pickle=DATA_FOLDER+'test'

    #train=ForceDataReader(train_pickle)
    #valid=ForceDataReader(validation_pickle)
    test=ForceDataReader(test_pickle)
    #return train, valid, test

    return test

if __name__=='__main__':
    NR_SAMPLES = 10
    train = load_dataset()
    data = train.get_batch_data(0, NR_SAMPLES)
    electrodes_raw, electrodes_tared, force = data
    print(electrodes_raw.shape)
    print(electrodes_tared.shape)
    print(force.shape)

    for i in range(NR_SAMPLES):
        print("electrodes raw: ", electrodes_raw[i])
        print("electrodes tared: ", electrodes_tared[i])
        print("output: ", force[i])
