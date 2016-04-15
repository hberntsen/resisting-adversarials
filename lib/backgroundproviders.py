import numpy as np
import h5py
import random

class ImageNet:
    def __init__(self, dataset='train'):
        self.provided_indexes = []
        self.data = np.array(h5py.File('/home/harm/Downloads/imagenet/%s.h5' % dataset,'r')['images'], copy=True)

    def generator(self):
        while True:
            index = random.randint(0, self.data.shape[0] - 1)
            self.provided_indexes.append(index)
            yield self.data[index]
