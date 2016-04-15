import numpy as np
import h5py
import sys
sys.path.append('../')
import lib.backgroundproviders as bgp
import random

def augment(db, numextra, data_type):
    newlabel = np.hstack((db['label'], np.full((numextra), 3, dtype='uint8')))
    del db['label']
    db.create_dataset('label', data=newlabel, compression='gzip', dtype='uint8')

    extra = np.full((numextra, 4), [0,0,0,1], dtype='uint8')
    newsplit = np.hstack((db['label_split'],np.zeros((db['label_split'].shape[0],1))))
    newsplit = np.vstack((newsplit, extra))
    del db['label_split']
    db.create_dataset('label_split', data=newsplit, compression='gzip', dtype='uint8')

    newparams = np.vstack((db['params'], np.full((numextra, 3), [0,0,0], dtype='float')))
    del db['params']
    db.create_dataset('params', data=newparams, compression='gzip')

    label_is_not_none = np.reshape(1*(newlabel != 3), (-1, 1))
    print(label_is_not_none.shape)
    db.create_dataset('label_is_not_none', data=label_is_not_none, dtype='uint8', compression='gzip')

    bg = bgp.ImageNet(data_type)
    images = np.array(list([x for _, x in zip(range(numextra), bg.generator())])).reshape(numextra,1,64,64)
    newimages = np.vstack((db['images'], images))
    del db['images']
    db.create_dataset('images', data=newimages, compression='gzip')

    assert len(db['background_indexes']) > 0
    newindexes = np.hstack((db['background_indexes'], bg.provided_indexes))
    assert len(newindexes) > 0
    del db['background_indexes']
    db.create_dataset('background_indexes', data=newindexes, compression='gzip')

random.seed(1234)
augment(h5py.File('./dataset_val.h5','a'), 10000, 'val')
augment(h5py.File('./dataset_train.h5','a'), 40000, 'train')
