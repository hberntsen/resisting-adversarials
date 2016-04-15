import h5py
import numpy as np
import os
from functools import reduce

DATA_WIDTH = 64;
DATA_HEIGHT = 64;
DATA_HXW=DATA_WIDTH*DATA_HEIGHT
DATA_PARAMS=3

def load(path):
    with h5py.File(path,'r') as f:
        alphas = np.array(f['alphas'], dtype='float32')
        alphas = alphas.reshape(alphas.shape[0], DATA_HXW)

        # params = params.reshape(params.shape[0],DATA_PARAMS)
        images = np.array(f['images'], dtype='uint8')
        if len(images.shape) != 4:
            images = images.reshape(images.shape[0],1,DATA_HEIGHT, DATA_WIDTH)

        if 'params' in f.keys():
            params = np.array(f['params'], dtype='float32')
            return {'alphas': alphas, 'params': params, 'images':images}
        else:
            label = np.array(f['label'], dtype='uint8')
            return {'alphas': alphas, 'images':images, 'label': label}

def write(filename, db, overwrite=False):
    if 'alphas' in db.keys() and len(db['alphas']) > 0:
        assert np.max(db['alphas']) < 1.1 and np.min(db['alphas']) >-0.1

    if not overwrite:
        if os.path.isfile(filename):
            filename = raw_input("Target file " + filename + " exists, enter alternate path:")

    with h5py.File(filename,'w') as f:
        if len(db['images'].shape) != 4:
            images = db['images'].reshape(db['images'].shape[0], 1, DATA_HEIGHT, DATA_WIDTH)
        else:
            images = db['images']
        f.create_dataset('images',data=images, dtype='uint8', compression='gzip')
        if len(db['alphas']) > 0:
            f.create_dataset('alphas',data=db['alphas'].reshape(db['alphas'].shape[0], DATA_HXW), compression='gzip')
        f.create_dataset('params',data=db['params'], compression='gzip')
        if 'background_indexes' in db.keys():
            f.create_dataset('background_indexes', data=db['background_indexes'], dtype='int', compression='gzip')
        if 'label' in db.keys():
            f.create_dataset('label', data=db['label'], dtype='uint8', compression='gzip')
        if 'label_split' in db.keys():
            f.create_dataset('label_split', data=db['label_split'], dtype='uint8', compression='gzip')

def empty_dataset(size=0):
    return {'alphas': np.empty((size,DATA_HXW)),
            'params': np.empty((size,DATA_PARAMS)),
            'label' : np.empty((size,3)),
            'images': np.empty((size,1,DATA_HEIGHT, DATA_WIDTH))}

def append2(d1,d2):
    r = {}
    print('appending datasets')
    for key in set(d1.keys()).intersection(d2.keys()):
        print ('key:', key, d1[key].shape, d2[key].shape)
        if(len(d1[key].shape) < 2):
            r[key] = np.append(d1[key], d2[key])
        else:
            r[key] = np.vstack((d1[key], d2[key]))
    return r

def append(loaded_datasets):
    return reduce(append2, loaded_datasets, empty_dataset())

def append_lowlevel(inputs, output):
    df = h5py.File(output, 'w')
    empty = empty_dataset()
    df.create_dataset('images',
            data=empty['images'],compression='gzip',dtype='uint8',
            maxshape=(None,1,DATA_HEIGHT,DATA_WIDTH))
    df.create_dataset('alphas',
            data=empty['alphas'],compression='gzip',dtype='float32',
            maxshape=(None,DATA_HXW))

    with h5py.File(inputs[0]) as i0:
        if 'label' in i0.keys():
            df.create_dataset('label',
                    data=empty['label'],compression='gzip',dtype='uint8',
                    maxshape=(None,3))
        elif 'params' in i0.keys():
            df.create_dataset('params',
                    data=empty['params'],compression='gzip',dtype='float32',
                    maxshape=(None,DATA_PARAMS))

    for key in df.keys():
        print(key)
        for iname in inputs:
            print('\t',iname)
            with h5py.File(iname,'r') as i:
                df[key].resize(df[key].shape[0] + i[key].shape[0],axis=0)
                df[key][-i[key].shape[0]:] = i[key]

def save_png(img, dest):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(DATA_WIDTH,DATA_HEIGHT))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    implot = plt.imshow(img.reshape(DATA_HEIGHT,DATA_WIDTH),interpolation='nearest',vmin=0,vmax=1, cmap='gist_gray')
    plt.axis('off')
    #plt.colorbar(implot)
    plt.savefig(dest)
    plt.close()

#s: see https://docs.scipy.org/doc/numpy/reference/generated/numpy.s_.html#numpy-s
def slice(dataset, s):
    r = {}
    for key in dataset.keys():
        r[key] = dataset[key][s]
    return r

