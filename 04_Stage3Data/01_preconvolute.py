import h5py
import numpy as np

np.random.seed(1)

#must be whole number when calculating (imwidth-kernel_size)/stride + 1
#images are square
kernel_size = 10
stride = 6
mult = (64-kernel_size)/6.0 + 1
alpha_min = 0.01

f = h5py.File('tmp/dataset_train.h5', 'r')
f= {'images': np.array(f['images']),
    'label': np.array(f['label'])}

print('#label=1', f['label'].sum())
print('#label', f['label'].shape[0])

newimg = np.empty((f['images'].shape[0] * mult * mult, 3, kernel_size, kernel_size), dtype='uint8')
newlabel = np.empty((f['images'].shape[0] * mult * mult), dtype='uint8')

def convolute(image, kernel_size, stride):
    steps_x = (image.shape[0] - kernel_size)/stride + 1
    result = np.empty((steps_x * steps_x, kernel_size, kernel_size), dtype='uint8')
    x = 0
    y = 0
    i = 0
    while y <= image.shape[0] - kernel_size:
        while x <= image.shape[1] - kernel_size:
            result[i] = image[y:y+kernel_size, x:x+kernel_size]
            i+=1
            x+=stride
        x=0
        y+=stride
    assert i == steps_x * steps_x
    return result

for i in range(f['images'].shape[0]):
    #index to write the new convolutions to
    index = np.s_[i*mult*mult:(i+1)*mult*mult]
    #convolute each channel
    for j in range(3):
        newimg[index, j] = convolute(f['images'][i,j], kernel_size, stride)
    newlabel[index] = f['label'][i]

assert np.sum(newlabel) == newlabel.shape[0] / 2

print('newimg', newimg.shape)
print('newimg corrected', newimg[:,2,:].shape)
#sum alpha channel
sums = np.sum(newimg[:,2,:], axis=(1,2))
print(sums.shape)
select = sums > kernel_size*kernel_size * alpha_min

selected_newimg = newimg[select,:]
selected_newlabel = newlabel[select]

newimg = selected_newimg
newlabel = selected_newlabel

with h5py.File('dataset_train_preconvoluted.h5', 'w') as outf:
    outf.create_dataset('images', data=selected_newimg, compression='gzip')
    outf.create_dataset('label', data=selected_newlabel, compression='gzip', dtype='uint8')

