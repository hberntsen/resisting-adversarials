import cv2
import json
import h5py
import sys

dataset = sys.argv[1] # train or val
width = 64
height = 64
index = open('imagenet_%s_index.json' % dataset, 'r')
files = json.load(index)

results = []

for f in files:
    print(f)
    img = cv2.imread(f, 0)
    resized = cv2.resize(img, (width, height))
    results.append(resized)

with h5py.File('%s.h5' % dataset, 'w') as f:
    f.create_dataset('images', data=results, dtype='uint8', compression='gzip')
