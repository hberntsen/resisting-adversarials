import h5py
import caffe
import numpy as np
caffe.set_mode_gpu()

np.random.seed(1)

classifier_data = h5py.File('../02_ClassifierData/dataset_train.h5', 'r')
f = {'images': np.expand_dims(np.array(classifier_data['images'])[:,0,:], axis=1),
     'label': np.array(classifier_data['label'])}
in_len = classifier_data['images'].shape[0]
out_len = 10000 * 4

out_data = h5py.File('tmp/dataset_train.h5', 'w')
net = caffe.Net('../00_PrototxtDeploy/generated/stage2.prototxt', '../03_MergeStage1-2/merged.caffemodel', caffe.TEST)

negative_label_indexes = np.random.choice(in_len, out_len/2, replace=False)
print('nli_shape', negative_label_indexes.shape)
positive_label_indexes = np.random.choice(in_len/4*3, out_len/2, replace=False)
out_images = np.zeros((out_len, 3 ,64, 64),dtype='float')
out_label = np.ones((out_len ),dtype='uint8')
out_label[0:negative_label_indexes.shape[0]] = 0

samples_per_forward = 250
for i in range(negative_label_indexes.shape[0]/samples_per_forward):
    net.blobs['image'].data[...] = f['images'][negative_label_indexes][i*samples_per_forward: (i+1)*samples_per_forward]
    labels = f['label'][negative_label_indexes][i*samples_per_forward: (i+1)*samples_per_forward]
    net.forward()

    bms = ['100', '010', '001']
    for j in range(samples_per_forward):
        choices = np.array(range(3))
        choices = choices[choices != labels[j]]
        model_choice = np.random.choice(choices)
        out_images[i*samples_per_forward + j][0] = np.clip(net.blobs['image'].data.reshape(samples_per_forward,64,64)[j], 0, 255)
        out_images[i*samples_per_forward + j][1] = np.clip(net.blobs['fromparam_%s_final_image' % (bms[model_choice])].data.reshape(samples_per_forward,64,64)[j], 0, 255)
        out_images[i*samples_per_forward + j][2] = net.blobs['fromparam_%s_deconv8_segm' % (bms[model_choice])].data.reshape(samples_per_forward,64,64)[j]

offset = negative_label_indexes.shape[0]
for i in range(positive_label_indexes.shape[0]/samples_per_forward):
    net.blobs['image'].data[...] = f['images'][positive_label_indexes][i*samples_per_forward: (i+1)*samples_per_forward]
    labels = f['label'][positive_label_indexes][i*samples_per_forward: (i+1)*samples_per_forward]
    net.forward()

    bms = ['100', '010', '001']
    for j in range(samples_per_forward):
        model_choice = labels[j]
        out_images[offset + i*samples_per_forward + j][0] = np.clip(net.blobs['image'].data.reshape(samples_per_forward,64,64)[j], 0, 255)
        out_images[offset + i*samples_per_forward + j][1] = np.clip(net.blobs['fromparam_%s_final_image' % (bms[model_choice])].data.reshape(samples_per_forward,64,64)[j], 0, 255)
        out_images[offset + i*samples_per_forward + j][2] = net.blobs['fromparam_%s_deconv8_segm' % (bms[model_choice])].data.reshape(samples_per_forward,64,64)[j]

out_data.create_dataset('label', data=out_label, compression='gzip', dtype='uint8')
out_data.create_dataset('images', data=out_images, compression='gzip', dtype='uint8')

out_data.close()
