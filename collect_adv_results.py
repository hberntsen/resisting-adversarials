import caffe
import h5py
import numpy as np
from skimage.measure import compare_ssim as ssim
import pickle

object_names = ['Monkey', 'Penguin', 'Airplane','None']
samples_per_object = 10000
d = h5py.File('02_ClassifierData/dataset_val.h5','r')
d = {'images': np.array(d['images']),
     'label': np.array(d['label']),
     'label_split': np.array(d['label_split'])
     }
merged_net_none_thresholds = [0.2, 0.45, 0.7]
max_steps = 10000

caffe.set_mode_gpu()

#code should also work on the _p network
classifier_net = caffe.Net('03_TrainClassifier/deploy.prototxt', '03_TrainClassifier/classifier_iter_40000.caffemodel', caffe.TEST)
classifier_net.blobs['images'].reshape(1,1,64,64)
classifier_net.blobs['label'].reshape(1)

class GradientSign:
    def __init__(self, net, inputs, targets, input_key, target_key, towards_target=False):

        net.blobs[target_key].data[...] = targets
        net.blobs[input_key].data[...] = inputs
        net.forward()

        self.net = net
        self.input_key = input_key
        self.towards_target = towards_target

    def step(self, step_size=1.0):
        #we assume the net has had a forward with this data
        #self.net.forward()
        self.net.backward()
        g = np.sign(self.net.blobs[self.input_key].diff)

        if self.towards_target:
            g = -g
        # - g to move the input towards the target
        # + g to move the input away from the target
        self.net.blobs[self.input_key].data[...] = \
            np.clip(self.net.blobs[self.input_key].data + g * step_size,0,255)
        self.net.forward()


all_results = {}

print('classifier_tonone')
results=[]
test_indexes = np.hstack([np.arange(x, x+samples_per_object) for x in np.arange(0,len(d['images']) - samples_per_object,samples_per_object)])

for i in test_indexes:
    input_img = d['images'][i][0]
    original_class = d['label_split'][i]
    original_class_num = d['label'][i]
    target_class_num = 3 # none
    steps = 0

    gs = GradientSign(classifier_net, input_img, target_class_num, 'images', 'label', True)

    while (np.argmax(classifier_net.blobs['softmax'].data[0]) != target_class_num).all():
        gs.step()
        steps += 1
        if steps >= max_steps:
            break

    if i % 1000 == 0:
        print(i)

    adv_img = np.array(classifier_net.blobs['images'].data[0,0],dtype='uint8')
    results.append([
            steps,
            np.copy(classifier_net.blobs['softmax'].data[0]),
            adv_img,
            ssim(input_img, adv_img, dynamic_range=255)])
all_results['classifier_tonone'] = results

print('classifier_notnonetonotnone')
#class -> some other class that is not none
results=[]
test_indexes = np.hstack([np.arange(x, x+samples_per_object) for x in np.arange(0,len(d['images']) - samples_per_object,samples_per_object)])

for class_offset in [1,2]:
    for i in test_indexes:
        input_img = d['images'][i][0]
        original_class = d['label_split'][i]
        original_class_num = d['label'][i]
        target_class_num = (original_class_num + class_offset) % 3 # none
        steps = 0

        gs = GradientSign(classifier_net, input_img, target_class_num, 'images', 'label', True)

        while np.argmax(classifier_net.blobs['softmax'].data[0]) == original_class_num or np.argmax(classifier_net.blobs['softmax'].data[0]) == 3:
            gs.step()
            steps += 1
            if steps >= max_steps:
                break

        if i % 1000 == 0:
            print(i)

        adv_img = np.array(classifier_net.blobs['images'].data[0,0],dtype='uint8')
        results.append([
                steps,
                np.copy(classifier_net.blobs['softmax'].data[0]),
                adv_img,
                ssim(input_img, adv_img, dynamic_range=255)])


all_results['classifier_notnonetonotnone'] = results
del classifier_net

merged_net = caffe.Net('00_PrototxtDeploy/generated/adversarial.prototxt', '06_MergeStage1-3/merged.caffemodel', caffe.TEST)
merged_net.forward()

def class_num(net_output, none_threshold):
    if (net_output < none_threshold).all():
        #last index is none
        return 3
    else:
        return int(np.argmax(net_output))

test_indexes = np.hstack([np.arange(x, x+samples_per_object) for x in np.arange(0,len(d['images']) - samples_per_object,samples_per_object)])
for merged_net_none_threshold in merged_net_none_thresholds:
    results=[]
    for i in test_indexes:
        input_img = d['images'][i][0]
        original_class = d['label_split'][i]
        original_class_num = d['label'][i]
        steps = 0

        gs = GradientSign(merged_net, input_img, [0,0,0], 'image', 'label_split', True)

        while class_num(merged_net.blobs['concat_classifications'].data[0], merged_net_none_threshold) != 3:
            gs.step()
            steps += 1
            if steps >= max_steps:
                break

        if i % 100 == 0:
            print(i)

        adv_img = np.array(merged_net.blobs['image'].data[0,0],dtype='uint8')
        results.append([
                steps,
                np.copy(merged_net.blobs['concat_classifications'].data[0]),
                adv_img,
                ssim(input_img, adv_img, dynamic_range=255)
            ])
    all_results['merged_tonone_t%f' % merged_net_none_threshold] = results
    pickle.dump( all_results, open( "all_results_adversarial_1.p", "wb" ), -1 )

print('merged_net_notnonetonotnone')
test_indexes = np.hstack([np.arange(x, x+samples_per_object) for x in np.arange(0,len(d['images']) - samples_per_object,samples_per_object)])
for merged_net_none_threshold in merged_net_none_thresholds:
    print('threshold', merged_net_none_threshold)
    results=[]
    for class_offset in [1,2]:
        print('class_offset', class_offset)
        for i in test_indexes:
            input_img = d['images'][i][0]
            original_class = d['label_split'][i]
            original_class_num = d['label'][i]
            target_class_num = (original_class_num + class_offset) % 3
            steps = 0

            gs = GradientSign(merged_net, input_img, [target_class_num == 0, target_class_num == 1, target_class_num == 2], 'image', 'label', True)

            while class_num(merged_net.blobs['concat_classifications'].data[0], merged_net_none_threshold) == original_class_num or class_num(merged_net.blobs['concat_classifications'].data[0],merged_net_none_threshold) == 3:
                gs.step()
                steps += 1
                if steps >= max_steps:
                    break

            if i % 10 == 0:
                print(i)

            adv_img = np.array(merged_net.blobs['image'].data[0,0],dtype='uint8')
            results.append([
                    steps,
                    np.copy(merged_net.blobs['concat_classifications'].data[0]),
                    adv_img,
                    ssim(input_img, adv_img, dynamic_range=255)
                ])
            if i % 1000 == 0:
                print("Dumping")
                all_results['merged_notnonetonotnone_t%f' % merged_net_none_threshold] = results
                pickle.dump( all_results, open( "all_results_adversarial_4.p", "wb" ), -1 )

    all_results['merged_notnonetonotnone_t%f' % merged_net_none_threshold] = results
    pickle.dump( all_results, open( "all_results_adversarial_4.p", "wb" ), -1 )
