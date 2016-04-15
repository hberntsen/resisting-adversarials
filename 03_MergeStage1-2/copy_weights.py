import caffe
caffe.set_mode_cpu()

net1 = caffe.Net('../02_TrainStage1/deploy.prototxt', '../02_TrainStage1/stage1_iter_40000.caffemodel', caffe.TEST)
net2 = caffe.Net('../02_TrainStage2/deploy.prototxt', '../02_TrainStage2/stage2_iter_500000.caffemodel', caffe.TEST)
merged = caffe.Net('../00_PrototxtDeploy/generated/stage2.prototxt', caffe.TEST)

def copy_params(target, prefix, source):
    for k in source.params.keys():
        for i in range(len(source.params[k])):
            key = "%s%s" % (prefix, k)
            if key in target.params.keys():
                target.params[key][i] = source.params[k][i]
            else:
                print("Key %s not found in target" % key)

copy_params(merged, 'toparam_100_', net1)
copy_params(merged, 'toparam_010_', net1)
copy_params(merged, 'toparam_001_', net1)
copy_params(merged, 'fromparam_100_', net2)
copy_params(merged, 'fromparam_010_', net2)
copy_params(merged, 'fromparam_001_', net2)

merged.save('merged.caffemodel')

