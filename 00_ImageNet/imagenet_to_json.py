import os
names = []
for root, dirnames, filenames in os.walk('/home/harm/Downloads/imagenet/ILSVRC2013_DET_val'):
    for f in filenames:
        names.append(os.path.join(root, f))
import json
a = json.dumps(names)
f = open ('imagenet_val_index.json', 'w')
f.write(a)
f.close()
