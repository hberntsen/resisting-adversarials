## The Artificial Mind's Eye: *Resisting Adversarials for Convolutional Neural Networks using Internal Projection*

This repository contains the source code to train the network that was used in
the [paper](https://www.techfak.uni-bielefeld.de/~fschleif/mlr/mlr_04_2016.pdf).

Training the networks
---------------------
In order to train and use the networks, you need Caffe with the clipping layer
code. This can be added by merging https://github.com/BVLC/caffe/pull/3791 into
your local version of Caffe.

The 00_Blender folder contains the files to render the 3D models in blender to
hdf5 files. Within the blender file is a python script to generate the
datasets. Every model generates 50000 samples. Save the results as ``airplane_s1.h5``,
``suzanne_s1.h5`` and ``tux_s1.h5`` in the 00_Blender folder. Each blender file
has an embedded python script to generate those.

The 00_ImageNet folder contains scripts to convert the ImageNet dataset to
hdf5. The scripts resize the images to 64x64 pixels and converts them to
greyscale. You need to change the paths to the ImageNet dataset both here and
in the lib folder.

We can now blend the imagenet backgrounds into the blender files. This is done
by running run.sh in the 01_Alphablend folder. 

The 00_PrototxtDeploy folder contains a script to generate prototxt files for
Caffe. Simply run the run.sh script to generate them.

To generate the data to train the direct classification network, run the run.sh
file in the 02_ClassifierData folder. We now have the data to train the
direct classifiers in the 03_TrainClassifier folders..
We now also have enough data to start training stage 1 and 2 of the network.
Training of the network is done by starting caffe in each folder, e.g.:
```
caffe.bin train -solver solver.prototxt |& tee log-$(date +%s).txt
```
After the networks have been trained, stage 1 and 2 can be merged. This is done
by executing the copy weights script in 03_MergeStage1-2. Based on this network
we will generate data for stage 3 of the network. The `run.sh` script in the
`04_Stage3Data` folder will do this. After this network has been trained we can
create the triple-staged network using the merge script in `06_MergeStage1-3`.

Generating adversarial images
-----------------------------
The `collect_adv_results.py` script generates adversarial images wich will be
stored in a pickle file.
