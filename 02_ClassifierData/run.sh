#!/bin/sh
cp ../01_Alphablend/dataset_train.h5 ../01_Alphablend/dataset_val.h5 .
python augment_nomodel_class.py
echo "`pwd`/dataset_train.h5" > dataset_train.txt
