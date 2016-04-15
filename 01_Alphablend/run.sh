#!/bin/sh
python alphablend_dataset.py dataset_train.h5 dataset_val.h5 dataset_train_black.h5 dataset_val_black.h5  
echo "`pwd`/dataset_train.h5" > dataset_train.txt
echo "`pwd`/dataset_train_black.h5" > dataset_train_black.txt
