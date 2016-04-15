#!/bin/sh
set -e
mkdir -p tmp
python 00_generate_trainset.py
python 01_preconvolute.py
echo "`pwd`/dataset_train_preconvoluted.h5" > dataset_train_preconvoluted.txt
