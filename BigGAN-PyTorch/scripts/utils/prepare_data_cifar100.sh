#!/bin/bash
python make_hdf5.py --dataset C100 --batch_size 50 #--dataset_root data
python calculate_inception_moments.py --dataset C100 #--dataset_root data
