#!/bin/bash
source ~/.bashrc
source /opt/conda/bin/activate torch
rm -rf ./metrics/*
echo "first para is $1"
nsys profile --stats=true --force-overwrite=true -o ./metrics/$1 -t cuda,nvtx,cublas,cudnn --cuda-memory-usage=true  python train.py --data $1
nsys stats --report gputrace --format csv,column --output ./metrics/$1,- ./metrics/$1.nsys-rep
cat ./metrics/$1_gputrace.csv | grep -E 'Name|ncclKernel_AllReduce_RING_LL_Sum_float' > ./metrics/$1_results.csv
