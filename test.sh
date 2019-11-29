#!/bin/bash
#test1:data=10,size=1024
CUDA_VISIBLE_DEVICES=4,5 python ./test.py --experiment_dir '/mnt/share/gengxiaoqi/2cycleGAN/output_test2/wsi_pair_patch' --samples_testing 'samples_testing3' --testA '4testA' --testB '4testB' --load_size 512 --crop_size 512
