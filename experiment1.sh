#!/bin/bash
#train1: data=100, load_size=286, crop_size=256
#CUDA_VISIBLE_DEVICES=1 python ./train2.py --dataset 'wsi_pair_patch' --datasets_dir '/mnt/share/gengxiaoqi/2cycleGAN/datasets' --trainA '1trainA'\
# --trainB '1trainB' --testA '0testA' --testB '0testB' --output_path '/mnt/share/gengxiaoqi/2cycleGAN/output_test2' --load_size 286 --crop_size 256

#train2: data=100, load_size=1024, crop_size=1024
CUDA_VISIBLE_DEVICES=7 python ./train.py --dataset 'wsi_pair_patch' --datasets_dir '/mnt/share/gengxiaoqi/2cycleGAN/datasets' --trainA '1trainA'\
 --trainB '1trainB' --testA '0testA' --testB '0testB' --output_path '/mnt/share/gengxiaoqi/2cycleGAN/output_test7' --load_size 512 --crop_size 512 

#train3: data=1000, load_size=512, crop_size=512
#CUDA_VISIBLE_DEVICES=1 python ./train2.py --dataset 'wsi_pair_patch' --datasets_dir '/mnt/share/gengxiaoqi/2cycleGAN/datasets' --trainA '0trainA'\
 #--trainB '0trainB' --testA '0testA' --testB '0testB' --output_path '/mnt/share/gengxiaoqi/2cycleGAN/output_test3' --load_size 512 --crop_size 512
