# <p align="center"> CycleGAN - Tensorflow 2 </p>

Tensorflow 2 implementation of CycleGAN.

Paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)

Author: [Jun-Yan Zhu ](https://people.eecs.berkeley.edu/~junyanz/) *et al.*

## Exemplar results

### summer2winter

row 1: summer -> winter -> reconstructed summer, row 2: winter -> summer -> reconstructed winter

<p align="center"> <img src="./pics/summer2winter.jpg" width="100%" /> </p>

### horse2zebra

row 1: horse -> zebra -> reconstructed horse, row 2: zebra -> horse -> reconstructed zebra

<p align="center"> <img src="./pics/horse2zebra.jpg" width="100%" /> </p>

### apple2orange

row 1: apple -> orange -> reconstructed apple, row 2: orange -> apple -> reconstructed orange

<p align="center"> <img src="./pics/apple2orange.jpg" width="100%" /> </p>

# Usage

- Prerequisites

    - Tensorflow 2.0 Alpha `pip install tensorflow-gpu==2.0.0-alpha0`
    - Tensorflow Addons `pip install tensorflow-addons`
    - (if you meet "tf.summary.histogram fails with TypeError" `pip install --upgrade tb-nightly`)
    - scikit-image, oyaml, tqdm
    - Python 3.6

- Dataset   
cycle-GAN need H&E and the IHC domains training set to train,so we set trainA is ki67 data, trainB is HE data, and sent them into network training at the same time.

| Date | wsi ID |Number of patches  |remark  |
| --- | --- | --- | --- |
| training_set |Total 19 wsi:'747672-1','721179-6','751086-3','745195-2','744059-3','749908-3','745514-2','748125-6','746836-3','746325-1','745262-3','750584-3','746074-2','749309-3','745586-4','745586-4','743947-2','742584-2','748259-3','724697-21'   |trainA_set(ki67 patches): 59738 trainB_set(HE patches): 55821 |no  |
|test_set  |Total 1 wsi:'749320-4'|testA_set(ki67 patches): 1539 testB_set(HE patches): 1873  | no |
    
- Example of training

    ```console
    CUDA_VISIBLE_DEVICES=7 python ./train.py --dataset 'wsi_pair_patch' --datasets_dir '/mnt/share/gengxiaoqi/2cycleGAN/datasets' --trainA '1trainA' --trainB '1trainB' --testA '0testA' --testB '0testB' --output_path '/mnt/share/gengxiaoqi/2cycleGAN/output_test7' --load_size 512 --crop_size 512
    ```

    - tensorboard for loss visualization

        ```console
        tensorboard --logdir ./output/summer2winter_yosemite/summaries --port 6006
        ```

- Example of testing

    ```console
    CUDA_VISIBLE_DEVICES=4,5 python ./test.py --experiment_dir '/mnt/share/gengxiaoqi/2cycleGAN/output_test2/wsi_pair_patch' --samples_testing 'samples_testing3' --testA '4testA' --testB '4testB' --load_size 512 --crop_size 512
    ```
