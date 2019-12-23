# <p align="center"> CycleGAN - Tensorflow 2 </p>

Tensorflow 2 implementation of CycleGAN.

Paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)

Author: [Jun-Yan Zhu ](https://people.eecs.berkeley.edu/~junyanz/) *et al.*


# Usage

- Prerequisites

    - Tensorflow 2.0  `pip install tensorflow-gpu==2.0.0`
    - Tensorflow Addons `pip install tensorflow-addons`
    - (if you meet "tf.summary.histogram fails with TypeError" `pip install --upgrade tb-nightly`)
    - scikit-image, oyaml, tqdm
    - Python 3.6

- Dataset   
    cycleGAN need unpaired H&E and KI67 domains data to train,so we set trainA is KI67 data, trainB is HE data, and sent them into network at the same time. The quantities of trainA and trainB are not required to be the same.
    
- Example of training

    ```console
    CUDA_VISIBLE_DEVICES=7 python ./train.py --dataset 'wsi_pair_patch' --datasets_dir '/mnt/share/gengxiaoqi/2cycleGAN/datasets' --trainA '1trainA' --trainB '1trainB' --testA '0testA' --testB '0testB' --output_path '/mnt/share/gengxiaoqi/2cycleGAN/output_test7' --load_size 512 --crop_size 512
    ```

    - tensorboard for loss visualization

        ```console
        tensorboard --logdir /mnt/share/gengxiaoqi/2cycleGAN/output_test7/wsi_pair_patch/summaries --port 6006
        ```

- Example of testing

    ```console
    CUDA_VISIBLE_DEVICES=4,5 python ./test.py --experiment_dir '/mnt/share/gengxiaoqi/2cycleGAN/output_test7/wsi_pair_patch' --samples_testing 'samples_testing' --testA '4testA' --testB '4testB' --load_size 512 --crop_size 512
    ```
