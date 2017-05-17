# CycleGAN

Tensorflow implementation of CycleGAN, mostly modified from https://github.com/XHUJOY/CycleGAN-tensorflow to a simpler version

Paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkssee](https://arxiv.org/pdf/1703.10593.pdf) 
Author: [Jun-Yan Zhu ](https://people.eecs.berkeley.edu/~junyanz/) *et al.*

## exemplar results on testset
- gif: horse -> zebra
![](./pics/horse2zebra.gif)
- row 1: horse -> zebra -> reconstructed horse, row 2: zebra -> horse -> reconstructed zebra
![](./pics/example_horse2zebra_1.jpg)
- row 1: apple -> orange -> reconstructed apple, row 2: orange -> apple -> reconstructed orange
![](./pics/example_apple2orange_1.jpg)

# Prerequisites
- tensorflow r1.0
- python 2.7

# Usage
```
cd CycleGAN-Tensorflow-Simple-master
```

## Download Datasets
- Download the horse2zebra dataset:
```bash
sh ./download_dataset.sh horse2zebra
```
- Download the apple2orange dataset:
```bash
sh ./download_dataset.sh apple2orange
```
- See download_dataset.sh for more datasets

## Train Example
```bash
python train.py --dataset=horse2zebra --gpu_id=0
```

## Test Example
```bash
python test.py --dataset=horse2zebra
```

## Gif2Gif
```bash
python gif2gif.py --gif=./pics/horse.gif --save_path=./pics/horse2zebra.gif --dataset=horse2zebra --direction=a2b
```