# CycleGAN

Tensorflow implementation of CycleGAN, mostly modified from https://github.com/XHUJOY/CycleGAN-tensorflow to a simpler version

Paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkssee](https://arxiv.org/pdf/1703.10593.pdf) 
Author: [Jun-Yan Zhu ](https://people.eecs.berkeley.edu/~junyanz/) *et al.*

## exemplar results on testset
- horse <-> zebra
![](./pics/example_horse2zebra_1.jpg)
- apple <-> orange
![](./pics/example_apple2orange_1.jpg)

# Prerequisites
- tensorflow r1.0

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