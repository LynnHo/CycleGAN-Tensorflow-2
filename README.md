***Recommendation***

- Our GAN based work for facial attribute editing - [AttGAN](https://github.com/LynnHo/AttGAN-Tensorflow).

***New***

- We re-implement CycleGAN by **TensorFlow 2.0 Alpha**! The old versions are here: [v1](https://github.com/LynnHo/CycleGAN-Tensorflow-PyTorch/tree/v1), [v0](https://github.com/LynnHo/CycleGAN-Tensorflow-PyTorch/tree/v0).

---

# <p align="center"> CycleGAN </p>

Tensorflow 2 implementation of CycleGAN.

Paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)

Author: [Jun-Yan Zhu ](https://people.eecs.berkeley.edu/~junyanz/) *et al.*

## Exemplar results

- gif: horse -> zebra

<p align="center"> <img src="./pics/horse2zebra.gif" /> </p>

- row 1: horse -> zebra -> reconstructed horse, row 2: zebra -> horse -> reconstructed zebra

<p align="center"> <img src="./pics/example_horse2zebra_1.jpg" /> </p>

- row 1: apple -> orange -> reconstructed apple, row 2: orange -> apple -> reconstructed orange

<p align="center"> <img src="./pics/example_apple2orange_1.jpg" /> </p>

# Usage

- Prerequisites
    - TensorFlow 2.0 Alpha
    - Python 3.6

- Dataset

    - Download the horse2zebra dataset:
    ```bash
    sh ./download_dataset.sh horse2zebra
    ```
    - Download the apple2orange dataset:
    ```bash
    sh ./download_dataset.sh apple2orange
    ```
    - See download_dataset.sh for more datasets

- Example of training
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=horse2zebra
```

- Example of training
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --dataset=horse2zebra
```
