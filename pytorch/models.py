from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import torch.nn as nn


def conv_norm_act(in_dim, out_dim, kernel_size, stride, padding=0,
                  norm=nn.BatchNorm2d, relu=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        norm(out_dim),
        relu())


def dconv_norm_act(in_dim, out_dim, kernel_size, stride, padding=0,
                   output_padding=0, norm=nn.BatchNorm2d, relu=nn.ReLU):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride,
                           padding, output_padding, bias=False),
        norm(out_dim),
        relu())


class Discriminator(nn.Module):

    def __init__(self, dim=64):
        super(Discriminator, self).__init__()

        lrelu = functools.partial(nn.LeakyReLU, negative_slope=0.2)
        conv_bn_lrelu = functools.partial(conv_norm_act, relu=lrelu)

        self.ls = nn.Sequential(nn.Conv2d(3, dim, 4, 2, 1), nn.LeakyReLU(0.2),
                                conv_bn_lrelu(dim * 1, dim * 2, 4, 2, 1),
                                conv_bn_lrelu(dim * 2, dim * 4, 4, 2, 1),
                                conv_bn_lrelu(dim * 4, dim * 8, 4, 1, (1, 2)),
                                nn.Conv2d(dim * 8, 1, 4, 1, (2, 1)))

    def forward(self, x):
        return self.ls(x)


class ResiduleBlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(ResiduleBlock, self).__init__()

        conv_bn_relu = conv_norm_act

        self.ls = nn.Sequential(nn.ReflectionPad2d(1),
                                conv_bn_relu(in_dim, out_dim, 3, 1),
                                nn.ReflectionPad2d(1),
                                nn.Conv2d(out_dim, out_dim, 3, 1),
                                nn.BatchNorm2d(out_dim))

    def forward(self, x):
        return x + self.ls(x)


class Generator(nn.Module):

    def __init__(self, dim=64):
        super(Generator, self).__init__()

        conv_bn_relu = conv_norm_act
        dconv_bn_relu = dconv_norm_act

        self.ls = nn.Sequential(nn.ReflectionPad2d(3),
                                conv_bn_relu(3, dim * 1, 7, 1),
                                conv_bn_relu(dim * 1, dim * 2, 3, 2, 1),
                                conv_bn_relu(dim * 2, dim * 4, 3, 2, 1),
                                ResiduleBlock(dim * 4, dim * 4),
                                ResiduleBlock(dim * 4, dim * 4),
                                ResiduleBlock(dim * 4, dim * 4),
                                ResiduleBlock(dim * 4, dim * 4),
                                ResiduleBlock(dim * 4, dim * 4),
                                ResiduleBlock(dim * 4, dim * 4),
                                ResiduleBlock(dim * 4, dim * 4),
                                ResiduleBlock(dim * 4, dim * 4),
                                ResiduleBlock(dim * 4, dim * 4),
                                dconv_bn_relu(dim * 4, dim * 2, 3, 2, 1, 1),
                                dconv_bn_relu(dim * 2, dim * 1, 3, 2, 1, 1),
                                nn.ReflectionPad2d(3),
                                nn.Conv2d(dim, 3, 7, 1),
                                nn.Tanh())

    def forward(self, x):
        return self.ls(x)
