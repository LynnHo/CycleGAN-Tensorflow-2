from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import shutil

import numpy as np
import torch


def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)


def cuda_devices(gpu_ids):
    gpu_ids = [str(i) for i in gpu_ids]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_ids)


def cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]


def save_checkpoint(state, save_path, is_best=False, max_keep=None):
    # save checkpoint
    torch.save(state, save_path)

    # deal with max_keep
    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, 'latest_checkpoint')

    save_path = os.path.basename(save_path)
    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [save_path + '\n'] + ckpt_list
    else:
        ckpt_list = [save_path + '\n']

    if max_keep is not None:
        for ckpt in ckpt_list[max_keep:]:
            ckpt = os.path.join(save_dir, ckpt[:-1])
            if os.path.exists(ckpt):
                os.remove(ckpt)
        ckpt_list[max_keep:] = []

    with open(list_path, 'w') as f:
        f.writelines(ckpt_list)

    # copy best
    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, 'best_model.ckpt'))


def load_checkpoint(ckpt_dir_or_file, map_location=None, load_best=False):
    if os.path.isdir(ckpt_dir_or_file):
        if load_best:
            ckpt_path = os.path.join(ckpt_dir_or_file, 'best_model.ckpt')
        else:
            with open(os.path.join(ckpt_dir_or_file, 'latest_checkpoint')) as f:
                ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
    else:
        ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


def reorganize(dataset_dir):
    dirs = {}
    dirs['trainA'] = os.path.join(dataset_dir, 'link_trainA')
    dirs['trainB'] = os.path.join(dataset_dir, 'link_trainB')
    dirs['testA'] = os.path.join(dataset_dir, 'link_testA')
    dirs['testB'] = os.path.join(dataset_dir, 'link_testB')
    mkdir(dirs.values())

    for key in dirs:
        try:
            os.remove(os.path.join(dirs[key], '0'))
        except:
            pass
        os.symlink(os.path.abspath(os.path.join(dataset_dir, key)),
                   os.path.join(dirs[key], '0'))

    return dirs


class ItemPool(object):

    def __init__(self, max_num=50):
        self.max_num = max_num
        self.num = 0
        self.items = []

    def __call__(self, in_items):
        """`in_items` is a list of item."""
        if self.max_num <= 0:
            return in_items
        return_items = []
        for in_item in in_items:
            if self.num < self.max_num:
                self.items.append(in_item)
                self.num = self.num + 1
                return_items.append(in_item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_num)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items
