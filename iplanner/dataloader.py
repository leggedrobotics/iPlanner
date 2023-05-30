# ======================================================================
# Copyright (c) 2023 Fan Yang
# Robotic Systems Lab, ETH Zurich
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ======================================================================

import os
import PIL
import torch
import numpy as np
import pypose as pp
from PIL import Image
from pathlib import Path
from random import sample
from operator import itemgetter
from torch.utils.data import Dataset, DataLoader

torch.set_default_dtype(torch.float32)


class MultiEpochsDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class PlannerData(Dataset):
    def __init__(self, root, max_episode, goal_step, train, ratio=0.9, max_depth=10.0, sensorOffsetX=0.0, transform=None, is_robot=True):
        super().__init__()
        self.transform = transform
        self.is_robot = is_robot
        self.max_depth = max_depth
        img_filename_list = []
        img_path = os.path.join(root, "depth")
        img_filename_list = [str(s) for s in Path(img_path).rglob('*.png')]
        img_filename_list.sort(key = lambda x : int(x.split("/")[-1][:-4]))

        odom_path = os.path.join(root, "odom_ground_truth.txt")
        odom_list = []
        offset_T = pp.identity_SE3()
        offset_T.tensor()[0] = sensorOffsetX
        with open(odom_path) as f:
            lines = f.readlines()
            for line in lines:
                odom = np.fromstring(line[1:-2], dtype=np.float32, sep=', ')
                odom = pp.SE3(odom)
                if is_robot:
                    odom = odom @ offset_T
                odom_list.append(odom)

        N = len(odom_list)

        self.img_filename = []
        self.odom_list    = []
        self.goal_list    = []

        for ahead in range(1, max_episode+1, goal_step):
            for i in range(N):
                odom = odom_list[i]
                goal = odom_list[min(i+ahead, N-1)]
                goal = (pp.Inv(odom) @ goal)
                # gp = goal.tensor()
                # if (gp[0] > 1.0 and gp[1]/gp[0] < 1.2 and gp[1]/gp[0] > -1.2 and torch.norm(gp[:3]) > 1.0):
                self.img_filename.append(img_filename_list[i])
                self.odom_list.append(odom.tensor())
                self.goal_list.append(goal.tensor())

        N = len(self.odom_list)

        indexfile = os.path.join(img_path, 'split.pt')
        is_generate_split = True;
        if os.path.exists(indexfile):
            train_index, test_index = torch.load(indexfile)
            if len(train_index)+len(test_index) == N:
                is_generate_split = False
            else:
                print("Data changed! Generate a new split file")
        if (is_generate_split):
            indices = range(N)
            train_index = sample(indices, int(ratio*N))
            test_index = np.delete(indices, train_index)
            torch.save((train_index, test_index), indexfile)

        if train == True:
            self.img_filename = itemgetter(*train_index)(self.img_filename)
            self.odom_list    = itemgetter(*train_index)(self.odom_list)
            self.goal_list    = itemgetter(*train_index)(self.goal_list)
        else:
            self.img_filename = itemgetter(*test_index)(self.img_filename)
            self.odom_list    = itemgetter(*test_index)(self.odom_list)
            self.goal_list    = itemgetter(*test_index)(self.goal_list)

        assert len(self.odom_list) == len(self.img_filename), "odom numbers should match with image numbers"
        

    def __len__(self):
        return len(self.img_filename)

    def __getitem__(self, idx):
        image = Image.open(self.img_filename[idx])
        if self.is_robot:
            image = np.array(image.transpose(PIL.Image.ROTATE_180))
        else:
            image = np.array(image)
        image[~np.isfinite(image)] = 0.0
        image = (image / 1000.0).astype("float32")
        image[image > self.max_depth] = 0.0
        # DEBUG show image
        # img = Image.fromarray((image * 255 / np.max(image)).astype('uint8'))
        # img.show()
        image = Image.fromarray(image)
        image = self.transform(image).expand(3, -1, -1)

        return image, self.odom_list[idx], self.goal_list[idx]