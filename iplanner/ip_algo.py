# ======================================================================
# Copyright (c) 2023 Fan Yang
# Robotic Systems Lab, ETH Zurich
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ======================================================================

import PIL
import math
import torch
import torchvision.transforms as transforms

from iplanner import traj_opt

class IPlannerAlgo:
    def __init__(self, args):
        super(IPlannerAlgo, self).__init__()
        self.config(args)

        self.depth_transform = transforms.Compose([
            transforms.Resize(tuple(self.crop_size)),
            transforms.ToTensor()])

        net, _ = torch.load(self.model_save, map_location=torch.device("cpu"))
        self.net = net.cuda() if torch.cuda.is_available() else net

        self.traj_generate = traj_opt.TrajOpt()
        return None

    def config(self, args):
        self.model_save = args.model_save
        self.crop_size  = args.crop_size
        self.sensor_offset_x = args.sensor_offset_x
        self.sensor_offset_y = args.sensor_offset_y
        self.is_traj_shift = False
        if math.hypot(self.sensor_offset_x, self.sensor_offset_y) > 1e-1:
            self.is_traj_shift = True
        return None


    def plan(self, image, goal_robot_frame):
        img = PIL.Image.fromarray(image)
        img = self.depth_transform(img).expand(1, 3, -1, -1)
        if torch.cuda.is_available():
            img = img.cuda()
            goal_robot_frame = goal_robot_frame.cuda()
        with torch.no_grad():
            keypoints, fear = self.net(img, goal_robot_frame)
        if self.is_traj_shift:
            batch_size, _, dims = keypoints.shape
            keypoints = torch.cat((torch.zeros(batch_size, 1, dims, device=keypoints.device, requires_grad=False), keypoints), axis=1)
            keypoints[..., 0] += self.sensor_offset_x
            keypoints[..., 1] += self.sensor_offset_y
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints , step=0.1)
        
        return keypoints, traj, fear, img
