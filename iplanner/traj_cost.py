# ======================================================================
# Copyright (c) 2023 Fan Yang
# Robotic Systems Lab, ETH Zurich
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ======================================================================

import torch
import pypose as pp
from tsdf_map import TSDF_Map
from traj_opt import TrajOpt
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)

class TrajCost:
    def __init__(self, gpu_id=0):
        self.tsdf_map = TSDF_Map(gpu_id)
        self.opt = TrajOpt()
        self.is_map = False
        return None

    def TransformPoints(self, odom, points):
        batch_size, num_p, _ = points.shape
        world_ps = pp.identity_SE3(batch_size, num_p, device=points.device, requires_grad=points.requires_grad)
        world_ps.tensor()[:, :, 0:3] = points
        world_ps = pp.SE3(odom[:, None, :]) @ pp.SE3(world_ps)
        return world_ps
    
    def SetMap(self, root_path, map_name):
        self.tsdf_map.ReadTSDFMap(root_path, map_name)
        self.is_map = True
        return

    def CostofTraj(self, waypoints, odom, goal, ahead_dist, alpha=0.5, beta=1.0, gamma=2.0, delta=5.0, obstalce_thred=0.5):
        batch_size, num_p, _ = waypoints.shape
        if self.is_map:
            world_ps = self.TransformPoints(odom, waypoints)
            norm_inds, _ = self.tsdf_map.Pos2Ind(world_ps)
            # Obstacle Cost
            cost_grid = self.tsdf_map.cost_array.T.expand(batch_size, 1, -1, -1)
            oloss_M = F.grid_sample(cost_grid, norm_inds[:, None, :, :], mode='bicubic', padding_mode='border', align_corners=False).squeeze(1).squeeze(1)
            oloss_M = oloss_M.to(torch.float32)
            oloss = torch.mean(torch.sum(oloss_M, axis=1))

            # Terrian Height loss
            height_grid = self.tsdf_map.ground_array.T.expand(batch_size, 1, -1, -1)
            hloss_M = F.grid_sample(height_grid, norm_inds[:, None, :, :], mode='bicubic', padding_mode='border', align_corners=False).squeeze(1).squeeze(1)
            hloss_M = torch.abs(waypoints[:, :, 2] - hloss_M)
            hloss = torch.mean(torch.sum(hloss_M, axis=1))

        # Goal Cost
        gloss = torch.norm(goal[:, :3] - waypoints[:, -1, :], dim=1)
        gloss = torch.mean(torch.log(gloss + 1.0))
        # gloss = torch.mean(gloss)
        
        # Motion Loss
        desired_wp = self.opt.TrajGeneratorFromPFreeRot(goal[:, None, 0:3], step=1.0/(num_p-1)) 
        desired_ds = torch.norm(desired_wp[:, 1:num_p, :] - desired_wp[:, 0:num_p-1, :], dim=2)
        wp_ds = torch.norm(waypoints[:, 1:num_p, :] - waypoints[:, 0:num_p-1, :], dim=2)
        mloss = torch.abs(desired_ds - wp_ds)
        mloss = torch.sum(mloss, axis=1)
        mloss = torch.mean(mloss)

        # Fear labels
        goal_dists = torch.cumsum(wp_ds, dim=1, dtype=wp_ds.dtype)
        floss_M = torch.clone(oloss_M)[:, 1:]
        floss_M[goal_dists > ahead_dist] = 0.0
        fear_labels = torch.max(floss_M, 1, keepdim=True)[0]
        fear_labels = (fear_labels > obstalce_thred).to(torch.float32)

        return alpha*oloss + beta*hloss + gamma*mloss + delta*gloss, fear_labels