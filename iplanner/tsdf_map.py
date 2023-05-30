# ======================================================================
# Copyright (c) 2023 Fan Yang
# Robotic Systems Lab, ETH Zurich
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ======================================================================

import open3d as o3d
import numpy as np
import torch
import os

torch.set_default_dtype(torch.float32)

class TSDF_Map:
    def __init__(self, gpu_id=0):
        self.map_init = False
        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(gpu_id))
        else:
            self.device = torch.device("cpu")
        self.pcd_tsdf = o3d.geometry.PointCloud()
        self.pcd_viz  = o3d.geometry.PointCloud()

    # def DirectLoadMap(self, tsdf_array, viz_points, start_xy, voxel_size, clear_dist):
    def DirectLoadMap(self, data, coord, params):
        self.voxel_size = params[0]
        self.clear_dist = params[1]
        self.start_x, self.start_y = coord
        self.tsdf_array = torch.tensor(data[0], device=self.device)
        self.num_x, self.num_y = self.tsdf_array.shape
        # visualization points
        self.viz_points = data[1]
        self.pcd_viz.points = o3d.utility.Vector3dVector(self.viz_points)
        self.ground_array = torch.tensor(data[2], device=self.device)
        # set cost map
        self.SetUpCostArray()
         # update pcd instance
        xv, yv = np.meshgrid(np.linspace(0, self.num_x * self.voxel_size, self.num_x), np.linspace(0, self.num_y * self.voxel_size, self.num_y), indexing="ij")
        T = np.concatenate((np.expand_dims(xv, axis=0), np.expand_dims(yv, axis=0)), axis=0)
        T = np.concatenate((T, np.expand_dims(self.cost_array.cpu(), axis=0)), axis=0)
        self.pcd_tsdf.points = o3d.utility.Vector3dVector(T.reshape(3, -1).T)
        
        self.map_init = True;

    def ShowTSDFMap(self, cost_map=True): # not run with cuda
        if not self.map_init:
            print("Error: cannot show map, map has not been init yet!")
            return;
        if cost_map:
            o3d.visualization.draw_geometries([self.pcd_tsdf])
        else:
            o3d.visualization.draw_geometries([self.pcd_viz])
        return

    def Pos2Ind(self, points):
        # points [torch shapes [num_p, 3]]
        start_xy = torch.tensor([self.start_x, self.start_y], dtype=torch.float64, device=points.device).expand(1, 1, -1)
        H = (points.tensor()[:, :, 0:2] - start_xy) / self.voxel_size
        # H = torch.round(H)
        mask = torch.logical_and((H > 0).all(axis=2), (H < torch.tensor([self.num_x, self.num_y], device=points.device)[None,None,:]).all(axis=2))
        return self.NormInds(H), H[mask, :]

    def NormInds(self, H):
        norm_matrix = torch.tensor([self.num_x/2.0, self.num_y/2.0], dtype=torch.float64, device=H.device)
        H = (H - norm_matrix) / norm_matrix
        return H

    def DeNormInds(self, NH):
        norm_matrix = torch.tensor([self.num_x/2.0, self.num_y/2.0], dtype=torch.float64, device=NH.device)
        NH = NH * norm_matrix + norm_matrix
        return NH

    def SaveTSDFMap(self, root_path, map_name):
        if not self.map_init:
            print("Error: map has not been init yet!")
            return;
        map_path    = os.path.join(*[root_path, "maps", "data",   map_name   + "_map.txt"])
        ground_path = os.path.join(*[root_path, "maps", "data",   map_name   + "_ground.txt"])
        params_path = os.path.join(*[root_path, "maps", "params", map_name + "_param.txt"])
        cloud_path  = os.path.join(*[root_path, "maps", "cloud",  map_name   + "_cloud.txt"])
        # save datas
        np.savetxt(map_path, self.tsdf_array.cpu())
        np.savetxt(ground_path, self.ground_array.cpu())
        np.savetxt(cloud_path, self.viz_points)
        params = [str(self.voxel_size), str(self.start_x), str(self.start_y), str(self.clear_dist)]
        with open(params_path, 'w') as f:
            for param in params:
                f.write(param)
                f.write('\n')
        print("TSDF Map saved.")
    
    def SetUpCostArray(self):
        self.cost_array = self.tsdf_array

    def ReadTSDFMap(self, root_path, map_name):
        map_path    = os.path.join(*[root_path, "maps", "data",   map_name   + "_map.txt"])
        ground_path = os.path.join(*[root_path, "maps", "data",   map_name   + "_ground.txt"])
        params_path = os.path.join(*[root_path, "maps", "params", map_name   + "_param.txt"])
        cloud_path  = os.path.join(*[root_path, "maps", "cloud",  map_name   + "_cloud.txt"])
        # open params file
        with open(params_path) as f:
            content = f.readlines()
        self.voxel_size = float(content[0])
        self.start_x    = float(content[1])
        self.start_y    = float(content[2])
        self.clear_dist = float(content[3])
        self.tsdf_array = torch.tensor(np.loadtxt(map_path), device=self.device)
        self.viz_points = np.loadtxt(cloud_path)
        self.ground_array = torch.tensor(np.loadtxt(ground_path), device=self.device)

        self.num_x, self.num_y = self.tsdf_array.shape
        # visualization points
        self.pcd_viz.points = o3d.utility.Vector3dVector(self.viz_points)
        # opne map array
        self.SetUpCostArray()
        # update pcd instance
        xv, yv = np.meshgrid(np.linspace(0, self.num_x * self.voxel_size, self.num_x), np.linspace(0, self.num_y * self.voxel_size, self.num_y), indexing="ij")
        T = np.concatenate((np.expand_dims(xv, axis=0), np.expand_dims(yv, axis=0)), axis=0)
        T = np.concatenate((T, np.expand_dims(self.cost_array.cpu().detach().numpy(), axis=0)), axis=0)
        wps = T.reshape(3, -1).T + np.array([self.start_x, self.start_y, 0.0])
        self.pcd_tsdf.points = o3d.utility.Vector3dVector(wps)

        self.map_init = True;
        return