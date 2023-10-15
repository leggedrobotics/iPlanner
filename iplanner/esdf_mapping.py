# ======================================================================
# Copyright (c) 2023 Fan Yang
# Robotic Systems Lab, ETH Zurich
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ======================================================================

import os
import cv2
import math
import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation as R


class CloudUtils:
    @staticmethod
    def create_open3d_cloud(points, voxel_size):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(voxel_size)
        return pcd

    @staticmethod
    def extract_cloud_from_image(P_matrix, im, T, min_dist=0.2, max_dist=50, scale=1000.0):
        p_inv = np.linalg.inv(P_matrix)
        im = im / scale
        im[im<min_dist] = 1e-3
        im[im>max_dist] = 1e-3

        T_z = np.concatenate((T, np.expand_dims(1.0/im, axis=0)), axis=0).reshape(4, -1)
        P = np.multiply(im.reshape(1, -1), p_inv.dot(T_z)).T[:,:3]
        return P

class CameraUtils:
    @staticmethod
    def compute_pixel_tensor(x_nums, y_nums):
        T = np.zeros([3, x_nums, y_nums])
        for u in range(x_nums):
            for v in range(y_nums):
                T[:, u, v] = np.array([u, v, 1.0])
        return T

    @staticmethod
    def compute_e_matrix(odom, is_flat_ground, cameraR, cameraT):
        Rc = R.from_quat(odom[3:])
        if is_flat_ground:
            euler = Rc.as_euler('xyz', degrees=False)
            euler[1] = 0.0
            Rc = R.from_euler("xyz", euler, degrees=False)
        Rc = Rc * cameraR
        C = (odom[:3] + cameraT).reshape(-1,1)
        Rc_t = Rc.as_matrix().T
        E = np.concatenate((Rc_t, -Rc_t.dot(C)), axis=1)
        E = np.concatenate((E, np.array([0.0, 0.0, 0.0, 1.0]).reshape(1,-1)), axis=0)
        return E

class DataUtils:
    @staticmethod
    def read_odom_list(odom_path):
        odom_list = []
        avg_height = 0.0
        with open(odom_path) as f:
            lines = f.readlines()
            for line in lines:
                odom = np.fromstring(line[1:-1], dtype=float, sep=', ')
                avg_height = avg_height + odom[2]
                odom_list.append(list(odom))
            avg_height =  avg_height / len(lines)
        return odom_list, avg_height

    @staticmethod
    def read_intrinsic(intrinsic_path):
        with open(intrinsic_path) as f:
            lines = f.readlines()
            elems = np.fromstring(lines[0][1:-1], dtype=float, sep=', ')
        if len(elems) == 12:
            P = np.array(elems).reshape(3, 4)
            K = np.concatenate((P, np.array([0.0, 0.0, 0.0, 1.0]).reshape(1,-1)), axis=0)
        else:
            K = np.array(elems).reshape(4, 4)
        return K

    @staticmethod
    def read_extrinsic(extrinsic_path):
        with open(extrinsic_path) as f:
            lines = f.readlines()
            elems = np.fromstring(lines[0][1:-1], dtype=float, sep=', ')
        CR = R.from_quat(elems[:4])
        CT = np.array(elems[4:])
        return CR, CT

    @staticmethod
    def load_point_cloud(path):
        return o3d.io.read_point_cloud(path)

    @staticmethod
    def prepare_output_folders(out_path, image_type):
        depth_im_path = os.path.join(out_path, image_type)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            os.makedirs(depth_im_path)
            os.makedirs(os.path.join(out_path, "maps", "cloud"))
            os.makedirs(os.path.join(out_path, "maps", "data"))
            os.makedirs(os.path.join(out_path, "maps", "params"))
        elif os.path.exists(depth_im_path):  # remove existing files
            for efile in os.listdir(depth_im_path):
                os.remove(os.path.join(depth_im_path, efile))
        return None
    
    @staticmethod
    def load_images(start_id, end_id, root_path, image_type):
        im_arr_list = []
        im_path = os.path.join(root_path, image_type)
        for idx in range(start_id, end_id):
            path = os.path.join(im_path, str(idx) + ".png")
            im = cv2.imread(path, cv2.IMREAD_ANYDEPTH).T
            im_arr_list.append(im)
        print("total number of images for reconstruction: {}".format(len(im_arr_list)))
        return im_arr_list

    @staticmethod
    def save_images(out_path, im_arr_list, image_type, is_transpose=True):
        for idx, img in enumerate(im_arr_list):
            if is_transpose:
                img = img.T
            cv2.imwrite(os.path.join(out_path, image_type, f"{idx}.png"), img)
        return None

    @staticmethod
    def save_odom_list(out_path, odom_list, start_id, num_images):
        with open(os.path.join(out_path, "odom_ground_truth.txt"), 'w') as f:
            for i in range(start_id, start_id + num_images):
                f.write(str(odom_list[i]) + "\n")
        return None

    @staticmethod
    def save_extrinsic(out_path, cameraR, cameraT):
        with open(os.path.join(out_path, "camera_extrinsic.txt"), 'w') as f:
            f.write(str(list(cameraR.as_quat()) + list(cameraT)) + "\n")
        return None

    @staticmethod
    def save_intrinsic(out_path, K):
        with open(os.path.join(out_path, "depth_intrinsic.txt"), 'w') as f:
            f.write(str(K.flatten().tolist()) + "\n")
        return None

    @staticmethod
    def save_point_cloud(out_path, pcd):
        o3d.io.write_point_cloud(os.path.join(out_path, "cloud.ply"), pcd)  # save point cloud
        return None

class TSDF_Creator:
    def __init__(self, input_path, voxel_size, robot_height, robot_size, clear_dist=1.0):
        self.initialize_path_and_properties(input_path, voxel_size, robot_height, robot_size, clear_dist)
        self.initialize_point_clouds()

    def initialize_path_and_properties(self, input_path, voxel_size, robot_height, robot_size, clear_dist):
        self.input_path = input_path
        self.is_map_ready = False
        self.clear_dist = clear_dist
        self.voxel_size = voxel_size
        self.robot_height = robot_height
        self.robot_size = robot_size

    def initialize_point_clouds(self):
        self.obs_pcd = o3d.geometry.PointCloud()
        self.free_pcd = o3d.geometry.PointCloud()
        
    def update_point_cloud(self, P_obs, P_free, is_downsample=False):
        self.obs_pcd.points  = o3d.utility.Vector3dVector(P_obs)
        self.free_pcd.points = o3d.utility.Vector3dVector(P_free)
        self.downsample_point_cloud(is_downsample)
        self.obs_points   = np.asarray(self.obs_pcd.points)
        self.free_points  = np.asarray(self.free_pcd.points)
        
    def read_point_from_file(self, file_name, is_filter=True):
        file_path = os.path.join(self.input_path, file_name)
        pcd_load = DataUtils.load_point_cloud(file_path)
        
        print("Running terrain analysis...")
        obs_p, free_p = self.terrain_analysis(np.asarray(pcd_load.points))
        self.update_point_cloud(obs_p, free_p, is_downsample=True)
        
        if is_filter:
            obs_p = self.filter_cloud(self.obs_points, num_nbs=50, std_ratio=2.0)
            self.update_point_cloud(obs_p, free_p)
        
        self.update_map_params()
        
    def downsample_point_cloud(self, is_downsample):
        if is_downsample:
            self.obs_pcd  = self.obs_pcd.voxel_down_sample(self.voxel_size)
            self.free_pcd = self.free_pcd.voxel_down_sample(self.voxel_size * 0.85)
            
    def update_map_params(self):
        self._handle_no_points()
        self._set_map_limits_and_start_coordinates()
        self._log_map_initialization()
        self.is_map_ready = True
        
    def terrain_analysis(self, input_points, ground_height=0.25):
        obs_points, free_points = self._initialize_point_arrays(input_points)
        obs_idx = free_idx = 0
        
        for p in input_points:
            p_height = p[2]
            if self._is_obstacle(p_height, ground_height):
                obs_points[obs_idx, :] = p
                obs_idx += 1
            elif self._is_free_space(p_height, ground_height):
                free_points[free_idx, :] = p
                free_idx += 1

        return obs_points[:obs_idx, :], free_points[:free_idx, :]
    
    def create_TSDF_map(self, sigma_smooth=2.5):
        if not self.is_map_ready:
            print("create tsdf map fails, no points received.")
            return

        free_map = np.ones([self.num_x, self.num_y])
        obs_map = self._create_obstacle_map()

        # create free place map
        free_I = self._index_array_of_points(self.free_points)
        free_map = self._create_free_space_map(free_I, free_map, sigma_smooth)

        free_map[obs_map > 0.3] = 1.0 # re-assign obstacles if they are in free space
        print("occupancy map generation completed.")

        # Distance Transform
        tsdf_array = self._distance_transform_and_smooth(free_map, sigma_smooth)

        viz_points = np.concatenate((self.obs_points, self.free_points), axis=0)
        # TODO: Use true terrain analysis module
        ground_array = np.ones([self.num_x, self.num_y]) * 0.0

        return [tsdf_array, viz_points, ground_array], [self.start_x, self.start_y], [self.voxel_size, self.clear_dist]
    
    def filter_cloud(self, points, num_nbs=100, std_ratio=1.0):
        pcd = self._convert_to_point_cloud(points)
        filtered_pcd = self._remove_statistical_outliers(pcd, num_nbs, std_ratio)
        return np.asarray(filtered_pcd.points)
    
    def visualize_cloud(self, pcd):
        o3d.visualization.draw_geometries([pcd])
    
    def _convert_to_point_cloud(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def _remove_statistical_outliers(self, pcd, num_nbs, std_ratio):
        filtered_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=num_nbs, std_ratio=std_ratio)
        return filtered_pcd
    
    def _create_obstacle_map(self):
        obs_map = np.zeros([self.num_x, self.num_y])
        obs_I = self._index_array_of_points(self.obs_points)
        for i in obs_I:
            obs_map[i[0], i[1]] = 1.0
        obs_map = gaussian_filter(obs_map, sigma=self.robot_size / self.voxel_size)
        obs_map /= np.max(obs_map + 1e-5) # normalize
        return obs_map
    
    def _create_free_space_map(self, free_I, free_map, sigma_smooth):
        for i in free_I:
            if i[0] >= 0 and i[0] < self.num_x and i[1] >= 0 and i[1] < self.num_y:
                free_map[i[0], i[1]] = 0
        free_map = gaussian_filter(free_map, sigma=sigma_smooth)
        free_map /= np.max(free_map) # normalize
        free_map[free_map < 0.7] = 0 # 0.683% is the probability of standard normal distribution
        return free_map
    
    def _distance_transform_and_smooth(self, free_map, sigma_smooth, is_log=True):
        dt_map = ndimage.distance_transform_edt(free_map)
        tsdf_array = gaussian_filter(dt_map, sigma=sigma_smooth)
        if is_log:
            tsdf_array = np.log(tsdf_array + 1.00001)
        return tsdf_array
    
    def _index_array_of_points(self, points):
        I = np.round((points[:, :2] - np.array([self.start_x, self.start_y])) / self.voxel_size).astype(int)
        return I
    
    def _initialize_point_arrays(self, input_points):
        return np.zeros(input_points.shape), np.zeros(input_points.shape)

    def _is_obstacle(self, p_height, ground_height):
        return (p_height > ground_height) and (p_height < self.robot_height * 1.5)

    def _is_free_space(self, p_height, ground_height):
        return p_height < ground_height and p_height > - ground_height
        
    def _handle_no_points(self):
        if (self.obs_points.shape[0] == 0):
            print("No points received.")
            return
        
    def _set_map_limits_and_start_coordinates(self):
        max_x, max_y, _ = np.amax(self.obs_points, axis=0) + self.clear_dist
        min_x, min_y, _ = np.amin(self.obs_points, axis=0) - self.clear_dist
        self.num_x = np.ceil((max_x - min_x) / self.voxel_size / 10).astype(int) * 10
        self.num_y = np.ceil((max_y - min_y) / self.voxel_size / 10).astype(int) * 10
        self.start_x = (max_x + min_x) / 2.0 - self.num_x / 2.0 * self.voxel_size
        self.start_y = (max_y + min_y) / 2.0 - self.num_y / 2.0 * self.voxel_size

    def _log_map_initialization(self):
        print("tsdf map initialized, with size: %d, %d" %(self.num_x, self.num_y))
        

class DepthReconstruction:
    def __init__(self, input_path, out_path, start_id, iters, voxel_size, max_range, is_max_iter=True):
        self._initialize_paths(input_path, out_path)
        self._initialize_parameters(voxel_size, max_range, is_max_iter)
        self._read_camera_params()
        
        # odom list read
        self.odom_list, self._avg_height = DataUtils.read_odom_list(self.input_path + "/odom_ground_truth.txt")
        
        N = len(self.odom_list)
        self.start_id = 0 if self.is_max_iter else start_id
        self.end_id = N if self.is_max_iter else min(start_id + iters, N)
        
        self.is_constructed = False
        print("Ready to read depth data.")

    # public methods
    def depth_map_reconstruction(self, is_output=False, is_flat_ground=False):
        self.im_arr_list = DataUtils.load_images(self.start_id, self.end_id, self.input_path, "depth")

        x_nums, y_nums = self.im_arr_list[0].shape
        T = CameraUtils.compute_pixel_tensor(x_nums, y_nums)
        pixel_nums = x_nums * y_nums

        print("start reconstruction...")
        self.points = np.zeros([(self.end_id - self.start_id + 1) * pixel_nums, 3])

        for idx, im in enumerate(self.im_arr_list):
            odom = self.odom_list[idx + self.start_id].copy()
            if is_flat_ground:
                odom[2] = self._avg_height
            E = CameraUtils.compute_e_matrix(odom, is_flat_ground, self.cameraR, self.cameraT)
            P_matrix = self.K.dot(E)
            if is_output:
                print("Extracting points from image: ", idx + self.start_id)
            self.points[idx * pixel_nums: (idx + 1) * pixel_nums, :] = CloudUtils.extract_cloud_from_image(
                P_matrix, im, T, max_dist=self.max_range)

        print("creating open3d geometry point cloud...")
        self.pcd = CloudUtils.create_open3d_cloud(self.points, self.voxel_size)
        self.is_constructed = True
        print("construction completed.")
    
    def show_point_cloud(self):
        if not self.is_constructed:
            print("no reconstructed cloud")
        o3d.visualization.draw_geometries([self.pcd])  # visualize point cloud
        
    def save_reconstructed_data(self, image_type="depth"):
        if not self.is_constructed:
            print("save points failed, no reconstructed cloud!")
            
        print("save output files to: " + self.out_path)
        DataUtils.prepare_output_folders(self.out_path, image_type)
        
        DataUtils.save_images(self.out_path, self.im_arr_list, image_type)
        DataUtils.save_odom_list(self.out_path, self.odom_list, self.start_id, len(self.im_arr_list))
        DataUtils.save_extrinsic(self.out_path, self.cameraR, self.cameraT)
        DataUtils.save_intrinsic(self.out_path, self.K)
        DataUtils.save_point_cloud(self.out_path, self.pcd)  # save point cloud
        print("saved cost map data.")
        
    @property
    def avg_height(self):
        return self._avg_height
    
    # private methods
    def _initialize_paths(self, input_path, out_path):
        self.input_path = input_path
        self.out_path = out_path

    def _initialize_parameters(self, voxel_size, max_range, is_max_iter):
        self.voxel_size = voxel_size
        self.is_max_iter = is_max_iter
        self.max_range = max_range

    def _read_camera_params(self):
        # Get Camera Parameters
        self.K = DataUtils.read_intrinsic(self.input_path + "/depth_intrinsic.txt")
        self.cameraR, self.cameraT = DataUtils.read_extrinsic(self.input_path + "/camera_extrinsic.txt")
        
        

