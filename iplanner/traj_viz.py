# ======================================================================
# Copyright (c) 2023 Fan Yang
# Robotic Systems Lab, ETH Zurich
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ======================================================================

import os
import cv2
import copy
import torch
import numpy as np
import pypose as pp
import open3d as o3d
from tsdf_map import TSDF_Map
from scipy.spatial.transform import Rotation as R
import open3d.visualization.rendering as rendering
from typing import List, Optional

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360
MESH_SIZE = 0.5

class TrajViz:    
    def __init__(self, root_path: str, map_name: str = "tsdf1", cameraTilt: float = 0.0, robot_name: str = "robot"):
        """
        Initialize TrajViz class.
        """
        self.is_map = False
        if map_name:
            self.tsdf_map = TSDF_Map()
            self.tsdf_map.ReadTSDFMap(root_path, map_name)
            self.is_map = True
            intrinsic_path = os.path.join(root_path, "depth_intrinsic.txt")
        else:
            intrinsic_path = os.path.join(root_path, robot_name + "_intrinsic.txt")
        self.SetCamera(intrinsic_path)
        self.camera_tilt = cameraTilt

    def TransformPoints(self, odom: torch.Tensor, points: torch.Tensor) -> pp.SE3:
        """
        Transform points in the trajectory.
        """
        batch_size, num_p, _ = points.shape
        world_ps = pp.identity_SE3(batch_size, num_p, device=points.device)
        world_ps.tensor()[:, :, 0:3] = points
        world_ps = pp.SE3(odom[:, None, :]) @ pp.SE3(world_ps)
        return world_ps

    def SetCamera(self, intrinsic_path: str, img_width: int = IMAGE_WIDTH, img_height: int = IMAGE_HEIGHT):
        """
        Set camera parameters.
        """
        with open(intrinsic_path) as f:
            lines = f.readlines()
            elems = np.fromstring(lines[0][1:-2], dtype=float, sep=', ')
        K = np.array(elems).reshape(-1, 4)
        self.camera = o3d.camera.PinholeCameraIntrinsic(img_width, img_height, K[0,0], K[1,1], K[0,2], K[1,2])

    def VizTrajectory(self, preds: torch.Tensor, waypoints: torch.Tensor, odom: torch.Tensor, goal: torch.Tensor, fear: torch.Tensor, cost_map: bool = True, visual_height: float = 0.5, mesh_size: float = MESH_SIZE) -> None:
        """
        Visualize the trajectory.

        Parameters:
        preds (tensor): Predicted trajectory points
        waypoints (tensor): Original waypoints
        odom (tensor): Odometry data
        goal (tensor): Goal position
        fear (tensor): Fear value per trajectory
        cost_map (bool): If True, visualize cost map; otherwise, visualize point cloud
        visual_height (float): Height offset for visualization
        mesh_size (float): Size of mesh objects in visualization
        """
        if not self.is_map:
            print("Cost map is missing.")
            return

        batch_size, _, _ = waypoints.shape

        # Transform to map frame
        preds_ws = self.TransformPoints(odom, preds)
        wp_ws    = self.TransformPoints(odom, waypoints)
        goal_ws  = pp.SE3(odom) @ pp.SE3(goal)

        # Convert to positions
        preds_ws = preds_ws.tensor()[:, :, 0:3].cpu().detach().numpy()
        wp_ws    = wp_ws.tensor()[:, :, 0:3].cpu().detach().numpy()
        goal_ws  = goal_ws.tensor()[:, 0:3].cpu().detach().numpy()

        visual_list = []
        if cost_map:
            visual_list.append(self.tsdf_map.pcd_tsdf)
        else:
            visual_list.append(self.tsdf_map.pcd_viz)
            visual_height /= 5.0
        
        # Visualize trajectory
        traj_pcd = o3d.geometry.PointCloud()
        wp_ws = wp_ws.reshape(-1, 3)
        wp_ws[:, 2] += visual_height
        traj_pcd.points = o3d.utility.Vector3dVector(wp_ws[:, 0:3])
        traj_pcd.paint_uniform_color([0.99, 0.1, 0.1])
        visual_list.append(traj_pcd)

        # Start and goal markers
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(mesh_size/1.5)  # start points
        small_sphere = o3d.geometry.TriangleMesh.create_sphere(mesh_size/3.0)  # successful trajectory points
        small_sphere_fear = o3d.geometry.TriangleMesh.create_sphere(mesh_size/3.0)  # unsuccessful trajectory points
        mesh_box = o3d.geometry.TriangleMesh.create_box(mesh_size, mesh_size, mesh_size)  # end points

        # Set mesh colors
        mesh_box.paint_uniform_color([1.0, 0.64, 0.0])
        small_sphere.paint_uniform_color([0.4, 1.0, 0.1])
        small_sphere_fear.paint_uniform_color([1.0, 0.4, 0.1])

        lines = []
        points = []
        for i in range(batch_size):
            lines.append([2*i, 2*i+1])
            gp = goal_ws[i, :]
            op = odom.numpy()[i, :]
            op[2] += visual_height
            gp[2] += visual_height
            points.append(gp[:3].tolist())
            points.append(op[:3].tolist())

            # Add visualization
            visual_list.append(copy.deepcopy(mesh_sphere).translate((op[0], op[1], op[2])))
            visual_list.append(copy.deepcopy(mesh_box).translate((gp[0]-mesh_size/2.0, gp[1]-mesh_size/2.0, gp[2]-mesh_size/2.0)))
            for j in range(preds_ws[i].shape[0]):
                kp = preds_ws[i, j, :]
                if fear[i, :] > 0.5:
                    visual_list.append(copy.deepcopy(small_sphere_fear).translate((kp[0], kp[1], kp[2] + visual_height)))
                else:
                    visual_list.append(copy.deepcopy(small_sphere).translate((kp[0], kp[1], kp[2] + visual_height)))   
        # Set line from odom to goal
        colors = [[0.99, 0.99, 0.1] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet(o3d.utility.Vector3dVector(points), o3d.utility.Vector2iVector(lines))
        line_set.colors = o3d.utility.Vector3dVector(colors)
        visual_list.append(line_set)
        
        o3d.visualization.draw_geometries(visual_list)
        return

    def VizImages(self, preds: torch.Tensor, waypoints: torch.Tensor, odom: torch.Tensor, goal: torch.Tensor, fear: torch.Tensor, images: torch.Tensor, visual_offset: float = 0.5, mesh_size: float = 0.3, is_shown: bool = True) -> List[np.ndarray]:
        """
        Visualize images of the trajectory.

        Parameters:
        preds (tensor): Predicted Key points
        waypoints (tensor): Trajectory waypoints
        odom (tensor): Odometry data
        goal (tensor): Goal position
        fear (tensor): Fear value per waypoint
        images (tensor): Image data
        visual_offset (float): Offset for visualizing images
        mesh_size (float): Size of mesh objects in images
        is_shown (bool): If True, show images; otherwise, return image list
        """
        batch_size, _, _ = waypoints.shape

        preds_ws = self.TransformPoints(odom, preds)
        wp_ws = self.TransformPoints(odom, waypoints)

        if goal.shape[-1] != 7:
            pp_goal = pp.identity_SE3(batch_size, device=goal.device)
            pp_goal.tensor()[:, 0:3] = goal
            goal = pp_goal.tensor()

        goal_ws  = pp.SE3(odom) @ pp.SE3(goal)

        # Detach to CPU
        preds_ws = preds_ws.tensor()[:, :, 0:3].cpu().detach().numpy()
        wp_ws    = wp_ws.tensor()[:, :, 0:3].cpu().detach().numpy()
        goal_ws  = goal_ws.tensor()[:, 0:3].cpu().detach().numpy()

        # Adjust height
        preds_ws[:, :, 2] -= visual_offset
        wp_ws[:, :, 2]    -= visual_offset
        goal_ws[:, 2]     -= visual_offset

        # Set material shader
        mtl = o3d.visualization.rendering.MaterialRecord()
        mtl.base_color = [1.0, 1.0, 1.0, 0.3]
        mtl.shader = "defaultUnlit"

        # Set meshes
        small_sphere      = o3d.geometry.TriangleMesh.create_sphere(mesh_size/20.0)  # trajectory points
        mesh_sphere       = o3d.geometry.TriangleMesh.create_sphere(mesh_size/5.0)  # successful predict points
        mesh_sphere_fear  = o3d.geometry.TriangleMesh.create_sphere(mesh_size/5.0)  # unsuccessful predict points
        mesh_box          = o3d.geometry.TriangleMesh.create_box(mesh_size, mesh_size, mesh_size)  # end points

        # Set colors
        small_sphere.paint_uniform_color([0.99, 0.2, 0.1])  # green
        mesh_sphere.paint_uniform_color([0.4, 1.0, 0.1])
        mesh_sphere_fear.paint_uniform_color([1.0, 0.64, 0.0])
        mesh_box.paint_uniform_color([1.0, 0.64, 0.1])

        # Init open3D render
        render = rendering.OffscreenRenderer(self.camera.width, self.camera.height)
        render.scene.set_background([0.0, 0.0, 0.0, 1.0])  # RGBA
        render.scene.scene.enable_sun_light(False)

        # compute veretx normals
        small_sphere.compute_vertex_normals()
        mesh_sphere.compute_vertex_normals()
        mesh_sphere_fear.compute_vertex_normals()
        mesh_box.compute_vertex_normals()
        
        wp_start_idx = 1
        cv_img_list = []

        for i in range(batch_size):
            # Add geometries
            gp = goal_ws[i, :]

            # Add goal marker
            goal_mesh = copy.deepcopy(mesh_box).translate((gp[0]-mesh_size/2.0, gp[1]-mesh_size/2.0, gp[2]-mesh_size/2.0))
            render.scene.add_geometry("goal_mesh", goal_mesh, mtl)

            # Add predictions
            for j in range(preds_ws[i, :, :].shape[0]):
                kp = preds_ws[i, j, :]
                if fear[i, :] > 0.5:
                    kp_mesh = copy.deepcopy(mesh_sphere_fear).translate((kp[0], kp[1], kp[2]))
                else:
                    kp_mesh = copy.deepcopy(mesh_sphere).translate((kp[0], kp[1], kp[2]))
                render.scene.add_geometry("keypose"+str(j), kp_mesh, mtl)

            # Add trajectory
            for k in range(wp_start_idx, wp_ws[i, :, :].shape[0]):
                wp = wp_ws[i, k, :]
                wp_mesh = copy.deepcopy(small_sphere).translate((wp[0], wp[1], wp[2]))
                render.scene.add_geometry("waypoint"+str(k), wp_mesh, mtl)

            # Set cameras
            self.CameraLookAtPose(odom[i, :], render, self.camera_tilt)

            # Project to image
            img_o3d = np.asarray(render.render_to_image())
            mask = (img_o3d < 10).all(axis=2)

            # Attach image
            c_img = images[i, :, :].expand(3, -1, -1)
            c_img = c_img.cpu().detach().numpy().transpose(1, 2, 0)
            c_img = (c_img * 255 / np.max(c_img)).astype('uint8')
            img_o3d[mask, :] = c_img[mask, :]
            img_cv2 = cv2.cvtColor(img_o3d, cv2.COLOR_RGBA2BGRA)
            cv_img_list.append(img_cv2)

            # Visualize image
            if is_shown: 
                cv2.imshow("Preview window", img_cv2)
                cv2.waitKey()

            # Clear render geometry
            render.scene.clear_geometry()        

        return cv_img_list

    def CameraLookAtPose(self, odom: torch.Tensor, render: o3d.visualization.rendering.OffscreenRenderer, tilt: float) -> None:
        """
        Set camera to look at at current odom pose.

        Parameters:
        odom (tensor): Odometry data
        render (OffscreenRenderer): Renderer object
        tilt (float): Tilt angle for camera
        """
        unit_vec = pp.identity_SE3(device=odom.device)
        unit_vec.tensor()[0] = 1.0
        tilt_vec = [0, 0, 0]
        tilt_vec.extend(list(R.from_euler('y', tilt, degrees=False).as_quat()))
        tilt_vec = torch.tensor(tilt_vec, device=odom.device, dtype=odom.dtype)
        target_pose = pp.SE3(odom) @ pp.SE3(tilt_vec) @ unit_vec
        camera_up = [0, 0, 1]  # camera orientation
        eye = pp.SE3(odom)
        eye = eye.tensor()[0:3].cpu().detach().numpy()
        target = target_pose.tensor()[0:3].cpu().detach().numpy()
        render.scene.camera.set_projection(self.camera.intrinsic_matrix, 0.1, 100.0, self.camera.width, self.camera.height)
        render.scene.camera.look_at(target, eye, camera_up)
        return
    