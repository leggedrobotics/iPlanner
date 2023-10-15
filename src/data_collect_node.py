# ======================================================================
# Copyright (c) 2023 Fan Yang
# Robotic Systems Lab, ETH Zurich
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ======================================================================


import rospy
import open3d as o3d
import message_filters
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import numpy as np
import ros_numpy
from scipy.spatial.transform import Rotation
# OpenCV2 for saving an image
import sys
import cv2
import tf
import os
import rospkg

rospack = rospkg.RosPack()
pack_path = rospack.get_path('iplanner_node')
planner_path = os.path.join(pack_path,'iplanner')
sys.path.append(pack_path)
sys.path.append(planner_path)

from iplanner.rosutil import ROSArgparse

class DataCollector:
    def __init__(self, args):
        super(DataCollector, self).__init__()
        self.__config(args)

        # internal values
        self.__tf_listener = tf.TransformListener()
        rospy.sleep(2.5) # wait for tf listener to be ready
        
        self.__odom_extrinsic = None
        self.__cv2_img_depth = np.ndarray([640, 360])
        self.__cv2_img_cam   = np.ndarray([640, 360])
        self.__odom_list = []
        self.__pcd = o3d.geometry.PointCloud()
        self.__init_check_dics = {"color": 0, "depth": 0, "scan_extrinsic": 0, "camera_extrinsic": 0, "odometry_extrinsic": 0}

        depth_sub = message_filters.Subscriber(self.__depth_topic, Image)
        image_sub = message_filters.Subscriber(self.__color_topic, Image)
        odom_sub  = message_filters.Subscriber(self.__odom_topic,  Odometry)
        scan_sub  = message_filters.Subscriber(self.__scan_topic,  PointCloud2)

        time_sync_thred = 0.01 # second 
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub, scan_sub, odom_sub], 50, time_sync_thred)
        ts.registerCallback(self.__syncCallback)

        # camera info subscriber
        rospy.Subscriber(self.__depth_info_topic, CameraInfo, self.__depthInfoCallback, (self.__depth_intrinc_path))
        rospy.Subscriber(self.__color_info_topic, CameraInfo, self.__colorInfoCallback, (self.__color_intrinc_path))

        print("deleting previous files, if any ...")
        folder_list = ["depth", "camera", "scan"]
        for folder_name in folder_list:
            dir_path = os.path.join(*[self.__root_path, folder_name])
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            for efile in os.listdir(dir_path):
                os.remove(os.path.join(dir_path, efile))

        self.__odom_file_name = os.path.join(self.__root_path, "odom_ground_truth.txt")
        open(self.__odom_file_name, 'w').close()  # clear txt file
        return

    def __config(self, args):
        self.__main_freq         = args.main_freq
        self.__root_path         = args.root_path
        self.__depth_topic       = args.depth_topic
        self.__color_topic       = args.color_topic
        self.__odom_topic        = args.odom_topic
        self.__scan_topic        = args.scan_topic
        self.__depth_info_topic  = args.depth_info_topic
        self.__color_info_topic  = args.color_info_topic
        self.__camera_frame_id   = args.camera_frame_id
        self.__scan_frame_id     = args.scan_frame_id
        self.__base_frame_id     = args.base_frame_id
        self.__odom_associate_id = args.odom_associate_id

        self.__depth_intrinc_path  = os.path.join(self.__root_path, "depth_intrinsic.txt")
        self.__color_intrinc_path  = os.path.join(self.__root_path, "color_intrinsic.txt")
        self.__camera_extrinc_path = os.path.join(self.__root_path, "camera_extrinsic.txt")
        self.__scan_extrinc_path   = os.path.join(self.__root_path, "scan_extrinsic.txt")
        # print(self.__depth_intrinc_path)
        return

    def spin(self):
        r = rospy.Rate(self.__main_freq)
        time_step = 0
        last_odom = []
        odom_file = open(self.__odom_file_name, 'w')
        while not rospy.is_shutdown():
            # listen to camera extrinsic
            if (self.__init_check_dics["camera_extrinsic"] == 0):
                try:
                    (pos, ori) = self.__tf_listener.lookupTransform(self.__base_frame_id, self.__camera_frame_id, rospy.Time(0))
                    self.__writeExtrinstic(pos, ori, self.__camera_extrinc_path, "camera_extrinsic")
                except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    print("Wait to get camera extrinsic.")
            if (self.__init_check_dics["scan_extrinsic"] == 0):
                try:
                    (pos, ori) = self.__tf_listener.lookupTransform(self.__base_frame_id, self.__scan_frame_id, rospy.Time(0))
                    self.__writeExtrinstic(pos, ori, self.__scan_extrinc_path, "scan_extrinsic")
                except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    print("Wait to get scan extrinsic.")
            # listen to odom extrinsic
            if (self.__init_check_dics["odometry_extrinsic"] == 0):
                try:
                    (pos, ori) = self.__tf_listener.lookupTransform(self.__base_frame_id, self.__odom_associate_id, rospy.Time(0))
                    self.__odom_extrinsic = np.eye(4)
                    self.__odom_extrinsic[:3, :3] = Rotation.from_quat(ori).as_matrix()
                    self.__odom_extrinsic[:3,3] = np.array(pos)
                    self.__init_check_dics["odometry_extrinsic"] = 1
                    print("odometry_extrinsic" + " save succeed.")
                except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    print("Wait to get odometry extrinsic.")
                
            # save images, depths and odom
            if len(self.__odom_list) > 0:
                if len(last_odom) > 0 and np.linalg.norm(np.array(last_odom) - np.array(self.__odom_list)) < 1e-2:
                    self.__odom_list.clear()
                    r.sleep()
                    continue
                odom_file.writelines(str(self.__odom_list))
                odom_file.write('\n')
                file_name_depth  = os.path.join(*[self.__root_path, "depth",  str(time_step) + ".png"])
                file_name_camera = os.path.join(*[self.__root_path, "camera", str(time_step) + ".png"])
                file_name_scan   = os.path.join(*[self.__root_path, "scan",   str(time_step) + ".ply"])
                self.__saveDepthImage(file_name_depth, self.__cv2_img_depth)
                cv2.imwrite(file_name_camera, self.__cv2_img_cam)
                o3d.io.write_point_cloud(file_name_scan, self.__pcd)
                time_step = time_step + 1
                print("save current idx: %d", time_step)
                last_odom = self.__odom_list.copy()
            r.sleep()
            
        odom_file.close()
        rospy.spin()
        return

    def __syncCallback(self, image, depth, scan, odom):
        if self.__init_check_dics["odometry_extrinsic"] == 0:
            return
        self.__cv2_img_cam = ros_numpy.numpify(image)
        self.__cv2_img_depth = ros_numpy.numpify(depth)
        self.__updateScanPoints(scan, self.__pcd)
        
        self.__cv2_img_depth[~np.isfinite(self.__cv2_img_depth)] = 0.0

        pos = odom.pose.pose.position
        ori = odom.pose.pose.orientation
        if not self.__odom_associate_id == self.__base_frame_id:
            trans = np.eye(4)
            trans[:3,:3] = Rotation.from_quat([ori.x, ori.y, ori.z, ori.w]).as_matrix()
            trans[:3,3]  = np.array([pos.x, pos.y, pos.z])
            trans = trans @ self.__odom_extrinsic
            ori = Rotation.from_matrix(trans[:3,:3]).as_quat().tolist()
            self.__odom_list = trans[:3,3].tolist()
            self.__odom_list.extend(ori)
        else:
            self.__odom_list = [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]
        return


    def __saveDepthImage(self, path, image, scale=1000.0):
        image[~np.isfinite(image)] = 0.0
        if not image.dtype.name == 'uint16':
            image = image * scale
            image = np.uint16(image)
        cv2.imwrite(path, image) 
        return

    def __updateScanPoints(self, pc_msg, pcd):
        pc_np = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pc_msg, remove_nans=True)
        pcd.clear()
        pcd.points = o3d.utility.Vector3dVector(pc_np)


    def __depthInfoCallback(self, depth_info, args):
        if (self.__init_check_dics["depth"] == 1):
            return
        # TODO: save intrinsic matrix in path
        path = args
        P = depth_info.P
        open(path, 'w').close()  # clear txt file
        fc = open(path, 'w')
        fc.writelines(str(P))
        fc.close()
        self.__init_check_dics["depth"] = 1
        return

    def __colorInfoCallback(self, color_info, args):
        if (self.__init_check_dics["color"] == 1):
            return
        # TODO: save intrinsic matrix in path
        path = args
        P = color_info.P
        open(path, 'w').close()  # clear txt file
        fc = open(path, 'w')
        fc.writelines(str(P))
        fc.close()
        self.__init_check_dics["color"] = 1
        return

    def __writeExtrinstic(self, pos, ori, path, name):
        extric_list = [ori[0], ori[1], ori[2], ori[3], pos[0], pos[1], pos[2]]
        open(path, 'w').close()  # clear txt file
        fc = open(path, 'w')
        fc.writelines(str(extric_list))
        fc.close()
        self.__init_check_dics[name] = 1
        print(name + " save succeed.")
        return
    
    
if __name__ == '__main__':
    # init global valuables for callback funcs
    # init ros node
    node_name = "data_collect_node"
    rospy.init_node(node_name, anonymous=False)

    parser = ROSArgparse(relative=node_name)
    parser.add_argument('main_freq',         type=int,   default=5,                                help="Main frequency of the path planner.")
    parser.add_argument('depth_topic',       type=str,   default='/rgbd_camera/depth/image',       help="Topic for depth image.")
    parser.add_argument('color_topic',       type=str,   default='/rgbd_camera/color/image',       help="Topic for color image.")
    parser.add_argument('odom_topic',        type=str,   default='/state_estimation',              help='Topic for odometry data.')
    parser.add_argument('scan_topic',        type=str,   default='/velodyne_points',               help='Topic for lidar point cloud data.')
    parser.add_argument('env_name',          type=str,   default='empty',                          help='Name of the environment, also used as the folder name for data.')
    parser.add_argument('depth_info_topic',  type=str,   default='/rgbd_camera/depth/camera_info', help='Topic for depth camera information.')
    parser.add_argument('color_info_topic',  type=str,   default='/rgbd_camera/color/camera_info', help='Topic for color camera information.')
    parser.add_argument('camera_frame_id',   type=str,   default='camera',                         help='Frame ID for the camera.')
    parser.add_argument('scan_frame_id',     type=str,   default='sensor',                         help='Frame ID for the lidar sensor.')
    parser.add_argument('base_frame_id',     type=str,   default='vehicle',                        help='Base frame ID.')
    parser.add_argument('odom_associate_id', type=str,   default='vehicle',                        help='Frame ID associated with odometry data.')

    rospack = rospkg.RosPack()
    pack_path = rospack.get_path("iplanner_node")
    args = parser.parse_args()
    args.root_path = os.path.join(*[pack_path, "iplanner", "data", "CollectedData", args.env_name])

    node = DataCollector(args)
    node.spin()