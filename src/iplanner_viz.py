# ======================================================================
# Copyright (c) 2023 Fan Yang
# Robotic Systems Lab, ETH Zurich
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ======================================================================

import os
import PIL
import sys
import torch
import rospy
import rospkg
import tf
import numpy as np
from sensor_msgs.msg import Image, Joy
from std_msgs.msg import Int16
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped
import ros_numpy

rospack = rospkg.RosPack()
pack_path = rospack.get_path('iplanner_node')
planner_path = os.path.join(pack_path,'iplanner')
sys.path.append(pack_path)
sys.path.append(planner_path)

from iplanner.ip_algo import IPlannerAlgo
from iplanner.rosutil import ROSArgparse
from iplanner import traj_viz

class iPlannerNode:
    def __init__(self, args):
        super(iPlannerNode, self).__init__()
        self.config(args)

        self.iplanner_algo = IPlannerAlgo(args)
        self.traj_viz = traj_viz.TrajViz(os.path.join(*[planner_path, 'camera_intrinsic']), 
                                         map_name=None,
                                         cameraTilt=args.camera_tilt,
                                         robot_name="robot")
        self.tf_listener = tf.TransformListener()
        
        rospy.sleep(2.5) # wait for tf listener to be ready

        self.image_time = rospy.get_rostime()
        self.is_goal_init = False
        self.ready_for_planning = False

        # planner status
        self.planner_status = Int16()
        self.planner_status.data = 0
        self.is_goal_processed = False
        self.is_smartjoy = False

        # fear reaction
        self.fear_buffter = 0
        self.is_fear_reaction = False
        
        rospy.Subscriber(self.depth_topic, Image, self.imageCallback)
        rospy.Subscriber(self.goal_topic, PointStamped, self.goalCallback)
        rospy.Subscriber("/joy", Joy, self.joyCallback, queue_size=10)

        status_topic = '/iplanner_status'

        # planning status topics
        self.status_pub = rospy.Publisher(status_topic, Int16, queue_size=10)
        # image visualizer
        self.img_pub = rospy.Publisher(self.image_topic, Image, queue_size=10)
        self.path_pub  = rospy.Publisher(self.path_topic, Path, queue_size=10)
        self.fear_path_pub = rospy.Publisher(self.path_topic + "_fear", Path, queue_size=10)

        rospy.loginfo("iPlanner Ready.")
        

    def config(self, args):
        self.main_freq   = args.main_freq
        self.depth_topic = args.depth_topic
        self.goal_topic  = args.goal_topic
        self.path_topic  = args.path_topic
        self.image_topic = args.image_topic
        self.camera_tilt = args.camera_tilt
        self.frame_id    = args.robot_id
        self.world_id    = args.world_id
        self.uint_type   = args.uint_type
        self.image_flip  = args.image_flip
        self.conv_dist   = args.conv_dist
        self.depth_max   = args.depth_max 
        # fear reaction
        self.is_fear_act = args.is_fear_act
        self.buffer_size = args.buffer_size
        self.ang_thred   = args.angular_thred
        self.track_dist  = args.track_dist
        self.joyGoal_scale = args.joyGoal_scale
        return 

    def spin(self):
        r = rospy.Rate(self.main_freq)
        while not rospy.is_shutdown():
            if self.ready_for_planning and self.is_goal_init:
                # network planning
                cur_image = self.img.copy()
                self.preds, self.waypoints, fear_output, img_process = self.iplanner_algo.plan(cur_image, self.goal_rb)
                # check goal less than converage range
                goal_np = self.goal_rb[0, :].cpu().detach().numpy()
                if (np.sqrt(goal_np[0]**2 + goal_np[1]**2) < self.conv_dist) and self.is_goal_processed and (not self.is_smartjoy):
                    self.ready_for_planning = False
                    self.is_goal_init = False
                    # planner status -> Success
                    if self.planner_status.data == 0:
                        self.planner_status.data = 1
                        self.status_pub.publish(self.planner_status)

                    rospy.loginfo("Goal Arrived")
                self.fear = torch.tensor([[0.0]], device=fear_output.device)
                if self.is_fear_act:
                    self.fear = fear_output
                    is_track_ahead = self.isForwardTraking(self.waypoints)
                    self.fearPathDetection(self.fear, is_track_ahead)
                    if self.is_fear_reaction:
                        rospy.logwarn_throttle(2.0, "current path prediction is invaild.")
                        # planner status -> Fails
                        if self.planner_status.data == 0:
                            self.planner_status.data = -1
                            self.status_pub.publish(self.planner_status)
                self.pubPath(self.waypoints, self.is_goal_init)
                # visualize image
                self.pubRenderImage(self.preds, self.waypoints, self.odom, self.goal_rb, self.fear, img_process)
            r.sleep()
        rospy.spin()

    def pubPath(self, waypoints, is_goal_init=True):
        path = Path()
        fear_path = Path()
        if is_goal_init:
            for p in waypoints.squeeze(0):
                pose = PoseStamped()
                pose.pose.position.x = p[0]
                pose.pose.position.y = p[1]
                pose.pose.position.z = p[2]
                path.poses.append(pose)
        # add header
        path.header.frame_id = fear_path.header.frame_id = self.frame_id
        path.header.stamp = fear_path.header.stamp = self.image_time
        # publish fear path
        if self.is_fear_reaction:
            fear_path.poses = path.poses.copy()
            path.poses = path.poses[:1]
        # publish path
        self.fear_path_pub.publish(fear_path)
        self.path_pub.publish(path)
        return

    def fearPathDetection(self, fear, is_forward):
        if fear > 0.5 and is_forward:
            if not self.is_fear_reaction:
                self.fear_buffter = self.fear_buffter + 1
        elif self.is_fear_reaction:
            self.fear_buffter = self.fear_buffter - 1
        if self.fear_buffter > self.buffer_size:
            self.is_fear_reaction = True
        elif self.fear_buffter <= 0:
            self.is_fear_reaction = False
        return None

    def isForwardTraking(self, waypoints):
        xhead = np.array([1.0, 0])
        phead = None
        for p in waypoints.squeeze(0):
            if torch.norm(p[0:2]).item() > self.track_dist:
                phead = np.array([p[0].item(), p[1].item()])
                phead /= np.linalg.norm(phead)
                break
        if phead is None or phead.dot(xhead) > 1.0 - self.ang_thred:
            return True
        return False

    def joyCallback(self, joy_msg):
        if joy_msg.buttons[4] > 0.9:
            rospy.loginfo("Switch to Smart Joystick mode ...")
            self.is_smartjoy = True
            # reset fear reaction
            self.fear_buffter = 0
            self.is_fear_reaction = False
        if self.is_smartjoy:
            if np.sqrt(joy_msg.axes[3]**2 + joy_msg.axes[4]**2) < 1e-3:
                # reset fear reaction
                self.fear_buffter = 0
                self.is_fear_reaction = False
                self.ready_for_planning = False
                self.is_goal_init = False
            else:
                joy_goal = PointStamped()
                joy_goal.header.frame_id = self.frame_id
                joy_goal.point.x = joy_msg.axes[4] * self.joyGoal_scale
                joy_goal.point.y = joy_msg.axes[3] * self.joyGoal_scale
                joy_goal.point.z = 0.0
                joy_goal.header.stamp = rospy.Time.now()
                self.goal_pose = joy_goal
                self.is_goal_init = True
                self.is_goal_processed = False
        return

    def goalCallback(self, msg):
        rospy.loginfo("Recevied a new goal")
        self.goal_pose = msg
        self.is_smartjoy = False
        self.is_goal_init = True
        self.is_goal_processed = False
        # reset fear reaction
        self.fear_buffter = 0
        self.is_fear_reaction = False
        # reste planner status
        self.planner_status.data = 0
        return

    def pubRenderImage(self, preds, waypoints, odom, goal, fear, image):
        if torch.cuda.is_available():
            odom = odom.cuda()
            goal = goal.cuda()
        image = self.traj_viz.VizImages(preds, waypoints, odom, goal, fear, image, is_shown=False)[0]
        ros_img = ros_numpy.msgify(Image, image, encoding='8UC4')
        self.img_pub.publish(ros_img)
        return None

    def imageCallback(self, msg):
        # rospy.loginfo("Received image %s: %d"%(msg.header.frame_id, msg.header.seq))
        self.image_time = msg.header.stamp
        frame = ros_numpy.numpify(msg)
        frame[~np.isfinite(frame)] = 0
        if self.uint_type:
            frame = frame / 1000.0
        frame[frame > self.depth_max] = 0.0
        # DEBUG - Visual Image
        # img = PIL.Image.fromarray((frame * 255 / np.max(frame[frame>0])).astype('uint8'))
        # img.show()
        if self.image_flip:
            frame = PIL.Image.fromarray(frame)
            self.img = np.array(frame.transpose(PIL.Image.ROTATE_180))
        else:
            self.img = frame
        # get odom from TF for camera image visualization 
        try:
            (odom, ori) = self.tf_listener.lookupTransform(self.world_id, self.frame_id, rospy.Time(0))
            odom.extend(ori)
            self.odom = torch.tensor(odom, dtype=torch.float32).unsqueeze(0)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("Fail to get odomemrty from tf.")
            return

        if self.is_goal_init:
            goal_robot_frame = self.goal_pose
            if not self.goal_pose.header.frame_id == self.frame_id:
                try:
                    goal_robot_frame.header.stamp = self.tf_listener.getLatestCommonTime(self.goal_pose.header.frame_id,
                                                                                         self.frame_id)
                    goal_robot_frame = self.tf_listener.transformPoint(self.frame_id, goal_robot_frame)
                except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    rospy.logerr("Fail to transfer the goal into robot frame.")
                    return
            self.goal_rb = torch.tensor([goal_robot_frame.point.x, 
                                         goal_robot_frame.point.y,
                                         goal_robot_frame.point.z], dtype=torch.float32)[None, ...]
        else:
            return
        self.ready_for_planning = True
        self.is_goal_processed  = True
        return

if __name__ == '__main__':

    node_name = "iplanner_node"
    rospy.init_node(node_name, anonymous=False)

    parser = ROSArgparse(relative=node_name)
    parser.add_argument('main_freq',         type=int,   default=5,                          help="Main frequency of the path planner.")
    parser.add_argument('model_save',        type=str,   default='/models/plannernet.pt',    help="Path to the saved model.")
    parser.add_argument('crop_size',         type=tuple, default=[360,640],                  help='Dimensions to crop the image to.')
    parser.add_argument('uint_type',         type=bool,  default=False,                      help="Flag to indicate if the image is in uint type.")
    parser.add_argument('depth_topic',       type=str,   default='/rgbd_camera/depth/image', help='ROS topic for depth image.')
    parser.add_argument('goal_topic',        type=str,   default='/way_point',               help='ROS topic for goal waypoints.')
    parser.add_argument('path_topic',        type=str,   default='/iplanner_path',           help='ROS topic for the iPlanner path.')
    parser.add_argument('image_topic',       type=str,   default='/path_image',              help='ROS topic for iPlanner image view.')
    parser.add_argument('camera_tilt',       type=float, default=0.0,                        help='Tilt angle of the camera.')
    parser.add_argument('robot_id',          type=str,   default='base',                     help='TF frame ID for the robot.')
    parser.add_argument('world_id',          type=str,   default='odom',                     help='TF frame ID for the world.')
    parser.add_argument('depth_max',         type=float, default=10.0,                       help='Maximum depth distance in the image.')
    parser.add_argument('is_sim',            type=bool,  default=True,                       help='Flag to indicate if the system is in a simulation setting.')
    parser.add_argument('image_flip',        type=bool,  default=True,                       help='Flag to indicate if the image is flipped.')
    parser.add_argument('conv_dist',         type=float, default=0.5,                        help='Convergence range to the goal.')
    parser.add_argument('is_fear_act',       type=bool,  default=True,                       help='Flag to indicate if fear action is enabled.')
    parser.add_argument('buffer_size',       type=int,   default=10,                         help='Buffer size for fear reaction.')
    parser.add_argument('angular_thred',     type=float, default=1.0,                        help='Angular threshold for turning.')
    parser.add_argument('track_dist',        type=float, default=0.5,                        help='Look-ahead distance for path tracking.')
    parser.add_argument('joyGoal_scale',     type=float, default=0.5,                        help='Scale for joystick goal distance.')
    parser.add_argument('sensor_offset_x',   type=float, default=0.0,                        help='Sensor offset on the X-axis.')
    parser.add_argument('sensor_offset_y',   type=float, default=0.0,                        help='Sensor offset on the Y-axis.')

    args = parser.parse_args()
    args.model_save = planner_path + args.model_save

    node = iPlannerNode(args)

    node.spin()
