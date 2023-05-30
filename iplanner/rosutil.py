# ==============================================================================
# Copyright <2019> <Chen Wang [https://chenwang.site], Carnegie Mellon University>
# Refer to: https://github.com/wang-chen/interestingness_ros/blob/master/script/rosutil.py
# ==============================================================================

import os
import rospy
import torch
import numpy as np

class ROSArgparse():
    def __init__(self, relative=None):
        self.relative = relative

    def add_argument(self, name, default, type=None, help=None):
        name = os.path.join(self.relative, name)
        if rospy.has_param(name):
            rospy.loginfo('Get param %s', name)
        else:
            rospy.logwarn('Couldn\'t find param: %s, Using default: %s', name, default)
        value = rospy.get_param(name, default)
        variable = name[name.rfind('/')+1:].replace('-','_')
        if isinstance(value, str):
            exec('self.%s=\'%s\''%(variable, value))
        else:
            exec('self.%s=%s'%(variable, value))

    def parse_args(self):
        return self


def msg_to_torch(data, shape=np.array([-1])):
    return torch.from_numpy(data).view(shape.tolist())


def torch_to_msg(tensor):
    return [tensor.view(-1).cpu().numpy(), tensor.shape]
