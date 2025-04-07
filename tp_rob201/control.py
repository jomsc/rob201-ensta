""" A set of robotics control functions """

import random
import numpy as np


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1

    laser_dist = lidar.get_sensor_values()
    speed = 0.4
    rotation_speed = 1.0

    
    middle = laser_dist[int(len(laser_dist)/2)]
    maximum = max(laser_dist)
    max_i = np.where(laser_dist == maximum)[0]
    middle_i = int(len(laser_dist)/2)

    if (abs(middle_i-max_i) > 30) and abs(middle-maximum) > 50:
        if max_i < middle_i:
            command = {"forward": 0,
               "rotation": -1*rotation_speed}
        else:
            command = {"forward": 0,
               "rotation": rotation_speed}

    else:
        command = {"forward": speed,
               "rotation": 0}

    return command


def potential_field_control(lidar, current_pose, goal_pose):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    Notes: As lidar and odom are local only data, goal and gradient will be defined either in
    robot (x,y) frame (centered on robot, x forward, y on left) or in odom (centered / aligned
    on initial pose, x forward, y on left)
    """
    # TODO for TP2

    command = {"forward": 0,
               "rotation": 0}

    return command
