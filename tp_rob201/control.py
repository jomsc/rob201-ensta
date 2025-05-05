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
    
    MIN_DIST = 50
    SWITCH_DIST = 100

    # GRADIENT OBJECTIF
    K_goal = 1

    dst = np.linalg.norm(goal_pose[:2]-current_pose[:2])
    if dst>SWITCH_DIST:
        grad_objectif = [
            K_goal*(goal_pose[0]-current_pose[0])/dst, 
            K_goal*(goal_pose[1]-current_pose[1])/dst
        ]
    
    else:
        grad_objectif = [
            K_goal*(goal_pose[0]-current_pose[0])**2/(2*dst), 
            K_goal*(goal_pose[1]-current_pose[1])**2/(2*dst)
        ]

    # GRADIENT OBSTACLE
    K_obs = 1000
    dst_safe = 500
    values = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()
    dst_obstacle = min(values)

    # This line is problematic: np.where() returns a tuple of arrays
    # angle_obstacle = angles[np.where(values==dst_obstacle)]
    # Instead, use np.argmin():
    min_idx = np.argmin(values)
    angle_obstacle = angles[min_idx]

    # Now these calculations will work correctly
    obstacle_x_robot = dst_obstacle * np.cos(angle_obstacle)
    obstacle_y_robot = dst_obstacle * np.sin(angle_obstacle)

    # Convert to the world frame using the robot's current pose
    cos_theta = np.cos(current_pose[2])
    sin_theta = np.sin(current_pose[2])
    obstacle_pose = [
        current_pose[0] + obstacle_x_robot * cos_theta - obstacle_y_robot * sin_theta,
        current_pose[1] + obstacle_x_robot * sin_theta + obstacle_y_robot * cos_theta
    ]

    # Only apply repulsive force if obstacle is within safety distance
    grad_obstacle = [0, 0]  # Initialize with zero
    if dst_obstacle < dst_safe:
        # Calculate the repulsive gradient
        # The force increases as distance decreases
        repulsive_magnitude = K_obs * (1/dst_obstacle - 1/dst_safe) / (dst_obstacle)
        
        # Calculate the vector from obstacle to robot (opposite of direction to obstacle)
        obstacle_vector = [
            current_pose[0] - obstacle_pose[0],
            current_pose[1] - obstacle_pose[1]
        ]
        
        if dst_obstacle > 0:  # Avoid division by zero
            obstacle_direction = [
                obstacle_vector[0] / dst_obstacle,
                obstacle_vector[1] / dst_obstacle
            ]
            
            grad_obstacle = [
                repulsive_magnitude * obstacle_direction[0],
                repulsive_magnitude * obstacle_direction[1]
            ]

    # GRADIENT FINAL

    grad = [
        grad_objectif[0]+grad_obstacle[0],
        grad_objectif[1]+grad_obstacle[1]
    ]

    target_angle = np.arctan2(grad[1], grad[0])
    
    angle_error = target_angle - current_pose[2]
    angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
    
    # CONTROLEUR PROPORTIONNEL
    K_p = 0.1
    forward = min(K_p * np.linalg.norm(grad), 1) if dst >= MIN_DIST else 0
    rotation = np.clip(K_p * angle_error, -1, 1)
    
    # COMMANDE
    if dst < MIN_DIST:
        command = {"forward": 0, "rotation": 0}
    else:
        command = {"forward": forward, "rotation": rotation}
    
    return command