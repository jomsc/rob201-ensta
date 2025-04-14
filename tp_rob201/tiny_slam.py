""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid

TINY_VALUE = -2
BIG_VALUE = 2
CLIP_VALUE = 40


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def _score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        # TODO for TP4

        lidar_val_pol = np.array(lidar.get_sensor_values())
        lidar_angles = np.array(lidar.get_ray_angles())

        # estimation des positions des détections dans le repère absolu
        pose = self.get_corrected_pose(pose)

        x_lidar = pose[0] + lidar_val_pol * np.cos(pose[2]+lidar_angles)
        y_lidar = pose[1] + lidar_val_pol * np.sin(pose[2]+lidar_angles)

        # suppression des détecions à distance maximale
        max_range_indexes = np.where(lidar_val_pol == lidar.max_range)
        x_lidar = np.delete(x_lidar, max_range_indexes)
        y_lidar = np.delete(y_lidar, max_range_indexes)

        # conversion dans les coordonnées grille
        x_map, y_map = self.grid.conv_world_to_map(x_lidar, y_lidar)
        
        # on enleve ce qui est en dehors de la grille
        x_outside = np.where(abs(x_map - self.grid.x_max_map/2) > self.grid.x_max_map/2)
        y_outside = np.where(abs(y_map - self.grid.y_max_map/2) > self.grid.y_max_map/2)
        outside_indexes = np.unique(np.concatenate((x_outside, y_outside)))

        x_map = np.delete(x_map, outside_indexes)
        y_map = np.delete(y_map, outside_indexes)

        pts_score = np.dstack((x_map, y_map))
        
        # calcul du score
        score = np.sum(self.grid.occupancy_map[pts_score])

        return score

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # TODO for TP4
        if odom_pose_ref is None:
            corrected_pose = odom_pose+self.odom_pose_ref
        else:
            corrected_pose = odom_pose+odom_pose_ref
        
        return corrected_pose

    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4

        best_score = 0

        return best_score

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        lidar_val_pol = np.array(lidar.get_sensor_values())
        lidar_angles = np.array(lidar.get_ray_angles())

        max_range_indexes = np.where(lidar_val_pol == lidar.max_range)

        x_lidar = pose[0] + lidar_val_pol * np.cos(pose[2]+lidar_angles)
        y_lidar = pose[1] + lidar_val_pol * np.sin(pose[2]+lidar_angles)
        
        for i in range(0, len(x_lidar)):
            self.grid.add_value_along_line(pose[0], pose[1], x_lidar[i], y_lidar[i], TINY_VALUE)

        # TO DO : modif probas
        
        x_lidar = np.delete(x_lidar, max_range_indexes)
        y_lidar = np.delete(y_lidar, max_range_indexes)

        self.grid.add_map_points(x_lidar, y_lidar, BIG_VALUE-TINY_VALUE)

        self.grid.occupancy_map = np.clip(self.grid.occupancy_map, -CLIP_VALUE, CLIP_VALUE)

