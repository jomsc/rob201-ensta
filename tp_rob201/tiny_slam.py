""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid

TINY_VALUE = -2
BIG_VALUE = 4
CLIP_VALUE = 40
LOC_MAX_ITER = 100


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

        x_lidar = pose[0] + lidar_val_pol * np.cos(pose[2]+lidar_angles)
        y_lidar = pose[1] + lidar_val_pol * np.sin(pose[2]+lidar_angles)

        # suppression des détecions à distance maximale
        max_range_indexes = np.where(lidar_val_pol == lidar.max_range)
        x_lidar = np.delete(x_lidar, max_range_indexes)
        y_lidar = np.delete(y_lidar, max_range_indexes)

        # conversion dans les coordonnées grille
        x_map, y_map = self.grid.conv_world_to_map(x_lidar, y_lidar)
        
        # on enleve ce qui est en dehors de la grille
        x_inside = np.where(x_map < self.grid.x_max_map)
        x_map = x_map[x_inside]
        y_map = y_map[x_inside]

        x2_inside = np.where(x_map > -1)
        x_map = x_map[x2_inside]
        y_map = y_map[x2_inside]

        y_inside = np.where(y_map < self.grid.y_max_map)
        x_map = x_map[y_inside]
        y_map = y_map[y_inside]

        y2_inside = np.where(y_map > -1)
        x_map = x_map[y2_inside]
        y_map = y_map[y2_inside]

        # calcul du score
        score = np.sum(self.grid.occupancy_map[x_map, y_map])

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
            ref = self.odom_pose_ref
        else:
            ref = odom_pose_ref
        
        corrected_pose = np.array([0, 0, 0])

        d_0 = np.linalg.norm(odom_pose[:2])
        alpha_0 = np.arctan2(odom_pose[1], odom_pose[0])

        corrected_pose[0] = ref[0] + d_0 * np.cos(ref[2]+alpha_0)
        corrected_pose[1] = ref[1] + d_0 * np.sin(ref[2]+alpha_0)
        corrected_pose[2] = ref[2] + odom_pose[2]

        return corrected_pose

    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        best_score = self._score(lidar, self.get_corrected_pose(raw_odom_pose))
        sigma = 2
        for i in range(LOC_MAX_ITER):
            offset = np.random.normal(0, sigma, (3, 1))
            odom_ref = self.odom_pose_ref
            odom_ref[0] += offset[0]
            odom_ref[1] += offset[1]
            odom_ref[2] += offset[2]

            new_score = self._score(lidar, self.get_corrected_pose(raw_odom_pose, odom_ref))

            if new_score > best_score:
                best_score = new_score
                self.odom_pose_ref = odom_ref

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

        x_values = pose[0] + lidar_val_pol * 0.9 * np.cos(pose[2]+lidar_angles)
        y_values = pose[1] + lidar_val_pol * 0.9 * np.sin(pose[2]+lidar_angles)
        
        for i in range(0, len(x_lidar)):
            self.grid.add_value_along_line(pose[0], pose[1], x_values[i], y_values[i], TINY_VALUE)
        
        x_lidar = np.delete(x_lidar, max_range_indexes)
        y_lidar = np.delete(y_lidar, max_range_indexes)

        self.grid.add_map_points(x_lidar, y_lidar, BIG_VALUE)

        self.grid.occupancy_map = np.clip(self.grid.occupancy_map, -CLIP_VALUE, CLIP_VALUE)

