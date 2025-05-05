"""
Planner class
Implementation of A*
"""

import numpy as np
import heapq

from occupancy_grid import OccupancyGrid


class Planner:
    """Simple occupancy grid Planner"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid


        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def get_neighbors(self, current_cell):
        """
        Returns the 8 neighboring cells of the current cell
        current_cell: tuple (x, y) representing the position in grid coordinates
        return: list of tuples [(x1, y1), (x2, y2), ...] representing valid neighboring cells
        """
        x, y = current_cell
        # Define the 8 possible directions (horizontal, vertical and diagonal)
        directions = [
            (0, 1),   # right
            (1, 0),   # down
            (0, -1),  # left
            (-1, 0),  # up
            (1, 1),   # down-right
            (1, -1),  # down-left
            (-1, 1),  # up-right
            (-1, -1)  # up-left
        ]
        
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # Check if the neighbor is within the grid boundaries
            if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                # Check if the cell is free (not occupied by an obstacle)
                if self.grid.occupancy_map[nx, ny] < 0.5 :  # Assuming < 0.5 is free space
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def heuristic(self, cell_1, cell_2):
        """
        Calculates the Euclidean distance between two cells
        cell_1, cell_2: tuples (x, y) representing positions in grid coordinates
        return: float, the Euclidean distance between the cells
        """
        return np.sqrt((cell_1[0] - cell_2[0])**2 + (cell_1[1] - cell_2[1])**2)

    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates (theta unused)
        goal : [x, y, theta] nparray, goal pose in world coordinates (theta unused)
        return: list of poses in world coordinates
        """
        # Convert world coordinates to map coordinates
        start_map_x, start_map_y = self.grid.conv_world_to_map(start[0], start[1])
        goal_map_x, goal_map_y = self.grid.conv_world_to_map(goal[0], goal[1])
        
        start_cell = (int(start_map_x), int(start_map_y))
        goal_cell = (int(goal_map_x), int(goal_map_y))
        
        # Initialize open set, came_from, g_score, and f_score
        open_set = []
        heapq.heappush(open_set, (0, start_cell))  # (f_score, cell)
        
        came_from = {}
        
        g_score = {}
        g_score[start_cell] = 0
        
        f_score = {}
        f_score[start_cell] = self.heuristic(start_cell, goal_cell)
        
        # For retrieving items from the heap
        open_set_hash = {start_cell}
        
        while open_set:
            # Get the node with the lowest f_score
            current_f_score, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            
            # If we've reached the goal, reconstruct the path
            if current == goal_cell:
                path = []
                while current in came_from:
                    # Convert map coordinates back to world coordinates before adding to path
                    world_x, world_y = self.grid.conv_map_to_world(current[0], current[1])
                    path.append(np.array([world_x, world_y, 0]))  # theta is set to 0
                    current = came_from[current]
                
                # Add the start position
                world_x, world_y = self.grid.conv_map_to_world(start_cell[0], start_cell[1])
                path.append(np.array([world_x, world_y, 0]))
                
                # Reverse the path to get start -> goal order
                path.reverse()
                return path
            
            # Check all neighbors
            for neighbor in self.get_neighbors(current):
                # Calculate the tentative g_score
                # For diagonal moves, the distance is sqrt(2)
                if abs(neighbor[0] - current[0]) == 1 and abs(neighbor[1] - current[1]) == 1:
                    tentative_g_score = g_score[current] + np.sqrt(2)
                else:
                    tentative_g_score = g_score[current] + 1
                
                # If this path to neighbor is better than any previous one
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # Record the best path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_cell)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        # If we got here, no path was found
        return [start, goal]  # Return direct path if no better path found

    def explore_frontiers(self):
        """ Frontier based exploration """
        goal = np.array([0, 0, 0])  # frontier to reach for exploration
        return goal
