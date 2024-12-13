import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.spatial import distance
from heapq import heappush, heappop
from matplotlib.backend_bases import MouseEvent

# File path
grid_file_path = r"D:\capstone\24_12_11\result_grid_test_size_min.csv"

# List to store manually clicked waypoints
clicked_waypoints = []

# A* Weighted Algorithm
def heuristic(a, b):
    return distance.euclidean(a, b)

def astar_weighted(grid, weights, start, goal):
    rows, cols = grid.shape
    open_set = []
    heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        _, current_cost, current = heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Return reversed path

        # Explore neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-connectivity
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor] == 1:
                new_cost = current_cost + weights[neighbor]
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heappush(open_set, (priority, new_cost, neighbor))
                    came_from[neighbor] = current

    return []  # No path found

# Interactive Click Handler
def onclick(event: MouseEvent):
    if event.xdata is not None and event.ydata is not None:
        # Get x and y coordinates
        x, y = int(event.xdata), int(event.ydata)
        clicked_waypoints.append((int(y), int(x)))  # Store as (row, column)

        # Display waypoint number on map
        waypoint_number = len(clicked_waypoints)
        ax_map.scatter(x, y, color='red', edgecolor='black', s=100)
        ax_map.text(x, y, f"{waypoint_number}", color='blue', fontsize=12, ha='center', va='center')
        print(f"Waypoint {waypoint_number}: (row={int(y)}, column={int(x)})")
        plt.draw()

        # Stop after 9 waypoints are selected
        if len(clicked_waypoints) >= 9:
            print("\n9 waypoints selected. Closing the map.")
            plt.close()

# Distance Calculation
def calculate_path_distance(path, resolution):
    """
    Calculate the total distance of a path in meters.

    Parameters:
        path (list of tuples): List of (row, column) coordinates.
        resolution (float): Resolution of the grid map (meters per pixel).

    Returns:
        float: Total path distance in meters.
    """
    total_distance = 0.0
    for i in range(len(path) - 1):
        current = path[i]
        next_point = path[i + 1]
        pixel_distance = distance.euclidean(current, next_point)
        step_distance = resolution * pixel_distance
        total_distance += step_distance
    return total_distance

# Main Execution
if not os.path.exists(grid_file_path):
    print(f"File not found: {grid_file_path}")
else:
    try:
        # Load the CSV file
        grid_data = pd.read_csv(grid_file_path, header=None)
        if grid_data.empty:
            raise ValueError("The CSV file is empty or improperly formatted.")

        # Clean and convert grid data
        grid_data_cleaned = grid_data.fillna(0).applymap(lambda x: 0 if x not in [0, 1] else x)
        grid_array = grid_data_cleaned.to_numpy()

        # Compute the distance transform for weighted grid
        distance_map = distance_transform_edt(grid_array)
        max_distance = distance_map.max()
        weighted_grid = max_distance - distance_map  # Lower cost = farther from walls

        # Interactive Waypoint Selection
        print("Select 9 waypoints by clicking on the map.")
        fig, (ax_map, ax_legend) = plt.subplots(
            2, 1, figsize=(12, 14), gridspec_kw={"height_ratios": [10, 1]}
        )
        ax_map.imshow(grid_array, cmap='gray', origin='upper')
        ax_map.set_title("Click to Select 9 Waypoints")
        ax_map.set_xlabel("X")
        ax_map.set_ylabel("Y")
        plt.connect('button_press_event', onclick)  # Connect click handler
        plt.subplots_adjust(hspace=0.4)
        plt.show()

        # Check if 9 waypoints were selected
        if len(clicked_waypoints) != 9:
            raise ValueError("You must select exactly 9 waypoints.")

        # Pathfinding between waypoints
        adjusted_paths = []
        failed_adjusted_paths = []
        path_distances = []
        resolution = 0.1  # Example resolution: 0.05 meters per pixel

        for i in range(len(clicked_waypoints) - 1):
            start = clicked_waypoints[i]
            goal = clicked_waypoints[i + 1]
            path = astar_weighted(grid_array, weighted_grid, start, goal)
            if path:
                adjusted_paths.append(path)
                path_distance = calculate_path_distance(path, resolution)
                path_distances.append(path_distance)
                print(f"Path {i + 1} Distance: {path_distance:.2f} meters")
            else:
                failed_adjusted_paths.append((i + 1, start, goal))
                print(f"Failed to generate path between {start} and {goal}")

        # Total distance calculation
        total_distance = sum(path_distances)
        print(f"\nTotal Distance for All Paths: {total_distance:.2f} meters")

        # Visualization of Grid Map with Adjusted Paths
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green', 'cyan', 'blue', 'darkblue']
        fig, (ax_map, ax_legend) = plt.subplots(
            2, 1, figsize=(12, 14), gridspec_kw={"height_ratios": [10, 1]}
        )
        ax_map.imshow(grid_array, cmap='gray', origin='upper')

        # Plot the paths
        for idx, path in enumerate(adjusted_paths):
            path_x, path_y = zip(*path)
            ax_map.plot(path_y, path_x, color=colors[idx], label=f"Path {idx + 1} ({path_distances[idx]:.2f}m)")

        # Overlay waypoints
        for wp, coord in enumerate(clicked_waypoints, 1):
            ax_map.scatter(coord[1], coord[0], color='white', edgecolor='black', s=100)
            ax_map.text(
                coord[1],
                coord[0],
                str(wp),
                color='black',
                fontsize=10,
                ha='center',
                va='center',
                bbox=dict(boxstyle="circle", fc="white", ec="black", lw=1),
            )

        # Legend in the bottom subplot
        ax_legend.axis("off")
        ax_legend.legend(
            handles=ax_map.get_legend_handles_labels()[0],
            labels=ax_map.get_legend_handles_labels()[1],
            loc="center",
            ncol=4,
            fontsize="small",
        )

        # Add title and labels
        ax_map.set_title("Grid Map with Adjusted Paths")
        ax_map.set_xlabel("X")
        ax_map.set_ylabel("Y")
        plt.show()

    except ValueError as ve:
        print(f"Value error: {ve}")
    except Exception as e:
        print(f"Error loading or processing CSV file: {e}")