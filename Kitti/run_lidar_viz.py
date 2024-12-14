import open3d as o3d
import numpy as np
import glob
import os
import time

# Function to load PCD files
def load_pcd(file_path):
    pcd = np.fromfile(file_path, dtype=np.float32)
    points = pcd.reshape(-1, 4)[:, :3]  # Keep only x, y, z for Kitti
    return points

# Set the directory containing the PCD files
input_dir =  "KITTI/Tracking/training/velodyne/0001/"
# input_dir = 'NuScenes/NuscenesData/mini/v1.0-mini/samples/LIDAR_TOP/'
pcd_files = glob.glob(os.path.join(input_dir, '*.bin'))
pcd_files.sort()

# Load point clouds into a list for sequential viewing
point_clouds = [load_pcd(pcd_file) for pcd_file in pcd_files]

# Downsample all point clouds for performance if needed
def downsample(points, target_points=1000):
    current_points = len(points)
    if current_points > target_points:
        nth_point = max(1, int(current_points / target_points))
        return points[::nth_point]
    return points

# Initialize Open3D visualizer with a specific window size
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(width=1024, height=768)  # Set your preferred window size here

# Initialize the first point cloud for display
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(downsample(point_clouds[0]))
vis.add_geometry(pcd)

# Access view control and render options
ctr = vis.get_view_control()
opt = vis.get_render_option()
opt.point_size = 2.0  # Adjust as needed

# Frame update function to update the point cloud in the visualizer
def update_vis(vis):
    for points in point_clouds:
        # Update point cloud data
        pcd.points = o3d.utility.Vector3dVector(downsample(points))
        vis.update_geometry(pcd)

        # Update the visualizer
        vis.get_render_option().background_color = [0, 0, 0]
        vis.poll_events()
        vis.update_renderer()
        
        time.sleep(0.01)  # Adjust for desired frame rate (e.g., 10 FPS)
    return False

# Set up key callback to quit visualization
vis.register_key_callback(ord("Q"), lambda vis: vis.close())

# Run the animation loop
update_vis(vis)  # Start the update function
vis.run()        # Keep the window open until closed
vis.destroy_window()
