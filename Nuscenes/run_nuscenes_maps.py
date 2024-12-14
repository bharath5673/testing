import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
import matplotlib.pyplot as plt

# Initialize NuScenes
nuscenes_data_path = 'mini/v1.0-mini/'
ver = 'v1.0-mini'
nusc = NuScenes(version=ver, dataroot=nuscenes_data_path, verbose=False)


# Layer names for rendering the map
layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
camera_channel = 'CAM_FRONT'

# Loop through all samples in the dataset
for sample in nusc.sample:
    sample_token = sample['token']  # Get the token for each sample

    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    location = log['location']

    # Initialize NuScenesMap for a specific map
    maps_path = 'nuScenes-map-expansion-v1.3'
    nusc_map = NuScenesMap(dataroot=maps_path, map_name=location)

    # Render the map onto the camera image using NuScenesMap
    fig, ax = nusc_map.render_map_in_image(nusc, sample_token, layer_names=layer_names, camera_channel=camera_channel)
    
    # Convert the Matplotlib figure to a NumPy array
    fig.canvas.draw()
    overlayed_img = np.array(fig.canvas.renderer.buffer_rgba())
    
    # Crop the image using the bounding box
    x, y, w, h = 0, 547, 900, 506
    cropped_img = overlayed_img[y:y+h, x:x+w]

    # Extract ego vehicle position from the sample
    camera_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    ego_pose = nusc.get('ego_pose', camera_data['ego_pose_token'])
    ego_position = ego_pose['translation'][:2]  # (x, y) position on the map
    ego_yaw = np.arctan2(*ego_pose['rotation'][:2])  # Calculate yaw from quaternion

    # Define patch_box dynamically centered around the ego vehicle's position
    patch_box = (ego_position[0], ego_position[1], 150, 150)  # Width and height set to 150x150 meters
    patch_angle = np.degrees(ego_yaw)  # Convert yaw to degrees for rotation
    canvas_size = (500, 500)  # Pixel size of the output mask

    map_mask = nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)
    map_mask = map_mask[0].astype(np.uint8) * 255

    # Step 1: Create a rotation matrix based on the ego vehicle's yaw
    rotation_matrix = cv2.getRotationMatrix2D((map_mask.shape[1] // 2, map_mask.shape[0] // 2), -patch_angle, 1)
    
    # Step 2: Apply the rotation to the map
    rotated_map_mask = cv2.warpAffine(map_mask, rotation_matrix, (map_mask.shape[1], map_mask.shape[0]))

    # Step 3: Flip the map if necessary to align the vehicle’s front with the top
    rotated_map_mask = cv2.flip(rotated_map_mask, 1)  # Flip horizontally if necessary

    # Step 4: Show the final map (with car in the center and dynamically rotated)
    cv2.imshow('Rotated Map Centered Around Car', rotated_map_mask)
    cv2.imshow('Cropped Map with Camera Image Overlay', cropped_img)

    # Wait for a key press to move to the next image
    key = cv2.waitKey(1)  # Adjust the delay if needed for smoother display
    if key == 27:  # ESC key to exit early
        break

    plt.close(fig)  # Close Matplotlib figure to release memory

cv2.destroyAllWindows()  # Close all OpenCV windows after the loop ends
