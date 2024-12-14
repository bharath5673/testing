import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pandas as pd
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.nuscenes import NuScenes
import tqdm
import os


# # Define image path and NuScenes data path
imgs_path = 'mini/v1.0-mini'
nuscenes_data_path = 'mini/v1.0-mini/'
ver = 'v1.0-mini'


# imgs_path = r"C:/Users/shara/OneDrive/Desktop/Projects/NUSCENES/v1.0-trainval_meta"
# nuscenes_data_path = r"C:/Users/shara/OneDrive/Desktop/Projects/NUSCENES/v1.0-trainval_meta"
# ver = 'v1.0-trainval'


output_video_path = 'nuscenes_visual_odo_trajactory_tracking.mp4'
frame_width, frame_height = 1600, 900
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 output
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))





# Initialize NuScenes dataset
nusc = NuScenes(version=ver, dataroot=nuscenes_data_path, verbose=False)
# Initialize NuScenes CAN Bus API
nusc_can = NuScenesCanBus(dataroot='can_bus')



nuscenes_images = []
for sample in nusc.sample:
    cam_token = sample['data']['CAM_FRONT']
    cam_data = nusc.get('sample_data', cam_token)
    img_name = cam_data['filename']

    scene_token = sample['scene_token']
    scene = nusc.get('scene', scene_token)
    scene_name = scene['name']
    nuscenes_images.append(imgs_path+'/'+img_name)



# Camera intrinsic parameters
# k = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02],
#               [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02],
#               [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]], dtype=np.float32)


k = np.array([[1.2664172e+03, 0.0000000e+00, 8.1626703e+02],
             [0.0000000e+00, 1.2664172e+03, 4.9150708e+02],
             [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]], dtype=np.float32)


# Feature parameters
kMinNumFeature = 3000
traj = np.zeros((600, 600, 3), dtype=np.uint8)
x_loc = []
z_loc = []

# Feature tracking using LK optical flow
def featureTracking(image_ref, image_cur, px_ref):
    lk_params = dict(winSize=(21, 21),
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)
    st = st.reshape(st.shape[0])
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]
    return kp1, kp2

# Initialize features from the first two frames
def process_first_frames(first_frame, second_frame, k):
    det = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
    kp1 = det.detect(first_frame)
    kp1 = np.array([x.pt for x in kp1], dtype=np.float32)

    kp1, kp2 = featureTracking(first_frame, second_frame, kp1)
    E, mask = cv2.findEssentialMat(kp2, kp1, k, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, kp2, kp1, k)
    kp1 = kp2
    return kp1, R, t


def picture_in_picture(main_image, overlay_image, img_ratio=3, border_size=3, x_margin=30, y_offset_adjust=-100):
    """
    Overlay an image onto a main image with a white border.
    
    Args:
        main_image_path (str): Path to the main image.
        overlay_image_path (str): Path to the overlay image.
        img_ratio (int): The ratio to resize the overlay image height relative to the main image.
        border_size (int): Thickness of the white border around the overlay image.
        x_margin (int): Margin from the right edge of the main image.
        y_offset_adjust (int): Adjustment for vertical offset.

    Returns:
        np.ndarray: The resulting image with the overlay applied.
    """
    # Load images
    if main_image is None or overlay_image is None:
        raise FileNotFoundError("One or both images not found.")

    # Resize the overlay image to 1/img_ratio of the main image height
    new_height = main_image.shape[0] // img_ratio
    new_width = int(new_height * (overlay_image.shape[1] / overlay_image.shape[0]))
    overlay_resized = cv2.resize(overlay_image, (new_width, new_height))

    # Add a white border to the overlay image
    overlay_with_border = cv2.copyMakeBorder(
        overlay_resized,
        border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )

    # Determine overlay position
    x_offset = main_image.shape[1] - overlay_with_border.shape[1] - x_margin
    y_offset = (main_image.shape[0] // 2) - overlay_with_border.shape[0] + y_offset_adjust

    # Overlay the image
    main_image[y_offset:y_offset + overlay_with_border.shape[0], x_offset:x_offset + overlay_with_border.shape[1]] = overlay_with_border

    return main_image





im_list = nuscenes_images.copy()
# Save list to a text file
output_file = 'available_imgs_list.txt'
with open(output_file, 'w') as f:
    for item in im_list:
        if os.path.exists(item):
            f.write(f"{item}\n")
print(f"List saved to {output_file}")

input_file = 'available_imgs_list.txt'
# Read the file and store the paths in a list
with open(input_file, 'r') as f:
    images_path = [line.strip() for line in f.readlines()]



seq00_list = sorted(images_path.copy())
first_frame = cv2.imread(seq00_list[0], 0)
second_frame = cv2.imread(seq00_list[1], 0)
kp1, cur_R, cur_t = process_first_frames(first_frame, second_frame, k)
last_frame = second_frame

# List to store trajectory points
trajectory_points = [(cur_t[0], cur_t[2])]  # Initial point (x, z)

# Loop through frames
for i in range(2, len(seq00_list)):
    new_frame = cv2.imread(seq00_list[i], 0)
    new_frame_rgb = cv2.imread(seq00_list[i])
    kp1, kp2 = featureTracking(last_frame, new_frame, kp1)
    E, mask = cv2.findEssentialMat(kp2, kp1, k, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, kp2, kp1, k)
    change = np.mean(np.abs(kp2 - kp1))
    
    if change > 5:
        cur_t = cur_t + cur_R.dot(t)
        cur_R = R.dot(cur_R)

    if kp1.shape[0] < kMinNumFeature:
        det = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        kp2 = det.detect(new_frame)
        kp2 = np.array([x.pt for x in kp2], dtype=np.float32)

    kp1 = kp2
    last_frame = new_frame

    # Update trajectory location
    x, y, z = cur_t[0], cur_t[1], cur_t[2]
    x_loc.append(x)
    z_loc.append(z)
    
    # Store the new trajectory point in reverse order (growing the tail backwards)
    trajectory_points.insert(0, (x, z))  # Insert new point at the front (back of the vehicle)

    # Draw tracking points on the current frame (new_frame)
    for point in kp2:
        draw_x, draw_y = int(point[0]), int(point[1])
        cv2.circle(new_frame_rgb, (draw_x, draw_y), 2, (0, 0, 255), 3)  # Draw keypoints in red

    # Draw trajectory on the trajectory image (fixed center and growing tail)
    for pt in trajectory_points:
        draw_x, draw_y = int(pt[0]) + 300, int(-pt[1]) + 500
        cv2.circle(traj, (draw_x, draw_y), 1, (0, 0, 255), 1)  # Draw trajectory points (red)

    # # Display the images
    # cv2.imshow('Road facing camera', new_frame_rgb)
    # cv2.imshow('Trajectory', traj)

    final_img =  picture_in_picture(new_frame_rgb, cv2.resize(traj.copy(), (1000, 800))) 
    cv2.imshow('Road facing camera & Trajectory', final_img)
    out.write(final_img)

    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and Destroy
cv2.destroyAllWindows()
cv2.imwrite('map.png', traj)
out.release()

# Plot Result
plt.figure(figsize=(8, 8), dpi=100)
plt.title("X Z Trajectory")
plt.xlabel("Z")
plt.ylabel("X")
plt.plot(x_loc, z_loc, label="Trajectory")
plt.legend()
plt.show()
