import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# Dataset path
dataset_path = 'KITTI/Tracking/training/image_03/0007/'

output_video_path = 'kitti_visual_odo_trajactory_tracking.mp4'
frame_width, frame_height = 1242, 750
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 output
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))




# Camera intrinsic parameters
k = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02],
              [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02],
              [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]], dtype=np.float32)

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

# Main loop
seq00_list = glob.glob(dataset_path + "*.png")  # Assuming the images are in PNG format
seq00_list.sort()

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

    final_img =  cv2.vconcat([new_frame_rgb, cv2.resize(traj.copy(), (1242, 375))]) 
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
