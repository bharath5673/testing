import numpy as np
import cv2
import matplotlib.pyplot as plt

# Input video path
# input_video_path = 'high_way_steer_data_video.mp4'
input_video_path = 'kitti_2011_09_26_drive_0084_sync.mp4'
output_video_path = 'kitti_2011_09_26_drive_0084_sync_video_visual_odo_trajactory_tracking.mp4'


cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 output
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height*2))


fov = 90                                                # field of view
fov_rad = np.deg2rad(fov)                               # Convert FOV to radians
fx = fy = frame_width / (2 * np.tan(fov_rad / 2))       # Compute focal lengths based on FOV
cx = frame_width / 2                                    # Principal points at the image center
cy = frame_height / 2

# Construct the intrinsic matrix
k = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float32)


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

# Read the first two frames
ret, first_frame = cap.read()
if not ret:
    print("Failed to read the video.")
    cap.release()
    exit()

first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
ret, second_frame = cap.read()
second_frame_gray = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)

kp1, cur_R, cur_t = process_first_frames(first_frame_gray, second_frame_gray, k)
last_frame_gray = second_frame_gray

# List to store trajectory points
trajectory_points = [(cur_t[0], cur_t[2])]  # Initial point (x, z)

# Loop through video frames
while True:
    ret, new_frame = cap.read()
    if not ret:
        break

    new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    kp1, kp2 = featureTracking(last_frame_gray, new_frame_gray, kp1)
    E, mask = cv2.findEssentialMat(kp2, kp1, k, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, kp2, kp1, k)
    change = np.mean(np.abs(kp2 - kp1))
    
    if change > 5:
        cur_t = cur_t + cur_R.dot(t)
        cur_R = R.dot(cur_R)

    if kp1.shape[0] < kMinNumFeature:
        det = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        kp2 = det.detect(new_frame_gray)
        kp2 = np.array([x.pt for x in kp2], dtype=np.float32)

    kp1 = kp2
    last_frame_gray = new_frame_gray

    # Update trajectory location
    x, y, z = cur_t[0], cur_t[1], cur_t[2]
    x_loc.append(x)
    z_loc.append(z)
    
    # Store the new trajectory point
    trajectory_points.append((x, z))

    # Draw tracking points on the current frame
    for point in kp2:
        draw_x, draw_y = int(point[0]), int(point[1])
        cv2.circle(new_frame, (draw_x, draw_y), 2, (0, 0, 255), 3)  # Draw keypoints in red

    # Draw trajectory on the trajectory image
    for pt in trajectory_points:
        draw_x, draw_y = int(pt[0]) + 300, int(-pt[1]) + 500
        cv2.circle(traj, (draw_x, draw_y), 1, (0, 0, 255), 2)  # Draw trajectory points

    # Concatenate and show the frames
    final_img = cv2.vconcat([new_frame, cv2.resize(traj.copy(), (frame_width, frame_height))]) 
    cv2.imshow('Road facing camera & Trajectory', final_img)
    out.write(final_img)


    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and Destroy
cap.release()
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
