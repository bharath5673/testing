**Document for KITTI Visual Odometry and Trajectory Tracking**

---

### Purpose
This Python script is designed to process the KITTI dataset for visual odometry tasks. It calculates the trajectory of a vehicle using feature tracking and camera motion estimation. The script processes sequential images from the KITTI dataset to extract camera motion and visualize the vehicle's trajectory over time.

---

### Key Features
1. **Feature Tracking**:
   - Uses Lucas-Kanade optical flow to track keypoints between consecutive frames.

2. **Camera Motion Estimation**:
   - Computes the essential matrix and recovers the relative pose (rotation and translation) between frames.

3. **Trajectory Visualization**:
   - Displays the vehicle's trajectory on a 2D plane.
   - Combines trajectory visualization with image frames for real-time feedback.

4. **Output Generation**:
   - Saves a video file (`kitti_visual_odo_trajactory_tracking.mp4`) that shows image frames and the corresponding trajectory.
   - Exports the trajectory as a PNG image (`map.png`) and plots X-Z trajectory using Matplotlib.

---

### Input Requirements
1. **Dataset**:
   - The script expects images from the KITTI dataset stored in the path specified by `dataset_path`.
   - Images should be named sequentially and in PNG format.

2. **Camera Intrinsic Parameters**:
   - The camera matrix `k` is hardcoded, matching KITTI's calibration file.

---

### Main Components

#### 1. **Feature Tracking**
   The `featureTracking` function:
   - Tracks features between two images using the Lucas-Kanade optical flow method.
   - Filters out points with low confidence using a status flag.

#### 2. **Pose Estimation**
   - Computes the essential matrix using `cv2.findEssentialMat`.
   - Recovers relative rotation (`R`) and translation (`t`) using `cv2.recoverPose`.

#### 3. **Trajectory Update**
   - Updates the vehicle's pose by chaining relative transformations (rotation and translation).
   - Accumulates trajectory points over time.

#### 4. **Visualization**
   - Draws tracked points on each frame for debugging.
   - Plots trajectory points on a 2D map, ensuring a fixed reference frame for the path.

#### 5. **Dynamic Feature Reinitialization**
   - If the number of tracked features drops below a threshold (`kMinNumFeature`), new features are detected.

---

### Execution Steps
1. Place the KITTI dataset images in the specified `dataset_path`.
2. Adjust the `dataset_path`, `frame_width`, and `frame_height` variables if necessary.
3. Run the script using Python.
4. Press `q` to exit the visualization prematurely.

---

### Outputs
1. **Video File**:
   - Combines frame-wise visualization with trajectory tracking.
   - Stored as `kitti_visual_odo_trajactory_tracking.mp4`.

2. **Static Trajectory Map**:
   - A 2D visualization of the trajectory saved as `map.png`.

3. **Trajectory Plot**:
   - Displays the X-Z trajectory using Matplotlib.

---

### Dependencies
- Python 3.x
- OpenCV
- Matplotlib
- Numpy

---

### Enhancements & Notes
1. **Error Handling**:
   - The script assumes the dataset is complete and sorted; ensure correct dataset preparation.

2. **Performance**:
   - Uses FAST feature detector; replace with alternative feature detectors if higher accuracy is required.

3. **Customization**:
   - Parameters like `kMinNumFeature`, `lk_params`, and `traj` dimensions can be adjusted for different datasets or resolutions.

4. **Future Work**:
   - Add 3D trajectory visualization.
   - Extend support for stereo visual odometry.

---

**Author**: Bharath

**Date**: December 2024

