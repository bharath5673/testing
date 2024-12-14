import os
import glob
import numpy as np
import cv2

import folium
from pyproj import Transformer
import matplotlib.pyplot as plt
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time



# Define directories
oxts_dir = '/home/bharath/Documents/Projects/KITTI/raw data/2011_09_26_drive_0084_sync/2011_09_26/2011_09_26_drive_0084_sync/oxts/data/'
image_dir = '/home/bharath/Documents/Projects/KITTI/raw data/2011_09_26_drive_0084_sync/2011_09_26/2011_09_26_drive_0084_sync/image_02/data'

# Get all images and IMU files, then sort them
image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
oxts_files = sorted(glob.glob(os.path.join(oxts_dir, '*.txt')))

# Load IMU data
imu_data = []
for file in oxts_files:
    with open(file, 'r') as f:
        data = list(map(float, f.readline().strip().split()))
        imu_data.append({
            'lat': data[0], 'lon': data[1], 'alt': data[2],
            'roll': data[3], 'pitch': data[4], 'yaw': data[5],
            'velocity': data[6:9], 'acceleration': data[9:12],
            'angular_velocity': data[12:15]
        })

# Check for mismatched file lengths
if len(image_files) != len(imu_data):
    print("Mismatch between the number of images and IMU data entries.")
    exit()






# Ensure map_temp folder exists
output_folder = "map_temp"
os.makedirs(output_folder, exist_ok=True)

# Define a projection based on nuScenes map's coordinate system (Adjust UTM zone)
utm_zone = 33  # Adjust based on the log location
proj_cartesian = f"+proj=utm +zone={utm_zone} +datum=WGS84"
proj_latlong = "+proj=longlat +datum=WGS84"
transformer = Transformer.from_crs(proj_cartesian, proj_latlong)


# Setup Selenium WebDriver (ensure ChromeDriver is installed)
options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)



# Initialize video writer
output_video_path = 'run_kitti_gps_output_map_pip.mp4'
fps = 30  # You can change this to match the FPS of your input video or set it manually
frame_width, frame_height = 1242, 375  # Change this to match your frame resolution
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for .mp4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))




map_center = [imu_data[0]['lat'], imu_data[0]['lon']] if imu_data else [1.0, 104.0]  
zoom_start = 22
folium_map = folium.Map(
    location=map_center, 
    tiles="OpenStreetMap", 
    zoom_start=zoom_start,
    control_scale=False,  # Disables scale control
    zoom_control=False    # Disables the zoom control (plus/minus buttons)
)



# # Initialize the marker to represent the moving object (e.g., a car)
# marker = folium.Marker(location=map_center, popup="Vehicle Position")
# marker.add_to(folium_map)

# Add a blue dot for the current position
marker =  folium.CircleMarker(location=map_center, radius=15, color="blue", fill=True, fill_color="blue", fill_opacity=0.8)
marker.add_to(folium_map)



# Create a path for the trajectory
trajectory_path = folium.PolyLine(locations=[(imu['lat'], imu['lon']) for imu in imu_data], color="blue", weight=5.5, opacity=0.1)
trajectory_path.add_to(folium_map)


# Save the initial map
html_path = os.path.join(output_folder, "map.html")
image_path = os.path.join(output_folder, "map_image.png")
folium_map.save(html_path)
# Render the HTML file to an image using Selenium (to simulate real-time updates)
driver.get(f"file://{os.path.abspath(html_path)}")



def picture_in_picture(main_image, overlay_image, img_ratio=4, border_size=3, x_margin=30, y_offset_adjust=-50):
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
    # print(overlay_image.shape)
    overlay_image = cv2.resize(overlay_image, (800, 300))
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





# Single loop to process images and IMU data
for img_file, imu in zip(image_files, imu_data):
    # Load the image
    image = cv2.imread(img_file)

    if image is None:
        print(f"Error loading image: {img_file}")
        continue

    # IMU data text
    text = (
        f"Lat: {imu['lat']:.6f}, Lon: {imu['lon']:.6f}\n"
        f"Alt: {imu['alt']:.2f}\n"
        f"Roll: {imu['roll']:.3f}, Pitch: {imu['pitch']:.3f}, Yaw: {imu['yaw']:.3f}"
    )
    

    ## map frame
    latitude, longitude = imu['lat'], imu['lon']
    # Update the marker's position
    marker.location = [latitude, longitude]
    # Recenter the map to the new position (simulating movement)
    folium_map.location = [latitude, longitude]
    # Save the updated map
    folium_map.save(html_path)
    
    # Render the updated map as an image
    driver.get(f"file://{os.path.abspath(html_path)}")
    driver.save_screenshot(image_path)        
    img = cv2.imread(image_path)
    
    ## p-i-p
    image = picture_in_picture(image, img)

    # Overlay the IMU data text on the image
    y0, dy = 30, 30
    for j, line in enumerate(text.split('\n')):
        y = y0 + j * dy
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    ## Display the image
    # cv2.imshow('map', img)
    cv2.imshow('Image with IMU Data', image)
    out.write(image)



    # Wait for a key press or automatically proceed after 500 ms
    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit early
        break

out.release()
cv2.destroyAllWindows()