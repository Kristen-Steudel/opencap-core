# Visualize the OpenPose Markers Trajectories
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utilsTRC
import utilsMedian

# Path to trc file
trc_file_path = r"G:\Shared drives\Stanford Football\March_2\subject2\MarkerData\OpenPose_default\3-cameras\PreAugmentation\ID2_S7_sprint.trc"

# Read the trc file
data = utilsTRC.trc_2_dict(trc_file_path)
# Extract time and markers
time = data['time']
markers = data['markers']

print("Time shape:", time.shape)
print("Markers keys:", markers.keys())

# Visualize the trajectories of a few markers
marker_list = ['Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'midHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel']

for marker_name in marker_list:
        if marker_name in markers:
            marker_data = markers[marker_name]
            time = data['time']
            
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            fig.suptitle(f'{marker_name} Trajectory', fontsize=16)
            
            # X trajectory
            axes[0].plot(time, marker_data[:, 0], 'b-', linewidth=2)
            axes[0].set_ylabel('X (mm)', fontsize=12)
            axes[0].set_title('X Position')
            axes[0].grid(True, alpha=0.3)
            
            # Y trajectory
            axes[1].plot(time, marker_data[:, 1], 'g-', linewidth=2)
            axes[1].set_ylabel('Y (mm)', fontsize=12)
            axes[1].set_title('Y Position')
            axes[1].grid(True, alpha=0.3)
            
            # Z trajectory
            axes[2].plot(time, marker_data[:, 2], 'r-', linewidth=2)
            axes[2].set_ylabel('Z (mm)', fontsize=12)
            axes[2].set_xlabel('Time (s)', fontsize=12)
            axes[2].set_title('Z Position')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()

# Turn on and off plot showing so not all these individual plots always show
#plt.show()

# Get outlier information
filtered, outliers = utilsMedian.median_filter_trajectory(markers['LBigToe'], window=5, return_outliers=True)
print(f"Frames with X outliers: {np.where(outliers['x'])[0]}")
plt.figure(figsize=(10, 4))
plt.plot(time, markers['LBigToe'][:, 0], 'b-', label='Original X')
plt.plot(time, filtered[:, 0], 'r-', label='Filtered X', linewidth=2)
plt.scatter(time[outliers['x']], markers['LBigToe'][outliers['x'], 0], color='red', s=100, marker='x', label='Outliers')
plt.title('LBigToe X Trajectory with Outliers')
plt.xlabel('Time (s)')
plt.ylabel('X Position (mm)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()