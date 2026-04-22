import os
import sys
import numpy as np
from scipy.signal import butter, filtfilt
import utilsDataman

# ============================================================================
# CONFIGURATION - ADJUST FOR EACH RUN
# ============================================================================
subject_num = 5
date = 'March_2'
session = '7'
trial_type = 'sprint'
filter_freq = 10  # Hz
coord_filter_freq = 10  # Hz
marker_filter_freq = 10  # Hz
angular_vel_filter_freq = 2  # Hz (for step detection)

# ============================================================================
# SETUP TRC FILE PATH
# ============================================================================

#trc_file_name = f'ID{subject_num}_S{session}_{trial_type}NoSync_LSTM.trc' # OR f'ID{subject_num}_S{session}_{trial_type}NoSync_LSTM.trc'
trc_file_name = 'ID5_S7_sprintNoSync_medFilt_LSTM.trc'
trc_file_path = os.path.join('G:\\Shared drives\\Stanford Football','AnalysisCompare','PostaugmentationMarkerFiles',trc_file_name)

# trc_file_path = os.path.join(
#     'G:\\Shared drives\\Stanford Football',
#     date,
#     f'subject{subject_num}',
#     'CleanedMarkerData',
#     'OpenPose_default',
#     '3-cameras',
#     'PostAugmentation_v0.2',
#     trc_file_name
# )

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    Apply a 4th order zero-lag Butterworth low-pass filter.

    Parameters:
    - data: array-like, the data to filter
    - cutoff: float, cutoff frequency in Hz
    - fs: float, sampling frequency in Hz
    - order: int, order of the filter (default 4)

    Returns:
    - y: array-like, filtered data
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def filter_trc_markers(trc_path, cutoff_freq=15):
    """
    Load a TRC file, apply low-pass filter to all marker coordinates, and save filtered version.

    Parameters:
    - trc_path: str, path to the input TRC file
    - cutoff_freq: float, cutoff frequency in Hz (default 15)
    """
    # Load the TRC file
    trc = utilsDataman.TRCFile(trc_path)

    # Get sampling frequency
    fs = trc.data_rate

    print(f"Filtering TRC file: {trc_path}")
    print(f"Sampling rate: {fs} Hz")
    print(f"Cutoff frequency: {cutoff_freq} Hz")

    # Filter each marker's x, y, z coordinates
    for marker in trc.marker_names:
        print(f"Filtering marker: {marker}")

        # Get original coordinates
        x = trc.marker(marker)[:, 0]
        y = trc.marker(marker)[:, 1]
        z = trc.marker(marker)[:, 2]

        # Apply filter
        x_filt = butter_lowpass_filter(x, cutoff_freq, fs)
        y_filt = butter_lowpass_filter(y, cutoff_freq, fs)
        z_filt = butter_lowpass_filter(z, cutoff_freq, fs)

        # Update the TRC data
        trc.data[marker + '_tx'] = x_filt
        trc.data[marker + '_ty'] = y_filt
        trc.data[marker + '_tz'] = z_filt

    # Create output path with suffix
    base, ext = os.path.splitext(trc_path)
    new_path = base + '_postaug_filt15Hz' + ext

    # Save the filtered TRC
    trc.write(new_path)
    print(f"Filtered TRC saved to: {new_path}")

if __name__ == "__main__":
    print(f"TRC file to filter: {trc_file_path}")

    if not os.path.exists(trc_file_path):
        print(f"Error: TRC file not found: {trc_file_path}")
        sys.exit(1)

    try:
        filter_trc_markers(trc_file_path)
    except Exception as e:
        print(f"Error processing TRC file: {e}")
        sys.exit(1)