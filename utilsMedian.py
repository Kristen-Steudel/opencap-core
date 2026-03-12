# utilsMedian.py
"""
Utility functions for median filtering of OpenPose marker trajectories.
Used to smooth trajectories and detect/replace outliers before LSTM augmentation.
"""

import numpy as np

threshold_val = 1  # Default MAD threshold for outlier detection (similar to ~3 sigma for normal distribution)

def detect_outliers_mad(x, threshold=threshold_val):
    """
    Detect outliers using Median Absolute Deviation (MAD).
    
    MAD is more robust to outliers than standard deviation.
    
    Parameters:
    -----------
    x : array-like
        Signal to analyze
    threshold : float
        Number of MADs away from median to consider as outlier.
        Default threshold_val is recommended (similar to ~3 sigma for normal distribution)
    
    Returns:
    --------
    outlier_mask : numpy.ndarray (bool)
        True where outliers detected
    """
    x = np.asarray(x, dtype=float)
    
    # Calculate median
    median = np.nanmedian(x)
    
    # Calculate MAD
    mad = np.nanmedian(np.abs(x - median))
    
    # Avoid division by zero
    if mad == 0:
        return np.zeros(len(x), dtype=bool)
    
    # Modified z-score
    modified_z_scores = 0.6745 * (x - median) / mad
    
    # Mark outliers
    outlier_mask = np.abs(modified_z_scores) > threshold
    
    return outlier_mask


def median_window_filter_with_outlier_detection(x, window, threshold=threshold_val, replace_outliers=True):
    """
    Moving median filter with outlier detection and replacement.
    
    Parameters:
    -----------
    x : array-like
        Signal to be filtered (1D array)
    window : int
        Number of samples to use for median estimation.
        Will be adjusted to even number if odd.
    threshold : float
        MAD threshold for outlier detection (default threshold_val)
    replace_outliers : bool
        If True, replace outliers with filtered values.
        If False, just return filtered signal.
    
    Returns:
    --------
    y : numpy.ndarray
        Median filtered signal
    outlier_mask : numpy.ndarray (bool)
        Boolean array indicating outlier positions
    """
    x = np.asarray(x, dtype=float)
    
    # Ensure window is even
    if window % 2:
        window = window - 1
    
    win2 = int(window / 2)
    n = len(x)
    
    # Detect outliers FIRST (globally)
    outlier_mask = detect_outliers_mad(x, threshold)
    
    # Create a cleaned version with outliers set to NaN for median calculation
    x_clean = np.copy(x)
    x_clean[outlier_mask] = np.nan
    
    # Initialize output
    y = np.copy(x)
    
    # Apply median filter to central region (using cleaned data)
    for ii in range(win2, n - win2):
        idx = np.arange(ii - win2, ii + win2, dtype=int)
        y[ii] = np.nanmedian(x_clean[idx])
    
    # Handle edges with smaller windows
    for ii in range(win2):
        y[ii] = np.nanmedian(x_clean[:ii + win2 + 1])
    for ii in range(n - win2, n):
        y[ii] = np.nanmedian(x_clean[ii - win2:])
    
    # Replace only outliers if requested
    if replace_outliers:
        result = np.copy(x)
        result[outlier_mask] = y[outlier_mask]
        return result, outlier_mask
    else:
        return y, outlier_mask


def median_window_filter(x, window):
    """
    Original moving median filter (backward compatible).
    Now uses outlier detection internally.
    """
    filtered, _ = median_window_filter_with_outlier_detection(
        x, window, threshold=threshold_val, replace_outliers=True
    )
    return filtered


def median_filter_trajectory(marker_data, window=5, threshold=threshold_val, return_outliers=False):
    """
    Apply median filter to 3D marker trajectory with outlier detection.
    
    Parameters:
    -----------
    marker_data : numpy.ndarray
        Array of shape (n_frames, 3) containing XYZ coordinates
    window : int
        Window size for median filter
    threshold : float
        MAD threshold for outlier detection
    return_outliers : bool
        If True, also return outlier mask
    
    Returns:
    --------
    filtered_data : numpy.ndarray
        Smoothed trajectory (same shape as input)
    outlier_masks : dict (optional)
        Dictionary with keys 'x', 'y', 'z' containing boolean outlier masks
    """
    marker_data = np.asarray(marker_data)
    
    if marker_data.ndim != 2 or marker_data.shape[1] != 3:
        raise ValueError(f"Expected shape (n_frames, 3), got {marker_data.shape}")
    
    filtered_data = np.zeros_like(marker_data)
    outlier_masks = {}
    
    # Apply median filter to each coordinate
    for i, coord in enumerate(['x', 'y', 'z']):
        filtered_data[:, i], outlier_masks[coord] = \
            median_window_filter_with_outlier_detection(
                marker_data[:, i], window, threshold, replace_outliers=True
            )
    
    if return_outliers:
        return filtered_data, outlier_masks
    return filtered_data


def median_filter_all_markers(markers_dict, window=5, threshold=threshold_val, verbose=False):
    """
    Apply median filter to all markers in a dictionary with outlier reporting.
    
    Parameters:
    -----------
    markers_dict : dict
        Dictionary with marker names as keys and (n_frames, 3) arrays as values
    window : int
        Window size for median filter
    threshold : float
        MAD threshold for outlier detection
    verbose : bool
        If True, print outlier statistics
    
    Returns:
    --------
    filtered_markers : dict
        Dictionary with same structure, containing filtered trajectories
    outlier_report : dict
        Dictionary with outlier counts per marker
    """
    filtered_markers = {}
    outlier_report = {}
    
    for marker_name, trajectory in markers_dict.items():
        filtered, outliers = median_filter_trajectory(
            trajectory, window, threshold, return_outliers=True
        )
        filtered_markers[marker_name] = filtered
        
        # Count outliers
        total_outliers = sum(mask.sum() for mask in outliers.values())
        outlier_report[marker_name] = {
            'total': total_outliers,
            'x': outliers['x'].sum(),
            'y': outliers['y'].sum(),
            'z': outliers['z'].sum()
        }
        
        if verbose and total_outliers > 0:
            print(f"{marker_name}: {total_outliers} outliers detected "
                  f"(X:{outliers['x'].sum()}, Y:{outliers['y'].sum()}, Z:{outliers['z'].sum()})")
    
    return filtered_markers, outlier_report


# Test/demo code
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Testing outlier detection and filtering...")
    
    # Test 1: Simple signal with outliers
    signal = np.array([1, 2, 3, 100, 5, 6, 7, 8, -50, 10], dtype=float)
    filtered, outliers = median_window_filter_with_outlier_detection(signal, window=3)
    
    print(f"Original: {signal}")
    print(f"Filtered: {filtered}")
    print(f"Outliers at indices: {np.where(outliers)[0]}")
    print(f"Outlier values: {signal[outliers]}")
    
    # Test 2: Signal with NaN and outliers
    signal_nan = np.array([1, 2, np.nan, 4, 5, 200, 7, 8, 9, 10], dtype=float)
    filtered_nan, outliers_nan = median_window_filter_with_outlier_detection(signal_nan, window=5)
    
    print(f"\nOriginal (with NaN): {signal_nan}")
    print(f"Filtered (with NaN): {filtered_nan}")
    print(f"Outliers detected: {outliers_nan}")
    
    # Test 3: 3D trajectory with outliers
    print("\nTesting median_filter_trajectory...")
    np.random.seed(42)
    trajectory = np.random.randn(100, 3) * 0.1  # Small noise
    trajectory += np.linspace(0, 1, 100)[:, np.newaxis]  # Smooth trend
    
    # Add specific outliers
    trajectory[25, :] = [10, 10, 10]  # Spike
    trajectory[50, 0] = -5  # X outlier only
    trajectory[75, :] = [-8, -8, -8]  # Another spike
    
    filtered_trajectory, outlier_masks = median_filter_trajectory(
        trajectory, window=5, threshold=threshold_val, return_outliers=True
    )
    
    print(f"Outliers detected - X: {outlier_masks['x'].sum()}, "
          f"Y: {outlier_masks['y'].sum()}, Z: {outlier_masks['z'].sum()}")
    
    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    for i, coord in enumerate(['X', 'Y', 'Z']):
        # Plot original
        axes[i].plot(trajectory[:, i], 'o-', alpha=0.4, markersize=3, 
                    label='Original', color='lightblue')
        
        # Highlight outliers
        outlier_idx = np.where(outlier_masks[coord.lower()])[0]
        axes[i].scatter(outlier_idx, trajectory[outlier_idx, i], 
                       color='red', s=100, marker='x', 
                       label=f'Outliers ({len(outlier_idx)})', zorder=5)
        
        # Plot filtered
        axes[i].plot(filtered_trajectory[:, i], '-', linewidth=2, 
                    label='Filtered', color='darkblue')
        
        axes[i].set_ylabel(f'{coord}')
        axes[i].legend(loc='upper left')
        axes[i].grid(True, alpha=0.3)
    
    axes[0].set_title('Median Filter with Outlier Detection on 3D Trajectory')
    axes[2].set_xlabel('Frame')
    plt.tight_layout()
    plt.show()
    
    # Test 4: Multiple markers
    print("\nTesting median_filter_all_markers...")
    markers = {
        'RAnkle': trajectory,
        'LAnkle': np.random.randn(100, 3) * 0.1
    }
    
    filtered, report = median_filter_all_markers(markers, window=5, verbose=True)
    print("\nOutlier Report:")
    for marker, counts in report.items():
        print(f"  {marker}: {counts}")