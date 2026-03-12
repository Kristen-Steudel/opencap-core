# utilsPlotting.py
"""
Utility functions for plotting OpenPose marker trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_single_marker_xyz(marker_name, original, filtered, time, save_path=None):
    """
    Plot X, Y, Z for a single marker.
    
    Parameters:
    -----------
    marker_name : str
        Name of the marker
    original : numpy.ndarray
        Original trajectory (n_frames, 3)
    filtered : numpy.ndarray
        Filtered trajectory (n_frames, 3)
    time : numpy.ndarray
        Time vector
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle(f'{marker_name} Trajectory - Median Filtered', fontsize=16)
    
    coords = ['X', 'Y', 'Z']
    
    for i, coord in enumerate(coords):
        axes[i].plot(time, original[:, i], 'b-', alpha=0.5, 
                    linewidth=1, label='Original')
        axes[i].plot(time, filtered[:, i], 'g-', linewidth=2, label='Filtered')
        
        axes[i].set_ylabel(f'{coord} Position (mm)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Time (s)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_all_markers_by_coordinate(markers, filtered_all, time, marker_list, 
                                   coord_idx=0, save_path=None):
    """
    Plot all markers for a single coordinate (X, Y, or Z).
    
    Parameters:
    -----------
    markers : dict
        Original marker data
    filtered_all : dict
        Filtered marker data
    time : numpy.ndarray
        Time vector
    marker_list : list
        List of marker names to plot
    coord_idx : int
        0 for X, 1 for Y, 2 for Z
    save_path : str, optional
        Path to save figure
    """
    coord_names = ['X', 'Y', 'Z']
    coord_name = coord_names[coord_idx]
    
    n_markers = len(marker_list)
    n_cols = 4
    n_rows = int(np.ceil(n_markers / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows*3))
    fig.suptitle(f'All Markers - {coord_name} Coordinate (Median Filtered)', 
                 fontsize=16)
    axes = axes.flatten()
    
    for idx, marker_name in enumerate(marker_list):
        if marker_name in markers:
            axes[idx].plot(time, markers[marker_name][:, coord_idx], 'b-', 
                          alpha=0.4, linewidth=1, label='Original')
            axes[idx].plot(time, filtered_all[marker_name][:, coord_idx], 'g-', 
                          linewidth=2, label='Filtered')
            
            axes[idx].set_title(marker_name, fontsize=10)
            axes[idx].set_xlabel('Time (s)', fontsize=8)
            axes[idx].set_ylabel(f'{coord_name} (mm)', fontsize=8)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].tick_params(labelsize=8)
            
            if idx == 0:
                axes[idx].legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(marker_list), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_all_markers_all_coords(markers, filtered_all, time, marker_list, 
                                save_path=None):
    """
    Plot all markers, all coordinates in one large figure.
    """
    # For each coordinate
    for coord_idx, coord_name in enumerate(['X', 'Y', 'Z']):
        plot_all_markers_by_coordinate(
            markers, filtered_all, time, marker_list, 
            coord_idx=coord_idx,
            save_path=f'all_markers_{coord_name}_filtered.png' if save_path else None
        )