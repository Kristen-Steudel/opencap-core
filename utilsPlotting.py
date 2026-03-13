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

def plot_all_markers_all_coords_one(markers, filtered_all, time, marker_list, 
                                save_path=None):
    """
    Plot all markers with x, y, z coordinates on the same subplot.
    x in red, y in green, z in blue
    Dotted lines for pre-filtered data, solid lines for post-filtered data.
    """
    n_markers = len(marker_list)
    
    # Calculate grid dimensions
    n_cols = 3
    n_rows = int(np.ceil(n_markers / n_cols))
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    
    # Flatten axes array for easier indexing
    if n_markers == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = np.array(axes)
    else:
        axes = axes.flatten()
    
    # Colors for x, y, z
    colors = ['red', 'green', 'blue']
    coord_names = ['X', 'Y', 'Z']
    
    # Plot each marker
    for marker_idx, marker_name in enumerate(marker_list):
        ax = axes[marker_idx]
        
        # Plot each coordinate (x, y, z)
        for coord_idx in range(3):
            # Pre-filtered data (dotted line)
            ax.plot(time, markers[marker_name][:, coord_idx], 
                   linestyle=':', color=colors[coord_idx], 
                   alpha=0.6, linewidth=1,
                   label=f'{coord_names[coord_idx]} (pre-filter)')
            
            # Post-filtered data (solid line)
            ax.plot(time, filtered_all[marker_name][:, coord_idx], 
                   linestyle='-', color=colors[coord_idx], 
                   linewidth=2,
                   label=f'{coord_names[coord_idx]} (filtered)')
        
        ax.set_title(f'{marker_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (mm)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide any unused subplots
    for idx in range(n_markers, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    