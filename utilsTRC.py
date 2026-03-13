"""
Simple adapter providing trc_2_dict for compatibility with scripts
that expect a small API for reading TRC files.
"""
import os
import numpy as np
from utilsDataman import TRCFile

def trc_2_dict(trc_path):
    """Read a TRC file and return a dict with 'time' and 'markers'.

    Returns:
        {
            'time': 1D numpy array of time stamps,
            'markers': { marker_name: (n_frames x 3) numpy array }
        }
    """
    if not os.path.exists(trc_path):
        raise FileNotFoundError(f"TRC file not found: {trc_path}")

    trc = TRCFile(trc_path)
    out = {}
    out['time'] = trc.time
    markers = {}
    for name in trc.marker_names:
        markers[name] = trc.marker(name)
    out['markers'] = markers
    return out

def dict_2_trc(trc_file_path, data, time, output_path):
    """
    Write marker data from dictionary format back to TRC file.
    
    Parameters:
    -----------
    trc_file_path : str
        Path to original TRC file (to copy header information)
    data : dict
        Dictionary containing 'time' and 'markers' keys
    output_path : str
        Path where the new TRC file will be saved
    """
    # Read original file to get header information
    with open(trc_file_path, 'r') as f:
        lines = f.readlines()
    
    # Extract header lines (first 5 lines typically)
    header_lines = lines[:5]
    
    # Get data
    time = time
    # Get data
    markers = data  # data is already the markers dictionary
    marker_list = list(markers.keys())
    
    n_frames = len(time)
    n_markers = len(marker_list)
    
    # Update header line 4 with correct marker count and names
    # Format: Frame#  Time  marker1    marker1    marker1    marker2    marker2    marker2 ...
    marker_names_line = "Frame#\tTime\t" + "\t\t\t".join(marker_list) + "\n"
    header_lines[3] = marker_names_line
    
    # Update header line 5 with X, Y, Z labels
    xyz_line = "\t\t" + "\t".join(["X" + str(i+1) + "\tY" + str(i+1) + "\tZ" + str(i+1) 
                                    for i in range(n_markers)]) + "\n"
    header_lines[4] = xyz_line
    
    # Update header line 3 with frame info
    # Format: DataRate CameraRate NumFrames NumMarkers Units OrigDataRate OrigDataStartFrame OrigNumFrames
    header_parts = header_lines[2].strip().split('\t')
    if len(header_parts) >= 3:
        header_parts[2] = str(n_frames)  # Update NumFrames
        header_parts[3] = str(n_markers)  # Update NumMarkers
        header_lines[2] = '\t'.join(header_parts) + '\n'
    
    # Write to new file
    with open(output_path, 'w') as f:
        # Write header
        f.writelines(header_lines)
        
        # Write data
        for frame_idx in range(n_frames):
            # Frame number and time
            line = f"{frame_idx + 1}\t{time[frame_idx]:.6f}\t"
            
            # Add marker coordinates
            coords = []
            for marker_name in marker_list:
                marker_data = markers[marker_name][frame_idx, :]
                coords.extend([f"{marker_data[0]:.6f}", 
                              f"{marker_data[1]:.6f}", 
                              f"{marker_data[2]:.6f}"])
            
            line += "\t".join(coords) + "\n"
            f.write(line)
    
    print(f"Successfully saved filtered TRC file to: {output_path}")