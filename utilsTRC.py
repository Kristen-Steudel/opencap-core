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
