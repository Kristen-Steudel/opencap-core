r"""
Plot x, y, z positions of PreAugmentation markers from TRC files.

Filter, spline-correct, and save cleaned TRC files. LBigToe/LSmallToe and RBigToe/RSmallToe
are grouped and edited together by default.

Interactive workflow:
1. Median filter: after quality check, preview original vs filtered; 'a' accept, 'n' reject
2. Overview: click subplot to select marker/group; 'u' undo last, 'U' undo all; 'd' done
3. Single-marker view: click curve for anchor points; 'p' toggle preview; 'a' apply; 'd' done
4. Save to Cleaned/<name>_<timestamp>.trc; run IK on cleaned TRCs for dynamics/ACL metrics

Usage:
  python clean_preaugment_markers.py --input-root "D:\MyDrive\Football\OpenCap"   # interactive review
  python clean_preaugment_markers.py --input-root "D:\MyDrive\Football\OpenCap" --no-review  # save figures only

Notes:
- This version is designed for **local** OpenCap-style folder trees (e.g., session folders containing
  `MarkerData/PreAugmentation/*.trc`). It does **not** require OpenCap cloud access.
- It operates on the **full trial time vector** (no ROI trimming).
"""

import csv
import os
import re
import sys
import time
from datetime import datetime
import argparse
import numpy as np
import matplotlib

# Import pyplot with a robust fallback when Qt is missing.
# If a Qt backend is configured but no Qt bindings are installed, we switch to Agg.
try:
    import matplotlib.pyplot as plt
except ImportError as e:
    msg = str(e).lower()
    if 'qt binding' in msg or 'qt5' in msg:
        try:
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception:
            raise
    else:
        raise

try:
    from scipy.interpolate import CubicSpline
except ImportError:
    CubicSpline = None

# Ensure script dir (Data/MATLAB) is first for local imports; add repo root for utilsTRC
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..'))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

# # REPLACE, USER INPUT BASE DIRECTORY
# from extract_acl_metrics import BASE_DIR, SESSION_IDS, PARTICIPANT_IDS

# # DELETE SECTION, PLOT SECTION I AM INTERESTED IN MY DATA, MY MARKERS
# from plot_kinematics_all_trials import (
#     MANUAL_ROI_TIMES,
#     SESSION_ID_TO_MANUAL_KEY,
#     SESSION_EXCLUDE_PREFIX,
#     TRIAL_TYPE_RC,
#     TRIAL_TYPE_SDJ,
#     discover_kinematics_trials,
#     _trial_type,
# )

try:
    from utilsTRC import TRCFile
except ImportError:
    TRCFile = None

# --- Input parameters ---
# CHANGE OUTPUT DIR
OUT_DIR = os.path.join('PreAugment_QC', 'Plots', 'PreAugment_markers_by_participant')  # where to save PNGs and edit log CSV
########################

# Base dir for discovering local PreAugmentation TRCs.
# Set PREAUGMENT_BASE_DIR env var or pass --input-root.
# Cleaned TRCs save next to originals in `.../MarkerData/PreAugmentation/Cleaned/`.
_OPENCAP_CORE_DATA = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..', '..', 'opencap-core', 'Examples', 'Data'))
if os.environ.get('PREAUGMENT_BASE_DIR'):
    PREAUGMENT_BASE_DIR_DEFAULT = os.environ.get('PREAUGMENT_BASE_DIR')  # env override
elif os.path.isdir(_OPENCAP_CORE_DATA):
    PREAUGMENT_BASE_DIR_DEFAULT = _OPENCAP_CORE_DATA  # opencap-core/Examples/Data
else:
    PREAUGMENT_BASE_DIR_DEFAULT = os.getcwd()

# Exclude markers whose names end with this (case-insensitive) from review
MARKER_EXCLUDE_SUFFIX = '_study'

# Toe marker pairs grouped in one subplot; edited together by default
TOE_GROUPS = (('LBigToe', 'LSmallToe'), ('RBigToe', 'RSmallToe'))

# Multi-Factor Quality Check: rigid marker pairs (fixed distance) for stretching/compression detection.
# Pairs span segments (e.g. humerus, foot) that have small normal variation but should not deviate greatly.
QUALITY_CHECK_MARKER_PAIRS = [  # (marker1, marker2) pairs; distance checked against QUALITY_TOLERANCE_M
    ('LBigToe', 'LSmallToe'),
    ('RBigToe', 'RSmallToe'),
    ('LBigToe', 'LHeel'),
    ('RBigToe', 'RHeel'),
    ('LSmallToe', 'LHeel'),
    ('RSmallToe', 'RHeel'),
    ('LBigToe', 'LAnkle'),
    ('RBigToe', 'RAnkle'),
    ('LSmallToe', 'LAnkle'),
    ('RSmallToe', 'RAnkle'),
    ('LWrist', 'LElbow'),
    ('RWrist', 'RElbow'),
    ('LElbow', 'LShoulder'),
    ('RElbow', 'RShoulder'),
]
QUALITY_TOLERANCE_M = 0.080 # was 0.050  # max allowed deviation (m) from median pair distance (e.g. 0.050 m = ~2 in)
# Velocity threshold (m/s): flags frames exceeding this speed as likely artifact.
# Literature: elite kicking foot velocities 5.2–18.3 m/s (Corcoran et al. 2024, doi:10.3390/sports12030074);
#   max toe velocity ~20 m/s (Barfield et al. 2002, doi:10.52082/jssm.1.3.72).
# Threshold above max reported values to allow legitimate fast movement while catching artifacts.
QUALITY_VEL_THRESHOLD_M_S = 25.0 # Was 20.0

# Sliding median filter: tight window to remove only very short (1-2 frame) artifacts.
MEDIAN_FILTER_WINDOW_FRAMES = 7 # 10 = 5 each side of center. Uses np.nanmedian (handles NaN).

# Spline behavior:
# Clamped boundary conditions (matching estimated endpoint slopes) can cause large
# oscillations/overshoot ("big sine waves") on noisy or long trajectories.
# A natural cubic spline is typically smoother for pre-augmentation cleaning.
SPLINE_USE_CLAMPED_BC = False


def check_data_quality(data, marker_names, marker_pairs, pair_names=None, tolerance=QUALITY_TOLERANCE_M, vel_threshold=QUALITY_VEL_THRESHOLD_M_S, data_rate_hz=None):
    """
    Multi-Factor Quality Check: velocity + rigid body constraints.

    Velocity: flags frames where each marker's speed exceeds threshold. Uses central difference for
    interior frames, forward difference for first frame, backward difference for last frame.
    Rigid: flags frames where marker-pair distance deviates from median (stretching/compression).

    Args:
        data: np.array (n_frames, n_markers, 3) in meters
        marker_names: list of marker names matching data columns
        marker_pairs: list of (i, j) indices for rigid pairs
        pair_names: optional list of (name1, name2) matching marker_pairs order; used for per-pair rigid failures
        tolerance: max allowed deviation (m) from median pair distance
        vel_threshold: max allowed speed (m/s) per marker
        data_rate_hz: sampling rate (Hz); required for m/s conversion. If None, assumes 100 Hz (common mocap default).
    Returns:
        is_bad: bool array (n_frames,), True where quality fails
        vel_scores: float array (n_frames,) - mean velocity across markers (for summary)
        rigid_scores: float array (n_frames,)
        is_bad_vel: bool array (n_frames,) - any marker velocity failed
        is_bad_rigid: bool array (n_frames,) - any rigid failure
        vel_failures_by_marker: list of (name, is_bad_array) for each marker
        rigid_failures_by_pair: list of (name1, name2, is_bad_array) for each pair (if pair_names given)
    """
    n_frames, n_markers = data.shape[0], data.shape[1]
    # Velocity (m/s) via finite differences: central for interior, forward for first, backward for last
    # Convert displacement to m/s: vel = displacement / dt (dt = 1/rate)
    rate = float(data_rate_hz) if data_rate_hz is not None else 100.0  # 100 Hz fallback if unknown
    dt = 1.0 / rate if rate > 0 else 1.0
    vel_failures_by_marker = []
    vel_per_marker = []
    is_bad_vel_any = np.zeros(n_frames, dtype=bool)
    for j in range(n_markers):
        pos = data[:, j, :]  # (n_frames, 3)
        vel_j = np.full(n_frames, np.nan, dtype=float)
        if n_frames >= 2:
            vel_j[0] = np.linalg.norm(pos[1] - pos[0], axis=-1) / dt   # forward: dx/dt
            vel_j[-1] = np.linalg.norm(pos[-1] - pos[-2], axis=-1) / dt  # backward: dx/dt
        if n_frames >= 3:
            central = np.linalg.norm(pos[2:] - pos[:-2], axis=-1) / (2.0 * dt)  # central: (x[i+1]-x[i-1])/(2*dt)
            vel_j[1:-1] = central
        vel_per_marker.append(vel_j)
        is_bad_j = np.isfinite(vel_j) & (vel_j > vel_threshold)
        is_bad_vel_any = is_bad_vel_any | is_bad_j
        name = marker_names[j] if j < len(marker_names) else f'm{j}'
        vel_failures_by_marker.append((name, is_bad_j))
    vel_scores = np.nanmean(vel_per_marker, axis=0) if vel_per_marker else np.zeros(n_frames)

    rigid_scores = np.zeros(n_frames)
    rigid_failures_by_pair = []
    if marker_pairs:
        for k, (m1, m2) in enumerate(marker_pairs):
            p1, p2 = data[:, m1, :], data[:, m2, :]
            pair_dist = np.linalg.norm(p1 - p2, axis=1)
            valid = ~np.isnan(pair_dist)
            is_bad_pair = np.zeros(n_frames, dtype=bool)
            if len(valid) > 0:
                ideal_dist = np.nanmedian(pair_dist)
                deviation = np.abs(pair_dist - ideal_dist)
                failed = np.where(valid, (deviation > tolerance).astype(float), 0)
                rigid_scores += failed
                is_bad_pair = failed > 0
            if pair_names is not None and k < len(pair_names):
                n1, n2 = pair_names[k]
                rigid_failures_by_pair.append((n1, n2, is_bad_pair))

    is_bad_vel = is_bad_vel_any
    is_bad_rigid = rigid_scores > 0
    is_bad = is_bad_vel | is_bad_rigid
    return is_bad, vel_scores, rigid_scores, is_bad_vel, is_bad_rigid, vel_failures_by_marker, rigid_failures_by_pair


def _markers_to_data_array(markers, marker_names):
    """Build (n_frames, n_markers, 3) array from markers dict. Uses NaN for missing."""
    names = [m for m in marker_names if m in markers]
    if not names:
        return None
    n_frames = len(markers[names[0]]['x'])
    data = np.full((n_frames, len(names), 3), np.nan)
    for j, m in enumerate(names):
        d = markers[m]
        data[:, j, 0] = d['x']
        data[:, j, 1] = d['y']
        data[:, j, 2] = d['z']
    return data, names


def _get_marker_pair_indices(marker_names, pairs_by_name):
    """Resolve (name1, name2) pairs to (i, j) indices. Skips pairs not in marker_names."""
    idx_map = {m: i for i, m in enumerate(marker_names)}
    out = []
    for n1, n2 in pairs_by_name:
        if n1 in idx_map and n2 in idx_map:
            out.append((idx_map[n1], idx_map[n2]))
    return out


def _get_marker_pairs_with_names(marker_names, pairs_by_name):
    """Return (indices, names) for pairs that exist. indices = [(i,j),...], names = [(n1,n2),...]."""
    idx_map = {m: i for i, m in enumerate(marker_names)}
    indices, names = [], []
    for n1, n2 in pairs_by_name:
        if n1 in idx_map and n2 in idx_map:
            indices.append((idx_map[n1], idx_map[n2]))
            names.append((n1, n2))
    return indices, names


def _find_contiguous_regions(mask):
    """Return list of (start_idx, end_idx) for contiguous True regions in mask."""
    bad = np.asarray(mask, dtype=float)
    edges = np.diff(np.concatenate([[0], bad, [0]]))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return list(zip(starts, ends))


def _draw_quality_bad_regions(ax, t_pct, is_bad_vel, is_bad_rigid):
    """
    Draw horizontal bars above trajectory: red=velocity fail, blue=rigid fail, purple=both.
    Draw in order: both (purple), velocity-only (red), rigid-only (blue).
    """
    if is_bad_vel is None and is_bad_rigid is None:
        return
    n = len(t_pct)
    if n == 0:
        return

    def draw_regions(mask, color, alpha=0.85):
        if mask is None or not np.any(mask):
            return
        for s, e in _find_contiguous_regions(mask):
            if s < n and e <= n:
                t_lo = t_pct[min(s, n - 1)]
                t_hi = t_pct[min(e - 1, n - 1)] if e > 0 else t_lo
                ax.axvspan(t_lo, t_hi, ymin=0.92, ymax=1.0, facecolor=color, alpha=alpha, zorder=0)

    # Both fail → purple (draw first so it's underneath)
    both = np.logical_and(is_bad_vel, is_bad_rigid) if (is_bad_vel is not None and is_bad_rigid is not None) else None
    draw_regions(both, '#cc66cc')
    # Velocity only → red
    vel_only = np.logical_and(is_bad_vel, ~is_bad_rigid) if (is_bad_vel is not None and is_bad_rigid is not None) else is_bad_vel
    draw_regions(vel_only, '#ff6666')
    # Rigid only → blue
    rigid_only = np.logical_and(~is_bad_vel, is_bad_rigid) if (is_bad_vel is not None and is_bad_rigid is not None) else is_bad_rigid
    draw_regions(rigid_only, '#6666ff')


# Layout: maximize window, square subplots (set_box_aspect(1)), no per-subplot legends
CELL_SIZE_IN = 5.0   # subplot size in inches (5 columns × fewer rows) was 3.5
# Wider single-marker spline editor figure (longer x-axis for easier blip picking).
SPLINE_EDITOR_WIDTH_PER_AXIS_IN = 8.0
SPLINE_EDITOR_HEIGHT_IN = 4.0


def _maximize_figure(fig):
    """Maximize the figure window. Uses timer so window exists when called."""
    def _do_maximize():
        try:
            plt.figure(fig.number)
            mgr = plt.get_current_fig_manager()
            if mgr is None:
                mgr = fig.canvas.manager
            if mgr is None:
                return
            backend = plt.get_backend().lower()
            if hasattr(mgr, 'window') and mgr.window is not None:
                win = mgr.window
                if 'qt' in backend and hasattr(win, 'showMaximized'):
                    win.showMaximized()
                elif 'tk' in backend and hasattr(win, 'state'):
                    win.state('zoomed')
                elif hasattr(win, 'showMaximized'):
                    win.showMaximized()
                elif hasattr(win, 'state') and callable(getattr(win, 'state', None)):
                    win.state('zoomed')
            elif 'wx' in backend and hasattr(mgr, 'frame'):
                mgr.frame.Maximize(True)
            elif hasattr(mgr, 'full_screen_toggle'):
                mgr.full_screen_toggle()
        except Exception:
            pass

    # Delay so window exists when show() starts event loop
    timer = fig.canvas.new_timer(interval=350)
    timer.single_shot = True
    timer.add_callback(_do_maximize)
    timer.start()


def _try_focus_figure(fig):
    """Try to focus the TkAgg figure so keypress events reach mpl_connect handlers."""
    def _do_focus():
        try:
            # TkAgg exposes the underlying tk widget
            if hasattr(fig.canvas, 'get_tk_widget'):
                w = fig.canvas.get_tk_widget()
                top = None
                try:
                    top = w.winfo_toplevel()
                except Exception:
                    top = None
                # Ensure the app window is raised as well.
                if top is not None:
                    try:
                        top.deiconify()
                    except Exception:
                        pass
                    try:
                        top.lift()
                    except Exception:
                        pass
                try:
                    w.focus_set()
                except Exception:
                    pass
                try:
                    w.focus_force()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            mgr = fig.canvas.manager
            if mgr is not None and hasattr(mgr, 'window') and mgr.window is not None:
                win = mgr.window
                # Some backends expose focus_force/lift
                for method_name in ('lift', 'focus_force', 'focus'):
                    if hasattr(win, method_name):
                        try:
                            getattr(win, method_name)()
                        except Exception:
                            pass
        except Exception:
            pass

    # Try immediately, then again shortly after the event loop starts.
    _do_focus()
    try:
        timer = fig.canvas.new_timer(interval=250)
        timer.single_shot = True
        timer.add_callback(_do_focus)
        timer.start()
    except Exception:
        pass



def _safe_basename_no_ext(path):
    return os.path.splitext(os.path.basename(path))[0]


def _split_session_trial_from_trc_path(trc_path):
    """
    Best-effort labels from a typical local OpenCap tree:
      <input-root>/<session_id>/MarkerData/PreAugmentation/<trial>.trc
    Returns (session_id, trial_name).
    """
    trial_name = _safe_basename_no_ext(trc_path)
    parts = os.path.normpath(trc_path).split(os.sep)
    session_id = ''
    try:
        idx = parts.index('MarkerData')
        if idx > 0:
            session_id = parts[idx - 1]
    except ValueError:
        # Fallback: use parent folder
        session_id = os.path.basename(os.path.dirname(trc_path))
    return session_id, trial_name


def discover_preaugment_trcs(input_root, include_cleaned=False, fixed_preaug_rel=None):
    """
    Direct discovery of `.trc` files in a known PreAugmentation folder.

    This workflow assumes TRCs always live at:
      <subjectDir>/<fixed_preaug_rel>/*.trc

    Where `input_root` can be either:
    - a subject folder (e.g., .../March_2/subject5), or
    - a date folder containing multiple subject folders (e.g., .../March_2)

    No recursive crawling is performed.
    Returns: list[str] absolute paths, sorted.
    """
    input_root = os.path.abspath(input_root)

    if not fixed_preaug_rel:
        raise ValueError(
            "Direct discovery requires --fixed-preaug-rel. Example: "
            r'"MarkerData\OpenPose_default\3-cameras\PreAugmentation"'
        )

    fixed_preaug_rel = os.path.normpath(fixed_preaug_rel)
    candidate_dirs = []

    # Case 1: input_root is a subject folder.
    p = os.path.join(input_root, fixed_preaug_rel)
    if os.path.isdir(p):
        candidate_dirs.append(p)
    else:
        # Case 2: input_root is a parent folder (e.g., date) containing subject folders.
        for name in os.listdir(input_root):
            if not name.lower().startswith('subject'):
                continue
            subj_dir = os.path.join(input_root, name)
            if not os.path.isdir(subj_dir):
                continue
            pp = os.path.join(subj_dir, fixed_preaug_rel)
            if os.path.isdir(pp):
                candidate_dirs.append(pp)

    out = []
    for d in candidate_dirs:
        try:
            for fn in os.listdir(d):
                if not fn.lower().endswith('.trc'):
                    continue
                full = os.path.join(d, fn)
                if os.path.isfile(full):
                    out.append(full)
        except Exception:
            continue

    # Optionally add TRCs from Cleaned/ under the same PreAugmentation folder.
    if include_cleaned:
        for d in candidate_dirs:
            cleaned_dir = os.path.join(d, "Cleaned")
            if not os.path.isdir(cleaned_dir):
                continue
            try:
                for fn in os.listdir(cleaned_dir):
                    if not fn.lower().endswith(".trc"):
                        continue
                    full = os.path.join(cleaned_dir, fn)
                    if os.path.isfile(full):
                        out.append(full)
            except Exception:
                continue

    return sorted(set(out))


def load_trc_full(trc_path):
    """
    Load TRC and return full time vector + markers.
    Returns (t_s, markers_dict, data_rate_hz) or (None, None, None) on failure.
    """
    if TRCFile is None:
        return None, None, None
    try:
        trc = TRCFile(trc_path)
    except Exception:
        return None, None, None
    t_s = np.asarray(trc.time, dtype=float)
    markers = {}
    for name in trc.marker_names:
        pos = trc.marker(name)
        markers[name] = {
            'x': np.asarray(pos[:, 0], dtype=float),
            'y': np.asarray(pos[:, 1], dtype=float),
            'z': np.asarray(pos[:, 2], dtype=float),
        }
    data_rate_hz = float(trc.data_rate) if hasattr(trc, 'data_rate') and trc.data_rate else None
    return t_s, markers, data_rate_hz


def filter_markers_for_review(marker_names):
    """Return list of marker names to include, excluding those ending with _study."""
    return [m for m in marker_names if not m.lower().endswith(MARKER_EXCLUDE_SUFFIX.lower())]


def build_marker_groups(marker_names):
    """
    Build list of (display_label, [marker_names]) for overview.
    LBigToe+LSmallToe and RBigToe+RSmallToe are grouped.
    """
    seen = set()
    groups = []
    for m in sorted(marker_names):
        if m in seen:
            continue
        for pair in TOE_GROUPS:
            if m in pair:
                group = [x for x in pair if x in marker_names]
                if group:
                    group.sort()
                    for g in group:
                        seen.add(g)
                    groups.append((f"{group[0]}/{group[1]}" if len(group) == 2 else group[0], group))
                break
        else:
            seen.add(m)
            groups.append((m, [m]))
    return groups


def _median_window_filter(x, window):
    """
    Moving median filter using np.nanmedian (handles NaN in motion capture).
    Edges preserved from original; center uses median of window.
    Non-causal, symmetric: output[ii] = median of window centered at ii.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    y = x.copy()
    k = int(window)
    if k < 2 or n < k:
        return y
    if k % 2 == 0:
        k -= 1
    win2 = k // 2
    for ii in range(win2, n - win2):
        idx = np.arange(ii - win2, ii + win2 + 1, dtype=int)
        y[ii] = np.nanmedian(x[idx])
    return y


def apply_median_filter(markers, window_frames=MEDIAN_FILTER_WINDOW_FRAMES):
    """
    Apply sliding-window median filter to all marker x,y,z. Returns new markers dict.
    Uses np.nanmedian for NaN-safe filtering (handles occluded/missing markers).
    """
    k = int(window_frames)
    if k % 2 == 0:
        k += 1
    out = {}
    for m, d in markers.items():
        out[m] = {}
        for comp in ['x', 'y', 'z']:
            arr = np.asarray(d[comp], dtype=float)
            out[m][comp] = _median_window_filter(arr, k)
    return out


def apply_median_filter_to_trc(trc, window_frames=MEDIAN_FILTER_WINDOW_FRAMES):
    """Apply sliding-window median filter to TRC marker data in-place."""
    k = int(window_frames)
    if k % 2 == 0:
        k += 1
    for m in trc.marker_names:
        for col in ['_tx', '_ty', '_tz']:
            col_name = m + col
            arr = np.asarray(trc.data[col_name], dtype=float)
            trc.data[col_name] = _median_window_filter(arr, k)


def apply_spline_to_trc(trc, marker_anchors):
    """
    marker_anchors: [(marker_name, [(t_s, x, y, z), ...]), ...]
    Replace TRC data for each marker. Returns list of (marker, t_first, t_last) for logging.
    Uses clamped boundary conditions to match data slope at anchor endpoints.
    """
    if CubicSpline is None:
        return []
    log = []
    time_full = np.asarray(trc.time)
    t_start = float(time_full[0])
    t_end = float(time_full[-1])
    duration = t_end - t_start if t_end > t_start else 1.0
    t_pct = (time_full - t_start) / duration * 100 if duration > 0 else np.zeros_like(time_full)
    for m, anchor_points in marker_anchors:
        if m not in trc.marker_names or len(anchor_points) < 2:
            continue
        pts = sorted(anchor_points, key=lambda p: p[0])
        t_first, t_last = pts[0][0], pts[-1][0]
        mask = (time_full >= t_first) & (time_full <= t_last)
        t_replace = time_full[mask]
        if len(t_replace) < 2:
            continue
        x_new, y_new, z_new = spline_values_at_t(
            pts, t_replace,
            t_pct=t_pct, data_x=trc.data[m + '_tx'], data_y=trc.data[m + '_ty'], data_z=trc.data[m + '_tz'],
            t_start=t_start, t_end=t_end
        )
        if x_new is None:
            continue
        trc.data[m + '_tx'][mask] = x_new
        trc.data[m + '_ty'][mask] = y_new
        trc.data[m + '_tz'][mask] = z_new
        log.append((m, t_first, t_last, [(p[0], p[1], p[2], p[3]) for p in pts]))
    return log


def spline_values_at_t(anchor_points, t_eval, t_pct=None, data_x=None, data_y=None, data_z=None, t_start=None, t_end=None):
    """
    Return (x, y, z) from cubic spline through anchor_points (t_s, x, y, z).
    Uses a natural cubic spline by default for stability/smoothness.

    Optional: if SPLINE_USE_CLAMPED_BC is enabled, spline endpoint derivatives are
    estimated from surrounding data via np.gradient so the curve blends with
    local slope.
    """
    if CubicSpline is None or len(anchor_points) < 2:
        return None, None, None
    pts = sorted(anchor_points, key=lambda p: p[0])
    t_anch = np.array([p[0] for p in pts])
    x_anch = np.array([p[1] for p in pts])
    y_anch = np.array([p[2] for p in pts])
    z_anch = np.array([p[3] for p in pts])
    # Default stable behavior.
    bc_type = 'natural'

    # Optional clamped BC mode (off by default).
    if SPLINE_USE_CLAMPED_BC:
        duration = (t_end - t_start) if (t_start is not None and t_end is not None) else 1.0
        if t_pct is not None and data_x is not None and len(t_pct) > 1 and duration > 0:
            t_pct_anch = (t_anch - t_start) / duration * 100
            scale = 100.0 / duration

            def get_deriv(data, t_pct_val):
                grad = np.gradient(data, t_pct)
                return np.interp(t_pct_val, t_pct, grad) * scale

            try:
                d_start_x = get_deriv(data_x, t_pct_anch[0])
                d_end_x = get_deriv(data_x, t_pct_anch[-1])
                d_start_y = get_deriv(data_y, t_pct_anch[0])
                d_end_y = get_deriv(data_y, t_pct_anch[-1])
                d_start_z = get_deriv(data_z, t_pct_anch[0])
                d_end_z = get_deriv(data_z, t_pct_anch[-1])
                bc_type = ((1, (d_start_x, d_start_y, d_start_z)), (1, (d_end_x, d_end_y, d_end_z)))
            except Exception:
                bc_type = 'natural'

    if bc_type == 'natural':
        cs_x = CubicSpline(t_anch, x_anch, bc_type='natural')
        cs_y = CubicSpline(t_anch, y_anch, bc_type='natural')
        cs_z = CubicSpline(t_anch, z_anch, bc_type='natural')
    else:
        d_sx, d_sy, d_sz = bc_type[0][1]
        d_ex, d_ey, d_ez = bc_type[1][1]
        cs_x = CubicSpline(t_anch, x_anch, bc_type=((1, d_sx), (1, d_ex)))
        cs_y = CubicSpline(t_anch, y_anch, bc_type=((1, d_sy), (1, d_ey)))
        cs_z = CubicSpline(t_anch, z_anch, bc_type=((1, d_sz), (1, d_ez)))
    return cs_x(t_eval), cs_y(t_eval), cs_z(t_eval)


# # MAPPING TRIAL TO UUID
# def build_trial_to_uuid(session_root):
#     """Build trial_name -> uuid mapping from Videos/Cam0/InputMedia/<trial>/<uuid>.mov."""
#     trial_to_uuid = {}
#     input_media = os.path.join(session_root, 'Videos', 'Cam0', 'InputMedia')
#     if not os.path.isdir(input_media):
#         return trial_to_uuid
#     for trial_name in os.listdir(input_media):
#         trial_dir = os.path.join(input_media, trial_name)
#         if not os.path.isdir(trial_dir):
#             continue
#         for f in os.listdir(trial_dir):
#             if f.lower().endswith(('.mov', '.mp4', '.avi')):
#                 match = re.search(
#                     r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
#                     f, re.I
#                 )
#                 if match:
#                     trial_to_uuid[trial_name] = match.group(0)
#                     break
#     return trial_to_uuid


# def find_trc_path(base_dir, session_id, trial_name):
#     """Return path to TRC file for (session_id, trial_name), or None."""
#     session_root = os.path.join(base_dir, session_id)
#     preaug_dir = os.path.join(session_root, 'MarkerData', 'PreAugmentation')
#     marker_dir = os.path.join(session_root, 'MarkerData')

#     # 1. PreAugmentation/<trial_name>.trc
#     p1 = os.path.join(preaug_dir, trial_name + '.trc')
#     if os.path.isfile(p1):
#         return p1

#     # 2. PreAugmentation/<uuid>.trc
#     trial_to_uuid = build_trial_to_uuid(session_root)
#     if trial_name in trial_to_uuid:
#         uuid = trial_to_uuid[trial_name]
#         p2 = os.path.join(preaug_dir, uuid + '.trc')
#         if os.path.isfile(p2):
#             return p2

#     # 3. MarkerData/<trial_name>.trc
#     p3 = os.path.join(marker_dir, trial_name + '.trc')
#     if os.path.isfile(p3):
#         return p3

#     return None


# def load_trc_roi(trc_path, t_min, t_max, t_start, t_end):
#     """
#     Load TRC and trim to [t_min, t_max]. Normalize time to % ROI.
#     Returns (t_pct, markers, data_rate_hz) or (None, None, None).
#     """
#     if TRCFile is None:
#         return None, None, None
#     try:
#         trc = TRCFile(trc_path)
#     except Exception:
#         return None, None, None
#     time = np.asarray(trc.time)
#     mask = (time >= t_min) & (time <= t_max)
#     if np.sum(mask) < 2:
#         return None, None, None
#     time_trimmed = time[mask]
#     duration = t_end - t_start
#     if duration <= 0:
#         duration = 1e-6
#     t_pct = (time_trimmed - t_start) / duration * 100.0
#     markers = {}
#     for name in trc.marker_names:
#         pos = trc.marker(name)
#         markers[name] = {
#             'x': np.asarray(pos[:, 0])[mask],
#             'y': np.asarray(pos[:, 1])[mask],
#             'z': np.asarray(pos[:, 2])[mask],
#         }
#     data_rate_hz = float(trc.data_rate) if hasattr(trc, 'data_rate') and trc.data_rate else None
#     return t_pct, markers, data_rate_hz


# def pct_to_seconds(t_pct, t_start, t_end):
#     """Convert % ROI to absolute time in seconds."""
#     duration = t_end - t_start
#     return t_start + (t_pct / 100.0) * duration


def collect_spline_interactive(marker_names, markers_dict, t_pct, t_start, t_end, coupled=True, markers_original=None,
                               is_bad_vel=None, is_bad_rigid=None):
    """
    Spline selection: click on curve to add anchor points. Returns marker_anchors for apply, or None.
    marker_names: list of 1 or 2 markers. coupled: apply same t-range to both (each has own x,y,z).
    markers_original: optional dict of original data (before splines) for comparison; plotted dashed/fine.
    is_bad_vel, is_bad_rigid: optional bool arrays for quality bars (velocity/rigid failures for these markers).
    """
    duration = t_end - t_start
    # anchor_points: list of (t_s, x, y, z) - for coupled we use first marker's coords as reference
    anchor_points = []
    scatter_artists = []
    preview_lines = []
    text_boxes = []
    apply_confirm = [False]
    last_key = [None]  # debug: show last key Matplotlib reports
    figsize_single = (
        SPLINE_EDITOR_WIDTH_PER_AXIS_IN * 3,
        SPLINE_EDITOR_HEIGHT_IN,
    )
    fig, axes = plt.subplots(1, 3, figsize=figsize_single, sharex=True, constrained_layout=True)
    label = '/'.join(marker_names) if len(marker_names) > 1 else marker_names[0]
    fig.suptitle(label, fontsize=11)

    colors_xyz = ('#e74c3c', '#27ae60', '#3498db')
    for ax_idx, (comp, lbl) in enumerate([('x', 'X'), ('y', 'Y'), ('z', 'Z')]):
        ax = axes[ax_idx]
        for mi, m in enumerate(marker_names):
            if m not in markers_dict:
                continue
            d = markers_dict[m]
            c = colors_xyz[ax_idx]
            ls = '-' if mi == 0 else '--'
            # Plot original (before splines) as dashed/fine for comparison when available
            orig = (markers_original or {}).get(m)
            if orig is not None and comp in orig:
                lbl_orig = 'original' if (mi == 0 and ax_idx == 0) else None
                ax.plot(t_pct, orig[comp], color=c, linewidth=0.8, linestyle=':', alpha=0.7, label=lbl_orig)
            ax.plot(t_pct, d[comp], color=c, linewidth=1.5, linestyle=ls, label=m if len(marker_names) > 1 else None)
        ax.set_ylabel(f'{lbl} (m)')
        ax.grid(True, alpha=0.3)
        # REPLA
        #ax.set_xlim(X_MIN_PCT, X_MAX_PCT)
        _draw_quality_bad_regions(ax, t_pct, is_bad_vel, is_bad_rigid)
    axes[2].set_xlabel('Time (s)')
    for ax in axes[:2]:
        ax.set_xlabel('')
    if len(marker_names) > 1 or markers_original:
        axes[0].legend(fontsize=8)

    show_preview = [True]  # Auto-preview when 2+ points; 'p' toggles off

    def update_display():
        for s in scatter_artists:
            s.remove()
        scatter_artists.clear()
        for pl in preview_lines:
            for l in pl:
                l.remove()
        preview_lines.clear()
        for tb in text_boxes:
            tb.remove()
        text_boxes.clear()

        # Draw selected points
        if coupled and anchor_points:
            for ax_idx in range(3):
                t_pct_pts = np.array([p[0][0] for p in anchor_points])
                vals = np.array([p[0][ax_idx + 1] for p in anchor_points])
                s = axes[ax_idx].scatter(t_pct_pts, vals, c='black', s=60, zorder=5, marker='o')
                scatter_artists.append(s)
        elif anchor_points:
            t_pct_pts = np.array([p[0] for p in anchor_points])
            for ax_idx, comp in enumerate(['x', 'y', 'z']):
                vals = np.array([p[ax_idx + 1] for p in anchor_points])
                s = axes[ax_idx].scatter(t_pct_pts, vals, c='black', s=60, zorder=5, marker='o')
                scatter_artists.append(s)

        # Draw preview if 2+ points and user pressed 'p' (use clamped bc to match data slope)
        if show_preview[0] and len(anchor_points) >= 2:
            pts = sorted([p[0] if coupled else p for p in anchor_points], key=lambda x: x[0])
            t_anch = np.array([p[0] for p in pts])
            t_s_span = np.linspace(t_anch[0], t_anch[-1], 80)
            t_pct_span = t_s_span
            m0 = marker_names[0]
            d0 = markers_dict.get(m0, {})
            x_s, y_s, z_s = spline_values_at_t(
                pts, t_s_span,
                t_pct=t_pct, data_x=d0.get('x'), data_y=d0.get('y'), data_z=d0.get('z'),
                t_start=t_start, t_end=t_end
            )
            if x_s is not None:
                pl = []
                pl.append(axes[0].plot(t_pct_span, x_s, 'k-', linewidth=2, alpha=0.8)[0])
                pl.append(axes[1].plot(t_pct_span, y_s, 'k-', linewidth=2, alpha=0.8)[0])
                pl.append(axes[2].plot(t_pct_span, z_s, 'k-', linewidth=2, alpha=0.8)[0])
                preview_lines.append(pl)

        status = f'Points: {len(anchor_points)}'
        if last_key[0] is not None:
            status += f'  | last key: {last_key[0]}'
        if apply_confirm[0]:
            status += '  >>> Press a again to apply spline'
        status += '\nu=undo | p=toggle preview | a=apply | d=done'
        tb = fig.text(0.5, 0.03, status, ha='center', fontsize=9,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        text_boxes.append(tb)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is None or event.inaxes not in axes:
            return
        x = event.xdata
        if x is None:
            return
        t_s = float(x)
        if coupled:
            pts = []
            for m in marker_names:
                if m in markers_dict:
                    d = markers_dict[m]
                    pts.append((t_s, np.interp(t_s, t_pct, d['x']), np.interp(t_s, t_pct, d['y']), np.interp(t_s, t_pct, d['z'])))
            if pts:
                anchor_points.append(pts)
        else:
            m = marker_names[0]
            d = markers_dict[m]
            xv = np.interp(t_s, t_pct, d['x'])
            yv = np.interp(t_s, t_pct, d['y'])
            zv = np.interp(t_s, t_pct, d['z'])
            anchor_points.append((t_s, xv, yv, zv))
        update_display()

    def on_key(event):
        key = (event.key or '').lower()
        last_key[0] = event.key
        if key == 'u':
            if anchor_points:
                anchor_points.pop()
                apply_confirm[0] = False
                update_display()
        elif key == 'a':
            if apply_confirm[0] and len(anchor_points) >= 2:
                if coupled:
                    result[0] = [(m, [p[mi] for p in anchor_points]) for mi, m in enumerate(marker_names)]
                else:
                    result[0] = [(marker_names[0], list(anchor_points))]
                plt.close(fig)
            elif len(anchor_points) >= 2:
                apply_confirm[0] = True
                update_display()
        elif key == 'p':
            if len(anchor_points) >= 2:
                show_preview[0] = not show_preview[0]
                update_display()
        elif key == 'c' and len(marker_names) > 1:
            # Toggle coupled - would need to re-enter; skip for now
            pass
        elif key == 'd':
            plt.close(fig)

    result = [None]

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    # TkAgg fallback: bind directly to Tk widget in case mpl's key_press_event isn't delivered.
    try:
        if hasattr(fig.canvas, 'get_tk_widget'):
            _w = fig.canvas.get_tk_widget()
            if _w is not None:
                def _tk_on_key(ev):
                    ch = getattr(ev, 'char', '') or ''
                    ks = getattr(ev, 'keysym', '') or ''
                    if not ch:
                        ch = ks
                    class _Evt:
                        pass
                    e = _Evt()
                    e.key = (ch or '').lower()
                    on_key(e)
                    return 'break'
                _w.bind('<KeyPress>', _tk_on_key)
                try:
                    top = _w.winfo_toplevel()
                    if top is not None:
                        top.bind('<KeyPress>', _tk_on_key, add='+')
                except Exception:
                    pass
    except Exception:
        pass
    _maximize_figure(fig)
    _try_focus_figure(fig)
    plt.show()

    return result[0]


def get_clicked_group(event, axes_list, group_labels):
    """Return (label, marker_names) if click was on one of the axes, else None."""
    if event.inaxes is None:
        return None
    for ax, (label, mnames) in zip(axes_list, group_labels):
        if event.inaxes == ax:
            return (label, mnames)
    return None

# Replace t_pct with time vector
def show_median_filter_preview(markers_orig, markers_filtered, t_pct, group_labels, window_frames, n_cols=5):
    """
    Show original vs median-filtered data. Blocks until user presses 'a' (accept) or 'n' (reject).
    Returns True if accepted, False if rejected.
    """
    result = [None]
    n_subplots = len(group_labels)
    n_rows = (n_subplots + n_cols - 1) // n_cols
    figsize = (n_cols * CELL_SIZE_IN, n_rows * CELL_SIZE_IN)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()

    for ax_idx, (label, mnames) in enumerate(group_labels):
        ax = axes[ax_idx]
        has_data = any(mk in markers_orig for mk in mnames)
        if not has_data:
            ax.set_visible(False)
            continue
        for mi, mk in enumerate(mnames):
            if mk not in markers_orig or mk not in markers_filtered:
                continue
            do_orig, df_orig = markers_orig[mk], markers_filtered[mk]
            lw = 1.5 if mi == 0 else 1
            ax.plot(t_pct, do_orig['x'], 'r:', linewidth=lw * 0.8, alpha=0.8, label='original' if (mi == 0 and ax_idx == 0) else None)
            ax.plot(t_pct, do_orig['y'], 'g:', linewidth=lw * 0.8, alpha=0.8)
            ax.plot(t_pct, do_orig['z'], 'b:', linewidth=lw * 0.8, alpha=0.8)
            ax.plot(t_pct, df_orig['x'], 'r-', linewidth=lw, label='filtered' if (mi == 0 and ax_idx == 0) else None)
            ax.plot(t_pct, df_orig['y'], 'g-', linewidth=lw)
            ax.plot(t_pct, df_orig['z'], 'b-', linewidth=lw)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (m)')
        #ax.set_xlim(X_MIN_PCT, X_MAX_PCT)
        ax.set_box_aspect(1)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    for idx in range(n_subplots, len(axes)):
        axes[idx].set_visible(False)
    if group_labels and any(mk in markers_orig for mk in group_labels[0][1]):
        axes[0].legend(fontsize=8)

    def on_key(event):
        k = (event.key or '').lower()
        if k == 'a':
            result[0] = True
            plt.close(fig)
        elif k == 'n':
            result[0] = False
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.suptitle(f'Median filter (window={window_frames} frames): Original (dashed) vs Filtered (solid)', fontsize=11)
    fig.text(0.5, 0.02, 'a = accept filter | n = reject (keep original)', ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    # TkAgg fallback for shortcuts.
    try:
        if hasattr(fig.canvas, 'get_tk_widget'):
            _w = fig.canvas.get_tk_widget()
            if _w is not None:
                def _tk_on_key(ev):
                    ch = getattr(ev, 'char', '') or ''
                    ks = getattr(ev, 'keysym', '') or ''
                    if not ch:
                        ch = ks
                    class _Evt:
                        pass
                    e = _Evt()
                    e.key = (ch or '').lower()
                    on_key(e)
                    return 'break'
                _w.bind('<KeyPress>', _tk_on_key)
                try:
                    top = _w.winfo_toplevel()
                    if top is not None:
                        top.bind('<KeyPress>', _tk_on_key, add='+')
                except Exception:
                    pass
    except Exception:
        pass
    _maximize_figure(fig)
    _try_focus_figure(fig)
    plt.show()
    plt.close(fig)
    return result[0] if result[0] is not None else False


def _draw_spline_region(ax, spline_regions, t_start, t_end):
    """Draw vertical lines for spline region start (orange) and end (red) on ax."""
    for t_first, t_last in spline_regions:
        ax.axvline(t_first, color='orange', linestyle='--', alpha=0.8)
        ax.axvline(t_last, color='red', linestyle='--', alpha=0.8)


def main():
    if TRCFile is None:
        print("utilsTRC.TRCFile not found. Ensure utilsTRC.py is on the path.")
        return

    parser = argparse.ArgumentParser(description="Interactive cleaning (median + spline) for local PreAugmentation TRCs.")
    parser.add_argument('--input-root', default=PREAUGMENT_BASE_DIR_DEFAULT,
                        help='Root folder to search for TRC files (defaults to PREAUGMENT_BASE_DIR env var, Examples/Data, or cwd).')
    parser.add_argument('--out-dir', default=OUT_DIR,
                        help='Where to save PNG overviews and edit-log CSV.')
    parser.add_argument('--include-cleaned', action='store_true',
                        help='Also include TRCs found under a Cleaned/ folder (default: skip).')
    parser.add_argument('--fixed-preaug-rel', default=None,
                        help=('If provided, skip recursive discovery and only load TRCs from this '
                              'relative PreAugmentation path inside each subject folder. Example: '
                              '"MarkerData\\OpenPose_default\\3-cameras\\PreAugmentation"'))
    parser.add_argument('--no-review', action='store_true',
                        help='Do not open interactive windows; only save per-trial overview PNGs.')
    parser.add_argument('--debug-discovery', action='store_true',
                        help='Print TRC discovery/loading counts to help match trials in your Excel sheet.')
    args, _unknown = parser.parse_known_args()

    input_root = args.input_root
    do_review = not args.no_review
    out_dir = args.out_dir

    trc_paths = discover_preaugment_trcs(
        input_root,
        include_cleaned=args.include_cleaned,
        fixed_preaug_rel=args.fixed_preaug_rel,
    )
    if not trc_paths:
        print(f"No TRC files found under: {input_root}")
        return
    if args.debug_discovery:
        print(f"[debug] Discovered PreAugmentation TRC files: {len(trc_paths)}")

    # Discover marker names from all TRCs (union across trials)
    all_marker_names = set()
    for trc_path in trc_paths:
        try:
            trc = TRCFile(trc_path)
            all_marker_names.update(trc.marker_names)
        except Exception:
            continue

    marker_names_flat = sorted(filter_markers_for_review(all_marker_names))
    marker_groups = build_marker_groups(marker_names_flat)
    if not marker_groups:
        print("No markers to review (all excluded?). Available:", sorted(all_marker_names)[:20], "...")
        return

    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    edit_log = []
    csv_path = os.path.join(out_dir, f'PreAugment_edits_{timestamp}.csv')

    def _save_edit_log():
        """Write edit_log to CSV. Format designed for human traceability of data processing."""
        if not edit_log:
            return
        # Columns: trial id, step order, action type, human-readable description, action-specific params
        fieldnames = ['participant_id', 'session_id', 'trial_name', 'trial_type', 'step', 'action',
                     'description', 'window_frames', 'marker_name', 't_first_s', 't_last_s', 'n_anchor_points']
        rows = []
        step_by_trial = {}
        for e in edit_log:
            key = (e.get('participant_id'), e.get('trial_name'))
            step_by_trial[key] = step_by_trial.get(key, 0) + 1
            step = step_by_trial[key]
            action = e.get('action', '')
            if action == 'median_filter':
                wf = e.get('window_frames', MEDIAN_FILTER_WINDOW_FRAMES)
                desc = f"Median filter applied to all markers (window={wf} frames, NaN-safe)"
                rows.append({'participant_id': e.get('participant_id'), 'session_id': e.get('session_id'),
                             'trial_name': e.get('trial_name'), 'trial_type': e.get('trial_type'),
                             'step': step, 'action': action, 'description': desc,
                             'window_frames': wf, 'marker_name': '', 't_first_s': '', 't_last_s': '', 'n_anchor_points': ''})
            elif action == 'spline':
                m = e.get('marker_name', '')
                t1, t2 = e.get('t_first', ''), e.get('t_last', '')
                n = len(e.get('anchor_points', []))
                t1_s = f"{t1:.3f}" if isinstance(t1, (int, float)) else str(t1)
                t2_s = f"{t2:.3f}" if isinstance(t2, (int, float)) else str(t2)
                desc = f"Spline correction: {m} from {t1_s}s to {t2_s}s ({n} anchor points)"
                rows.append({'participant_id': e.get('participant_id'), 'session_id': e.get('session_id'),
                             'trial_name': e.get('trial_name'), 'trial_type': e.get('trial_type'),
                             'step': step, 'action': action, 'description': desc,
                             'window_frames': '', 'marker_name': m, 't_first_s': t1_s, 't_last_s': t2_s, 'n_anchor_points': n})
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Edit log: {csv_path}")

    all_trials_flat = []
    n_loaded = 0
    n_skipped = 0
    skipped_paths = []
    for trc_path in trc_paths:
        t_s, markers, data_rate_hz = load_trc_full(trc_path)
        if t_s is None or markers is None or len(t_s) < 2:
            n_skipped += 1
            if args.debug_discovery and len(skipped_paths) < 25:
                skipped_paths.append(trc_path)
            continue
        n_loaded += 1
        sid, trial_name = _split_session_trial_from_trc_path(trc_path)
        pid = sid  # best-effort; for this local workflow we treat session as participant label
        t_start, t_end = float(t_s[0]), float(t_s[-1])
        all_trials_flat.append({
            'participant_id': pid,
            'session_id': sid,
            'trial_name': trial_name,
            'trial_type': '',
            'trc_path': trc_path,
            't_start': t_start,
            't_end': t_end,
            't_pct': t_s,   # NOTE: kept variable name to minimize changes; now stores time (s)
            'markers': markers,
            'data_rate_hz': data_rate_hz,
        })

    if not all_trials_flat:
        print("TRC discovery succeeded but none could be loaded.")
        return
    if args.debug_discovery:
        print(f"[debug] Loaded trials (usable TRCFile): {n_loaded}")
        print(f"[debug] Skipped trials (load/time issues): {n_skipped}")
        if skipped_paths:
            print("[debug] Example skipped TRCs:")
            for p in skipped_paths:
                print(f"  - {p}")

    for trial_idx, trial in enumerate(all_trials_flat):
        pid = trial['participant_id']
        sid = trial['session_id']
        tname = trial['trial_name']
        ttype = trial['trial_type']
        trc_path = trial['trc_path']
        t_start = trial['t_start']
        t_end = trial['t_end']
        t_pct = trial['t_pct']  # time (s)
        markers = trial['markers']
        print(f"\n--- Trial {trial_idx + 1}/{len(all_trials_flat)}: {pid} {tname} ---")

        n_subplots = len(marker_groups)
        group_labels = [(label, mnames) for label, mnames in marker_groups]

        selected = [None]
        marker_to_splines = {}  # marker_name -> [(t_first, t_last), ...]
        median_filter_applied = [False]
        median_filter_params = [None]
        trc = None

        # Median filter preview FIRST (before overview figure) to avoid blank Figure 1
        markers_filtered = apply_median_filter(markers, MEDIAN_FILTER_WINDOW_FRAMES)
        if do_review and TRCFile is not None:
            accepted = show_median_filter_preview(markers, markers_filtered, t_pct, group_labels, MEDIAN_FILTER_WINDOW_FRAMES)
            if accepted:
                trc = TRCFile(trc_path)
                apply_median_filter_to_trc(trc, MEDIAN_FILTER_WINDOW_FRAMES)
                time = np.asarray(trc.time)
                t_pct = time
                markers = {m: {'x': np.asarray(trc.data[m + '_tx'].copy()),
                               'y': np.asarray(trc.data[m + '_ty'].copy()),
                               'z': np.asarray(trc.data[m + '_tz'].copy())}
                           for m in trc.marker_names}
                trial['t_pct'] = t_pct
                trial['markers'] = markers
                median_filter_applied[0] = True
                median_filter_params[0] = {'window_frames': MEDIAN_FILTER_WINDOW_FRAMES}
                edit_log.append({'participant_id': pid, 'session_id': sid, 'trial_name': tname, 'trial_type': ttype,
                                 'action': 'median_filter', 'window_frames': MEDIAN_FILTER_WINDOW_FRAMES})
                print(f"  Median filter applied (window={MEDIAN_FILTER_WINDOW_FRAMES} frames)")

        # Create overview figure now (after median filter preview)
        n_cols = 3
        n_rows = (n_subplots + n_cols - 1) // n_cols
        figsize = (n_cols * CELL_SIZE_IN, n_rows * CELL_SIZE_IN)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
        axes = np.atleast_1d(axes).flatten()

        def on_overview_click(event):
            r = get_clicked_group(event, axes[:n_subplots], group_labels)
            if r:
                selected[0] = r
                plt.close(fig)

        # Debug: show last key we receive in the window (helps verify TkAgg key routing).
        _inst_text = [None]

        def on_overview_key(event):
            k = (event.key or '').lower()
            try:
                if _inst_text[0] is not None:
                    _inst_text[0].set_text(f'{_inst} | last key: {event.key!r}')
                    fig.canvas.draw_idle()
            except Exception:
                pass
            if k == 'd':
                selected[0] = ('DONE', [])
                plt.close(fig)

        def draw_overview(spline_regions_by_marker):
            m = trial['markers']  # use current trial state (updated after splines/filter)
            # Multi-Factor Quality Check: velocity + rigid constraints (per-marker / per-pair)
            vel_failures_by_marker, rigid_failures_by_pair = [], []
            data_arr, data_marker_names = _markers_to_data_array(m, marker_names_flat)
            if data_arr is not None:
                pair_indices, pair_names = _get_marker_pairs_with_names(data_marker_names, QUALITY_CHECK_MARKER_PAIRS)
                _, _, _, _, _, vel_failures_by_marker, rigid_failures_by_pair = check_data_quality(
                    data_arr, data_marker_names, pair_indices, pair_names=pair_names, data_rate_hz=trial.get('data_rate_hz'))
            for ax_idx, (label, mnames) in enumerate(group_labels):
                ax = axes[ax_idx]
                has_data = any(mk in m for mk in mnames)
                if not has_data:
                    ax.set_visible(False)
                    continue
                for mi, mk in enumerate(mnames):
                    if mk not in m:
                        continue
                    d = m[mk]
                    ax.plot(t_pct, d['x'], 'r-', linewidth=1.5 if mi == 0 else 1)
                    ax.plot(t_pct, d['y'], 'g-', linewidth=1.5 if mi == 0 else 1)
                    ax.plot(t_pct, d['z'], 'b-', linewidth=1.5 if mi == 0 else 1)
                regs = []
                for mk in mnames:
                    regs.extend(spline_regions_by_marker.get(mk, []))
                _draw_spline_region(ax, regs, t_start, t_end)
                # Velocity bars only on subplots whose markers exceeded velocity threshold
                # Rigid bars only on subplots whose markers are in a failing pair
                mnames_set = set(mnames)
                is_bad_vel_local = None
                if vel_failures_by_marker:
                    # Velocity is central difference global check
                    masks = [is_bad for name, is_bad in vel_failures_by_marker if name in mnames_set]
                    if masks:
                        is_bad_vel_local = np.logical_or.reduce(masks)
                is_bad_rigid_local = None
                if rigid_failures_by_pair:
                    # Could skip the pairs check when starting
                    masks = [is_bad for n1, n2, is_bad in rigid_failures_by_pair if n1 in mnames_set or n2 in mnames_set]
                    if masks:
                        is_bad_rigid_local = np.logical_or.reduce(masks)
                _draw_quality_bad_regions(ax, t_pct, is_bad_vel_local, is_bad_rigid_local)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Position (m)')
                #ax.set_xlim(X_MIN_PCT, X_MAX_PCT)
                ax.set_box_aspect(1)
                n_spl = sum(1 for mk in mnames for _ in spline_regions_by_marker.get(mk, []))
                if n_spl:
                    title = f'{label} ✓ edited ({n_spl} spl)'
                    ax.set_facecolor('#e8f5e9')
                else:
                    title = label
                    ax.set_facecolor('white')
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
            for idx in range(n_subplots, len(axes)):
                axes[idx].set_visible(False)

        # Multi-Factor Quality Check runs inside draw_overview; print summary before first review
        data_arr, data_marker_names = _markers_to_data_array(markers, marker_names_flat)
        if data_arr is not None:
            pair_indices, pair_names = _get_marker_pairs_with_names(data_marker_names, QUALITY_CHECK_MARKER_PAIRS)
            is_bad, _, _, is_bad_vel, is_bad_rigid, _, _ = check_data_quality(
                data_arr, data_marker_names, pair_indices, pair_names=pair_names, data_rate_hz=trial.get('data_rate_hz'))
            n_bad = np.sum(is_bad)
            if n_bad > 0:
                n_vel, n_rigid = np.sum(is_bad_vel), np.sum(is_bad_rigid)
                print(f"  Quality check: {n_bad}/{len(is_bad)} frames flagged (vel:{n_vel}, rigid:{n_rigid})")

        draw_overview(marker_to_splines)
        med_status = ' [MEDIAN]' if median_filter_applied[0] else ''
        fig.suptitle(f'{pid} {tname}{med_status}', fontsize=12)
        _inst = 'Click marker to edit | u/U=undo | d=done | Red=velocity fail | Blue=rigid fail | Purple=both'
        _inst_text[0] = fig.text(0.5, 0.02, _inst, ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        out_path = os.path.join(out_dir, f'PreAugment_{pid}_{tname}_{timestamp}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(out_path)

        if not do_review:
            plt.close(fig)
            continue

        fig.canvas.mpl_connect('button_press_event', on_overview_click)
        fig.canvas.mpl_connect('key_press_event', on_overview_key)
        # TkAgg fallback for shortcuts.
        try:
            if hasattr(fig.canvas, 'get_tk_widget'):
                _w = fig.canvas.get_tk_widget()
                if _w is not None:
                    def _tk_on_key(ev):
                        ch = getattr(ev, 'char', '') or ''
                        ks = getattr(ev, 'keysym', '') or ''
                        if not ch:
                            ch = ks
                        class _Evt:
                            pass
                        e = _Evt()
                        e.key = (ch or '').lower()
                        on_overview_key(e)
                        return 'break'

                    _w.bind('<KeyPress>', _tk_on_key)
                    try:
                        top = _w.winfo_toplevel()
                        if top is not None:
                            top.bind('<KeyPress>', _tk_on_key, add='+')
                    except Exception:
                        pass
        except Exception:
            pass
        _maximize_figure(fig)
        _try_focus_figure(fig)
        plt.show()
        plt.close(fig)

        if not median_filter_applied[0]:
            trc = None
        trial_edits = []

        while True:
            choice_label, choice_mnames = selected[0] if selected[0] else (None, [])
            selected[0] = None

            if choice_label == 'DONE' or choice_label is None:
                break

            if choice_mnames:
                coupled = len(choice_mnames) > 1
                # Original marker data (before splines this session) for comparison plot
                markers_original = {m: {'x': markers[m]['x'].copy(), 'y': markers[m]['y'].copy(), 'z': markers[m]['z'].copy()}
                                   for m in choice_mnames if m in markers}
                while True:
                    # Always use latest from trial so display shows applied splines
                    t_pct = trial['t_pct']
                    markers = trial['markers']
                    # Quality bars for edit view: compute vel/rigid failures for choice_mnames
                    is_bad_vel_edit, is_bad_rigid_edit = None, None
                    data_arr, data_marker_names = _markers_to_data_array(markers, marker_names_flat)
                    if data_arr is not None:
                        pair_indices, pair_names = _get_marker_pairs_with_names(data_marker_names, QUALITY_CHECK_MARKER_PAIRS)
                        _, _, _, _, _, vel_failures_by_marker, rigid_failures_by_pair = check_data_quality(
                            data_arr, data_marker_names, pair_indices, pair_names=pair_names, data_rate_hz=trial.get('data_rate_hz'))
                        mnames_set = set(choice_mnames)
                        if vel_failures_by_marker:
                            masks = [is_bad for name, is_bad in vel_failures_by_marker if name in mnames_set]
                            if masks:
                                is_bad_vel_edit = np.logical_or.reduce(masks)
                        if rigid_failures_by_pair:
                            masks = [is_bad for n1, n2, is_bad in rigid_failures_by_pair if n1 in mnames_set or n2 in mnames_set]
                            if masks:
                                is_bad_rigid_edit = np.logical_or.reduce(masks)
                    marker_anchors = collect_spline_interactive(choice_mnames, markers, t_pct, t_start, t_end, coupled=coupled, markers_original=markers_original,
                                                               is_bad_vel=is_bad_vel_edit, is_bad_rigid=is_bad_rigid_edit)
                    if not marker_anchors:
                        break
                    # Apply spline (marker_anchors is not None)
                    if trc is None:
                        trc = TRCFile(trc_path)
                    apply_spline_to_trc(trc, marker_anchors)
                    # Build fresh markers dict with spline-applied values (ensures display updates)
                    markers_new = {mm: {'x': markers[mm]['x'].copy(), 'y': markers[mm]['y'].copy(), 'z': markers[mm]['z'].copy()}
                                  for mm in markers}
                    for m, anchors in marker_anchors:
                        t_first = min(p[0] for p in anchors)
                        t_last = max(p[0] for p in anchors)
                        pts = [(p[0], p[1], p[2], p[3]) for p in anchors]
                        marker_to_splines.setdefault(m, []).append((t_first, t_last))
                        entry = {'participant_id': pid, 'session_id': sid, 'trial_name': tname, 'trial_type': ttype,
                                 'action': 'spline', 'marker_name': m, 't_first': t_first, 't_last': t_last,
                                 'anchor_points': pts}
                        edit_log.append(entry)
                        trial_edits.append(entry)
                        if m in markers_new:
                            mask = (t_pct >= t_first) & (t_pct <= t_last)
                            if np.any(mask):
                                t_s_eval = t_pct[mask]
                                d = markers_new[m]
                                xv, yv, zv = spline_values_at_t(
                                    anchors, t_s_eval,
                                    t_pct=t_pct, data_x=d['x'], data_y=d['y'], data_z=d['z'],
                                    t_start=t_start, t_end=t_end
                                )
                                if xv is not None:
                                    markers_new[m]['x'][mask] = xv
                                    markers_new[m]['y'][mask] = yv
                                    markers_new[m]['z'][mask] = zv
                    trial['markers'] = markers_new
                    continue

            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
            axes = np.atleast_1d(axes).flatten()
            draw_overview(marker_to_splines)
            status = f"Splines: {list(marker_to_splines.keys())}" if marker_to_splines else "None"
            fig.suptitle(f'{pid} {tname} — {status}', fontsize=11)
            _inst2 = 'u=undo last | U=undo all | Click marker | d=done | Red=vel | Blue=rigid | Purple=both'
            _inst2_text = fig.text(0.5, 0.02, _inst2, ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

            def on_click2(event):
                r = get_clicked_group(event, axes[:n_subplots], group_labels)
                if r:
                    selected[0] = r
                    plt.close(fig)

            def on_key2(event):
                k = event.key or ''
                k_lower = k.lower()
                try:
                    _inst2_text.set_text(f'{_inst2} | last key: {event.key!r}')
                    fig.canvas.draw_idle()
                except Exception:
                    pass
                if k_lower == 'd':
                    selected[0] = ('DONE', [])
                    plt.close(fig)
                elif k_lower == 'u' and (trial_edits or (k == 'U' and median_filter_applied[0])):
                    if k == 'U':
                        trial_edits.clear()
                        edit_log[:] = [e for e in edit_log if e.get('participant_id') != pid or e.get('trial_name') != tname]
                        median_filter_applied[0] = False
                        median_filter_params[0] = None
                        selected[0] = ('REDRAW', [])
                    else:
                        trial_edits.pop()
                        if edit_log and edit_log[-1].get('action') == 'spline':
                            edit_log.pop()
                        selected[0] = ('REDRAW', [])
                    plt.close(fig)
                    time.sleep(0.1)  # Let backend flush pending callbacks (avoids Tk idle_draw error)

            fig.canvas.mpl_connect('button_press_event', on_click2)
            fig.canvas.mpl_connect('key_press_event', on_key2)
            # TkAgg fallback for shortcuts.
            try:
                if hasattr(fig.canvas, 'get_tk_widget'):
                    _w = fig.canvas.get_tk_widget()
                    if _w is not None:
                        def _tk_on_key(ev):
                            ch = getattr(ev, 'char', '') or ''
                            ks = getattr(ev, 'keysym', '') or ''
                            if not ch:
                                ch = ks
                            class _Evt:
                                pass
                            e = _Evt()
                            e.key = (ch or '').lower()
                            on_key2(e)
                            return 'break'
                        _w.bind('<KeyPress>', _tk_on_key)
                        try:
                            top = _w.winfo_toplevel()
                            if top is not None:
                                top.bind('<KeyPress>', _tk_on_key, add='+')
                        except Exception:
                            pass
            except Exception:
                pass
            _maximize_figure(fig)
            _try_focus_figure(fig)
            plt.show()
            plt.close(fig)

            if selected[0] and selected[0][0] == 'REDRAW':
                selected[0] = None
                if trc is not None:
                    trc = TRCFile(trc_path)
                    if median_filter_applied[0]:
                        apply_median_filter_to_trc(trc, MEDIAN_FILTER_WINDOW_FRAMES)
                    marker_to_splines = {}
                    for e in trial_edits:
                        if e.get('action') == 'spline':
                            m = e['marker_name']
                            marker_to_splines.setdefault(m, []).append((e['t_first'], e['t_last']))
                            apply_spline_to_trc(trc, [(m, e['anchor_points'])])
                    time = np.asarray(trc.time)
                    t_pct = time
                    markers = {m: {'x': np.asarray(trc.data[m + '_tx'].copy()),
                                   'y': np.asarray(trc.data[m + '_ty'].copy()),
                                   'z': np.asarray(trc.data[m + '_tz'].copy())}
                               for m in trc.marker_names}
                    trial['t_pct'] = t_pct
                    trial['markers'] = markers
                continue
            if selected[0] and selected[0][0] == 'DONE':
                break

        if trc is not None and (marker_to_splines or median_filter_applied[0]):
            cleaned_dir = os.path.join(os.path.dirname(trc_path), 'Cleaned')
            os.makedirs(cleaned_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(trc_path))[0]
            out_trc = os.path.join(cleaned_dir, f'{base}_{timestamp}.trc')
            trc.write(out_trc)
            print(f"Saved: {out_trc}")
            _save_edit_log()

    if edit_log:
        print("\nNext steps for pipeline:")
        print("  1. Run your LSTM augmentation on these cleaned marker TRCs")
        print("  2. Run your post-augmentation low-pass filter on the LSTM output")


def _test_spline_reload():
    """Verify spline application and marker reload flow works (no GUI)."""
    if TRCFile is None or CubicSpline is None:
        print("SKIP: TRCFile or CubicSpline not available")
        return True

    # Local-first: use discovered TRCs under PREAUGMENT_BASE_DIR_DEFAULT
    trc_paths = discover_preaugment_trcs(PREAUGMENT_BASE_DIR_DEFAULT, include_cleaned=False)
    if not trc_paths:
        print("SKIP: No TRC files discovered for test")
        return True
    trc_path = trc_paths[0]
    trc = TRCFile(trc_path)
    if not trc.marker_names:
        print("SKIP: No markers in TRC")
        return True
    m = trc.marker_names[0]
    time = np.asarray(trc.time)
    t_start, t_end = float(time[0]), float(time[-1])
    t_first = t_start + 0.2 * (t_end - t_start)
    t_last = t_start + 0.5 * (t_end - t_start)
    x0 = np.interp(t_first, time, trc.data[m + '_tx'])
    y0 = np.interp(t_first, time, trc.data[m + '_ty'])
    z0 = np.interp(t_first, time, trc.data[m + '_tz'])
    x1 = np.interp(t_last, time, trc.data[m + '_tx'])
    y1 = np.interp(t_last, time, trc.data[m + '_ty'])
    z1 = np.interp(t_last, time, trc.data[m + '_tz'])
    anchors = [(t_first, x0, y0, z0), (t_last, x1, y1, z1)]
    before = trc.data[m + '_tx'].copy()
    apply_spline_to_trc(trc, [(m, anchors)])
    after = trc.data[m + '_tx'].copy()
    changed = not np.allclose(before, after)
    mask = np.ones_like(time, dtype=bool)
    t_pct = time[mask]
    markers = {mm: {'x': np.asarray(trc.data[mm + '_tx'][mask].copy()),
                    'y': np.asarray(trc.data[mm + '_ty'][mask].copy()),
                    'z': np.asarray(trc.data[mm + '_tz'][mask].copy())}
               for mm in trc.marker_names}
    ok = np.allclose(markers[m]['x'], trc.data[m + '_tx'][mask])
    if changed and ok:
        print("PASS: spline applied and reload matches TRC")
        return True
    if not changed:
        print("WARN: spline did not change data (anchors may be degenerate)")
    if not ok:
        print("FAIL: reload does not match TRC")
    return ok


if __name__ == '__main__':
    if '--test' in sys.argv:
        ok = _test_spline_reload()
        sys.exit(0 if ok else 1)
    main()
