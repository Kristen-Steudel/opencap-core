"""
Batch-trim motion-trial videos into per-stride clips using cleaned step times.

Reads left/right foot step CSVs from each subject's CleanedKinematics outputs,
builds same-foot stride intervals (consecutive contacts on one side), and writes
one trimmed video per stride per camera. Strides are numbered from the latest
stride (highest end time) as stride_1, counting up for earlier strides.

Does not modify BatchTrimVideos.py outputs or source sprint trials. Skips any
stride folder/video that already exists unless OVERWRITE_EXISTING = True.

Example output (subject2, trial ID2_S7_sprint):
    .../Videos/Cam1b/InputMedia/ID2_S7_sprint_trimmed_stride_1/
        ID2_S7_sprint_trimmed_stride_1.mp4
    .../Videos/Cam1b/InputMedia/ID2_S7_sprint_trimmed_stride_2/
        ID2_S7_sprint_trimmed_stride_2.mp4

Run:
    python Process/BatchTrimVideosByStride.py
"""

import csv
import json
import os
import re
import subprocess

# Reuse ffmpeg helpers from BatchTrimVideos (import does not run its main()).
from BatchTrimVideos import (
    FPS,
    FFMPEG_PATH,
    find_input_video,
    find_motion_trial,
    get_video_fps,
    trim_video,
    verify_frame_count,
)

# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------
DATA_ROOT = r'G:\Shared drives\Stanford Football\March_16'
CAMERAS = ['Cam1b', 'Cam4b', 'Cam7b']

# Relative to subject folder, e.g. subject2/CleanedKinematics/Outputs/
STEP_TIMES_SUBDIR = os.path.join('CleanedKinematics', 'Outputs')
STEP_TIMES_LEFT_NAME = 'step_times_left.csv'
STEP_TIMES_RIGHT_NAME = 'step_times_right.csv'

# Process only these subject IDs, or None for every subject{N} under DATA_ROOT.
ACTIVE_SUBJECTS = [2]

# If False, never overwrite an existing stride mp4 (safe default).
OVERWRITE_EXISTING = False

# Optional padding (seconds) added before/after each stride interval.
PAD_BEFORE_S = 0.0
PAD_AFTER_S = 0.0


def discover_subject_numbers(data_root):
    if not os.path.isdir(data_root):
        return []
    nums = []
    for name in os.listdir(data_root):
        match = re.fullmatch(r'subject(\d+)', name, flags=re.IGNORECASE)
        if match and os.path.isdir(os.path.join(data_root, name)):
            nums.append(int(match.group(1)))
    return sorted(nums)


def load_step_times_csv(csv_path):
    """Return sorted list of step times (seconds) from a step_times_*.csv file."""
    times = []
    if not os.path.isfile(csv_path):
        return times
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return times
        time_key = None
        for key in reader.fieldnames:
            if key.strip().lower() == 'time':
                time_key = key
                break
        if time_key is None:
            raise ValueError(f"No 'time' column in {csv_path}")
        for row in reader:
            val = row.get(time_key, '').strip()
            if val:
                times.append(float(val))
    return sorted(times)


def same_foot_stride_intervals(step_times):
    """Consecutive same-foot contacts -> (start_s, end_s) intervals."""
    if len(step_times) < 2:
        return []
    return [(step_times[i], step_times[i + 1]) for i in range(len(step_times) - 1)]


def build_numbered_strides(left_csv, right_csv, pad_before=0.0, pad_after=0.0):
    """
    Merge left/right stride intervals, sort by end time descending.
    stride_1 = latest (highest end time).
    """
    intervals = []
    for path, side in ((left_csv, 'left'), (right_csv, 'right')):
        for start_s, end_s in same_foot_stride_intervals(load_step_times_csv(path)):
            intervals.append({
                'start_s': max(0.0, start_s - pad_before),
                'end_s': end_s + pad_after,
                'side': side,
            })

    if not intervals:
        return []

    intervals.sort(key=lambda x: x['end_s'], reverse=True)
    numbered = []
    for stride_num, interval in enumerate(intervals, start=1):
        numbered.append({
            'stride_num': stride_num,
            'start_s': interval['start_s'],
            'end_s': interval['end_s'],
            'side': interval['side'],
        })
    return numbered


def stride_folder_name(trial_name, stride_num):
    return f'{trial_name}_trimmed_stride_{stride_num}'


def step_times_paths(subject_dir):
    out_dir = os.path.join(subject_dir, STEP_TIMES_SUBDIR)
    return (
        os.path.join(out_dir, STEP_TIMES_LEFT_NAME),
        os.path.join(out_dir, STEP_TIMES_RIGHT_NAME),
    )


def trim_subject_strides(data_root, cameras, subject_num):
    subject_dir = os.path.join(data_root, f'subject{subject_num}')
    videos_dir = os.path.join(subject_dir, 'Videos')
    left_csv, right_csv = step_times_paths(subject_dir)

    if not os.path.isfile(left_csv) and not os.path.isfile(right_csv):
        print(f'  SKIPPING subject {subject_num} — no step time CSVs in '
              f'{os.path.join(subject_dir, STEP_TIMES_SUBDIR)}')
        return

    strides = build_numbered_strides(
        left_csv, right_csv, pad_before=PAD_BEFORE_S, pad_after=PAD_AFTER_S)
    if not strides:
        print(f'  SKIPPING subject {subject_num} — fewer than two steps per foot; '
              'no stride intervals')
        return

    trial_name = find_motion_trial(videos_dir, cameras[0])
    if trial_name is None:
        print(f'  SKIPPING subject {subject_num} — no source motion trial found')
        return

    print(f'\n{"=" * 70}')
    print(f'Subject {subject_num} | trial {trial_name} | {len(strides)} strides')
    print(f'  Left CSV:  {left_csv}')
    print(f'  Right CSV: {right_csv}')
    print(f'{"=" * 70}')

    for stride in strides:
        stride_num = stride['stride_num']
        out_trial = stride_folder_name(trial_name, stride_num)
        start_frame = int(round(stride['start_s'] * FPS))
        end_frame = int(round(stride['end_s'] * FPS))
        frame_count = end_frame - start_frame
        if frame_count <= 0:
            print(f'  stride_{stride_num}: SKIPPING — non-positive duration '
                  f'({stride["start_s"]:.3f}-{stride["end_s"]:.3f} s)')
            continue

        print(f'\n  stride_{stride_num} ({stride["side"]}): '
              f'{stride["start_s"]:.3f}-{stride["end_s"]:.3f} s '
              f'-> frames {start_frame}-{end_frame} ({frame_count} frames)')

        for cam in cameras:
            input_dir = os.path.join(videos_dir, cam, 'InputMedia', trial_name)
            input_path = find_input_video(input_dir, trial_name)
            if input_path is None:
                print(f'    [{cam}] SKIPPING — source video not found')
                continue

            output_dir = os.path.join(
                videos_dir, cam, 'InputMedia', out_trial)
            output_path = os.path.join(output_dir, out_trial + '.mp4')

            if os.path.isfile(output_path) and not OVERWRITE_EXISTING:
                print(f'    [{cam}] EXISTS — skipping {output_path}')
                continue

            os.makedirs(output_dir, exist_ok=True)
            trim_video(input_path, start_frame, frame_count, output_path)

            actual = verify_frame_count(output_path)
            if actual and actual != frame_count:
                print(f'    [{cam}] WARNING: expected {frame_count} frames, '
                      f'wrote {actual}')


def main():
    subject_nums = discover_subject_numbers(DATA_ROOT)
    if ACTIVE_SUBJECTS is not None:
        subject_nums = [n for n in subject_nums if n in ACTIVE_SUBJECTS]
        missing = [n for n in ACTIVE_SUBJECTS if n not in subject_nums]
        for n in missing:
            print(f'WARNING: subject{n} not found under {DATA_ROOT}')

    print(f'\n{"#" * 70}')
    print('BatchTrimVideosByStride')
    print(f'Data root: {DATA_ROOT}')
    print(f'Subjects: {subject_nums}')
    print(f'Cameras: {CAMERAS}')
    print(f'Overwrite existing: {OVERWRITE_EXISTING}')
    print(f'{"#" * 70}')

    for subject_num in subject_nums:
        trim_subject_strides(DATA_ROOT, CAMERAS, subject_num)

    print(f'\n{"=" * 70}')
    print('Stride trimming complete.')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()
