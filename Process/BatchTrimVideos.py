"""
Batch-trim motion-trial videos for Stanford Football collection days.

Each collection day has its own subject trim dictionary. Subjects with no trim
note in your table are omitted and skipped. Cameras run at 120 Hz:
    seconds * 120 = frames

Output layout (same as NeutralTrialTrim):
    .../Videos/<Cam>/InputMedia/<trial>_trimmed/<trial>_trimmed.mp4

Set ACTIVE_DAYS to the day(s) you want to process. Optionally set
ACTIVE_SUBJECTS = [4] to trim only subject4. Then run:
    python Process/BatchTrimVideos.py

When a collection day has include_all_subjects=True, every subject folder
under data_root is processed. Subjects with trim notes are trimmed; subjects
without trim notes get a full-length copy renamed to <trial>_trimmed so
downstream code can always glob for *_trimmed trials.
"""

import subprocess
import os
import json
import re
import shutil

FFMPEG_PATH = 'ffmpeg'
FPS = 120

# Which collection day(s) to process. Keys must match COLLECTION_DAYS below.
ACTIVE_DAYS = ['March_16']

# Optional: limit to specific subject IDs, e.g. [4] for subject4 only.
# Set to None to process all subjects for the active day(s).
ACTIVE_SUBJECTS = None


def f(seconds):
    """Convert seconds to frame index at 120 Hz."""
    return int(round(seconds * FPS))


def T(end_s):
    """Trim from start of video to end_s seconds."""
    return {'start_frame': f(3), 'end_frame': f(end_s)}


def R(start_s, end_s):
    """Trim from start_s to end_s seconds."""
    return {'start_frame': f(start_s), 'end_frame': f(end_s)}


# ---------------------------------------------------------------------------
# Collection-day configs
# ---------------------------------------------------------------------------
COLLECTION_DAYS = {
    'March_16': {
        'data_root': r'G:\Shared drives\Stanford Football\March_16',
        'cameras': ['Cam1b', 'Cam4b', 'Cam7b'],
        'subject_trims': {
            13: R(4.571, 5.058)
            # 2: T(5.5),
            # 3: T(5.5),
            # 4: T(5.5),
            # 6: T(5.5),
            # 7: T(5.5),
            # 8: T(5.5),
            # 9: T(5.5),
            # 11: T(5.5),
            # 12: R(4, 9),          # trim false start (seconds 4–9)
            # 16: R(3, 6),          # trim start (seconds 3–6)
            # 22: T(5),             # trim to end at 5 s
            # 23: T(5.5),
            # 25: T(5.5),
            # 26: T(6),
            # 27: R(1, 5.5),
            # 28: R(1, 5.5),
            # 29: T(4.5),
            # 31: R(1, 5),
            # 32: T(5),
            # 33: R(1, 5),
            # 34: R(2, 6),
            # 36: T(5),
            # 41: T(5),
            # 46: R(2, 5.5),
            # 47: R(1, 5),
            # 49: R(2, 6.5),
            # 51: R(1, 6),
            # 52: R(4, 7.5),
            # 53: T(5),
            # 54: R(1, 5),
            # 55: R(1, 5),
            # 56: R(1, 5),
            # 57: R(2, 6),
            # 59: T(5),
            # 60: R(1.5, 5),
            # 62: T(4.5),
            # 63: T(4.5),
            # 64: T(4.5),
            # 65: R(0.5, 5.5),
            # 66: R(1, 5.5),
            # 68: T(4.5),
            # 18: "can trim end" — no specific range, skipped
        },
    },

    'March_2': {
        'data_root': r'G:\Shared drives\Stanford Football\March_2',
        'cameras': ['Cam1b', 'Cam4b', 'Cam7b'],
        # Process every subject{N} folder; subjects without trim notes get a
        # full-length copy under <trial>_trimmed (same duration, renamed only).
        'include_all_subjects': True,
        'subject_trims': {
            4:  T(7),
            8:  T(7.5),
            9:  R(2.5, 7.5),
            11: T(6.5),
            12: R(2, 7.5),
            13: R(1, 6),
            16: R(1, 6),
            19: R(2, 6.5),
            22: T(5.5),
            25: T(6),
            26: R(1, 6.5),
            27: T(5.5),
            28: R(1, 6.5),
            29: R(1, 6),
            31: R(1, 6),
            32: T(6.5),
            33: T(6.5),
            34: R(1, 6),
            35: R(2, 7),
            36: R(1, 6.5),
            41: R(1, 6.5),
            44: R(1, 6),
            45: R(1, 6.5),
            46: R(1, 6),
            47: R(1, 6.5),
            49: R(1, 6),
            51: R(1, 7),
            52: R(2, 7),
            53: R(1, 6),
            54: R(2, 6.5),
            55: R(1, 6),
            56: R(2, 6.5),
            57: R(2, 6.5),
            59: R(2, 6),
            60: R(1, 6),
            62: R(1, 7),
            63: R(1, 6),
            64: R(2, 7),
            65: R(1, 6.5),
            66: R(1, 6),
            68: R(2, 6.5),
        },
    },

    'February_23': {
        'data_root': r'G:\Shared drives\Stanford Football\February_23',
        'cameras': ['Cam1b', 'Cam4b', 'Cam7b'],
        'subject_trims': {
            2:  R(1, 5.5),
            3:  T(5.5),
            4:  R(1, 5.5),
            5:  T(5),
            7:  R(5, 9.5),
            8:  R(2, 8),
            9:  R(2, 6.5),
            11: T(6.5),
            12: R(1, 6),
            13: T(6),
            16: R(1, 6.5),
            19: R(2, 7.5),
            22: R(2, 6.5),
            25: T(5.5),
            26: R(1, 7),
            27: R(1.5, 6.5),
            28: R(1.5, 6.5),
            29: R(1.5, 6),
            31: R(1, 6),
            32: R(2, 6.5),
            33: R(2, 8),
            34: T(5.5),
            36: R(1, 5.5),
            41: R(1, 6),
            44: R(1, 6),
            45: R(1, 5.5),
            46: R(1.5, 6.5),
            47: R(3, 8),
            49: R(2, 6.5),
            51: R(1, 6),
            52: R(1, 6),
            54: R(1, 6),
            55: R(2, 7),
            56: R(2, 6),
            57: R(2, 6),
            58: R(2, 6),
            59: R(1, 6),
            60: R(1, 5.5),
            62: R(2, 7),
            63: R(2, 6),
            66: R(3, 7),
            68: R(1, 6),
        },
    },

    'February_9': {
        'data_root': r'G:\Shared drives\Stanford Football\February_9',
        'cameras': ['Cam1b', 'Cam4b', 'Cam7b'],
        'subject_trims': {
            2:  T(6), #  WARNING: Expected 360 but got 180!
            3:  R(1, 6),
            4:  R(1, 6),
            5:  R(1, 6),
            6:  R(3, 8),
            7:  R(1, 6),
            8:  R(1, 7),
            9:  R(5, 11),
            10: R(1, 6),
            11: R(1, 6),
            12: R(1, 6),
            14: R(3, 7.5),
            16: T(5),
            17: T(5.5),
            18: T(5.5),
            19: R(1, 7),
            25: T(6),
            29: R(1, 7),
            31: R(1, 6),
            32: T(6),
            33: R(1, 6),
            34: R(1, 6),
            36: T(6),
            38: R(1, 6),
            39: R(1, 6.5),
            41: R(1, 6),
            45: R(1, 6),
            47: R(8, 13),
            49: R(1, 6),
            51: R(13, 19),
            52: R(2, 7),
            53: T(6),
            54: R(1, 6),
            56: T(6),
            57: R(12, 17),
            58: T(6),
            60: R(1, 6),
            62: R(8, 13),
            63: R(2, 6.5),
            64: R(1, 6),
            66: R(1, 6),
            # 23, 55: MISSING VIDEOS — skipped
        },
    },

    'February_2': {
        'data_root': r'G:\Shared drives\Stanford Football\February_2',
        'cameras': ['Cam1b', 'Cam4b', 'Cam7b'],
        'subject_trims': {
            2:  T(6),
            3:  T(6),
            4:  T(6),
            5:  T(6),
            7:  T(6),
            8:  T(6),
            9:  T(7),
            10:  T(6),
            11:  T(6),
            12: T(7),
            13: T(7),
            14: T(7),
            16: T(6),
            17: R(3, 8),
            18: T(6),
            19: R(2, 7),
            22: T(7),
            23: T(6),
            25: R(1, 7),
            26: T(8),
            27: T(6),
            28: T(6),
            29: R(2, 8),
            31: T(6),
            32: T(6),
            33: R(3, 8),
            34: T(6),
            36: R(1, 7),
            38: R(4, 10),
            39: T(6.5),
            41: T(6),
            44: R(1, 9),
            45: R(1, 6),
            46: R(1, 6),
            47: R(1, 8),
            48: T(7),
            49: R(1, 6),
            51: R(1, 6),
            52: R(5, 11),
            53: R(2, 7),
            54: R(1, 6),
            55: R(2, 6), 
            56: R(2, 7), 
            57: T(6.5),          # notes say 6.5 or 7 s — using 6.5
            58: T(7.5),
            60: T(7),
            62: R(2, 6),
            63: R(2, 6), 
            64: R(3, 7),
            66: R(1, 5),
            69: R(1, 8),
            # 14: ran through, no decel — no trim range given, skipped
            # 33: incorrect motion — skipped
        },
    },
}


# ---------------------------------------------------------------------------
# Trim helpers (same logic as NeutralTrialTrim / OpenCapCameraTestTrim)
# ---------------------------------------------------------------------------
def verify_frame_count(video_file):
    try:
        command = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-count_frames', '-show_entries', 'stream=nb_read_frames',
            '-print_format', 'json', video_file,
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        return int(info['streams'][0]['nb_read_frames'])
    except Exception as e:
        print(f"  Warning: Could not verify frame count: {e}")
        return None


def get_video_fps(input_file):
    try:
        command = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', '-select_streams', 'v:0', input_file,
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        streams = json.loads(result.stdout).get('streams', [])
        if not streams:
            return None
        for key in ('avg_frame_rate', 'r_frame_rate'):
            frac = streams[0].get(key)
            if frac and '/' in frac:
                num, den = frac.split('/')
                fps = float(num) / float(den)
                if fps > 0:
                    return fps
        return None
    except Exception as e:
        print(f"WARNING: Could not detect FPS for {input_file}: {e}")
        return None


def trim_video(input_file, start_frame, frame_count, output_file, fps=None):
    if fps is None:
        fps = get_video_fps(input_file)
        if fps is None:
            print(f"WARNING: Using default {FPS} fps for {input_file}")
            fps = float(FPS)

    select_filter = f'select=gte(n\\,{start_frame}),setpts=PTS-STARTPTS'
    command = [
        FFMPEG_PATH, '-y', '-i', input_file,
        '-vf', select_filter,
        '-frames:v', str(frame_count),
        '-r', f'{fps:.6f}',
        '-c:v', 'libx264', '-crf', '18',
        '-g', '30', '-keyint_min', '30',
        '-force_key_frames', 'expr:gte(n,0)',
        '-movflags', '+faststart', '-an',
        output_file,
    ]

    start_time = start_frame / fps
    duration = frame_count / fps
    print(f"\n  Processing: {os.path.basename(input_file)}")
    print(f"    Frames {start_frame}-{start_frame + frame_count} "
          f"({frame_count} frames, {start_time:.1f}-{start_time + duration:.1f} s)")
    print(f"    Output: {output_file}")

    try:
        subprocess.run(command, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("    SUCCESS")
    except subprocess.CalledProcessError as e:
        print(f"    ERROR: {e.stderr.decode()}")
    except FileNotFoundError:
        print(f"    ERROR: FFmpeg not found at '{FFMPEG_PATH}'")


def find_motion_trial(videos_dir, ref_cam):
    """Return the motion-trial folder name under InputMedia (auto-detect)."""
    input_media = os.path.join(videos_dir, ref_cam, 'InputMedia')
    if not os.path.isdir(input_media):
        return None

    skip = ('extrinsics', 'static', 'trimmed')
    trials = sorted([
        t for t in os.listdir(input_media)
        if os.path.isdir(os.path.join(input_media, t))
        and not any(s in t.lower() for s in skip)
    ])

    if len(trials) == 1:
        return trials[0]
    if len(trials) > 1:
        print(f"  WARNING: Multiple motion trials found {trials}; using '{trials[0]}'")
        return trials[0]
    return None


def find_input_video(trial_dir, trial_name):
    for ext in ('.mp4', '.avi', '.mov'):
        path = os.path.join(trial_dir, trial_name + ext)
        if os.path.isfile(path):
            return path
    return None


def discover_subject_numbers(data_root):
    """Return sorted subject IDs from subject{N} folders under data_root."""
    if not os.path.isdir(data_root):
        return []
    subject_nums = []
    for name in os.listdir(data_root):
        match = re.fullmatch(r'subject(\d+)', name, flags=re.IGNORECASE)
        if match and os.path.isdir(os.path.join(data_root, name)):
            subject_nums.append(int(match.group(1)))
    return sorted(subject_nums)


def copy_video_unchanged(input_file, output_file):
    """Copy the full source video to the trimmed output path (no re-encode)."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"\n  Copying (no trim): {os.path.basename(input_file)}")
    print(f"    Output: {output_file}")
    shutil.copy2(input_file, output_file)
    print("    SUCCESS")


def copy_subject_as_trimmed(day_name, data_root, cameras, subject_num):
    """Create <trial>_trimmed outputs as full-length copies of the source trial."""
    videos_dir = os.path.join(data_root, f'subject{subject_num}', 'Videos')
    trial_name = find_motion_trial(videos_dir, cameras[0])
    if trial_name is None:
        print(f"  SKIPPING subject {subject_num} — no motion trial found under {videos_dir}")
        return

    output_trial_name = trial_name + '_trimmed'

    print(f"\n{'='*70}")
    print(f"{day_name} | Subject {subject_num} | trial '{trial_name}' "
          f"-> full-length copy as '{output_trial_name}'")
    print(f"{'='*70}")

    for cam in cameras:
        input_dir = os.path.join(videos_dir, cam, 'InputMedia', trial_name)
        input_path = find_input_video(input_dir, trial_name)

        if input_path is None:
            print(f"  [{cam}] SKIPPING — input not found in {input_dir}")
            continue

        output_dir = os.path.join(videos_dir, cam, 'InputMedia', output_trial_name)
        output_path = os.path.join(output_dir, output_trial_name + '.mp4')
        copy_video_unchanged(input_path, output_path)

        actual_frames = verify_frame_count(output_path)
        source_frames = verify_frame_count(input_path)
        if actual_frames and source_frames:
            print(f"    Verified: {actual_frames} frames copied "
                  f"(source had {source_frames})")
            if actual_frames != source_frames:
                print("    WARNING: Copied frame count differs from source!")


def trim_subject(day_name, data_root, cameras, subject_num, trim_cfg):
    start_frame = trim_cfg['start_frame']
    end_frame = trim_cfg['end_frame']
    frame_count = end_frame - start_frame
    videos_dir = os.path.join(data_root, f'subject{subject_num}', 'Videos')

    trial_name = find_motion_trial(videos_dir, cameras[0])
    if trial_name is None:
        print(f"  SKIPPING subject {subject_num} — no motion trial found under {videos_dir}")
        return

    output_trial_name = trial_name + '_trimmed'

    print(f"\n{'='*70}")
    print(f"{day_name} | Subject {subject_num} | trial '{trial_name}' "
          f"-> frames {start_frame}-{end_frame} "
          f"({frame_count} frames, {start_frame/FPS:.1f}-{end_frame/FPS:.1f} s)")
    print(f"{'='*70}")

    for cam in cameras:
        input_dir = os.path.join(videos_dir, cam, 'InputMedia', trial_name)
        input_path = find_input_video(input_dir, trial_name)

        if input_path is None:
            print(f"  [{cam}] SKIPPING — input not found in {input_dir}")
            continue

        output_dir = os.path.join(videos_dir, cam, 'InputMedia', output_trial_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_trial_name + '.mp4')

        trim_video(input_path, start_frame, frame_count, output_path)

        actual_frames = verify_frame_count(output_path)
        if actual_frames:
            print(f"    Verified: {actual_frames} frames written")
            if actual_frames != frame_count:
                print(f"    WARNING: Expected {frame_count} but got {actual_frames}!")


def process_day(day_name):
    cfg = COLLECTION_DAYS[day_name]
    subject_trims = cfg['subject_trims']
    include_all_subjects = cfg.get('include_all_subjects', False)

    if include_all_subjects:
        subject_nums = discover_subject_numbers(cfg['data_root'])
    else:
        subject_nums = sorted(subject_trims)

    if ACTIVE_SUBJECTS is not None:
        subject_nums = [n for n in subject_nums if n in ACTIVE_SUBJECTS]
        missing = [n for n in ACTIVE_SUBJECTS if n not in subject_nums]
        for n in missing:
            print(f"  WARNING: subject{n} not found under {cfg['data_root']}")

    print(f"\n{'#'*70}")
    print(f"Collection day: {day_name}")
    print(f"Data root: {cfg['data_root']}")
    print(f"Subjects to process: {len(subject_nums)}")
    if ACTIVE_SUBJECTS is not None:
        print(f"  Filtered to ACTIVE_SUBJECTS: {ACTIVE_SUBJECTS}")
    if include_all_subjects:
        untrimmed = [n for n in subject_nums if n not in subject_trims]
        print(f"  With trim notes: {len(subject_trims)}")
        print(f"  Full-length copy as _trimmed: {len(untrimmed)}")
    print(f"{'#'*70}")

    for subject_num in subject_nums:
        if subject_num in subject_trims:
            trim_subject(
                day_name, cfg['data_root'], cfg['cameras'],
                subject_num, subject_trims[subject_num],
            )
        elif include_all_subjects:
            copy_subject_as_trimmed(
                day_name, cfg['data_root'], cfg['cameras'], subject_num,
            )
        else:
            print(f"  SKIPPING subject {subject_num} — no trim note for this day")


def main():
    for day_name in ACTIVE_DAYS:
        if day_name not in COLLECTION_DAYS:
            raise ValueError(
                f"Unknown day '{day_name}'. "
                f"Choose from: {list(COLLECTION_DAYS.keys())}")
        process_day(day_name)

    print(f"\n{'='*70}")
    print("All requested collection days complete!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()