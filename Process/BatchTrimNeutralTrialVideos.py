import subprocess
import os
import json

# --- Configuration ---
DATA_ROOT  = r'G:\Shared drives\Stanford Football\January_9'
TRIAL_NAME = 'static'
CAMERAS    = ['Cam1', 'Cam4', 'Cam7']
FFMPEG_PATH = 'ffmpeg'

# Trim settings per subject. Cameras record at 120 Hz so seconds * 120 = frames.
# Subjects not listed here have no trim and are skipped entirely.
#
#   start_frame : first frame to keep  (e.g. 1 s → 120)
#   end_frame   : first frame NOT kept (e.g. 4 s → 480), frame count = end - start
#
# Time ranges used:
#   "trim to 3 s"     →   0 – 360  (3 s window from the very start)
#   "trim to 1–4 s"   → 120 – 480  (cut moving/head-turn at start)
#   "trim to 2–5 s"   → 240 – 600  (cut motion at start)
#   "trim to 3–5 s"   → 360 – 600  (cut motion at start, keep still section)
SUBJECT_TRIMS = {
    16: {'start_frame':   0, 'end_frame': 360},   # trim to 3 s
    17: {'start_frame':   0, 'end_frame': 360},   # trim to 3 s
    18: {'start_frame': 240, 'end_frame': 600},   # trim to 2–5 s
    20: {'start_frame':   0, 'end_frame': 360},   # trim to 3 s
    22: {'start_frame': 120, 'end_frame': 480},   # trim to 1–4 s (cut moving)
    23: {'start_frame':   0, 'end_frame': 360},   # trim to 3 s
    24: {'start_frame':   0, 'end_frame': 360},   # trim to 3 s
    25: {'start_frame':   0, 'end_frame': 360},   # trim to 3 s
    28: {'start_frame': 240, 'end_frame': 600},   # trim to 2–5 s
    29: {'start_frame':   0, 'end_frame': 360},   # trim to 3 s (head nod after 3 s)
    31: {'start_frame': 120, 'end_frame': 480},   # trim to 1–4 s (step at start)
    32: {'start_frame':   0, 'end_frame': 360},   # trim to 3 s
    33: {'start_frame':   0, 'end_frame': 360},   # trim to 3 s
    34: {'start_frame': 240, 'end_frame': 600},   # trim to 2–5 s (head rotation at start)
    36: {'start_frame':   0, 'end_frame': 360},   # trim to 3 s
    38: {'start_frame':   0, 'end_frame': 360},   # trim to 3 s
    39: {'start_frame':   0, 'end_frame': 360},   # trim to 3 s
    43: {'start_frame': 360, 'end_frame': 600},   # trim to 3–5 s
    44: {'start_frame':   0, 'end_frame': 360},   # trim to 3 s
    45: {'start_frame':   0, 'end_frame': 360},   # trim to 3 s
    46: {'start_frame': 240, 'end_frame': 600},   # trim to 2–5 s
    47: {'start_frame': 360, 'end_frame': 600},   # trim to 3–5 s
    49: {'start_frame':   0, 'end_frame': 360},   # trim to 3 s
    53: {'start_frame':   0, 'end_frame': 360},   # trim to 3 s
    54: {'start_frame': 120, 'end_frame': 480},   # trim to 1–4 s (steps at start)
    56: {'start_frame':   0, 'end_frame': 360},   # trim to 3 s (different extrinsics)
    57: {'start_frame': 120, 'end_frame': 480},   # trim to 1–4 s (head turns at start)
    58: {'start_frame': 240, 'end_frame': 600},   # trim to 2–5 s (opens hands at start)
    59: {'start_frame': 120, 'end_frame': 480},   # trim to 1–4 s (steps at start)
    60: {'start_frame': 120, 'end_frame': 480},   # trim to 1–4 s (steps at start)
    63: {'start_frame':   0, 'end_frame': 360},   # trim to 3 s
    64: {'start_frame':   0, 'end_frame': 360},   # trim to 3 s
    65: {'start_frame':   0, 'end_frame': 360},   # trim to 3 s (odd elbow bend)
    68: {'start_frame':   0, 'end_frame': 360},   # trim to 3 s
}
# ---


def verify_frame_count(video_file):
    """Counts actual frames in output video."""
    try:
        command = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-count_frames',
            '-show_entries', 'stream=nb_read_frames',
            '-print_format', 'json',
            video_file
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        return int(info['streams'][0]['nb_read_frames'])
    except Exception as e:
        print(f"  Warning: Could not verify frame count: {e}")
        return None


def get_video_fps(input_file):
    """Detects the FPS of a video using ffprobe."""
    try:
        command = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            '-select_streams', 'v:0',
            input_file
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        streams = info.get('streams', [])
        if not streams:
            return None
        stream = streams[0]
        for key in ('avg_frame_rate', 'r_frame_rate'):
            if key in stream:
                frac = stream.get(key)
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
    """Trims a video to a specified frame range.

    Uses a select filter (not -ss seeking) so that:
      - The cut is perfectly frame-accurate regardless of keyframe positions.
      - The output always starts with a full keyframe (no corrupt/undecodable
        first frames, which happen when -ss is placed after -i).
      - No time/FPS arithmetic is involved, so cameras with slightly different
        reported frame rates are all cut to the exact same content frame.
    """
    if fps is None:
        fps = get_video_fps(input_file)
        if fps is None:
            print(f"WARNING: Using default 120 fps for {input_file}")
            fps = 120.0

    select_filter = f'select=gte(n\\,{start_frame}),setpts=PTS-STARTPTS'

    command = [
        FFMPEG_PATH,
        '-y',
        '-i', input_file,
        '-vf', select_filter,
        '-frames:v', str(frame_count),
        '-r', f'{fps:.6f}',
        '-c:v', 'libx264',
        '-crf', '18',
        '-g', '30',
        '-keyint_min', '30',
        '-force_key_frames', 'expr:gte(n,0)',
        '-movflags', '+faststart',
        '-an',
        output_file
    ]

    start_time = start_frame / fps
    duration   = frame_count / fps
    print(f"\n  Processing: {os.path.basename(input_file)}")
    print(f"    Frames {start_frame}–{start_frame + frame_count} "
          f"({frame_count} frames, {start_time:.1f}–{start_time + duration:.1f} s)")
    print(f"    Output: {output_file}")

    try:
        subprocess.run(command, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"    SUCCESS")
    except subprocess.CalledProcessError as e:
        print(f"    ERROR: {e.stderr.decode()}")
    except FileNotFoundError:
        print(f"    ERROR: FFmpeg not found at '{FFMPEG_PATH}'")


def trim_subject(subject_num, start_frame, end_frame):
    """Trim the static trial for one subject across all cameras."""
    frame_count       = end_frame - start_frame
    base_dir          = os.path.join(DATA_ROOT, f'subject{subject_num}', 'Videos')
    output_trial_name = TRIAL_NAME + '_trimmed'

    print(f"\n{'='*70}")
    print(f"Subject {subject_num} — frames {start_frame}–{end_frame} "
          f"({frame_count} frames, {start_frame/120:.1f}–{end_frame/120:.1f} s)")
    print(f"{'='*70}")

    for cam in CAMERAS:
        input_dir  = os.path.join(base_dir, cam, 'InputMedia', TRIAL_NAME)
        input_path = os.path.join(input_dir, TRIAL_NAME + '.mp4')

        if not os.path.isfile(input_path):
            print(f"  [{cam}] SKIPPING — input not found: {input_path}")
            continue

        # Save to a NEW trial folder so OpenCap treats it as a separate trial.
        output_dir  = os.path.join(base_dir, cam, 'InputMedia', output_trial_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_trial_name + '.mp4')

        trim_video(input_path, start_frame, frame_count, output_path)

        actual_frames = verify_frame_count(output_path)
        if actual_frames:
            print(f"    Verified: {actual_frames} frames written")
            if actual_frames != frame_count:
                print(f"    WARNING: Expected {frame_count} but got {actual_frames}!")


def main():
    """Loop through all subjects that need trimming and process each one."""
    print(f"{'='*70}")
    print(f"Batch neutral trial trim — {len(SUBJECT_TRIMS)} subjects to process")
    print(f"{'='*70}")

    for subject_num in sorted(SUBJECT_TRIMS):
        cfg = SUBJECT_TRIMS[subject_num]
        trim_subject(subject_num, cfg['start_frame'], cfg['end_frame'])

    print(f"\n{'='*70}")
    print("All subjects complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()