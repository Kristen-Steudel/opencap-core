"""
Delete OpenCap pipeline cache files so subjects can be reprocessed from scratch.

The "Load extrinsics ... already existing" message means this file still exists:
    subjectN/Videos/Cam*/cameraIntrinsicsExtrinsics.pickle

Deleting pickles under March_2 does nothing if dataDir points to March_2_alternate.

Usage:
    conda run -n opencap python ClearSessionPipelineOutputs.py

Set DRY_RUN = False to actually delete files.
"""

import glob
import os
import re
import shutil

DATA_ROOT = r"G:\Shared drives\Stanford Football\March_2_alternate"

# None = all subject* folders under DATA_ROOT.
SUBJECT_NUMBERS = [2, 3, 4]

# 'extrinsics'  -> only cameraIntrinsicsExtrinsics.pickle (force recalibration)
# 'motion'      -> extrinsics + OpenPose/MarkerData for motion trials (not ExtrinsicsTrial)
# 'full'        -> all generated pipeline outputs under each subject (keeps InputMedia mp4s)
CLEAR_LEVEL = "motion"

DRY_RUN = True

CAMERAS = ["Cam1b", "Cam4b", "Cam7b"]
RESOLUTIONS = ["default", "1x1008_4scales", "1x736_2scales", "1x732_2scales"]
POSE_SETUP = "OpenPose_default"
CAMERA_SETUP = "3-cameras"


def subject_dirs(data_root, subject_numbers=None):
    pattern = os.path.join(data_root, "subject*", "")
    paths = []
    for path in sorted(glob.glob(pattern)):
        subject = os.path.basename(path.rstrip("/\\"))
        if not re.fullmatch(r"subject\d+", subject):
            continue
        if subject_numbers is not None:
            num = int(subject.replace("subject", ""))
            if num not in subject_numbers:
                continue
        paths.append(path)
    return paths


def motion_trial_names(subject_dir):
    cam_dir = os.path.join(subject_dir, "Videos", CAMERAS[0], "InputMedia")
    if not os.path.isdir(cam_dir):
        return []
    trials = []
    for name in os.listdir(cam_dir):
        if not os.path.isdir(os.path.join(cam_dir, name)):
            continue
        if "extrinsics" in name.lower():
            continue
        trials.append(name)
    return trials


def remove_path(path, removed):
    if not os.path.exists(path):
        return
    label = path
    if DRY_RUN:
        print(f"  DELETE {label}")
    else:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        print(f"  DELETED {label}")
    removed.append(path)


def clear_extrinsics(subject_dir, removed):
    for cam in CAMERAS:
        pickle_path = os.path.join(
            subject_dir, "Videos", cam, "cameraIntrinsicsExtrinsics.pickle")
        remove_path(pickle_path, removed)


def clear_trial_openpose(subject_dir, trial, removed):
    for cam in CAMERAS:
        cam_dir = os.path.join(subject_dir, "Videos", cam)
        for res in RESOLUTIONS:
            for folder in (
                f"OutputPkl_{res}",
                f"OutputMedia_{res}",
                f"OutputJsons_{res}",
            ):
                remove_path(os.path.join(cam_dir, folder, trial), removed)
        rotated = os.path.join(
            cam_dir, "InputMedia", trial, trial + "_rotated.avi")
        remove_path(rotated, removed)


def clear_trial_markers(subject_dir, trial, removed):
    marker_root = os.path.join(subject_dir, "MarkerData")
    if not os.path.isdir(marker_root):
        return
    for pose_root in glob.glob(os.path.join(marker_root, "OpenPose*")):
        for subfolder in ("PreAugmentation", "PostAugmentation_v0.3"):
            folder = os.path.join(pose_root, CAMERA_SETUP, subfolder)
            if not os.path.isdir(folder):
                continue
            for path in glob.glob(os.path.join(folder, trial + "*")):
                remove_path(path, removed)
    remove_path(os.path.join(subject_dir, "VisualizerVideos", trial), removed)
    remove_path(
        os.path.join(subject_dir, "VisualizerJsons", trial, trial + ".json"),
        removed)


def clear_full_subject(subject_dir, removed):
    for name in (
        "MarkerData",
        "OpenSimData",
        "VisualizerVideos",
        "VisualizerJsons",
        "CleanedMarkerData",
    ):
        remove_path(os.path.join(subject_dir, name), removed)
    for cam_dir in glob.glob(os.path.join(subject_dir, "Videos", "Cam*")):
        for pattern in ("OutputPkl_*", "OutputMedia_*", "OutputJsons_*"):
            for folder in glob.glob(os.path.join(cam_dir, pattern)):
                remove_path(folder, removed)
        for avi in glob.glob(
                os.path.join(cam_dir, "InputMedia", "*", "*_rotated.avi")):
            remove_path(avi, removed)
    clear_extrinsics(subject_dir, removed)


def main():
    subjects = subject_dirs(DATA_ROOT, SUBJECT_NUMBERS)
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"CLEAR_LEVEL: {CLEAR_LEVEL}")
    print(f"DRY_RUN: {DRY_RUN}")
    print(f"Subjects: {len(subjects)}\n")

    removed = []
    for subject_dir in subjects:
        subject = os.path.basename(subject_dir.rstrip("/\\"))
        print(f"=== {subject} ===")
        if CLEAR_LEVEL == "extrinsics":
            clear_extrinsics(subject_dir, removed)
        elif CLEAR_LEVEL == "motion":
            clear_extrinsics(subject_dir, removed)
            for trial in motion_trial_names(subject_dir):
                print(f"  trial {trial}")
                clear_trial_openpose(subject_dir, trial, removed)
                clear_trial_markers(subject_dir, trial, removed)
        elif CLEAR_LEVEL == "full":
            clear_full_subject(subject_dir, removed)
        else:
            raise ValueError(
                f"Unknown CLEAR_LEVEL: {CLEAR_LEVEL}")

    print(f"\nDone. paths targeted: {len(removed)}")
    if DRY_RUN and removed:
        print("Set DRY_RUN = False to apply deletions.")


if __name__ == "__main__":
    main()
