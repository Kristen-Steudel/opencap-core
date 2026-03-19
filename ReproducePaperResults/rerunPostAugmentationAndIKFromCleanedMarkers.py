"""
Rerun post-augmentation marker creation + IK using cleaned PreAugmentation TRCs.

This script is meant to be run AFTER `clean_preaugment_markers.py`.

What it does per subject:
1. Finds cleaned TRCs under:
   <subjectDir>/MarkerData/OpenPose_<resolutionPoseDetection>/<cameraSetup>/PreAugmentation/Cleaned/*.trc
2. Groups multiple cleaned versions per trial and selects the latest one.
3. Runs marker augmentation (`augmentTRC`) on the cleaned TRC.
4. Runs inverse kinematics (`runIKTool`) on the augmented (post-augmentation) TRC.
5. Writes outputs to:
   - <subjectDir>/CleanedMarkerData/.../PostAugmentation_<augmenterModel>/*.trc
   - <subjectDir>/CleanedKinematics/.../Kinematics/*.mot

Notes / assumptions:
- This script assumes a scaled OpenSim model already exists (the normal pipeline runs `scaleModel=False`
  for dynamic/walking trials and relies on scaling from the static trial).
- It currently targets OpenPose results (matching how your cleaning/paper scripts are configured on Windows).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Ensure repo root is on sys.path before local imports (utils.py, etc.).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils import importMetadata
from utilsAugmenter import augmentTRC
from utilsOpenSim import runIKTool


@dataclass(frozen=True)
class CleanedTrialInput:
    cleaned_trc_path: str
    trial_id: str  # trial_id used for augmented output naming
    cleaned_base_key: str  # cleaned grouping key (original pre-aug stem without timestamp suffix)


_CLEANED_TIMESTAMP_RE = re.compile(
    r"^(?P<base>.+)_(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$"
)


def _pick_latest_by_base(cleaned_trc_paths: List[str]) -> Dict[str, str]:
    """
    Returns mapping: base_key -> latest cleaned_trc_path
    where base_key is the cleaned grouping key (timestamp suffix removed when possible).
    """
    latest: Dict[str, Tuple[float, str]] = {}
    for p in cleaned_trc_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        m = _CLEANED_TIMESTAMP_RE.match(stem)
        base_key = m.group("base") if m else stem
        mtime = os.path.getmtime(p)
        prev = latest.get(base_key)
        if prev is None or mtime > prev[0]:
            latest[base_key] = (mtime, p)
    return {k: v[1] for k, v in latest.items()}


def _derive_trial_id_from_preaug_stem(preaug_stem: str) -> str:
    """
    main.py writes pre-aug TRC as: <trial_id> + "NoSync.trc"
    so we strip the "NoSync" suffix to recover trial_id.
    """
    if preaug_stem.endswith("NoSync"):
        return preaug_stem[: -len("NoSync")]
    return preaug_stem


def _discover_openpose_cleaned_dirs(
    subject_dir: str,
    camera_setups: Optional[List[str]] = None,
    resolution_pose_detection: Optional[List[str]] = None,
) -> List[Tuple[str, str, str]]:
    """
    Finds directories:
      MarkerData/OpenPose_<resolutionPoseDetection>/<cameraSetup>/PreAugmentation/Cleaned

    Returns list of tuples: (pose_dir_name, camera_setup, cleaned_preaug_dir)
    where pose_dir_name looks like: OpenPose_default / OpenPose_1x1008_4scales, ...
    """
    marker_data_root = os.path.join(subject_dir, "MarkerData")
    if not os.path.isdir(marker_data_root):
        return []

    out: List[Tuple[str, str, str]] = []
    for pose_dir in os.listdir(marker_data_root):
        pose_dir_path = os.path.join(marker_data_root, pose_dir)
        if not os.path.isdir(pose_dir_path):
            continue
        if not pose_dir.startswith("OpenPose_"):
            continue
        resolution = pose_dir[len("OpenPose_") :]

        if resolution_pose_detection is not None and resolution not in resolution_pose_detection:
            continue

        for camera_setup in os.listdir(pose_dir_path):
            camera_setup_path = os.path.join(pose_dir_path, camera_setup)
            if not os.path.isdir(camera_setup_path):
                continue
            if camera_setups is not None and camera_setup not in camera_setups:
                continue

            cleaned_dir = os.path.join(
                camera_setup_path, "PreAugmentation", "Cleaned"
            )
            if os.path.isdir(cleaned_dir):
                out.append((pose_dir, camera_setup, cleaned_dir))
    return out


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _file_exists_nonempty(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def run_for_subject(
    subject_dir: str,
    repo_root: str,
    dest_marker_root: str,
    dest_kin_root: str,
    augmenter_model: str,
    camera_setups: Optional[List[str]],
    resolution_pose_detection: Optional[List[str]],
    trial_ids: Optional[List[str]],
    overwrite: bool,
) -> None:
    session_metadata_path = os.path.join(subject_dir, "sessionMetadata.yaml")
    if not os.path.isfile(session_metadata_path):
        raise FileNotFoundError(f"Missing session metadata: {session_metadata_path}")

    sessionMetadata = importMetadata(session_metadata_path)

    mass_kg = float(sessionMetadata["mass_kg"])
    height_m = float(sessionMetadata["height_m"])
    openSimModel = sessionMetadata["openSimModel"]

    # In main.py:
    # - PostAugmentation folder name uses `augmentermodel` (webapp/config) or `augmenter_model` argument (e.g. v0.2).
    # - TRC output filename suffix uses `markerAugmentationSettings.markerAugmenterModel` (e.g. LSTM).
    markerAugmenterModelName = sessionMetadata["markerAugmentationSettings"]["markerAugmenterModel"]
    if not isinstance(markerAugmenterModelName, str):
        markerAugmenterModelName = str(markerAugmenterModelName)

    augmenterModel = sessionMetadata.get("augmentermodel", augmenter_model)
    if not isinstance(augmenterModel, str):
        augmenterModel = str(augmenterModel)

    pose_detector_dirs = _discover_openpose_cleaned_dirs(
        subject_dir=subject_dir,
        camera_setups=camera_setups,
        resolution_pose_detection=resolution_pose_detection,
    )
    if not pose_detector_dirs:
        print(f"[{os.path.basename(subject_dir)}] No cleaned PreAugmentation TRCs found.")
        return

    # OpenSim IK setup file matches main.py:
    suffix_model = "_shoulder" if "shoulder" in openSimModel else ""
    ik_setup_filename = f"Setup_IK{suffix_model}.xml"  # suffix_model is '' or '_shoulder'
    ik_setup_path = os.path.join(repo_root, "opensimPipeline", "IK", ik_setup_filename)
    if not os.path.isfile(ik_setup_path):
        raise FileNotFoundError(f"Missing IK setup file: {ik_setup_path}")

    augmenterDir = os.path.join(repo_root, "MarkerAugmenter")
    if not os.path.isdir(augmenterDir):
        raise FileNotFoundError(f"Missing MarkerAugmenter dir: {augmenterDir}")

    for pose_dir_name, camera_setup, cleaned_preaug_dir in pose_detector_dirs:
        cleaned_trc_paths = [
            os.path.join(cleaned_preaug_dir, fn)
            for fn in os.listdir(cleaned_preaug_dir)
            if fn.lower().endswith(".trc")
        ]
        if not cleaned_trc_paths:
            print(f"[{os.path.basename(subject_dir)}] {cleaned_preaug_dir} has no .trc files.")
            continue

        latest_by_base = _pick_latest_by_base(cleaned_trc_paths)
        trial_inputs: List[CleanedTrialInput] = []
        for base_key, latest_path in latest_by_base.items():
            trial_id = _derive_trial_id_from_preaug_stem(base_key)
            trial_inputs.append(
                CleanedTrialInput(
                    cleaned_trc_path=latest_path,
                    trial_id=trial_id,
                    cleaned_base_key=base_key,
                )
            )

        # Paths for scaled model (must already exist).
        # main.py computes openSimFolderName = OpenSimData/<pose_dir_name>/<camera_setup> (for OpenPose)
        openSimDir = os.path.join(subject_dir, "OpenSimData", pose_dir_name, camera_setup)
        scaled_model_path = os.path.join(openSimDir, "Model", f"{openSimModel}_scaled.osim")
        if not os.path.isfile(scaled_model_path):
            raise FileNotFoundError(
                f"Scaled OpenSim model not found (needed for IK): {scaled_model_path}\n"
                f"Make sure static trial scaling has been run successfully before re-running IK."
            )

        for inp in sorted(trial_inputs, key=lambda x: x.trial_id):
            if trial_ids is not None and inp.trial_id not in trial_ids:
                continue
            augmented_filename = f"{inp.trial_id}_{markerAugmenterModelName}.trc"
            augmented_out_dir = os.path.join(
                dest_marker_root,
                pose_dir_name,
                camera_setup,
                f"PostAugmentation_{augmenterModel}",
            )
            augmented_out_trc = os.path.join(augmented_out_dir, augmented_filename)

            motion_out_dir = os.path.join(
                dest_kin_root, pose_dir_name, camera_setup, "Kinematics"
            )
            _ensure_dir(augmented_out_dir)
            _ensure_dir(motion_out_dir)

            mot_expected_path = os.path.join(
                motion_out_dir, f"{os.path.splitext(augmented_filename)[0]}.mot"
            )

            if not overwrite and _file_exists_nonempty(augmented_out_trc) and _file_exists_nonempty(mot_expected_path):
                print(
                    f"[{os.path.basename(subject_dir)}] Skip (exists) trial={inp.trial_id}"
                )
                continue

            if overwrite:
                # Re-run cleanly by deleting expected outputs if present.
                for p in [augmented_out_trc, mot_expected_path]:
                    try:
                        if os.path.isfile(p):
                            os.remove(p)
                    except OSError:
                        pass

            print(
                f"[{os.path.basename(subject_dir)}] Trial {inp.trial_id}: "
                f"cleaned='{os.path.basename(inp.cleaned_trc_path)}' -> augmented + IK"
            )

            _ = augmentTRC(
                inp.cleaned_trc_path,
                mass_kg,
                height_m,
                augmented_out_trc,
                augmenterDir=augmenterDir,
                augmenterModelName=markerAugmenterModelName,
                augmenter_model=augmenter_model,
                offset=True,
            )

            runIKTool(
                ik_setup_path,
                scaled_model_path,
                augmented_out_trc,
                motion_out_dir,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerun augmentation + IK using cleaned PreAugmentation marker TRCs."
    )
    parser.add_argument(
        "--data-root",
        default=r"G:\Shared drives\Stanford Football\March_2",
        help="Dataset root containing subject folders like subject2, subject3, ...",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        type=int,
        default=[2],
        help="Subject numbers (e.g., --subjects 2 3 4).",
    )
    parser.add_argument(
        "--augmenter-model",
        default="v0.2",
        help="Feature-set version passed to augmentTRC (e.g., v0.1/v0.2).",
    )
    parser.add_argument(
        "--camera-setup",
        nargs="*",
        default=None,
        help="Filter which camera setup(s) to process (e.g., --camera-setup 3-cameras).",
    )
    parser.add_argument(
        "--resolution-pose",
        nargs="*",
        default=None,
        help="Filter which OpenPose resolution(s) to process (e.g., --resolution-pose default 1x1008_4scales).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing augmented TRCs and IK .mot outputs in CleanedMarkerData/CleanedKinematics.",
    )
    parser.add_argument(
        "--trial-id",
        nargs="*",
        default=None,
        help="Optional trial_id filter (e.g., --trial-id 12026d04-... or --trial-id trialA trialB).",
    )

    args = parser.parse_args()

    repo_root = REPO_ROOT
    # Ensure repo root is importable when running from a different working directory.
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    for subj in args.subjects:
        subject_dir = os.path.join(args.data_root, f"subject{subj}")
        if not os.path.isdir(subject_dir):
            print(f"[subject{subj}] Missing directory: {subject_dir}")
            continue

        dest_marker_root = os.path.join(subject_dir, "CleanedMarkerData")
        dest_kin_root = os.path.join(subject_dir, "CleanedKinematics")
        _ensure_dir(dest_marker_root)
        _ensure_dir(dest_kin_root)

        print(f"=== Subject {subj} ===")
        run_for_subject(
            subject_dir=subject_dir,
            repo_root=repo_root,
            dest_marker_root=dest_marker_root,
            dest_kin_root=dest_kin_root,
            augmenter_model=args.augmenter_model,
            camera_setups=args.camera_setup,
            resolution_pose_detection=args.resolution_pose,
            trial_ids=args.trial_id,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()

