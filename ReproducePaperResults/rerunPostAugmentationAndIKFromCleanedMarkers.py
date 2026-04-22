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


def _parse_pose_and_camera_from_trc_path(trc_path: str) -> Tuple[str, str]:
    """
    Best-effort extraction of pose_dir_name and camera_setup from a TRC path like:
      ...\\CleanedMarkerData\\OpenPose_default\\3-cameras\\PostAugmentation_v0.2\\file.trc
    or
      ...\\MarkerData\\OpenPose_default\\3-cameras\\PostAugmentation_v0.2\\file.trc
    """
    parts = os.path.normpath(trc_path).split(os.sep)
    pose_dir = None
    for i, p in enumerate(parts):
        if p.startswith("OpenPose_"):
            pose_dir = p
            # Next segment is usually camera setup (e.g. 3-cameras)
            if i + 1 < len(parts):
                return pose_dir, parts[i + 1]
            break
    raise ValueError(f"Could not infer pose_dir/camera_setup from path: {trc_path}")


def run_ik_only_for_postaug_trcs(
    subject_dir: str,
    repo_root: str,
    dest_kin_root: str,
    postaug_trc_paths: List[str],
    overwrite: bool,
) -> None:
    session_metadata_path = os.path.join(subject_dir, "sessionMetadata.yaml")
    if not os.path.isfile(session_metadata_path):
        raise FileNotFoundError(f"Missing session metadata: {session_metadata_path}")

    sessionMetadata = importMetadata(session_metadata_path)
    openSimModel = sessionMetadata["openSimModel"]

    suffix_model = "_shoulder" if "shoulder" in openSimModel else ""
    ik_setup_filename = f"Setup_IK{suffix_model}.xml"
    ik_setup_path = os.path.join(repo_root, "opensimPipeline", "IK", ik_setup_filename)
    if not os.path.isfile(ik_setup_path):
        raise FileNotFoundError(f"Missing IK setup file: {ik_setup_path}")

    processed_any = False
    for trc_path in postaug_trc_paths:
        if not os.path.isfile(trc_path):
            print(f"[{os.path.basename(subject_dir)}] Missing TRC: {trc_path}")
            continue

        pose_dir_name, camera_setup = _parse_pose_and_camera_from_trc_path(trc_path)

        openSimDir = os.path.join(subject_dir, "OpenSimData", pose_dir_name, camera_setup)
        scaled_model_path = os.path.join(openSimDir, "Model", f"{openSimModel}_scaled.osim")
        if not os.path.isfile(scaled_model_path):
            raise FileNotFoundError(
                f"Scaled OpenSim model not found (needed for IK): {scaled_model_path}\n"
                f"Make sure static trial scaling has been run successfully before re-running IK."
            )

        motion_out_dir = os.path.join(dest_kin_root, pose_dir_name, camera_setup, "Kinematics")
        _ensure_dir(motion_out_dir)

        base = os.path.splitext(os.path.basename(trc_path))[0]
        mot_expected_path = os.path.join(motion_out_dir, f"{base}.mot")

        if not overwrite and _file_exists_nonempty(mot_expected_path):
            print(f"[{os.path.basename(subject_dir)}] Skip IK (exists): {base}")
            continue

        if overwrite and os.path.isfile(mot_expected_path):
            try:
                os.remove(mot_expected_path)
            except OSError:
                pass

        print(f"[{os.path.basename(subject_dir)}] IK only: {os.path.basename(trc_path)}")
        runIKTool(
            ik_setup_path,
            scaled_model_path,
            trc_path,
            motion_out_dir,
        )
        processed_any = True

    if not processed_any:
        print(f"[{os.path.basename(subject_dir)}] No IK outputs generated (no matching TRCs or all skipped).")


def run_for_subject(
    subject_dir: str,
    repo_root: str,
    dest_marker_root: str,
    dest_kin_root: str,
    augmenter_model: str,
    camera_setups: Optional[List[str]],
    resolution_pose_detection: Optional[List[str]],
    trial_ids: Optional[List[str]],
    trial_stems: Optional[List[str]],
    keep_nosync_in_output: bool,
    overwrite: bool,
) -> None:
    processed_any = False
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
            # Allow filtering by either the normalized trial_id (NoSync stripped)
            # or the exact cleaned stem (e.g., ID5_S7_sprintNoSync).
            if trial_ids is not None and inp.trial_id not in trial_ids:
                continue
            if trial_stems is not None and inp.cleaned_base_key not in trial_stems:
                continue
            out_trial_stem = inp.cleaned_base_key if keep_nosync_in_output else inp.trial_id
            augmented_filename = f"{out_trial_stem}_{markerAugmenterModelName}.trc"
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

            processed_any = True

    if not processed_any:
        if trial_ids:
            print(
                f"[{os.path.basename(subject_dir)}] No trials matched --trial-id filter: {trial_ids}. "
                f"(Tip: if your cleaned TRC stems contain 'NoSync', pass either form; the script normalizes it.)"
            )
        else:
            print(f"[{os.path.basename(subject_dir)}] No trials were processed.")


def _normalize_trial_ids(trial_ids: Optional[List[str]]) -> Optional[List[str]]:
    """
    Normalize user-provided trial ids to match how this script derives them.
    In OpenCap local outputs, pre-augmentation stems often end with 'NoSync'.
    This script strips 'NoSync' when deriving trial_id, so we do the same here.
    """
    if not trial_ids:
        return None
    out = []
    for t in trial_ids:
        t = str(t)
        if t.endswith("NoSync"):
            t = t[: -len("NoSync")]
        out.append(t)
    return out


def main() -> None:
    # ---------------------------------------------------------------------------
    # Edit these constants instead of using terminal arguments.
    # ---------------------------------------------------------------------------

    # Dataset root containing subjectN folders (e.g. subject5).
    DATA_ROOT = r"G:\Shared drives\Stanford Football\March_2"

    # Which subject numbers to process.
    SUBJECTS = [5]

    # Pre-augmentation TRC to augment + run IK on.
    # Set to a path string to process one specific file, or None to
    # discover all cleaned TRCs under PreAugmentation/Cleaned/.
    PREAUG_TRC = r"G:\Shared drives\Stanford Football\March_2\subject5\MarkerData\OpenPose_default\3-cameras\PreAugmentation\ID5_S7_sprintNoSync.trc"

    # If you already have a post-augmented TRC and just want IK, set this.
    # Set to None to run full augmentation + IK.
    POSTAUG_TRC = None

    # Scaled OpenSim model. Set to a path string to use a specific model,
    # or None to infer from sessionMetadata.yaml (default).
    MODEL_PATH = r"G:\Shared drives\Stanford Football\March_2\subject5\OpenSimData\OpenPose_default\3-cameras\Model\LaiUhlrich2022_scaled.osim"

    # Augmenter model version (usually "v0.2").
    AUGMENTER_MODEL = "v0.2"

    # Camera setup filter, e.g. ["3-cameras"]. None = process all found.
    CAMERA_SETUP = ["3-cameras"]

    # OpenPose resolution filter, e.g. ["default"]. None = process all found.
    RESOLUTION_POSE = ["default"]

    # Destination folder names under each subject dir.
    DEST_MARKER_FOLDER = "CleanedMarkerData"
    DEST_KIN_FOLDER = "CleanedKinematics"

    # If True, keep "NoSync" in output filenames.
    KEEP_NOSYNC_IN_OUTPUT = True

    # Overwrite existing outputs.
    OVERWRITE = True

    # ---------------------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Rerun augmentation + IK using cleaned PreAugmentation marker TRCs."
    )
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--subjects", nargs="*", type=int, default=SUBJECTS)
    parser.add_argument("--augmenter-model", default=AUGMENTER_MODEL)
    parser.add_argument("--camera-setup", nargs="*", default=CAMERA_SETUP)
    parser.add_argument("--resolution-pose", nargs="*", default=RESOLUTION_POSE)
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE)
    parser.add_argument("--trial-id", nargs="*", default=None)
    parser.add_argument("--trial-stem", nargs="*", default=None)
    parser.add_argument("--keep-nosync-in-output", action="store_true", default=KEEP_NOSYNC_IN_OUTPUT)
    parser.add_argument("--postaug-trc", nargs="*", default=[POSTAUG_TRC] if POSTAUG_TRC else None)
    parser.add_argument("--preaug-trc", default=PREAUG_TRC or None,
                        help="Specific pre-augmentation TRC to process (augmentation + IK).")
    parser.add_argument("--model-path", default=MODEL_PATH or None,
                        help="Override scaled OpenSim model path (otherwise inferred from sessionMetadata).")
    parser.add_argument("--dest-marker-folder", default=DEST_MARKER_FOLDER)
    parser.add_argument("--dest-kin-folder", default=DEST_KIN_FOLDER)

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

        dest_marker_root = os.path.join(subject_dir, args.dest_marker_folder)
        dest_kin_root = os.path.join(subject_dir, args.dest_kin_folder)
        _ensure_dir(dest_marker_root)
        _ensure_dir(dest_kin_root)

        print(f"=== Subject {subj} ===")
        norm_trial_ids = _normalize_trial_ids(args.trial_id)

        if args.postaug_trc:
            run_ik_only_for_postaug_trcs(
                subject_dir=subject_dir,
                repo_root=repo_root,
                dest_kin_root=dest_kin_root,
                postaug_trc_paths=args.postaug_trc,
                overwrite=args.overwrite,
            )
            continue

        # If a specific pre-augmentation TRC was given, derive trial_stem from it
        # and run only that trial.
        extra_trial_stems = list(args.trial_stem or [])
        if args.preaug_trc and os.path.isfile(args.preaug_trc):
            stem = os.path.splitext(os.path.basename(args.preaug_trc))[0]
            if stem not in extra_trial_stems:
                extra_trial_stems.append(stem)
            print(f"  Specific pre-aug TRC: {args.preaug_trc} (stem={stem})")

        run_for_subject(
            subject_dir=subject_dir,
            repo_root=repo_root,
            dest_marker_root=dest_marker_root,
            dest_kin_root=dest_kin_root,
            augmenter_model=args.augmenter_model,
            camera_setups=args.camera_setup,
            resolution_pose_detection=args.resolution_pose,
            trial_ids=norm_trial_ids,
            trial_stems=extra_trial_stems if extra_trial_stems else args.trial_stem,
            keep_nosync_in_output=args.keep_nosync_in_output,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()

