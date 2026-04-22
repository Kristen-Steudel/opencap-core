import os
import sys

# Ensure repo root is importable.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Edit these before each run.
# ---------------------------------------------------------------------------

# Pre-augmentation TRC (the ~20-marker file from OpenPose/MMpose).
INPUT_TRC = r"G:\Shared drives\Stanford Football\AnalysisCompare\PreaugmentationMarkerFiles\ID5_S7_sprintNoSync_medFilt.trc"
# Where to save the post-augmentation TRC (~43 markers).
# Leave empty to auto-name as <stem>_LSTM.trc next to the input file.
OUTPUT_TRC = r"G:\Shared drives\Stanford Football\AnalysisCompare\PostaugmentationMarkerFiles\ID5_S7_sprintNoSync_medFilt_LSTM.trc"

# Path to sessionMetadata.yaml for this subject.
# Mass (kg) and height (m) are read from it automatically.
# Leave empty to search upward from INPUT_TRC automatically.
SESSION_METADATA_PATH = r"G:\Shared drives\Stanford Football\AnalysisCompare\sessionMetadata.yaml"

# Augmenter model version. Options: "v0.2", "v0.3" (v0.3 is the latest default).
AUGMENTER_MODEL = "v0.3"

# Augmenter model name (subfolder under MarkerAugmenter/). Usually "LSTM".
AUGMENTER_MODEL_NAME = "LSTM"

# ---------------------------------------------------------------------------


def main():
    inp = INPUT_TRC.strip()
    if not inp:
        print("Set INPUT_TRC at the top of augment_trc.py.")
        sys.exit(1)
    if not os.path.isfile(inp):
        print(f"Error: Input TRC not found:\n  {inp}")
        sys.exit(1)

    # Locate sessionMetadata.yaml
    meta_path = SESSION_METADATA_PATH.strip()
    if not meta_path:
        meta_path = SESSION_METADATA_PATH
    if not meta_path or not os.path.isfile(meta_path):
        print(
            "Error: Could not find sessionMetadata.yaml.\n"
            "Set SESSION_METADATA_PATH at the top of augment_trc.py."
        )
        sys.exit(1)

    from utils import importMetadata
    metadata = importMetadata(meta_path)
    mass_kg = float(metadata["mass_kg"])
    height_m = float(metadata["height_m"])
    augmenter_model_name_from_meta = metadata.get(
        "markerAugmentationSettings", {}
    ).get("markerAugmenterModel", AUGMENTER_MODEL_NAME)

    out = OUTPUT_TRC.strip()
    if not out:
        stem = os.path.splitext(os.path.basename(inp))[0]
        out = os.path.join(
            os.path.dirname(inp), f"{stem}_{augmenter_model_name_from_meta}.trc"
        )

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    augmenter_dir = os.path.join(_REPO_ROOT, "MarkerAugmenter")
    if not os.path.isdir(augmenter_dir):
        print(f"Error: MarkerAugmenter folder not found: {augmenter_dir}")
        sys.exit(1)

    print("Running marker augmentation:")
    print(f"  Input TRC:        {inp}")
    print(f"  Output TRC:       {out}")
    print(f"  sessionMetadata:  {meta_path}")
    print(f"  Mass:   {mass_kg} kg")
    print(f"  Height: {height_m} m")
    print(f"  Augmenter: {augmenter_model_name_from_meta} {AUGMENTER_MODEL}", flush=True)

    from utilsAugmenter import augmentTRC

    augmentTRC(
        pathInputTRCFile=inp,
        subject_mass=mass_kg,
        subject_height=height_m,
        pathOutputTRCFile=out,
        augmenterDir=augmenter_dir,
        augmenterModelName=augmenter_model_name_from_meta,
        augmenter_model=AUGMENTER_MODEL,
        offset=True,
    )

    if os.path.isfile(out):
        size_kb = os.path.getsize(out) / 1024
        print(f"\nSuccess! Post-augmentation TRC saved to:\n  {out}  ({size_kb:.1f} KB)")
    else:
        print("\nWarning: augmentTRC finished but output file was not found.")
        sys.exit(1)


if __name__ == "__main__":
    main()
