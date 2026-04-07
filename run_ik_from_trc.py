import os
import sys
import argparse
from utilsOpenSim import runIKTool

def main():
    parser = argparse.ArgumentParser(description='Run Inverse Kinematics on a TRC file using a scaled OpenSim model.')
    parser.add_argument('model_path', help='Path to the scaled OpenSim model file (.osim)')
    parser.add_argument('trc_path', help='Path to the post-augmentation marker TRC file')
    parser.add_argument('--output_dir', '-o', help='Output directory for IK results (default: same as TRC directory)', default=None)
    parser.add_argument('--ik_setup', '-s', help='Path to IK setup XML file (default: uses generic Setup_IK.xml)', default=None)

    args = parser.parse_args()

    model_path = args.model_path
    trc_path = args.trc_path
    output_dir = args.output_dir or os.path.dirname(trc_path)
    ik_setup_path = args.ik_setup

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    if not os.path.exists(trc_path):
        print(f"Error: TRC file not found: {trc_path}")
        sys.exit(1)

    # If no custom IK setup, use default
    if ik_setup_path is None:
        # Assume default IK setup in opensimPipeline
        base_dir = os.path.dirname(os.path.abspath(__file__))
        opensim_pipeline_dir = os.path.join(base_dir, 'opensimPipeline')
        ik_setup_path = os.path.join(opensim_pipeline_dir, 'IK', 'Setup_IK.xml')

        if not os.path.exists(ik_setup_path):
            print(f"Error: Default IK setup file not found: {ik_setup_path}")
            print("Please provide a custom IK setup file using --ik_setup")
            sys.exit(1)

    print(f"Running IK with:")
    print(f"  Model: {model_path}")
    print(f"  TRC: {trc_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  IK setup: {ik_setup_path}")

    try:
        # Run IK
        path_output_mot, path_model_ik = runIKTool(
            ik_setup_path, model_path, trc_path, output_dir
        )
        print(f"Success! IK results saved to: {path_output_mot}")

    except Exception as e:
        print(f"Error running IK: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()