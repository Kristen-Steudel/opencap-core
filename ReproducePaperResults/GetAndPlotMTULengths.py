import os
import sys
# Ensure repository root is on sys.path so imports from repo root succeed
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import glob
import utilsKinematics
import utilsProcessing
import utilsAuthentication

import numpy as np
import pandas as pd
import utilsAPI

# plotting
import matplotlib
import matplotlib.pyplot as plt


def calculate_and_plot_bflh_mtu_length(trial_folder_path):
    """
    Calculates and plots the Biceps Femoris Long Head (BFLH) MTU length
    for both legs from a kinematics file using OpenCap-processing.

    Args:
        trial_folder_path (str): Path to the session folder that contains
                                 `OpenSimData/Kinematics/<trial>.mot` (or
                                 directly a `.mot` file).
    """

    # Locate the kinematics (.mot) file inside the session/trial folder.
    kin_dir = os.path.join(trial_folder_path, 'OpenSimData', 'OpenPose_default','3-cameras', 'Kinematics')
    mot_files = []
    if os.path.isdir(kin_dir):
        mot_files = glob.glob(os.path.join(kin_dir, '*.mot'))
    if not mot_files:
        mot_files = glob.glob(os.path.join(trial_folder_path, '*.mot'))
    if not mot_files:
        print(f"No .mot file found under '{trial_folder_path}' or its OpenSimData/Kinematics subfolder.")
        return

    mot_path = mot_files[0]
    trial_name = os.path.splitext(os.path.basename(mot_path))[0]

    # 1. Load Kinematics Data
    print(f"Loading kinematics for trial: {trial_name} from {mot_path}...")
    try:
        # utilsKinematics.Kinematics expects (sessionDir, trialName, modelName=None,...)
        kinematics = utilsKinematics.Kinematics(trial_folder_path, trial_name)
    except FileNotFoundError as e:
        print(f"Error loading kinematics: {e}")
        return

    # 2. Get Muscle-Tendon Lengths
    # This returns a pandas DataFrame where rows are time steps and columns are muscle names.
    muscle_tendon_lengths = kinematics.get_muscle_tendon_lengths()

    # 3. Identify BFLH for both legs
    # Muscle names typically follow the OpenSim convention (e.g., 'bflh_r' for right, 'bflh_l' for left)
    bflh_muscle_names = ['bflh_r', 'bflh_l']

    # Check if the muscle names exist in the DataFrame columns
    if not all(muscle in muscle_tendon_lengths.columns for muscle in bflh_muscle_names):
        print(f"Muscle columns not found. Expected: {bflh_muscle_names}. Available: {list(muscle_tendon_lengths.columns)[:10]}...")
        return

    bflh_mtu_lengths = muscle_tendon_lengths[bflh_muscle_names]

    print(f"Successfully extracted BFLH MTU lengths (Shape: {bflh_mtu_lengths.shape})")

    # 4. Plot
    plot_mtu_lengths(bflh_mtu_lengths, trial_name)


def plot_mtu_lengths(df_mtu_lengths, trial_name):
    """
    Plot and save MTU lengths DataFrame.
    """
    # Ensure an output folder exists for saved plots
    plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(df_mtu_lengths.index, df_mtu_lengths['bflh_r'], label='BFLH Right')
    plt.plot(df_mtu_lengths.index, df_mtu_lengths['bflh_l'], label='BFLH Left')
    plt.title(f'BFLH MTU Length vs. Time for {trial_name}')
    plt.xlabel('Time (s)')
    plt.ylabel('MTU Length (m)')
    plt.legend()
    plt.grid(True)

    # Save plot to file (PNG + PDF)
    safe_name = ''.join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in str(trial_name)).strip()
    png_path = os.path.join(plots_dir, f"{safe_name}_bflh_mtu.png")
    pdf_path = os.path.join(plots_dir, f"{safe_name}_bflh_mtu.pdf")
    try:
        plt.savefig(png_path, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved plots to: {png_path} and {pdf_path}")
    except Exception as e:
        print(f"Warning: could not save plot to file: {e}")

    # Show interactively if a display is available
    try:
        plt.show()
    except Exception:
        # Non-interactive/backend environment: continue silently
        pass

    print("--- Plotting function executed ---")


# --- Execution Example (runs when script executed directly) ---
if __name__ == '__main__':
    # NOTE: Replace this `trial_path` with your session folder (the parent of OpenSimData)
    trial_path = r'G:\Shared drives\Stanford Football\January_12\subject2'
    calculate_and_plot_bflh_mtu_length(trial_path)