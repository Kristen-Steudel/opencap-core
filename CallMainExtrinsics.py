
import os 
import yaml
import pickle

from utils import importMetadata
from utils import getDataDirectory
from utilsCheckerSony import computeAverageIntrinsics
from utilsCheckerSony import saveCameraParameters
from main import calcExtrinsicsFromVideo
from main import main

# Example of how to call the function from outside the main definition
if __name__ == '__main__':
    try:
        main(
            sessionName='subject1_Session0',
            trialName='ExtrinsicsTrial',
            trial_id='Calib1',
            extrinsicsTrial=True, # This is the crucial flag to only run calibration
            # cameras_to_use can be left as default 'all' if all 3 cameras are available
        )
        print("Camera extrinsics calculation complete.")
    except Exception as e:
        print(f"An error occurred: {e}")