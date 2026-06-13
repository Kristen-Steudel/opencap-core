import os
import pickle
import cv2

base = r"G:\Shared drives\Stanford Football\March_2\subject4\Videos"
trial = "ID4_S7_sprint_trimmed"

for cam in ["Cam1b", "Cam4b", "Cam7b"]:
    camdir = os.path.join(base, cam)
    mp4 = os.path.join(camdir, "InputMedia", trial, trial + ".mp4")
    avi = os.path.join(camdir, "InputMedia", trial, trial + "_rotated.avi")
    pkl = os.path.join(camdir, "OutputPkl_default", trial, trial + "_rotated_pp.pkl")
    print(f"=== {cam} ===")
    for label, p in [("mp4", mp4), ("rotated.avi", avi), ("pkl", pkl)]:
        print(f"  {label} exists: {os.path.isfile(p)}")
    for label, p in [("mp4", mp4), ("rotated.avi", avi)]:
        if os.path.isfile(p):
            cap = cv2.VideoCapture(p)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ok = cap.isOpened()
            cap.release()
            print(f"  {label} frames: {n}, opened: {ok}")
    if os.path.isfile(pkl):
        with open(pkl, "rb") as f:
            frames = pickle.load(f)
        print(f"  pkl frames: {len(frames)}")
        if frames:
            print(f"  people in frame0: {len(frames[0])}")
