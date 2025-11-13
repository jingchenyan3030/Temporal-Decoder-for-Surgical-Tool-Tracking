import os
import shutil
from pathlib import Path
from natsort import natsorted

# ======= Paths =======
root = Path("/data/home/hao/chenyan/data")
src_root = root / "training_act_data"
dst_root = root / "training_act_data_by_action"
ACTIONS = ["dissect"]

SRC_ROOT = root / "training_data"                
DST_ROOT = root / "reconstruction_by_action" 

#ACTIONS = ["grasp", "clip", "cut", "dissect"]

def copy_depth_maps():
    for action in ACTIONS:
        action_dir = dst_root / action
        if not action_dir.exists():
            print(f"[Skip] Action folder not found: {action_dir}")
            continue

        for case_dir in natsorted(action_dir.iterdir()):
            if not case_dir.is_dir():
                continue

            src_case_dir = src_root / case_dir.name
            src_depth_dir = src_case_dir / "depth_maps"

            if not src_case_dir.exists():
                print(f"[Skip] Source case not found: {src_case_dir}")
                continue
            if not src_depth_dir.exists():
                print(f"[Skip] depth_maps not found in: {src_case_dir}")
                continue

            dst_depth_dir = case_dir / "depth_maps"
            if dst_depth_dir.exists():
                print(f"[Warning] {dst_depth_dir} already exists, skipping.")
                continue

            try:
                shutil.copytree(src_depth_dir, dst_depth_dir)
                print(f"[OK] Copied → {dst_depth_dir}")
            except Exception as e:
                print(f"[Error] Failed to copy {src_case_dir.name}: {e}")

def copy_kpt_depth_maps():
    for action in ACTIONS:
        action_dir = DST_ROOT / action
        if not action_dir.exists():
            print(f"[Skip] Action folder not found: {action_dir}")
            continue
        for case_dir in natsorted(d for d in action_dir.iterdir() if d.is_dir()):
            for video_dir in natsorted(d for d in case_dir.iterdir() if d.is_dir()):
                src_case_dir = SRC_ROOT / case_dir.name
                src_video_dir = src_case_dir / video_dir.name
                src_depth_dir = src_video_dir / "depth_maps"

                if not src_case_dir.exists():
                    print(f"[Skip] Source case not found: {src_case_dir}")
                    continue
                if not src_video_dir.exists():
                    print(f"[Skip] Source video not found: {src_video_dir}")
                    continue
                if not src_depth_dir.exists():
                    print(f"[Skip] depth_maps not found in: {src_video_dir}")
                    continue

                dst_depth_dir = video_dir / "depth_maps"
                if dst_depth_dir.exists():
                    print(f"[Warning] {dst_depth_dir} already exists, skipping.")
                    continue

                try:
                    shutil.copytree(src_depth_dir, dst_depth_dir)
                    print(f"[OK] Copied → {dst_depth_dir}")
                except Exception as e:
                    print(f"[Error] Failed to copy {src_case_dir.name}/{src_video_dir.name}: {e}")
       
if __name__ == "__main__":
    copy_kpt_depth_maps()
