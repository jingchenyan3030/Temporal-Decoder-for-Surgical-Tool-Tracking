import os
import shutil
import pandas as pd
from pathlib import Path
from natsort import natsorted
import json
import argparse

actions = ['grasp','clip','cut','dissect']
MOVE_FILES = False  


def copy_images_pose_and_visible(video_path: Path, dst_video_dir: Path):
    img_dir = video_path / 'images'
    pose_dir = video_path / 'pose_map'
    depth_dir = video_path / 'depth_maps'

    dst_img_dir = dst_video_dir / 'images'
    dst_pose_dir = dst_video_dir / 'pose_map'
    dst_depth_dir = dst_video_dir / 'depth_maps'
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_pose_dir.mkdir(parents=True, exist_ok=True)
    dst_depth_dir.mkdir(parents=True, exist_ok=True)

    for src_dir, dst_dir in [(img_dir, dst_img_dir), (pose_dir, dst_pose_dir), (depth_dir, dst_depth_dir)]:
        if src_dir.exists():
     
            try:
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            except Exception:
                for p in src_dir.rglob('*'):
                    if p.is_file():
                        rel = p.relative_to(src_dir)
                        (dst_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, dst_dir / rel)
      
            if MOVE_FILES:
                shutil.rmtree(src_dir, ignore_errors=True)
        else:
            print(f"[SKIP] Missing {src_dir.relative_to(video_path)} in {video_path}")

    
    cand_jsons = [
        video_path / 'visible_frames.json',
        video_path / 'visible_frame.json'
    ]
    copied_vjson = False
    for jf in cand_jsons:
        if jf.exists():
            try:
                shutil.copy2(jf, dst_video_dir / jf.name)
                if MOVE_FILES:
                    try:
                        jf.unlink(missing_ok=True)
                    except TypeError:
                  
                        if jf.exists():
                            jf.unlink()
                copied_vjson = True
                print(f"[COPY] {jf.name} -> {dst_video_dir/jf.name}")
                break
            except Exception as e:
                print(f"[WARN] Failed to copy {jf.name}: {e}")
    if not copied_vjson:
        print(f"[INFO] No visible_*.json in {video_path}")

# ===== kpt =====
def reconstruct_kpt(src_root, dst_root):
    for case_dir in natsorted(src_root.iterdir()):
        if not case_dir.is_dir():
            continue

        mapping_path = case_dir / 'clip_to_video_mapping.csv'
        if not mapping_path.exists():
            print(f"Mapping file not found in {case_dir}, skipping.")
            continue

        df = pd.read_csv(mapping_path)
        for _, row in df.iterrows():
            action = str(row['action']).strip()
            video_path = Path(str(row['out_dir']).strip())
            if action not in actions:
                continue
            if video_path == Path('None') or not video_path.exists():
                print(f"Video path invalid for action {action} in {case_dir}, skipping.")
                continue

            case_name = case_dir.name
            video_name = video_path.name
            dst_dir = dst_root / action / case_name / video_name
            dst_dir.mkdir(parents=True, exist_ok=True)


            copy_images_pose_and_visible(video_path, dst_dir)

            print(f"[OK] {case_name}/{video_name} -> {action}/")

# ===== act =====
def reconstruct_act(src_root, dst_root):
    for case_dir in natsorted(src_root.iterdir()):
        if not case_dir.is_dir():
            continue

        subdirs = [d for d in case_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 0:
            print(f"[SKIP] No subdir in {case_dir}")
            continue

        for action_dir in subdirs:
            action = action_dir.name
            dst_dir = dst_root / action
            dst_dir.mkdir(parents=True, exist_ok=True)

            for subfolder_name in ['depth_maps', 'images', 'pose_map']:
                src_subdir = action_dir / subfolder_name
                dst_subdir = dst_dir / subfolder_name
                if src_subdir.exists():
                    shutil.copytree(src_subdir, dst_subdir, dirs_exist_ok=True)
                    print(f"[COPY] {src_subdir} -> {dst_subdir}")
                    if MOVE_FILES:
                        shutil.rmtree(src_subdir, ignore_errors=True)
                else:
                    print(f"[SKIP] Missing {subfolder_name} in {action_dir}")

            print(f"[OK] {action_dir.name} copied selected folders.")

# ===== main =====
def main():
    parser = argparse.ArgumentParser(description="Reconstruction by Action Script")
    parser.add_argument('--dataset', type=str, choices=['kpt','act'], required=True,
                        help="Dataset type: 'kpt' or 'act'")
    args = parser.parse_args()

    if args.dataset == 'kpt':
        src_root = Path('/data/home/hao/chenyan/data/training_data')
        dst_root = Path('/data/home/hao/chenyan/data/reconstruction_by_action')
        reconstruct_kpt(src_root, dst_root)
    elif args.dataset == 'act':
        src_root = Path('/data/home/hao/chenyan/data/training_act_data')
        dst_root = Path('/data/home/hao/chenyan/data/reconstruction_act')
        reconstruct_act(src_root, dst_root)

if __name__ == "__main__":
    main()
