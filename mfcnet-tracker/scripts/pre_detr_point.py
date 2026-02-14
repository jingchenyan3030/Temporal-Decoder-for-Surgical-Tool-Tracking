import os, re, csv, math, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import argparse
import shutil
import re

W_tar = 320
H_tar = 256
def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        assert "imageDimensions" in data and "points" in data, "Invalid JSON format"
        W_ori = data["imageDimensions"]["width"]
        H_ori = data["imageDimensions"]["height"]
    return W_ori, H_ori, data

'''
{
'1':[p1,p2,p3]
'2':[p4,p5],...}
'''
def group_by_frame(points):
    frames = {}
    for point in points:
        frame_id = point["frameIndex"]
        if frame_id not in frames:
            frames[frame_id] = []
        frames[frame_id].append(point)
    return frames

def rescale_points(x, y, W_ori, H_ori, W_tar=W_tar, H_tar=H_tar):
    x_new = x * W_tar / W_ori
    y_new = y * H_tar / H_ori
    return x_new, y_new

def dump_json(out_dir, W_ori, H_ori, frames, W_tar=W_tar, H_tar=H_tar):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = sorted(frames.keys(), key=lambda k: int(k))
    pad = max(3, len(str(len(frame_ids))))

    for seq_id, points in enumerate(frame_ids, start=1):
         items = []
         for point in frames[points]:
             x_rescaled, y_rescaled = rescale_points(point["x"], point["y"], W_ori, H_ori, W_tar, H_tar)
             items.append({
             "x": x_rescaled,
             "y": y_rescaled,
             "label": point["type"],
             "visibility": point["vis"],
             })

         with open(out_dir / f"frame_{seq_id:0{pad}d}.json", "w") as f:
            json.dump(items, f, indent=4)

def load_mapping_rows(mapping_csv_path):
    rows = []
    with open(mapping_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "json_file": row["json_file"],
                "video_idx": int(row["video_idx"]),
                "action": row["action"],
            })
    return rows

def normalize_case_name(name: str) -> str:
    # cholec80_01 -> cholec01
    m = re.fullmatch(r"cholec80_(\d+)", name)
    if m:
        return f"cholec{int(m.group(1)):02d}"
    return name

if __name__ == "__main__":
    SRC_ROOT = "/data/home/hao/chenyan/ori_data/surg_act_09232025"
    DST_ROOT = "/data/home/hao/chenyan/data/0923_training_data"
    OUT_ROOT  = "/data/home/hao/chenyan/data/0923_by_action" 
 
    for case_name in os.listdir(SRC_ROOT):
        case_path = os.path.join(SRC_ROOT, case_name)
        if not os.path.isdir(case_path):
            continue
        ann_dir = os.path.join(case_path, "annotation")
        if not os.path.exists(ann_dir):
            continue
        case_dst = os.path.join(DST_ROOT, normalize_case_name(case_name))    
        mapping_csv = os.path.join(case_dst, "clip_to_video_mapping.csv")
        mapping = load_mapping_rows(mapping_csv)
        for row in mapping:
            json_file = row["json_file"]
            video_idx = row["video_idx"]
            action = row["action"]

            json_path = os.path.join(ann_dir, json_file)
            if not os.path.isfile(json_path):
                print(f"[warn] missing json: {json_path}")
                continue

            out_video_dir = os.path.join(OUT_ROOT, action, normalize_case_name(case_name), f"video_{video_idx:03d}")
            out_points_dir = os.path.join(out_video_dir, "points_detr")

            W_ori, H_ori, data = read_json(json_path)
            frames = group_by_frame(data["points"])
            print(f"  [WRITE] {action}/{normalize_case_name(case_name)}/video_{video_idx:03d} <- {json_file}  (frames={len(frames)})")
            dump_json(out_points_dir, W_ori, H_ori, frames)




