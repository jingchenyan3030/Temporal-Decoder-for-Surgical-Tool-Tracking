import os
import json
import csv
import re
import shutil
import argparse
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

CLASS_MAP = {
    "tool_tip": 0,
    "tool_anchor": 1,
}


def normalize_case_prefix(case_prefix: str) -> str:
    case_prefix = case_prefix.lower()
    m = re.fullmatch(r"cholec[_]?(\d+)", case_prefix)
    if m:
        return f"cholec80_{m.group(1)}"
    return case_prefix


def generate_multi_class_heatmap(points, types, H, W, sigma=10):
    C = len(CLASS_MAP)
    heatmaps = np.zeros((C, H, W), dtype=np.float32)

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    for pt, t in zip(points, types):
        if t not in CLASS_MAP:
            continue
        c = CLASS_MAP[t]
        x, y = pt
        gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        heatmaps[c] = np.maximum(heatmaps[c], gaussian)

    return heatmaps


def load_annotations_single_json(json_path, img_dir, start_f=None):
    json_path = Path(json_path)

    img_names = sorted([
        p for p in os.listdir(img_dir)
        if p.lower().endswith((".jpg", ".png", ".jpeg"))
    ])
    assert len(img_names) > 0, f"No images found in {img_dir}"

    first_img = cv2.imread(os.path.join(img_dir, img_names[0]))
    assert first_img is not None, f"Failed to read first image in {img_dir}"
    tgt_h, tgt_w = first_img.shape[:2]

    with open(json_path, "r") as f:
        data = json.load(f)

    orig_w = data.get("imageDimensions", {}).get("width", tgt_w)
    orig_h = data.get("imageDimensions", {}).get("height", tgt_h)
    sx, sy = float(tgt_w) / float(orig_w), float(tgt_h) / float(orig_h)

    if start_f is None:
        raise ValueError(f"[KPT] start_f is None for {json_path}.")
    start_f = int(start_f)

    frame_groups = defaultdict(list)
    for p in data.get("points", []):
        if not p.get("vis", True):
            continue
        if p.get("type", "") == "contact":
            continue

        raw_f = int(p["frameIndex"])
        local_f = raw_f - start_f

        if local_f < 0 or local_f >= len(img_names):
            continue

        frame_groups[local_f].append(p)

    if len(frame_groups) == 0:
        print(f"[WARN] No valid frames found in annotation: {json_path}")
        return [], {}

    annotations = []
    obj_to_class = {0: "tool"}

    for local_f, pts in frame_groups.items():
        pts_xy = []
        types = []

        for p in pts:
            x = float(p["x"]) * sx
            y = float(p["y"]) * sy
            pts_xy.append([x, y])
            types.append(p.get("type", ""))

        annotations.append({
            "frame_idx": local_f,
            "obj_id": 0,
            "points": np.array(pts_xy, dtype=np.float32),
            "types": types,
            "labels": np.ones(len(pts_xy), dtype=np.int32),
            "cls": "tool",
        })

    annotations.sort(key=lambda a: a["frame_idx"])
    return annotations, obj_to_class


def load_kpt_mapping(csv_path):
    entries = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            json_name = row[1]
            action = row[2]
            start_f = int(row[3])
            end_f = int(row[4])
            video_path = row[6].strip()

            video_path = video_path.replace(
                "/data/home/hao/chenyan/data/training_data/",
                 "/mnt/sda1/datasets/chenyan/fallout_data/data/0923_by_action/grasp/"
            )

            entries.append({
                "json_file": json_name,
                "action": action,
                "video_path": video_path,
                "start_frame": start_f,
                "end_frame": end_f,
            })
    return entries


def save_gaussian_heatmaps_only(save_dir, orig_img_folder, annotations=None, sigma=10):
    os.makedirs(save_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(orig_img_folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    print(f"[DEBUG] Saving gaussian heatmaps for {len(image_files)} frames -> {save_dir}")

    for frame_idx, img_name in enumerate(image_files):
        img_path = os.path.join(orig_img_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Failed to read image {img_path}, skip.")
            continue

        H, W = img.shape[:2]

        ann_points = []
        ann_types = []

        if annotations is not None:
            for ann in annotations:
                if ann["frame_idx"] == frame_idx:
                    pts = ann["points"]
                    ts = ann.get("types", ["tool_tip"] * len(pts))
                    for p, t in zip(pts, ts):
                        ann_points.append(p)
                        ann_types.append(t)

        heatmap = generate_multi_class_heatmap(
            points=ann_points,
            types=ann_types,
            H=H,
            W=W,
            sigma=sigma,
        )

        heatmap = heatmap[[CLASS_MAP["tool_tip"], CLASS_MAP["tool_anchor"]]]

        heatmap_npy_path = os.path.join(save_dir, f"frame_{frame_idx+1:03d}.npy")
        np.save(heatmap_npy_path, heatmap.astype(np.float32))

    print(f"[INFO] Done: saved {len(image_files)} npy files to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, default="clip",
                        choices=["clip", "cut", "grasp", "dissect"])
    parser.add_argument("--sigma", type=float, default=10)
    args = parser.parse_args()

    base_train = "/mnt/sda1/datasets/chenyan/fallout_data/data/0923_by_action"
    base_json = "/home/chenyan/fallout_data/ori_data/surg_act_09232025"
    csv_root = "/home/chenyan/fallout_data/data/0923_training_data"

    action = args.action
    action_dir = os.path.join(base_train, action)

    if not os.path.exists(action_dir):
        raise FileNotFoundError(f"No action dir found for {action_dir}")

    for case_name in os.listdir(action_dir):
        case_path = os.path.join(action_dir, case_name)
        if not os.path.isdir(case_path):
            continue
        if case_name.lower() in ["test", "test_multiframe"]:
            continue

        json_folder = os.path.join(base_json, normalize_case_prefix(case_name), "annotation")
        csv_path = os.path.join(csv_root, case_name, "clip_to_video_mapping.csv")

        if not os.path.exists(csv_path):
            print(f"[WARN] Missing csv: {csv_path}")
            continue
        if not os.path.exists(json_folder):
            print(f"[WARN] Missing annotation folder: {json_folder}")
            continue

        kpt_entries = load_kpt_mapping(csv_path)

        for entry in kpt_entries:
            if entry["action"] != action:
                continue

            json_path = os.path.join(json_folder, entry["json_file"])
            video_dir_l = entry["video_path"]

            if not os.path.exists(video_dir_l):
                print(f"[WARN] No video dir found for {video_dir_l}, skipping.")
                continue
            if not os.path.exists(json_path):
                print(f"[WARN] No json file found for {json_path}, skipping.")
                continue

            video_dir = os.path.join(video_dir_l, "images")
            if not os.path.exists(video_dir):
                print(f"[WARN] No image folder found for {video_dir}, skipping.")
                continue

            video_name = Path(video_dir_l).name
            save_dir = Path(base_train) / action / case_name / video_name / "sam_results"

            image_files = sorted([
                f for f in os.listdir(video_dir)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ])

            if save_dir.exists():
                npy_files = sorted([
                    f for f in os.listdir(save_dir)
                    if f.startswith("frame_") and f.endswith(".npy")
                ])
                if len(npy_files) == 3 *len(image_files):
                    print(f"[SKIP] {case_name}/{video_name}: npy count matches images.")
                    continue
                print(f"[REGEN] {case_name}/{video_name}: removing incomplete save_dir.")
                shutil.rmtree(save_dir)

            annotations, obj_to_class = load_annotations_single_json(
                json_path=json_path,
                img_dir=video_dir,
                start_f=entry["start_frame"],
            )

            save_gaussian_heatmaps_only(
                save_dir=str(save_dir),
                orig_img_folder=video_dir,
                annotations=annotations,
                sigma=args.sigma,
            )