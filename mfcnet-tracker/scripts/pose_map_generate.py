"""
Author: Chenyan
"""
import os, re, csv, math, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import argparse
import shutil


RGB_BG    = (0,   0,   0  )  # background: black
RGB_TIP   = (255, 255, 0  )  # tip: yellow
RGB_ANCH  = (255, 0,   0  )  # anchor：red
RGB_CONT  = (0,   255, 0  )  # contact：green

RADIUS = 5
SIGMA     = 2.0                    
CLIP_R    = int(3*SIGMA)   
target_w, target_h = 640, 480


types = ["tool_tip", "tool_anchor", "contact"]
types_channel = {t:i for i, t in enumerate(types)}

# ========== Basic I/O ===========
def count_png_frames(dir_path: str) -> int:
    """Count how many .png files are in a directory."""
    if not os.path.isdir(dir_path):
        return 0
    return len([f for f in os.listdir(dir_path) if f.lower().endswith(".png")])

def clear_dir(path: str):
    """Delete the whole directory if it exists (used for full regeneration)."""
    if os.path.isdir(path):
        shutil.rmtree(path)



def get_expected_frame_count_from_merged(merged_json_path: str) -> Optional[int]:
    if not os.path.exists(merged_json_path):
        return None

    try:
        with open(merged_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {merged_json_path}: {e}")
        return None

    frames = data.get("frames", {})
    if not frames:
        return None

    # merged.json 
    expected = len(frames)
    return expected



def read_json(JSON_PATH):
    with open(JSON_PATH,'r',encoding='utf-8') as f:
        data = json.load(f)
    assert "imageDimensions" in data and "points" in data, "lack imageDimensions or points"
    W = int(round(data["imageDimensions"]["width"]))
    H = int(round(data["imageDimensions"]["height"]))
    assert W > 0 and H > 0, "illegal dimension"
    return data, W, H


def joint_points_per_frame(points):
    frames = {}

    for p in points:
        frame_id = int(p["frameIndex"])
        if frame_id not in frames:
            frames[frame_id] = []

        frames[frame_id].append(p)
    return frames

def draw_circle(pose_rgb, x, y, r, color):
    cv2.circle(pose_rgb, (int(round(x)), int(round(y))), int(r), color, -1, lineType=cv2.LINE_8)

def get_pose_per_frame(points, H, W, r):
    pose_rgb = np.zeros((target_h, target_w, 3), np.uint8)
    sx, sy = target_w / float(W), target_h / float(H)
    r_scaled = r
    for point_type, color in (("tool_tip", RGB_TIP), ("tool_anchor", RGB_ANCH), ("contact", RGB_CONT)):
        for p in points:
            if p.get("type") != point_type:
                continue
            if p.get("vis", True) is False:
                continue
            x, y = p["x"], p["y"]
            if 0 <= x < W and 0 <= y < H:
                x2, y2 = int(round(x * sx)), int(round(y * sy))
                if 0 <= x2 < target_w and 0 <= y2 < target_h:
                    draw_circle(pose_rgb, x2, y2, r_scaled, color)
    return pose_rgb



def save_pose(OUT_DIR, seg_idx, frame_idx, pose_rgb):
    seg_dir = os.path.join(OUT_DIR,f"video_{seg_idx:03d}","pose_map")
    os.makedirs(seg_dir, exist_ok=True)

    bgr = cv2.cvtColor(pose_rgb,cv2.COLOR_RGB2BGR)
    out_path = os.path.join(seg_dir, f"frame_{frame_idx:03d}.png")
    ok = cv2.imwrite(out_path,bgr)

    if not ok:
        raise IOError(f"false: {out_path}")


def _parse_action_bounds(fname: str):

    m = re.match(r"([A-Za-z]+)_(\d+)_(\d+)\.json$", fname)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))

    act_m = re.match(r'([A-Za-z]+)', fname)
    act = act_m.group(1) if act_m else "unknown"
    nums = re.findall(r'(\d+)', fname)
    s = int(nums[0]) if nums else -1
    e = int(nums[1]) if len(nums) >= 2 else -1
    return act, s, e

# =======ACT:merge all jsons into one json with frames{} =====
def merge_json(JSON_DIR):
    """
    Merge all jsons under JSON_DIR into a single dict with 'frames' field.

    New logic:
    - Determine crop_start  = min start_frame from non-'none' jsons
    - Determine crop_end    = max end_frame   from non-'none' jsons
    - Final valid range is [crop_start, crop_end]
    - Only keep points whose frameIndex ∈ [crop_start, crop_end]
    """
    json_files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]
    json_files.sort()

    crop_start = None
    crop_end   = None

    # -------- 1) Find crop_start and crop_end from non-none json files --------
    for fname in json_files:
        action, s, e = _parse_action_bounds(fname)

        if action.lower() == "none":
            continue

        if s >= 0 and e >= 0:
            if crop_start is None or s < crop_start:
                crop_start = s
            if crop_end is None or e > crop_end:
                crop_end = e

    if crop_start is None:
        raise ValueError(f"No non-'none' json files found in {JSON_DIR}")

    if crop_end is None:
        raise ValueError(f"No valid end_frame found in non-'none' jsons in {JSON_DIR}")

    print(f"[INFO] merge_json: valid frame range = [{crop_start}, {crop_end}] in {JSON_DIR}")

    # -------- 2) Merge jsons but only keep frames inside the valid interval --------
    frames = {}
    W = H = None

    for fname in json_files:
        json_path = os.path.join(JSON_DIR, fname)
        data, W, H = read_json(json_path)

        for p in data.get("points", []):
            fid = int(p["frameIndex"])

            # skip if out of valid interval
            if fid < crop_start or fid > crop_end:
                continue

            if fid not in frames:
                frames[fid] = []
            frames[fid].append(p)

    # -------- 3) Sort points per frame for stability --------
    for fid in frames:
        frames[fid].sort(key=lambda q: (q.get("type", ""), q["x"], q["y"]))

    # -------- 4) Build merged dict --------
    if W is None or H is None:
        raise ValueError(f"No valid json found under {JSON_DIR}")

    merged = {
        "imageDimensions": {"width": W, "height": H},
        "frames": {}
    }

    for fid in sorted(frames.keys()):
        merged["frames"][str(fid)] = frames[fid]

    return merged




# ====== run_all ========
def run_all(JSON_DIR, OUT_DIR, RADIUS, dataset,case_name=None):
    json_files = sorted([f for f in os.listdir(JSON_DIR) if f.endswith(".json")])
    print(f"Found {len(json_files)} json files")


    if dataset == "KPT": 
        os.makedirs(OUT_DIR, exist_ok=True)
        mapping_csv = os.path.join(OUT_DIR, "clip_to_video_mapping.csv")
        with open(mapping_csv, "w", newline="") as fcsv:
            wr = csv.writer(fcsv)
            wr.writerow(["video_idx", "json_file", "action", "start_frame", "end_frame", "n_frames", "out_dir"])

            for seg_idx, json_file in enumerate(json_files, start=1):
                json_path = os.path.join(JSON_DIR, json_file)
                data, W, H = read_json(json_path)
                frames = joint_points_per_frame(data["points"])
                frame_seq = sorted(frames.keys())
                if not frame_seq:
                    continue

                for local_idx, frame_id in enumerate(frame_seq, start=1):
                    pose_rgb = get_pose_per_frame(frames[frame_id], H, W, RADIUS)
                    save_pose(OUT_DIR, seg_idx, local_idx, pose_rgb)

                action, s, e = _parse_action_bounds(json_file)
                out_dir_seg = os.path.join(OUT_DIR, f"video_{seg_idx:03d}")
                wr.writerow([seg_idx, json_file, action, s, e, len(frame_seq), out_dir_seg])

                print(f"[OK] {json_file} -> video_{seg_idx:03d}/pose_map/*.png")
    elif dataset == "ACT":
        os.makedirs(OUT_DIR, exist_ok = True)
        # make pose_idr--posemap file
        pose_idr = os.path.join(OUT_DIR,"pose_map")
        os.makedirs(pose_idr, exist_ok= True)

        # merge.json
        merged = merge_json(JSON_DIR)
        merge_json_dir = os.path.join(OUT_DIR,"merged.json")
        with open(merge_json_dir,"w") as f:
            json.dump(merged, f,indent=2) 
        print(f"wrote{merge_json_dir}")

        # take dict -- change into list -- draw pose_map
        frame_dict = merged["frames"]
        W = merged["imageDimensions"]["width"]
        H = merged["imageDimensions"]["height"]

        
        frame_idx = sorted([int(j) for j in frame_dict.keys()])

        for fid in frame_idx:
            pts = frame_dict[str(fid)]

            out_idx = fid + 1
            pose_rgb = get_pose_per_frame(pts, H, W, RADIUS)
            out_path = os.path.join(pose_idr, f"frame_{out_idx:03d}.png")
            cv2.imwrite(out_path, cv2.cvtColor(pose_rgb, cv2.COLOR_RGB2BGR))

        print(f"[OK] merged -> {pose_idr}/frame_*.png (total {len(frame_idx)})")


    elif dataset == "KPT_action":
        assert case_name is not None, "KPT_action needs case_name from ds_name"
        lower_name = case_name.lower()
        if "grasp" in lower_name or "grab" in lower_name:
            action = "grasp"
        elif "clip" in lower_name or "clipping" in lower_name:
            action = "clip"
        elif "cut" in lower_name:
            action = "cut"
        elif "dissect" in lower_name:
            action = "dissect"
        else:
            action = "unknown"

        if action == "unknown":
            print(f"[WARN] Skip {case_name} (no valid action)")
        
        
        out_dir_action = os.path.join(OUT_DIR, action)
        out_dir_case = os.path.join(out_dir_action, case_name)
        pose_dir = os.path.join(out_dir_case, "pose_map")
        img_dir = os.path.join(out_dir_case, "images")

        os.makedirs(pose_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        json_files = sorted([f for f in os.listdir(JSON_DIR) if f.endswith(".json")])
        print(f"Found {len(json_files)} json files in {case_name}")

        frame_counter = 1

        for json_file in json_files:
            if json_file.lower().startswith("none"):
                print(f"[SKIP] {json_file}")
                continue

            json_path = os.path.join(JSON_DIR, json_file)
            data, W, H = read_json(json_path)
            frames = joint_points_per_frame(data["points"])
            frame_seq = sorted(frames.keys())
            if not frame_seq:
                continue

            for frame_id in frame_seq:
                pose_rgb = get_pose_per_frame(frames[frame_id], H, W, RADIUS)
                pose_out = os.path.join(pose_dir, f"frame_{frame_counter:03d}.png")
                cv2.imwrite(pose_out, cv2.cvtColor(pose_rgb, cv2.COLOR_RGB2BGR))
                frame_counter += 1

            print(f"[OK] pose_map created from {json_file}")


            clip_name = os.path.splitext(json_file)[0] + ".mp4"
            clip_path = os.path.join(JSON_DIR.replace("annotation", "clips"), clip_name)
            if not os.path.exists(clip_path):
                print(f"[WARN] No mp4 found for {clip_name}")
                continue

            extract_frames(clip_path, img_dir) 
            print(f"[OK] Extracted frames from {clip_name} -> {img_dir}")




def normalize_name(name: str) -> str:
    m = re.fullmatch(r"cholec80_(\d+)", name)
    if m:
        return f"cholec{int(m.group(1)):02d}"
    return name




# cut frame--kpt
def extract_frames(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)

        out_path = os.path.join(out_dir, f"frame_{idx:03d}.png")
        cv2.imwrite(out_path, frame)
        idx += 1
    cap.release()
    print(f"[OK] {video_path} -> {idx-1} frames")

# cut fram--act
def extract_first_clip_frames(clip_dir, out_dir):
    mp4s = sorted([f for f in os.listdir(clip_dir) if f.lower().endswith(".mp4")])
    if not mp4s:
        print(f"[WARN] No mp4 under {clip_dir}")
        return
    first_mp4 = os.path.join(clip_dir, mp4s[0])
    extract_frames(first_mp4, out_dir)  



def process_dataset(ds_name, clip_dir, mapping_csv, dst_root):
  
    with open(mapping_csv, newline="") as f:
        reader = csv.DictReader(f)
        mapping = list(reader)

    for row in mapping:
        video_idx = int(row["video_idx"])
        json_file = row["json_file"]
        base_name = os.path.splitext(json_file)[0] + ".mp4"   
        video_path = os.path.join(clip_dir, base_name)

        if not os.path.exists(video_path):
            print(f"[WARN] {video_path} not found")
            continue

        out_dir = os.path.join(dst_root, ds_name, f"video_{video_idx:03d}", "images")
        extract_frames(video_path, out_dir)




if __name__ == "__main__":
   ap = argparse.ArgumentParser()
   ap.add_argument("--dataset",type= str, default="KPT", choices=["KPT","ACT","ACT_action","KPT_action"], help="choose json format")
   args = ap.parse_args()

   if args.dataset =="KPT":
       SRC_ROOT = "/data/home/hao/chenyan/ori_data/surg_act_raw_stage_1"
       DST_ROOT = "/data/home/hao/chenyan/data/training_data_raw_stage_1"
   elif args.dataset =="ACT":
       SRC_ROOT = "/data/home/hao/chenyan/ori_data/keypoint_act"
       DST_ROOT = "/data/home/hao/chenyan/data/training_act_data"   
   elif args.dataset =="ACT_action":
       SRC_ROOT = "/data/home/hao/chenyan/ori_data/keypoint_act"
       DST_ROOT = "/data/home/hao/chenyan/data/training_act_data_by_action"
   

   allow_list = []  
   
   for ds_name in sorted(os.listdir(SRC_ROOT)):
        full_path = os.path.join(SRC_ROOT,ds_name)
        if not os.path.isdir(full_path):
            continue
        if allow_list and ds_name not in allow_list:
            continue
        if ds_name == "index.json":
            continue


        if args.dataset =="KPT":  
                ann_dir = os.path.join(SRC_ROOT, ds_name, "annotation")
                clip_dir = os.path.join(SRC_ROOT, ds_name, "clips")
                out_dir = os.path.join(DST_ROOT, normalize_name(ds_name))
                os.makedirs(out_dir, exist_ok=True)
                print(f"\n=== Processing {ds_name} -> {out_dir} ===")
                run_all(ann_dir, out_dir, RADIUS, args.dataset,case_name=None)      
                mapping_csv = os.path.join(out_dir, "clip_to_video_mapping.csv")  
                process_dataset(normalize_name(ds_name), clip_dir, mapping_csv, DST_ROOT)
        elif args.dataset == "ACT":
            ann_dir = os.path.join(SRC_ROOT, ds_name, "annotation")
            clip_dir = os.path.join(SRC_ROOT, ds_name, "clips")
            out_dir = os.path.join(DST_ROOT, normalize_name(ds_name))
            os.makedirs(out_dir, exist_ok=True)

            print(f"\n=== Processing {ds_name} -> {out_dir} ===")

            merged_path = os.path.join(out_dir, "merged.json")
            pose_dir = os.path.join(out_dir, "pose_map")
            out_image_dir = os.path.join(out_dir, "images")

            # By default we assume we need to regenerate both pose maps and images
            need_pose = True
            need_images = True
            
            expected_frames = get_expected_frame_count_from_merged(merged_path)

            if expected_frames is not None:
                # Count existing png frames
                pose_count = count_png_frames(pose_dir)
                img_count  = count_png_frames(out_image_dir)

                if pose_count == expected_frames and img_count == expected_frames:
                    # Both pose_map and images are complete -> skip everything
                    print(
                        f"[SKIP] pose_map and images for {ds_name} "
                        f"(pose_map={pose_count}, images={img_count}, "
                        f"expected={expected_frames})"
                    )
                    need_pose   = False
                    need_images = False
                else:
                    # Any mismatch -> fully clear this case and regenerate from scratch
                    print(
                        f"[INFO] Mismatch for {ds_name}: "
                        f"pose_map={pose_count}, images={img_count}, "
                        f"expected={expected_frames}. Will FULLY regenerate."
                    )
                    clear_dir(pose_dir)
                    clear_dir(out_image_dir)
                    if os.path.exists(merged_path):
                        os.remove(merged_path)
                    need_pose   = True
                    need_images = True
            else:
                print(
                    f"[INFO] No valid merged.json for {ds_name}, "
                    "will generate pose_map and images from scratch."
                )
                need_pose   = True
                need_images = True

            # Regenerate pose_map (this also rewrites merged.json using the NEW logic)
            if need_pose:
                run_all(ann_dir, out_dir, RADIUS, args.dataset, case_name=None)

            # Regenerate images from the first clip
            if need_images:
                extract_first_clip_frames(clip_dir, out_image_dir)


        elif args.dataset == "ACT_action":
            ann_dir = os.path.join(SRC_ROOT, ds_name, "annotation")
            clip_dir = os.path.join(SRC_ROOT, ds_name, "clips")

            case_name = normalize_name(ds_name)
            lower_name = case_name.lower()

            # infer action
            if "grasp" in lower_name or "grab" in lower_name:
                action = "grasp"
            elif "clip" in lower_name or "clipping" in lower_name:
                action = "clip"
            elif "cut" in lower_name:
                action = "cut"
            elif "dissect" in lower_name:
                action = "dissect"
            else:
                print(f"[WARN] Skip {case_name} (unknown action)")
                continue

            # output directory
            out_case_dir = os.path.join(DST_ROOT, action, case_name)
            pose_dir     = os.path.join(out_case_dir, "pose_map")
            img_dir      = os.path.join(out_case_dir, "images")
            merged_path  = os.path.join(out_case_dir, "merged.json")

            os.makedirs(out_case_dir, exist_ok=True)

            print(f"\n=== Processing {ds_name} -> {out_case_dir} ===")

            # ===== skip or regenerate =====
            expected = get_expected_frame_count_from_merged(merged_path)

            need_pose = True
            need_imgs = True

            if expected is not None:
                pose_count = count_png_frames(pose_dir)
                img_count  = count_png_frames(img_dir)

                if pose_count == expected and img_count == expected:
                    print(f"[SKIP] {case_name}: already complete ({expected} frames)")
                    need_pose = False
                    need_imgs = False
                else:
                    print(f"[INFO] mismatch: pose={pose_count}, img={img_count}, expected={expected}, regenerate...")
                    clear_dir(pose_dir)
                    clear_dir(img_dir)
                    if os.path.exists(merged_path):
                        os.remove(merged_path)

            # ===== regenerate pose_map + merged.json =====
            if need_pose:
                os.makedirs(pose_dir, exist_ok=True)

                merged = merge_json(ann_dir)
                with open(merged_path, "w") as f:
                    json.dump(merged, f, indent=2)

                frame_dict = merged["frames"]
                W = merged["imageDimensions"]["width"]
                H = merged["imageDimensions"]["height"]

                for fid in sorted(int(k) for k in frame_dict.keys()):
                    pts = frame_dict[str(fid)]
                    out_idx = fid + 1
                    pose_rgb = get_pose_per_frame(pts, H, W, RADIUS)
                    cv2.imwrite(
                        os.path.join(pose_dir, f"frame_{out_idx:03d}.png"),
                        cv2.cvtColor(pose_rgb, cv2.COLOR_RGB2BGR)
                    )
                print(f"[OK] pose_map generated")

            # ===== regenerate images =====
            if need_imgs:
                os.makedirs(img_dir, exist_ok=True)
                extract_first_clip_frames(clip_dir, img_dir)
                print(f"[OK] images extracted")

        elif args.dataset == "KPT_action":
            ann_dir = os.path.join(SRC_ROOT, ds_name, "annotation")
            clip_dir = os.path.join(SRC_ROOT, ds_name, "clips")
            out_dir = DST_ROOT
            case_name = normalize_name(ds_name)
            os.makedirs(out_dir, exist_ok=True)
            print(f"\n=== Processing {ds_name} -> {out_dir} ===")
            run_all(ann_dir, out_dir, RADIUS, args.dataset, case_name=case_name)

            



