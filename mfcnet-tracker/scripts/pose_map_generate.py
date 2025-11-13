import os, re, csv, math, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import argparse



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
    json_files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]
    json_files.sort()

    frames = {} 
    for i in json_files:
        json_file_dir = os.path.join(JSON_DIR,i)
        data, W, H = read_json(json_file_dir)

        for p in data.get("points",[]):
            fid = int(p["frameIndex"])
            if fid not in frames:
                frames[fid] =[]
            frames[fid].append(p)

    for fid in frames:
        frames[fid].sort(key = lambda q:(q.get("type",""), q["x"],q["y"]))

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
        frame_idx = []
        for j in frame_dict:
            num = int(j)
            frame_idx.append(num)
        
        frame_idx.sort()

        count = 1
        for fid in frame_idx:
            pts = merged["frames"][str(fid)]
            pose_rgb = get_pose_per_frame(pts,H,W,RADIUS)
            out_path = os.path.join(pose_idr,f"frame_{count:03d}.png")
            cv2.imwrite(out_path, cv2.cvtColor(pose_rgb,cv2.COLOR_RGB2BGR))
            count += 1

        print(f"[OK] merged -> {pose_idr}/frame_*.png (total {len(frame_idx)})")

    elif dataset == "ACT_action":
        assert case_name is not None, "ACT_action needs case_name from ds_name"
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
                run_all(ann_dir, out_dir, RADIUS, args.dataset,case_name=None)
                out_image_dir = os.path.join(out_dir,"images")
                extract_first_clip_frames(clip_dir, out_image_dir)
        elif args.dataset == "ACT_action":
            ann_dir = os.path.join(SRC_ROOT, ds_name, "annotation")
            clip_dir = os.path.join(SRC_ROOT, ds_name, "clips")
            out_dir = DST_ROOT  
            case_name = normalize_name(ds_name)
            os.makedirs(out_dir, exist_ok=True)
            print(f"\n=== Processing {ds_name} -> {out_dir} ===")
            run_all(ann_dir, out_dir, RADIUS, args.dataset, case_name=case_name)
        elif args.dataset == "KPT_action":
            ann_dir = os.path.join(SRC_ROOT, ds_name, "annotation")
            clip_dir = os.path.join(SRC_ROOT, ds_name, "clips")
            out_dir = DST_ROOT
            case_name = normalize_name(ds_name)
            os.makedirs(out_dir, exist_ok=True)
            print(f"\n=== Processing {ds_name} -> {out_dir} ===")
            run_all(ann_dir, out_dir, RADIUS, args.dataset, case_name=case_name)

            



