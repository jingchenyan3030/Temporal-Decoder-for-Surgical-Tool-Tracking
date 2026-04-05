import json, cv2, numpy as np,torch, os
from pathlib import Path
import json
import tqdm
from sam2.build_sam import build_sam2_video_predictor
from collections import defaultdict
from hydra import initialize_config_dir, compose
from pathlib import Path
import shutil
import csv, re

CLASS_MAP = {
    "tool_tip": 0,
    "tool_anchor": 1,
}

def get_pred_size_from_json(pred_data):
    for frame_key, frame_info in pred_data.items():
        if "W_pred" in frame_info and "H_pred" in frame_info:
            return int(frame_info["W_pred"]), int(frame_info["H_pred"])
    raise ValueError("No W_pred / H_pred found in pred_points.json")

def load_pred_points_json(pred_json_path):
    pred_json_path = Path(pred_json_path)
    if not pred_json_path.exists():
        raise FileNotFoundError(f"pred_points.json not found: {pred_json_path}")
    with open(pred_json_path, "r") as f:
        data = json.load(f)
    return data

def frame_key_to_idx(frame_key):
    frame_name = Path(frame_key).stem   # frame_001
    return int(frame_name.split("_")[-1]) - 1

def build_frame_prompts_from_pred_json(pred_data, selected_frames, tip_thresh=0.5):
    selected_frames = set(selected_frames)
    frame_prompts = {}

    for frame_key, frame_info in pred_data.items():
        frame_idx = frame_key_to_idx(frame_key)
        if frame_idx not in selected_frames:
            continue

        pts = []
        lbs = []

        anchor_list = frame_info.get("anchor", [])
        tip_list = frame_info.get("tip", [])

        if len(anchor_list) > 0:
            ax, ay, _ = anchor_list[0]
            pts.append([float(ax), float(ay)])
            lbs.append(1)
     
        trusted_tips = [
            t for t in tip_list
            if len(t) >= 3 and float(t[2]) > tip_thresh
        ]

        for t in trusted_tips:
            tx, ty, _ = t
            pts.append([float(tx), float(ty)])
            lbs.append(1)

        if len(pts) == 0:
            continue

        frame_prompts[frame_idx] = {
            "points": np.array(pts, dtype=np.float32),
            "labels": np.array(lbs, dtype=np.int32),
        }

    return frame_prompts



def build_kpt_index(csv_root):
    """
    return:
      dict[video_name] -> list of entries
    """
    index = defaultdict(list)

    for case in os.listdir(csv_root):
        csv_path = os.path.join(csv_root, case, "clip_to_video_mapping.csv")
        if not os.path.exists(csv_path):
            continue

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                video_path = row[-1]
                video_name = Path(video_path).name

                index[video_name].append({
                    "case": case,
                    "json_file": row[1],
                    "action": row[2],
                })
    return index


def normalize_case_prefix(case_prefix: str) -> str:
    case_prefix = case_prefix.lower()
    m = re.fullmatch(r"cholec[_]?(\d+)", case_prefix)
    if m:
        return f"cholec80_{m.group(1)}"
    return case_prefix



# ========= load_annotations_from_folder ''KPT'' ===========
def load_annotations_single_json(json_path, img_dir, start_f=None):
    json_path = Path(json_path)

    img_names = sorted([p for p in os.listdir(img_dir)
                        if p.lower().endswith(('.jpg', '.png', '.jpeg'))])
    assert len(img_names) > 0, f"No images found in {img_dir}"

    first_img = cv2.imread(os.path.join(img_dir, img_names[0]))
    tgt_h, tgt_w = first_img.shape[:2]

    data = json.load(open(json_path, "r"))

    # ---------- Step 0: scale ----------
    orig_w = data.get("imageDimensions", {}).get("width", tgt_w)
    orig_h = data.get("imageDimensions", {}).get("height", tgt_h)
    sx, sy = float(tgt_w) / float(orig_w), float(tgt_h) / float(orig_h)
    if start_f is None:
        raise ValueError(f"[KPT] start_f is None for {json_path}. You must pass entry['start_frame'].")
    start_f = int(start_f)

    # ---------- Step 1: collect all raw frames ----------
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
        print("[WARN] No valid frames found in annotation.")
        return [], {}

    # ---------- Step 3: build annotations ----------
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
            "obj_id": 0,                         # single tool
            "points": np.array(pts_xy, dtype=np.float32),
            "types": types,
            "labels": np.ones(len(pts_xy), dtype=np.int32),
            "cls": "tool"
        })
    annotations.sort(key=lambda a: a["frame_idx"])
    return annotations, obj_to_class



#========= load_annotations_from_folder ''ACT'' ===========
def load_annotations_from_folder(folder_path, img_dir):
    folder_path = Path(folder_path)
    json_files = sorted([p for p in folder_path.glob("*.json") if p.is_file()])
    annotations = []
    obj_to_class = {}
    obj_id = 0

    img_names = sorted([p for p in os.listdir(img_dir) if p.lower().endswith(('.jpg', '.png', '.jpeg'))])
    first_img = cv2.imread(os.path.join(img_dir, img_names[0]))
    tgt_h, tgt_w = first_img.shape[:2]

    all_raw_frames = set()

    for jf in json_files:
        data = json.load(open(jf, "r"))
        for p in data.get("points", []):
            if not p.get("vis", True): 
                continue
            if p.get("type") == "contact":
                continue
            all_raw_frames.add(p["frameIndex"])

    if len(all_raw_frames) == 0:
        print("[WARN] No valid frames found in annotation.")
        return [], {}
    
    sorted_frames = sorted(all_raw_frames)
    sorted_frames = sorted_frames[:len(img_names)]   
    raw2local = {raw: i for i, raw in enumerate(sorted_frames)}

    # === Step 2: second pass — build annotations ===
    for jf in json_files:
        data = json.load(open(jf, "r"))

        tool_name = data.get("tool", f"obj_{obj_id}")
        points = data.get("points", [])

        frame_groups = defaultdict(list)

        for p in points:
            if not p.get("vis", True):
                continue
            if p.get("type", "") == "contact":
                continue

            # scale coords
            orig_w = data.get("imageDimensions", {}).get("width", tgt_w)
            orig_h = data.get("imageDimensions", {}).get("height", tgt_h)
            sx, sy = float(tgt_w)/float(orig_w), float(tgt_h)/float(orig_h)

            x = float(p["x"]) * sx
            y = float(p["y"]) * sy

            # === KEY PART: convert frameIndex to local frame ===
            raw_f = p["frameIndex"]
            if raw_f not in raw2local:
                continue

            local_f = raw2local[raw_f]
            frame_groups[local_f].append({
                "x": x,
                "y": y,
                "type": p.get("type", "")
            })

        # add annotations
        for local_f, pts in frame_groups.items():
            annotations.append({
                "frame_idx": local_f,
                "obj_id": obj_id,
                "points": np.array([[pt["x"], pt["y"]] for pt in pts], dtype=np.float32),
                "types": [pt["type"] for pt in pts],
                "labels": np.ones(len(pts), dtype=np.int32),
                "cls": tool_name,
            })

        obj_to_class[obj_id] = tool_name
        obj_id += 1

    return annotations, obj_to_class


#========= choose_first_frame ===========
def choose_first_frames(good_frames, annotations, early_window=30, debug_name=None):
    if len(annotations) == 0:
        return sorted(set(good_frames))

    min_f = min(ann["frame_idx"] for ann in annotations)

    early_counts = defaultdict(int)
    for ann in annotations:
        frame_idx = ann["frame_idx"]
        if frame_idx <= min_f + early_window:  
            early_counts[frame_idx] += len(ann["points"])

    if early_counts:
        early_best = max(early_counts.items(), key=lambda x: x[1])[0]
        good_frames = list(good_frames) + [early_best]

    good_frames = sorted(set(good_frames))

    if debug_name is not None:
        print(f"[DEBUG] {debug_name} - min_f={min_f}, early_window={early_window}, good_frames={good_frames}")
    return good_frames


#========== chooose_good_frame ===========
def choose_good_frames_from_pred_json(pred_data, num_frames=5, tip_thresh=0.5):
    candidate_frames = []

    for frame_key, frame_info in pred_data.items():
        anchor_list = frame_info.get("anchor", [])
        tip_list = frame_info.get("tip", [])

        if len(anchor_list) == 0:
            continue

        anchor = anchor_list[0]
        anchor_score = float(anchor[2])

        trusted_tips = [
            t for t in tip_list
            if len(t) >= 3 and float(t[2]) > tip_thresh
        ]

        if len(trusted_tips) == 0:
            continue

        frame_score = anchor_score + sum(float(t[2]) for t in trusted_tips)
        frame_idx = frame_key_to_idx(frame_key)
        candidate_frames.append((frame_idx, frame_score))

    if len(candidate_frames) == 0:
        return []

    candidate_frames.sort(key=lambda x: x[1], reverse=True)

    good_frames = [f for f, _ in candidate_frames[:num_frames]]
    good_frames = sorted(set(good_frames))
    return good_frames

#========== Propogate ===========

def prepare_prompts_for_frame(annotations, frame_idx):
    frame_anns = [ann for ann in annotations if ann["frame_idx"] == frame_idx]
    if not frame_anns:
        return None, None
    point_coords = np.concatenate([ann["points"] for ann in frame_anns], axis=0)
    point_labels = np.concatenate([ann["labels"] for ann in frame_anns], axis=0)
    return point_coords, point_labels


def run_propagation_from_prompts(predictor, inference_state, frame_prompts, obj_id=0):
    results = {}
    predictor.reset_state(inference_state)

    added_any = False

    for frame_idx in sorted(frame_prompts.keys()):
        prompt = frame_prompts[frame_idx]
        print(f"[INFO] Add prompts from frame {frame_idx}, num_pts={len(prompt['points'])}")

        predictor.add_new_points_or_box(
            inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=prompt["points"],
            labels=prompt["labels"]
        )
        added_any = True

    if not added_any:
        print("[WARN] No prompts added for propagation.")
        return {}

    for out_frame_idx, out_obj_idx, out_mask_logits in predictor.propagate_in_video(inference_state):
        frame_results = []
        for i, oid in enumerate(out_obj_idx):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()
            frame_results.append((oid, mask))
        results[out_frame_idx] = frame_results

    return results


def save_masks_only(results, save_dir, orig_img_folder=None, num_frames=None,
                    save_overlay=False, overlay_alpha=0.4, overlay_color=(0, 255, 0)):
    os.makedirs(save_dir, exist_ok=True)
    results = dict(results)

    if num_frames is None:
        assert orig_img_folder is not None
        num_frames = len([
            f for f in os.listdir(orig_img_folder)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])

    for frame_idx in range(num_frames):
        frame_results = results.get(frame_idx, [])

        orig_img = None
        if orig_img_folder is not None:
            img_path = os.path.join(orig_img_folder, f"{frame_idx:06d}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(orig_img_folder, f"{frame_idx:06d}.png")
            if os.path.exists(img_path):
                orig_img = cv2.imread(img_path)

        if orig_img is None:
            print(f"[WARN] No image found for frame {frame_idx}, skip.")
            continue

        H, W = orig_img.shape[:2]

        if len(frame_results) == 0:
            merged_mask = np.zeros((H, W), dtype=np.uint8)
        else:
            merged_mask = None
            for obj_id, mask in frame_results:
                if not isinstance(mask, np.ndarray):
                    continue
                mask = mask.squeeze().astype(bool)
                if merged_mask is None:
                    merged_mask = mask.copy()
                else:
                    merged_mask = np.logical_or(merged_mask, mask)

            if merged_mask is None:
                merged_mask = np.zeros((H, W), dtype=bool)

            merged_mask = merged_mask.astype(np.uint8)

            # ===== Bad-frame filtering: reject unrealistically large masks =====
            mask_area = merged_mask.sum()
            total_area = H * W
            area_ratio = mask_area / float(total_area)

            if area_ratio > 0.30:  
                merged_mask = np.zeros((H, W), dtype=np.uint8)

        frame_name = f"frame_{frame_idx+1:03d}"

        # 1) save npy
        np.save(os.path.join(save_dir, f"{frame_name}.npy"), merged_mask.astype(np.uint8))

        # 2) optional overlay
        if save_overlay:
            overlay_img = orig_img.copy()

            # color mask image
            color_mask = np.zeros_like(orig_img, dtype=np.uint8)
            color_mask[merged_mask > 0] = overlay_color

            # blend only masked region
            blended = cv2.addWeighted(orig_img, 1.0, color_mask, overlay_alpha, 0)

            overlay_img[merged_mask > 0] = blended[merged_mask > 0]

            # optional contour for easier viewing
            contours, _ = cv2.findContours(
                (merged_mask * 255).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay_img, contours, -1, (0, 0, 255), 1)

            cv2.imwrite(
                os.path.join(save_dir, f"{frame_name}_overlay.jpg"),
                overlay_img
            )
 

def convert_images_to_jpg_and_resize(img_dir, cache_dir, target_w, target_h):
    img_dir = Path(img_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ])

    for idx, file in enumerate(files):
        img = cv2.imread(str(file))
        if img is None:
            print(f"[WARN] Failed to read image: {file}")
            continue

        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        save_name = f"{idx:06d}.jpg"
        cv2.imwrite(
            str(cache_dir / save_name),
            img,
            [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )

    return str(cache_dir)


def load_kpt_mapping(csv_path):
    entries = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            json_name  = row[1]
            action     = row[2]
            start_f    = int(row[3])
            end_f      = int(row[4])
            video_path = row[6].strip()

            video_path = video_path.replace(
    "/data/home/hao/chenyan/data/training_data/",
    "/home/chenyan/fallout_data/data/0923_by_action_2/clip/"
)

            entries.append({
                "json_file": json_name,
                "action": action,
                "video_path": video_path,
                "start_frame": start_f,
                "end_frame": end_f,
            })
    return entries


#========== Setup environment ===========
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

if device.type == 'cuda':
    torch.autocast(device_type='cuda', dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == 'mps':
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
#========== Load SAM2 model ===========
sam2_checkpoint = 'checkpoints/sam2.1_hiera_large.pt'
model_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml'
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",type=str, required=True, choices=["act","kpt"], help="Dataset type: 'act' or 'kpt'")
parser.add_argument(
    "--action",
    type=str,
    default="clip",
    choices=["clip", "cut", "grasp", "dissect"],
    help="Action to process (default: clip)"
)
args = parser.parse_args()

#========== Main pipeline ===========

if __name__ == "__main__":
    if args.dataset == "kpt":
        base_root = Path("/mnt/sda1/datasets/chenyan/fallout_data/data/0923_by_action_2") / args.action

        if not base_root.exists():
            print(f"[WARN] No action dir found for {base_root}, skipping.")
            exit()

        for case_dir in sorted(base_root.iterdir()):
            if not case_dir.is_dir():
                continue
            if case_dir.name.lower() in ["test", "test_multiframe"]:
                continue

            for video_dir_l in sorted(case_dir.iterdir()):
                if not video_dir_l.is_dir():
                    continue

                video_dir = video_dir_l / "images"
                pred_json_path = video_dir_l / "pred_points.json"
                cache_dir = video_dir_l / "images_jpg"

                if not video_dir.exists():
                    print(f"[WARN] No pred image dir found for {video_dir}, skipping.")
                    continue

                if not pred_json_path.exists():
                    print(f"[WARN] No pred_points.json found for {pred_json_path}, skipping.")
                    continue

      
                save_dir = video_dir_l / "mask_coarse"

                image_files = sorted([
                    f for f in os.listdir(video_dir)
                    if f.lower().endswith((".jpg", ".png", ".jpeg"))
                ])

          
                if save_dir.exists():
                    npy_files = sorted([f for f in os.listdir(save_dir) if f.endswith(".npy")])
                    if len(npy_files) == 2 * len(image_files):
                        print(f"[SKIP] {video_dir_l.name}: mask count matches images ({len(image_files)}), skip.")
                        continue
                    else:
                        print(f"[REGEN] {video_dir_l.name}: npy={len(npy_files)} != images={len(image_files)}, regenerating.")
                        shutil.rmtree(save_dir)

                # PNG -> JPG
                pred_data = load_pred_points_json(pred_json_path)
                W_pred, H_pred = get_pred_size_from_json(pred_data)
                print(f"[INFO] Resize images to prediction size: W={W_pred}, H={H_pred}")

                # images -> resized jpg cache
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                jpg_video_dir = convert_images_to_jpg_and_resize(
                    str(video_dir),
                    str(cache_dir),
                    target_w=W_pred,
                    target_h=H_pred
                )

                inference_state = predictor.init_state(video_path=jpg_video_dir)

                good_frames = choose_good_frames_from_pred_json(
                    pred_data,
                    num_frames=5,
                    tip_thresh=0.5
                )

                if len(good_frames) == 0:
                    print(f"[WARN] No valid prompt frames for {video_dir_l}, skipping.")
                    del inference_state
                    torch.cuda.empty_cache()
                    if cache_dir.exists():
                        shutil.rmtree(cache_dir)
                    continue

                print(f"[INFO] Selected good frames: {good_frames}")

                frame_prompts = build_frame_prompts_from_pred_json(
                    pred_data,
                    selected_frames=good_frames,
                    tip_thresh=0.5
                )

                # propagation
                results = run_propagation_from_prompts(
                    predictor,
                    inference_state,
                    frame_prompts,
                    obj_id=0
                )

                del inference_state
                torch.cuda.empty_cache()

                if not results:
                    print(f"[WARN] Empty results for {video_dir_l.name}, skip saving.")
                    if cache_dir.exists():
                        shutil.rmtree(cache_dir)
                    continue

                target_case = "case_001_video_part_001_segment_3"

                save_masks_only(
                    results,
                    str(save_dir),
                    orig_img_folder=jpg_video_dir,
                    save_overlay=True
                )

                print(f"[INFO] Saved masks to {save_dir}/")

                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                    print(f"[INFO] Removed temporary cache directory {cache_dir}")
                                            