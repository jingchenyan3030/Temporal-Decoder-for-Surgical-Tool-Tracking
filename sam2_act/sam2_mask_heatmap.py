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


def generate_multi_class_heatmap(mask, points, types, sigma=6):
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)

    H, W = mask.shape
    C = len(CLASS_MAP)
    heatmaps = np.zeros((C, H, W), dtype=np.float32)

    # ===== Part 1: fixed mask brightness =====
    mask_value = 0.04
    for c in range(C):
        heatmaps[c][mask] = mask_value

    if points is not None:
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        mask_float = mask.astype(np.float32)

        for (pt,t) in zip(points,types):
            if t not in CLASS_MAP:
                continue
            c = CLASS_MAP[t]
            x, y = pt
            gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
            gaussian *= mask_float
            heatmaps[c] += gaussian

    return heatmaps


def generate_multi_class_weightmap(mask, points, types, sigma=6, bg_w=0.04, mask_w=0.1, gauss_w = 15.0, out_dtype= np.float16):
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)

    H, W = mask.shape
    C = len(CLASS_MAP)
    weightmaps = np.full((C, H, W), fill_value=bg_w, dtype=np.float32)

    # ===== Part 1: fixed mask brightness =====
    for c in range(C):
        weightmaps[c][mask] = mask_w

    if points is not None and len(points) > 0:
        yy, xx = np.meshgrid(np.arange(H, dtype=np.float32),
                             np.arange(W, dtype=np.float32),
                             indexing='ij')
        mask_float = mask.astype(np.float32)

        for (pt,t) in zip(points,types):
            if t not in CLASS_MAP:
                continue
            c = CLASS_MAP[t]
            x, y = pt
            gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
            gaussian *= mask_float
            weightmaps[c] +=  gaussian * gauss_w

    return weightmaps.astype(out_dtype)

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
def choose_good_frames(annotations, obj_to_class, num_frames=5):
    max_points_per_obj = defaultdict(int)
    for ann in annotations:
        n_points = len(ann["points"])
        max_points_per_obj[ann["obj_id"]] = max(max_points_per_obj[ann["obj_id"]], n_points)

    frame_obj_counts = defaultdict(lambda: defaultdict(int))
    for ann in annotations:
        frame_obj_counts[ann["frame_idx"]][ann["obj_id"]] += len(ann["points"])

    all_objs = set(obj_to_class.keys())
    good_frames = []
    candidate_frames = []

    for frame_idx, obj_counts in frame_obj_counts.items():
        score = sum(obj_counts.get(obj, 0) / max_points_per_obj[obj] for obj in all_objs)
        candidate_frames.append((frame_idx, score))

      
        if set(obj_counts.keys()) == all_objs and all(
            obj_counts[obj] == max_points_per_obj[obj] for obj in all_objs
        ):
            good_frames.append(frame_idx)

    good_frames.sort()

    if len(good_frames) < num_frames:
        candidate_frames.sort(key=lambda x: x[1], reverse=True)
        extra_frames = [f for f, s in candidate_frames if f not in good_frames]
        good_frames.extend(extra_frames[:num_frames - len(good_frames)])
        good_frames.sort()

    if len(good_frames) > num_frames:
        idxs = np.linspace(0, len(good_frames) - 1, num_frames).astype(int)
        good_frames = [good_frames[i] for i in idxs]

    return good_frames

#========== Propogate ===========

def prepare_prompts_for_frame(annotations, frame_idx):
    frame_anns = [ann for ann in annotations if ann["frame_idx"] == frame_idx]
    if not frame_anns:
        return None, None
    point_coords = np.concatenate([ann["points"] for ann in frame_anns], axis=0)
    point_labels = np.concatenate([ann["labels"] for ann in frame_anns], axis=0)
    return point_coords, point_labels


def run_propagation(predictor, inference_state, annotations, selected_frames):
    results = {}
    predictor.reset_state(inference_state)

    added = set()
    added_any = False   
    for frame_idx in sorted(selected_frames):
        frame_anns = [ann for ann in annotations if ann["frame_idx"] == frame_idx]
        print(f"[INFO] Add prompts from frame {frame_idx} with {len(frame_anns)} objs...")

        for ann in frame_anns:
            predictor.add_new_points_or_box(
                inference_state,
                frame_idx=frame_idx,
                obj_id=ann["obj_id"],
                points=ann["points"],
                labels=ann["labels"]
            )
            added_any = True
    if not added_any:
        print("[WARN] No prompts added for propagation. Check your annotations and selected_frames.")
        return {}
    for out_frame_idx, out_obj_idx, out_mask_logits in predictor.propagate_in_video(inference_state):
        frame_results = []
        for i, obj_id in enumerate(out_obj_idx):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()
            frame_results.append((obj_id, mask))
        results[out_frame_idx] = frame_results

    return results


def save_results(results, save_dir, orig_img_folder=None, annotations=None,
                 num_frames=None, empty_weight_zero=True):
    os.makedirs(save_dir, exist_ok=True)
    results = dict(results)

    # 1) sum of all frames
    if num_frames is None:
        assert orig_img_folder is not None
        num_frames = len([f for f in os.listdir(orig_img_folder)
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    for frame_idx in range(num_frames):
        frame_results = results.get(frame_idx, [])   

        # 2) read image
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

        # 3) merge mask
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

        # 4) keypoints
        ann_points, ann_types = [], []
        if annotations is not None:
            for ann in annotations:
                if ann["frame_idx"] == frame_idx:
                    pts = ann["points"]
                    ts = ann.get("types", ["tool_tip"] * len(pts))
                    for p, t in zip(pts, ts):
                        ann_points.append(p)
                        ann_types.append(t)

        # 5) heatmap
        sigma = 6
        heatmap = generate_multi_class_heatmap(
            mask=merged_mask,
            points=ann_points,
            types=ann_types,
            sigma=sigma
        )
        heatmap = heatmap[[CLASS_MAP["tool_tip"], CLASS_MAP["tool_anchor"]]]
        np.save(os.path.join(save_dir, f"frame_{frame_idx+1:03d}.npy"),
                heatmap.astype(np.float16))

        # 6) weightmap
        if empty_weight_zero and (len(ann_points) == 0) and (merged_mask.sum() == 0):
            weightmap = np.zeros((2, H, W), dtype=np.float16)
        else:
            weightmap = generate_multi_class_weightmap(
                mask=merged_mask,
                points=ann_points,
                types=ann_types,
                sigma=sigma,
                bg_w=0.03,
                mask_w=0.1,
                gauss_w=10.0,
                out_dtype=np.float16
            )
            weightmap = weightmap[[CLASS_MAP["tool_tip"], CLASS_MAP["tool_anchor"]]]
        np.save(os.path.join(save_dir, f"frame_{frame_idx+1:03d}_weight.npy"),
                weightmap.astype(np.float16))

        # ===== DEBUG: overlay heatmap on the original image for every frame =====
        tip_map = heatmap[0]   # channel 0: tool_tip
        anc_map = heatmap[1]   # channel 1: tool_anchor

        # Print quick stats (optional but useful)
        print(
            f"[DEBUG][frame {frame_idx}] "
            f"mask_sum={int(merged_mask.sum())}, num_kpts={len(ann_points)}, "
            f"tip(min/max)=({tip_map.min():.4f},{tip_map.max():.4f}), "
            f"anchor(min/max)=({anc_map.min():.4f},{anc_map.max():.4f})"
        )

        def to_u8(x: np.ndarray) -> np.ndarray:
            vmin, vmax = float(x.min()), float(x.max())
            x01 = (x - vmin) / (vmax - vmin + 1e-6)
            return (x01 * 255.0).astype(np.uint8)

        # Convert heatmaps to uint8 and apply colormap
        tip_u8 = to_u8(tip_map)
        anc_u8 = to_u8(anc_map)

        tip_color = cv2.applyColorMap(tip_u8, cv2.COLORMAP_VIRIDIS)
        anc_color = cv2.applyColorMap(anc_u8, cv2.COLORMAP_VIRIDIS)

        # Overlay on the original image
        tip_overlay = cv2.addWeighted(orig_img, 0.7, tip_color, 0.3, 0)
        anc_overlay = cv2.addWeighted(orig_img, 0.7, anc_color, 0.3, 0)

        # Save overlay images
        cv2.imwrite(os.path.join(save_dir, f"frame_{frame_idx+1:03d}_overlay_tip.png"), tip_overlay)
        cv2.imwrite(os.path.join(save_dir, f"frame_{frame_idx+1:03d}_overlay_anchor.png"), anc_overlay)
        # ======= Debug overlay end =======

def generate_combine_heatmap(mask, keypoints=None, sigma=6):
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)

    H, W = mask.shape
    heatmap = np.zeros((H, W), dtype=np.float32)

    # ===== Part 1: fixed mask brightness =====
    mask_value = 0.04
    heatmap[mask] = mask_value

    # ===== Part 2: CV Gaussian =====
    if keypoints is not None and len(keypoints) > 0:
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        for (x, y) in keypoints:
            # CV Gaussian -- (peak = 1)
            gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
            gaussian *= mask.astype(np.float32)
            heatmap += gaussian

        heatmap[~mask] = 0


    return heatmap

 


def convert_png_to_jpg(png_dir, cache_dir):
    png_dir = Path(png_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(png_dir.glob("*.png"))
    for idx, file in enumerate(files):
        img = cv2.imread(str(file))
       
        save_name = f"{idx:06d}.jpg"
        cv2.imwrite(str(cache_dir / save_name), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

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

    if args.dataset == "act":
        # Step 1: Load annotations
        base_train = '/data/home/hao/chenyan/data/act_training_data_by_action'
        base_json  = '/data/home/hao/chenyan/ori_data/keypoint_act'
        actions = ['clip','cut','grasp']

        for action in actions:

            action_dir = os.path.join(base_train, action)
            if not os.path.exists(action_dir):
                print(f"[WARN] No action dir found for {action}, skipping.")
                continue

            # Loop through each case in the action folder
            for case_name in os.listdir(action_dir):

                case_path = os.path.join(action_dir, case_name)
                if not os.path.isdir(case_path):
                    continue
                if case_name.lower() in ["test", "test_multiframe"]:
                    continue

                # ====== SKIP IF sigma10 EXISTS ======
                sam_results_dir = os.path.join(case_path, "sam_results")
                skip_case = False

                if os.path.exists(sam_results_dir):
                    for f in os.listdir(sam_results_dir):
                        if f.endswith("sigma8.png"):
                            print(f"[SKIP] {case_name}: sigma8 output found, skipping.")
                            skip_case = True
                            break

                if skip_case:
                    continue    
                # ======================================


                # ---------- Setup paths ----------
                video_dir = os.path.join(case_path, "images")
                cache_dir = os.path.join(case_path, "images_jpg")
                json_folder = os.path.join(base_json, normalize_case_prefix(case_name), "annotation")

                if not os.path.exists(video_dir):
                    continue
                if not os.path.exists(json_folder):
                    print(f"[WARN] No annotation folder found for {case_name}, skipping.")
                    continue

                # Convert PNG → JPG
                video_dir = convert_png_to_jpg(video_dir, cache_dir) 

                # Initialize SAM2
                inference_state = predictor.init_state(video_path=video_dir)

                # Load annotations
                annotations, obj_to_class = load_annotations_from_folder(
                    json_folder, img_dir=video_dir
                )

                merged_results = {}

                # ---------- Process Each Object ----------
                for tool_id, tool_name in obj_to_class.items():

                    tool_annotations = [ann for ann in annotations if ann["obj_id"] == tool_id]
                    good_frames = choose_good_frames(tool_annotations, {tool_id: tool_name}, 5)
                    good_frames = choose_first_frames(good_frames, tool_annotations, 10)

                    if not good_frames:
                        continue

                    inference_state = predictor.init_state(video_path=video_dir)

                    results = run_propagation(
                        predictor, inference_state, tool_annotations, good_frames
                    )

                    for frame_idx, frame_data in results.items():
                        if frame_idx not in merged_results:
                            merged_results[frame_idx] = []
                        merged_results[frame_idx].extend(frame_data)

                # ---------- Save Results ----------
                case_dir = Path(video_dir).parent
                save_dir = case_dir / "sam_results"

                if save_dir.exists():
                    print(f"[INFO] Found existing {save_dir}, removing old results...")
                    shutil.rmtree(save_dir)

                save_results(
                    merged_results,
                    str(save_dir),
                    orig_img_folder=video_dir,
                    annotations=annotations
                )

                print(f"[INFO] Saved masks and heatmaps to {save_dir}/")

                # Remove temporary JPG folder
                if "images_jpg" in cache_dir and os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
                    print(f"[INFO] Removed temporary cache directory {cache_dir}")
                else:
                    print(f"[WARN] Cache directory {cache_dir} does not exist or was not created by this script.")

    elif args.dataset == "kpt":
        # Step 1: Load annotations
        base_train = '/home/chenyan/fallout_data/data/0923_by_action_2'
        base_json  = '/home/chenyan/fallout_data/ori_data/surg_act_09232025'
        csv_root =  "/home/chenyan/fallout_data/data/0923_training_data" 
        actions = [args.action]
        for action in actions:
            action_dir = os.path.join(base_train, action)
            if not os.path.exists(action_dir):
                print(f"[WARN] No action dir found for {action}, skipping.")
                continue
            # Loop through each case in the action folder
            for case_name in os.listdir(action_dir):
                json_folder = os.path.join(base_json, normalize_case_prefix(case_name), "annotation")
                case_path = os.path.join(action_dir, case_name)
                if not os.path.isdir(case_path):
                    continue
                if case_name.lower() in ["test", "test_multiframe"]:
                    continue

                csv_path = os.path.join(csv_root, case_name, "clip_to_video_mapping.csv")
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

                    # ---------- Setup paths ----------
                    video_dir = os.path.join(video_dir_l, "images")
                    cache_dir = os.path.join(video_dir_l, "images_jpg")

                    if not os.path.exists(video_dir):
                        continue
                    if not os.path.exists(json_folder):
                        print(f"[WARN] No annotation folder found for {case_name}, skipping.")
                        continue

                    # ---------- Check existing sam_results completeness ----------
                    video_name = Path(video_dir_l).name
                    save_dir = Path(base_train) / action / case_name / video_name / "sam_results"

                    image_files = sorted([
                        f for f in os.listdir(video_dir)
                        if f.lower().endswith((".jpg", ".png"))
                    ])

                    if save_dir.exists():
                        npy_files = sorted([f for f in os.listdir(save_dir) if f.endswith(".npy")])

                        if len(npy_files) != 3 *len(image_files):
                            print(
                                f"[REGEN] {case_name}/{video_name}: "
                                f"npy={len(npy_files)} != images={len(image_files)}, regenerating."
                            )
                            shutil.rmtree(save_dir)
                        else:
                            print(
                                f"[SKIP] {case_name}/{video_name}: "
                                f"npy count matches images ({len(image_files)}), skip."
                            )
                            continue

                    #------------ Convert PNG → JPG---------------
                    if os.path.exists(cache_dir):
                        shutil.rmtree(cache_dir)
                    video_dir = convert_png_to_jpg(video_dir, cache_dir)


                    #--------------- Initialize SAM2---------------
                    inference_state = predictor.init_state(video_path=video_dir)

                    annotations, obj_to_class = load_annotations_single_json(
                        json_path, img_dir=video_dir, start_f=entry["start_frame"]
                    )



                    # ---------- Process Each Object ----------
                    for tool_id, tool_name in obj_to_class.items():

                        tool_annotations = [ann for ann in annotations if ann["obj_id"] == tool_id]
                        if len(tool_annotations) == 0:
                            print(f"[WARN] No annotations for tool_id {tool_id} in {json_path}, skipping.")
                            continue
                        good_frames = choose_good_frames(tool_annotations, {tool_id: tool_name}, 5)
                        good_frames = choose_first_frames(good_frames, tool_annotations, early_window=30)

                
                        annotated_frames = sorted({ann["frame_idx"] for ann in tool_annotations})
                        anchor = annotated_frames[0]
                        good_frames = sorted(set([anchor] + good_frames))

                        results = run_propagation(
                            predictor, inference_state, tool_annotations, good_frames
                        )
                    # ===== video done =====
                    del inference_state
                    torch.cuda.empty_cache()

                    # ---------- Save Results ----------
                    video_name = Path(video_dir_l).name
                    save_dir = Path(base_train) / action / case_name / video_name / "sam_results"

                    if save_dir.exists():
                        print(f"[INFO] Found existing {save_dir}, removing old results...")
                        shutil.rmtree(save_dir)

                    if not results:
                        print(f"[WARN] Empty results for {case_name}/{video_name}, skip saving.")
                        continue
                    save_results(
                        results,
                        str(save_dir),
                        orig_img_folder=video_dir,
                        annotations=annotations
                    )

                    print(f"[INFO] Saved masks and heatmaps to {save_dir}/")
                    last_save_dir = Path(base_train) / action / case_name / video_name / "heatmap_sam"
                    if last_save_dir.exists():
                        shutil.rmtree(last_save_dir)

                    # Remove temporary JPG folder
                    if "images_jpg" in cache_dir and os.path.exists(cache_dir):
                        shutil.rmtree(cache_dir)
                        print(f"[INFO] Removed temporary cache directory {cache_dir}")
                    else:
                        print(f"[WARN] Cache directory {cache_dir} does not exist or was not created by this script.")
                                          