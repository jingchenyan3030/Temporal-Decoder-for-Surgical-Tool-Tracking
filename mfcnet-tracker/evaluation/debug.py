import os, csv, json
import numpy as np
import cv2
import re







import os, json
from collections import Counter
def dump_dropped_missing_json(gt_dict, pred_norm, case_name, dump_dir):
    gt_keys = sorted(gt_dict.keys())
    pred_keys = sorted(pred_norm.keys())

    gt_set = set(gt_keys)
    pred_set = set(pred_keys)

    common = sorted(gt_set & pred_set)
    dropped_pred = sorted(pred_set - gt_set)   # pred 有但 GT 没 -> 会被过滤
    missing_pred = sorted(gt_set - pred_set)   # GT 有但 pred 没

    def get_video(k):
        k = k.replace("\\", "/")
        parts = k.split("/")
        return parts[1] if len(parts) >= 2 else "UNKNOWN"

    drop_cnt = Counter(get_video(k) for k in dropped_pred)
    miss_cnt = Counter(get_video(k) for k in missing_pred)

    os.makedirs(dump_dir, exist_ok=True)

    dropped_path = os.path.join(dump_dir, f"{case_name}_dropped_pred.json")
    missing_path = os.path.join(dump_dir, f"{case_name}_missing_pred.json")
    summary_path = os.path.join(dump_dir, f"{case_name}_keydiff_summary.json")

    # 1) dropped_pred.json：只存 dropped_pred + 分组统计
    dropped_obj = {
        "case": case_name,
        "counts": {
            "gt": len(gt_set),
            "pred": len(pred_set),
            "common": len(common),
            "dropped_pred": len(dropped_pred),
            "coverage_common_over_gt": float(len(common) / (len(gt_set) + 1e-8)),
        },
        "dropped_pred_by_video": dict(drop_cnt),
        "dropped_pred_keys": dropped_pred,  # 全量
    }
    with open(dropped_path, "w", encoding="utf-8") as f:
        json.dump(dropped_obj, f, indent=2)

    # 2) missing_pred.json：只存 missing_pred + 分组统计
    missing_obj = {
        "case": case_name,
        "counts": {
            "gt": len(gt_set),
            "pred": len(pred_set),
            "common": len(common),
            "missing_pred": len(missing_pred),
        },
        "missing_pred_by_video": dict(miss_cnt),
        "missing_pred_keys": missing_pred,  # 全量
    }
    with open(missing_path, "w", encoding="utf-8") as f:
        json.dump(missing_obj, f, indent=2)

    # 3) 可选：summary.json（很小，终端也会打印这里的信息）
    summary_obj = {
        "case": case_name,
        "gt": len(gt_set),
        "pred": len(pred_set),
        "common": len(common),
        "dropped_pred": len(dropped_pred),
        "missing_pred": len(missing_pred),
        "coverage_common_over_gt": float(len(common) / (len(gt_set) + 1e-8)),
        "top10_dropped_videos": drop_cnt.most_common(10),
        "top10_missing_videos": miss_cnt.most_common(10),
        "sample_gt_keys": gt_keys[:5],
        "sample_pred_keys": pred_keys[:5],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, indent=2)

    # 终端只打印摘要 + 路径
    print("\n========== KEY DIFF SUMMARY ==========")
    print("[case]", case_name)
    print("GT =", len(gt_set), "Pred =", len(pred_set), "Common =", len(common))
    print("Dropped_pred =", len(dropped_pred), "Missing_pred =", len(missing_pred))
    print("Coverage(common/GT) =", float(len(common) / (len(gt_set) + 1e-8)))
    print("Top10 dropped videos:", drop_cnt.most_common(10))
    print("Top10 missing videos:", miss_cnt.most_common(10))
    print("\nSaved JSON:")
    print(" ", dropped_path)
    print(" ", missing_path)
    print(" ", summary_path)

    return common, dropped_pred, missing_pred







def normalize_case_prefix(case_prefix: str) -> str:
    """
    Only convert short name like 'cholec29' -> 'cholec80_29'.
    Keep everything else unchanged (e.g., 'cholec80_29', 'case_059_...', etc.)
    """
    m = re.fullmatch(r"cholec(\d+)", case_prefix)
    if m:
        return f"cholec80_{m.group(1)}"
    return case_prefix

def normalize_path_key(rel_path: str) -> str:
    rel_path = rel_path.replace("\\", "/")
    parts = rel_path.split("/", 1)
    if len(parts) == 1:
        return normalize_case_prefix(parts[0])
    case_prefix, rest = parts
    return normalize_case_prefix(case_prefix) + "/" + rest


def back_normalize_name(name: str) -> str:
    m = re.fullmatch(r"cholec80_(\d+)", name)
    if m:
        return f"cholec{int(m.group(1)):02d}"
    return name

def normalize_name(case_name):
    if case_name.startswith("cholec") and "_" not in case_name:
        num = case_name.replace("cholec", "")
        return f"cholec80_{num}"
    return case_name


def joint_points_per_frame(points):
    frames = {}
    for p in points:
        fid = int(p["frameIndex"])
        frames.setdefault(fid, []).append(p)
    return frames


def build_kpt_gt(case_name, training_data_root, ori_data_root):
    gt = {}
    csv_path = os.path.join(training_data_root, case_name, 'clip_to_video_mapping.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Mapping file not found: {csv_path}")
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            json_file = row['json_file']
            action_file = row['action']
            out_dir_old = row['out_dir']
            n_frames = int(row['n_frames'])

            case_name = normalize_name(case_name)
            video_dir = os.path.basename(out_dir_old)
            out_dir_new = os.path.join(training_data_root, case_name, video_dir)

            json_path  = os.path.join(ori_data_root, case_name, 'annotation', json_file)
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"JSON file not found: {json_path}")
            
            with open(json_path, "r", encoding="utf-8") as jf:
                data = json.load(jf)

            W_orig = int(round(data["imageDimensions"]["width"]))
            H_orig = int(round(data["imageDimensions"]["height"]))

            frames = joint_points_per_frame(data["points"])
            frames_seq = sorted(frames.keys())

            for local_idx, fid in enumerate(frames_seq, start=1):
                pts_raw = frames[fid]

                raw_tip = [pt for pt in pts_raw if pt.get("type") == "tool_tip"]
                raw_anchor = [pt for pt in pts_raw if pt.get("type") == "tool_anchor"]

                if len(raw_tip) == 0 or len(raw_anchor) == 0:
                    continue

                if any(pt.get("vis") is not True for pt in raw_tip):
                    continue
                if any(pt.get("vis") is not True for pt in raw_anchor):
                    continue

                tip_pts = []
                anchor_pts = []
                for pt in pts_raw:
                    if not pt.get("vis", True):
                        continue
                    t = pt.get("type","")
                    if t == "tool_tip":
                        tip_pts.append([pt["x"], pt["y"]])
                    elif t == "tool_anchor":
                        anchor_pts.append([pt["x"], pt["y"]])
                    else:
                        continue

                tip_arr = (
                    np.array(tip_pts, dtype=np.float32)
                    if tip_pts else np.zeros((0, 2), dtype=np.float32)
                )
                anchor_arr = (
                    np.array(anchor_pts, dtype=np.float32)
                    if anchor_pts else np.zeros((0, 2), dtype=np.float32)
                )

                img_path_abs = os.path.join(
                    out_dir_new,
                    "images",
                    f"frame_{local_idx:03d}.png"
                )
                rel_img_path = os.path.relpath(img_path_abs, training_data_root)

                gt[rel_img_path] = {
                    "W_orig": W_orig,
                    "H_orig": H_orig,
                    "tip":    tip_arr,
                    "anchor": anchor_arr,
                }

    return gt

def rescale_pred_to_gt(pred_points, pred_size, ori_size):
    W_pred, H_pred = pred_size
    W_gt, H_gt     = ori_size

    scale_x = W_gt / float(W_pred)
    scale_y = H_gt / float(H_pred)

    scaled_points = pred_points.copy()
    scaled_points[:, 0] = scaled_points[:, 0] * scale_x
    scaled_points[:, 1] = scaled_points[:, 1] * scale_y

    return scaled_points

def greedy_nearest_match(gt_points, pred_points, dist_thresh=None):
    gt_points = np.asarray(gt_points, dtype=np.float32)
    pred_points = np.asarray(pred_points, dtype=np.float32) 

    Ng = gt_points.shape[0]
    Np = pred_points.shape[0]
    if Ng == 0 or Np == 0:
        return [], list(range(Ng)), list(range(Np))

    diff = gt_points[:, None, :] - pred_points[None, :, :]
    D = np.linalg.norm(diff, axis=-1)

    matches = []
    matched_gt = set()
    matched_pred = set()

    while True:
        gi, pj  = np.unravel_index(np.argmin(D, axis=None), D.shape)
        min_dist = D[gi, pj]

        if not np.isfinite(min_dist):
            break
        if dist_thresh is not None and min_dist > dist_thresh:
            break
        if gi in matched_gt or pj in matched_pred:
            D[gi, pj] = np.inf
            continue

        matches.append((gi, pj, min_dist))
        matched_gt.add(gi)
        matched_pred.add(pj)
        D[gi, :] = np.inf
        D[:, pj] = np.inf

    unmatched_gt = [i for i in range(Ng) if i not in matched_gt]
    unmatched_pred = [j for j in range(Np) if j not in matched_pred]
    return matches, unmatched_gt, unmatched_pred

def build_kpt_pred_from_json(pred_json_path):
    if not os.path.exists(pred_json_path):
        raise FileNotFoundError(f"Pred json not found: {pred_json_path}")

    with open(pred_json_path, "r", encoding="utf-8") as f:
        raw_pred = json.load(f)

    pred = {}

    for rel_img_path, info in raw_pred.items():
        W_pred = int(info.get("W_pred"))
        H_pred = int(info.get("H_pred"))

        tip_raw = info.get("tip", [])
        tip_xy = [[pt[0], pt[1]] for pt in tip_raw]
        tip_arr = (
            np.array(tip_xy, dtype=np.float32)
            if tip_xy else np.zeros((0, 2), dtype=np.float32)
        )

        anchor_raw = info.get("anchor", [])
        anchor_xy = [[pt[0], pt[1]] for pt in anchor_raw]
        anchor_arr = (
            np.array(anchor_xy, dtype=np.float32)
            if anchor_xy else np.zeros((0, 2), dtype=np.float32)
        )

        pred[rel_img_path] = {
            "W_pred": W_pred,
            "H_pred": H_pred,
            "tip":    tip_arr,
            "anchor": anchor_arr,
        }

    return pred

def rescale_points_generic(points, src_size, dst_size):
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] == 0:
        return points
    W_src, H_src = src_size
    W_dst, H_dst = dst_size

    scale_x = W_dst / float(W_src)
    scale_y = H_dst / float(H_src)

    out = points.copy()
    out[:, 0] = out[:, 0] * scale_x
    out[:, 1] = out[:, 1] * scale_y
    return out

def normalize_pred_key(rel_img_path, case_name):
    rel_img_path = rel_img_path.replace("\\", "/")
    if case_name in rel_img_path:
        idx = rel_img_path.index(case_name)
        return rel_img_path[idx:]
    return os.path.join(case_name, rel_img_path.lstrip("./").lstrip("/"))

def build_kpt_pred_for_case(pred_root_dir, case_name):
    case_dir = os.path.join(pred_root_dir, case_name)
    if not os.path.isdir(case_dir):
        raise FileNotFoundError(f"Pred case dir not found: {case_dir}")

    pred = {}

    for video_name in sorted(os.listdir(case_dir)):
        video_dir = os.path.join(case_dir, video_name)
        if not os.path.isdir(video_dir):
            continue

        json_path = os.path.join(video_dir, "pred_points.json")
        if not os.path.exists(json_path):
            continue

        one_pred = build_kpt_pred_from_json(json_path)

        for rel_img_path, info in one_pred.items():
            key = normalize_pred_key(rel_img_path, case_name)
            pred[key] = info

    return pred

from scipy.optimize import linear_sum_assignment

def hungarian_match(gt_points, pred_points, dist_thresh=None):
    gt_points = np.asarray(gt_points, dtype=np.float32)
    pred_points = np.asarray(pred_points, dtype=np.float32)

    Ng = gt_points.shape[0]
    Np = pred_points.shape[0]

    if Ng == 0 or Np == 0:
        return [], list(range(Ng)), list(range(Np))

    diff = gt_points[:, None, :] - pred_points[None, :, :]
    D = np.linalg.norm(diff, axis=-1)

    row_ind, col_ind = linear_sum_assignment(D)

    matches = []
    matched_gt = set()
    matched_pred = set()

    for gi, pj in zip(row_ind, col_ind):
        dist = D[gi, pj]
        if dist_thresh is not None and dist > dist_thresh:
            continue
        matches.append((gi, pj, dist))
        matched_gt.add(gi)
        matched_pred.add(pj)

    unmatched_gt = [i for i in range(Ng) if i not in matched_gt]
    unmatched_pred = [j for j in range(Np) if j not in matched_pred]

    return matches, unmatched_gt, unmatched_pred

def match_points(gt_points, pred_points, dist_thresh=None, method="greedy"):
    if method == "hungarian":
        return hungarian_match(gt_points, pred_points, dist_thresh)
    else:
        return greedy_nearest_match(gt_points, pred_points, dist_thresh)

def visualize_one_frame(
        img_path,
        gt_tip, gt_anchor,
        pred_tip, pred_anchor,
        tip_matches, tip_un_gt, tip_un_pred,
        anchor_matches, anchor_un_gt, anchor_un_pred,
        save_path
):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    Yellow = (0, 255, 255)
    Red    = (0, 0, 255)

    for x, y in gt_tip:
        cx, cy = int(x), int(y)
        cv2.circle(img, (cx, cy), 4, Yellow, -1)

    for x, y in pred_tip:
        cx, cy = int(x), int(y)
        cv2.drawMarker(img, (cx, cy), Yellow, markerType=cv2.MARKER_CROSS, markerSize=8, thickness=2)

    for gi, pj, _ in tip_matches:
        gx, gy = gt_tip[gi]
        px, py = pred_tip[pj]
        cv2.line(img, (int(gx), int(gy)), (int(px), int(py)), Yellow, 2)

    for x, y in gt_anchor:
        cx, cy = int(x), int(y)
        cv2.circle(img, (cx, cy), 4, Red, -1)

    for x, y in pred_anchor:
        cx, cy = int(x), int(y)
        cv2.drawMarker(img, (cx, cy), Red, markerType=cv2.MARKER_CROSS, markerSize=8, thickness=2)
    
    for gi, pj, _ in anchor_matches:
        gx, gy = gt_anchor[gi]
        px, py = pred_anchor[pj]
        cv2.line(img, (int(gx), int(gy)), (int(px), int(py)), Red, 2)

    if save_path is None:
        print("[WARN] save_path is None, skip saving.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)
    print(f"[INFO] Visualization saved to: {save_path}")

def test_single_case(
        case_name,
        training_data_root,
        ori_data_root,
        pred_root_dir,
        dist_thresh=None,
        match_method="hungarian",
        vis_n_frames=None,
        vis_out_dir="/data/home/hao/chenyan/data/match_vis_mse",
):
    gt_dict = build_kpt_gt(case_name, training_data_root, ori_data_root)
    pred_dict = build_kpt_pred_for_case(pred_root_dir, case_name)

    '''
        ############ debug
        print("\n[DEBUG STEP 1]")
        print("[DEBUG] case_name =", case_name)
        print("[DEBUG] expected pred case_dir =", os.path.join(pred_root_dir, case_name))
        print("[DEBUG] pred_dict size BEFORE filtering =", len(pred_dict))
        if len(pred_dict) > 0:
            k0 = next(iter(pred_dict.keys()))
            print("[DEBUG] sample pred key BEFORE filtering =", k0)
        #############
    '''

    pred_raw = build_kpt_pred_for_case(pred_root_dir, case_name)
    pred_norm = {normalize_path_key(k): v for k, v in pred_raw.items()}

    print("[DEBUG] sample GT keys:", list(gt_dict.keys())[:5])
    print("[DEBUG] sample Pred keys:", list(pred_norm.keys())[:5])

    common_keys, dropped_pred, missing_pred = dump_dropped_missing_json(
        gt_dict, pred_norm, case_name,
        dump_dir="/data/home/hao/chenyan/data/key_debug"
    )

    pred_dict = {k: pred_norm[k] for k in common_keys}



    '''
    gt_keys = set(gt_dict.keys())
    pred_keys = set(pred_dict.keys())
    common_keys = sorted(gt_keys & pred_keys)
    
        print("[DEBUG] GT frames =", len(gt_keys))
        print("[DEBUG] Pred frames =", len(pred_keys))
        print("[DEBUG] Common(frames used for eval) =", len(common_keys))
        print("[DEBUG] Coverage =", len(common_keys) / (len(gt_keys) + 1e-8))


        ############ debug
        print("\n[DEBUG STEP 2]")
        print("[DEBUG] gt_dict size =", len(gt_dict))
        print("[DEBUG] pred_dict size AFTER filtering =", len(pred_dict))
        ############


        print("len(gt_dict) =", len(gt_dict))
        print("len(pred_dict) =", len(pred_dict))
        print("sample GT key:", list(gt_dict.keys())[:5])
        print("sample Pred key:", list(pred_dict.keys())[:5])
    '''
    stats = {
        "tip":    {"TP": 0, "FP": 0, "FN": 0, "sum_l1":0.0,"sum_l2":0.0,"num_match":0},
        "anchor": {"TP": 0, "FP": 0, "FN": 0, "sum_l1":0.0,"sum_l2":0.0,"num_match":0},
    }

    totals = {
        "tip": {"gt_points": 0, "pred_points": 0},
        "anchor": {"gt_points": 0, "pred_points": 0},
    }

    vis_done = 0

    for rel_img_path in common_keys:
        gt_info = gt_dict[rel_img_path]
        pred_info = pred_dict[rel_img_path]

        W_gt, H_gt = gt_info["W_orig"], gt_info["H_orig"]
        W_pred, H_pred = pred_info["W_pred"], pred_info["H_pred"]

        # tip
        gt_tip = gt_info["tip"]
        pred_tip = pred_info["tip"]
        pred_tip_rescaled = rescale_pred_to_gt(pred_tip, (W_pred, H_pred), (W_gt, H_gt))

        totals["tip"]["gt_points"]   += int(gt_tip.shape[0])
        totals["tip"]["pred_points"] += int(pred_tip_rescaled.shape[0])


        tip_matches, tip_un_gt, tip_un_pred = match_points(
            gt_tip, pred_tip_rescaled, dist_thresh, match_method
        )

        stats["tip"]["TP"] += len(tip_matches)
        stats["tip"]["FN"] += len(tip_un_gt)     # only unmatched Pred points contribute to FN
        stats["tip"]["FP"] += len(tip_un_pred)

        for gi, pj, dist in tip_matches:
            stats["tip"]["sum_l2"] += dist
            stats["tip"]["sum_l1"] += np.linalg.norm(gt_tip[gi] - pred_tip_rescaled[pj], ord=1)
            stats["tip"]["num_match"] += 1

        # anchor
        gt_anchor = gt_info["anchor"]
        pred_anchor = pred_info["anchor"]
        pred_anchor_rescaled = rescale_pred_to_gt(pred_anchor, (W_pred, H_pred), (W_gt, H_gt))

        totals["anchor"]["gt_points"]   += int(gt_anchor.shape[0])
        totals["anchor"]["pred_points"] += int(pred_anchor_rescaled.shape[0])

        anchor_matches, anchor_un_gt, anchor_un_pred = match_points(
            gt_anchor, pred_anchor_rescaled, dist_thresh, match_method
        )

        stats["anchor"]["TP"] += len(anchor_matches)
        stats["anchor"]["FN"] += len(anchor_un_gt)
        stats["anchor"]["FP"] += len(anchor_un_pred)

        for gi, pj, dist in anchor_matches:
            stats["anchor"]["sum_l2"] += dist
            stats["anchor"]["sum_l1"] += np.linalg.norm(gt_anchor[gi] - pred_anchor_rescaled[pj], ord=1)
            stats["anchor"]["num_match"] += 1        


        if (vis_n_frames is None) or (vis_done < vis_n_frames):
            rel_img_path_for_disk = rel_img_path   
            m = re.fullmatch(r"cholec(\d+)", case_name)
            if m:
                case_disk = case_name
                case_key  = normalize_case_prefix(case_name)  
                rel_img_path_for_disk = rel_img_path.replace(case_key + "/", case_disk + "/", 1)
            img_abs_path_for_disk = os.path.join(training_data_root, rel_img_path_for_disk)
            img = cv2.imread(img_abs_path_for_disk)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_abs_path_for_disk}")

            H_img, W_img = img.shape[:2]

            # Rescale for visualization only
            gt_tip_vis = rescale_points_generic(gt_tip, (W_gt, H_gt), (W_img, H_img))
            gt_anchor_vis = rescale_points_generic(gt_anchor, (W_gt, H_gt), (W_img, H_img))
            pred_tip_vis = rescale_points_generic(pred_tip_rescaled, (W_gt, H_gt), (W_img, H_img))
            pred_anchor_vis = rescale_points_generic(pred_anchor_rescaled, (W_gt, H_gt), (W_img, H_img))

            save_path = os.path.join(
                vis_out_dir,
                case_name.replace("/", "_") + f"_{vis_done:04d}.png"
            )

            visualize_one_frame(
                img_abs_path_for_disk,
                gt_tip_vis, gt_anchor_vis,
                pred_tip_vis, pred_anchor_vis,
                tip_matches, tip_un_gt, tip_un_pred,
                anchor_matches, anchor_un_gt, anchor_un_pred,
                save_path=save_path,
            )
            vis_done += 1

    def _print_stats(key):
        TP = stats[key]["TP"]
        FP = stats[key]["FP"]
        FN = stats[key]["FN"]
        num_match = stats[key]["num_match"]

        if num_match > 0:
            avg_l2 = stats[key]["sum_l2"] / num_match
            avg_l1 = stats[key]["sum_l1"] / num_match
        else:
            avg_l2 = 0.0
            avg_l1 = 0.0

        print(f"[{key.upper()}] Avg L2 error: {avg_l2:.4f}, Avg L1 error: {avg_l1:.4f}")
        precision = TP / (TP + FP + 1e-8)
        recall    = TP / (TP + FN + 1e-8)
        print(f"[{key.upper()}] TP={TP}, FP={FP}, FN={FN}, "
              f"precision={precision:.4f}, recall={recall:.4f}")
        GTN = totals[key]["gt_points"]
        PRN = totals[key]["pred_points"]
        print(f"[{key.upper()}] Total points on common frames: GT={GTN}, Pred={PRN}, "
            f"TP+FN={TP+FN}, TP+FP={TP+FP}")

    print(f"\n=== Single-case evaluation: {case_name} ===")
    _print_stats("tip")
    _print_stats("anchor")

    return stats

def summarize_stats(stats):
    summary = {}
    for key in ["tip", "anchor"]:
        TP = stats[key]["TP"]
        FP = stats[key]["FP"]
        FN = stats[key]["FN"]
        num_match = stats[key]["num_match"]
        sum_l2 = float(stats[key]["sum_l2"])
        sum_l1 = float(stats[key]["sum_l1"])

        if num_match > 0:
            avg_l2 = float(sum_l2 / num_match)
            avg_l1 = float(sum_l1 / num_match)
        else:
            avg_l2 = 0.0
            avg_l1 = 0.0

        precision = TP / (TP + FP + 1e-8)
        recall    = TP / (TP + FN + 1e-8)

        summary[key] = {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "num_match": num_match,
            "avg_l1": avg_l1,
            "avg_l2": avg_l2,
            "precision": precision,
            "recall": recall,
        }
    return summary

if __name__ == "__main__":
    case_name = "cholec29"

    training_data_root = "/data/home/hao/chenyan/data/0923_training_data"
    ori_data_root = "/data/home/hao/chenyan/ori_data/surg_act_09232025"
    pred_root_dir = "/data/home/hao/chenyan/data/0923_test_multiframe_mse/grasp"

    stats = test_single_case(
        case_name,
        training_data_root,
        ori_data_root,
        pred_root_dir,
        dist_thresh=50.0,
        match_method="hungarian",
        vis_n_frames=10,
        vis_out_dir="/data/home/hao/chenyan/data/match_vis_mse",
    )

    print("\n=== Summary dict (printed only, not saved) ===")
    print(json.dumps({case_name: summarize_stats(stats)}, indent=2))



