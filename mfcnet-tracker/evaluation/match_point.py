#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, json
import numpy as np
import re

from scipy.optimize import linear_sum_assignment

# =========================
# Canonical evaluation size
# =========================
CANON_W, CANON_H = 640, 480


# =========================
# Path / name normalization
# =========================
def normalize_case_prefix(case_prefix: str) -> str:
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


def normalize_name(case_name: str) -> str:
    # "cholec29" -> "cholec80_29"
    if case_name.startswith("cholec") and "_" not in case_name:
        num = case_name.replace("cholec", "")
        return f"cholec80_{num}"
    return case_name


def normalize_pred_key(rel_img_path: str, case_name: str) -> str:
    rel_img_path = rel_img_path.replace("\\", "/")
    if case_name in rel_img_path:
        idx = rel_img_path.index(case_name)
        return rel_img_path[idx:]
    return os.path.join(case_name, rel_img_path.lstrip("./").lstrip("/"))


# =========================
# Geometry utilities
# =========================
def rescale_points_generic(points, src_size, dst_size):
    """
    points: (N,2) in src_size -> (N,2) in dst_size
    src_size/dst_size: (W,H)
    """
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] == 0:
        return points

    W_src, H_src = src_size
    W_dst, H_dst = dst_size

    sx = W_dst / float(W_src)
    sy = H_dst / float(H_src)

    out = points.copy()
    out[:, 0] = out[:, 0] * sx
    out[:, 1] = out[:, 1] * sy
    return out


def to_canon(points_xy, src_size):
    """Map points from (W_src,H_src) -> (CANON_W,CANON_H)."""
    return rescale_points_generic(points_xy, src_size, (CANON_W, CANON_H))


# =========================
# GT building
# =========================
def joint_points_per_frame(points):
    frames = {}
    for p in points:
        fid = int(p["frameIndex"])
        frames.setdefault(fid, []).append(p)
    return frames


def build_kpt_gt(case_name, training_data_root, ori_data_root):
    """
    Return:
      gt[rel_img_path] = {
        "W_orig": int, "H_orig": int,
        "tip": (Nt,2) float32,
        "anchor": (Na,2) float32
      }
    rel_img_path is relative to training_data_root.
    """
    gt = {}
    csv_path = os.path.join(training_data_root, case_name, "clip_to_video_mapping.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Mapping file not found: {csv_path}")

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            json_file = row["json_file"]
            out_dir_old = row["out_dir"]

            case_name_norm = normalize_name(case_name)
            video_dir = os.path.basename(out_dir_old)
            out_dir_new = os.path.join(training_data_root, case_name_norm, video_dir)

            json_path = os.path.join(ori_data_root, case_name_norm, "annotation", json_file)
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

                # filter: must have both tip & anchor, and all vis True for those
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
                    t = pt.get("type", "")
                    if t == "tool_tip":
                        tip_pts.append([pt["x"], pt["y"]])
                    elif t == "tool_anchor":
                        anchor_pts.append([pt["x"], pt["y"]])

                tip_arr = np.array(tip_pts, dtype=np.float32) if tip_pts else np.zeros((0, 2), dtype=np.float32)
                anchor_arr = np.array(anchor_pts, dtype=np.float32) if anchor_pts else np.zeros((0, 2), dtype=np.float32)

                img_path_abs = os.path.join(out_dir_new, "images", f"frame_{local_idx:03d}.png")
                rel_img_path = os.path.relpath(img_path_abs, training_data_root)

                gt[rel_img_path] = {
                    "W_orig": W_orig,
                    "H_orig": H_orig,
                    "tip": tip_arr,
                    "anchor": anchor_arr,
                }

    return gt


# =========================
# Pred building
# =========================
def build_kpt_pred_from_json(pred_json_path):
    """
    raw_pred[rel_img_path] = {
      "W_pred": int, "H_pred": int,
      "tip": [[x,y],...], "anchor":[[x,y],...]
    }
    """
    if not os.path.exists(pred_json_path):
        raise FileNotFoundError(f"Pred json not found: {pred_json_path}")

    with open(pred_json_path, "r", encoding="utf-8") as f:
        raw_pred = json.load(f)

    pred = {}
    for rel_img_path, info in raw_pred.items():
        W_pred = int(info.get("W_pred"))
        H_pred = int(info.get("H_pred"))

        tip_xy = info.get("tip", [])
        tip_arr = np.array([[pt[0], pt[1]] for pt in tip_xy], dtype=np.float32) if tip_xy else np.zeros((0, 2), dtype=np.float32)

        anchor_xy = info.get("anchor", [])
        anchor_arr = np.array([[pt[0], pt[1]] for pt in anchor_xy], dtype=np.float32) if anchor_xy else np.zeros((0, 2), dtype=np.float32)

        pred[rel_img_path] = {
            "W_pred": W_pred,
            "H_pred": H_pred,
            "tip": tip_arr,
            "anchor": anchor_arr,
        }
    return pred


def build_kpt_pred_for_case(pred_root_dir, case_name):
    """
    Traverse:
      pred_root_dir/case_name/*/pred_points.json
    Merge all pred_points.json into one dict.
    """
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


# =========================
# Matching (Hungarian / Greedy)
# =========================
def greedy_nearest_match(gt_points, pred_points, dist_thresh=None):
    gt_points = np.asarray(gt_points, dtype=np.float32)
    pred_points = np.asarray(pred_points, dtype=np.float32)

    Ng = gt_points.shape[0]
    Np = pred_points.shape[0]
    if Ng == 0 or Np == 0:
        return [], list(range(Ng)), list(range(Np))

    D = np.linalg.norm(gt_points[:, None, :] - pred_points[None, :, :], axis=-1)

    matches = []
    matched_gt = set()
    matched_pred = set()

    while True:
        gi, pj = np.unravel_index(np.argmin(D, axis=None), D.shape)
        min_dist = D[gi, pj]

        if not np.isfinite(min_dist):
            break
        if dist_thresh is not None and min_dist > dist_thresh:
            break
        if gi in matched_gt or pj in matched_pred:
            D[gi, pj] = np.inf
            continue

        matches.append((gi, pj, float(min_dist)))
        matched_gt.add(gi)
        matched_pred.add(pj)
        D[gi, :] = np.inf
        D[:, pj] = np.inf

    unmatched_gt = [i for i in range(Ng) if i not in matched_gt]
    unmatched_pred = [j for j in range(Np) if j not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


def hungarian_match(gt_points, pred_points, dist_thresh=None):
    gt_points = np.asarray(gt_points, dtype=np.float32)
    pred_points = np.asarray(pred_points, dtype=np.float32)

    Ng = gt_points.shape[0]
    Np = pred_points.shape[0]
    if Ng == 0 or Np == 0:
        return [], list(range(Ng)), list(range(Np))

    D = np.linalg.norm(gt_points[:, None, :] - pred_points[None, :, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(D)

    matches = []
    matched_gt = set()
    matched_pred = set()

    for gi, pj in zip(row_ind, col_ind):
        dist = float(D[gi, pj])
        if dist_thresh is not None and dist > dist_thresh:
            continue
        matches.append((int(gi), int(pj), dist))
        matched_gt.add(int(gi))
        matched_pred.add(int(pj))

    unmatched_gt = [i for i in range(Ng) if i not in matched_gt]
    unmatched_pred = [j for j in range(Np) if j not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


def match_points(gt_points, pred_points, dist_thresh=None, method="hungarian"):
    if method == "greedy":
        return greedy_nearest_match(gt_points, pred_points, dist_thresh)
    return hungarian_match(gt_points, pred_points, dist_thresh)


# =========================
# Evaluation
# =========================
def init_stats():
    return {
        "tip":    {"TP": 0, "FP": 0, "FN": 0, "sum_l1": 0.0, "sum_l2": 0.0, "num_match": 0},
        "anchor": {"TP": 0, "FP": 0, "FN": 0, "sum_l1": 0.0, "sum_l2": 0.0, "num_match": 0},
    }


def accumulate_stats(global_stats, new_stats):
    for key in ["tip", "anchor"]:
        for info in ["TP", "FP", "FN", "sum_l1", "sum_l2", "num_match"]:
            global_stats[key][info] += new_stats[key][info]
    return global_stats


def summarize_stats(stats):
    summary = {}
    for key in ["tip", "anchor"]:
        TP = stats[key]["TP"]
        FP = stats[key]["FP"]
        FN = stats[key]["FN"]
        num_match = stats[key]["num_match"]
        sum_l2 = float(stats[key]["sum_l2"])
        sum_l1 = float(stats[key]["sum_l1"])

        avg_l2 = float(sum_l2 / num_match) if num_match > 0 else 0.0
        avg_l1 = float(sum_l1 / num_match) if num_match > 0 else 0.0
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)

        summary[key] = {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "num_match": num_match,
            "avg_l1": avg_l1,
            "avg_l2": avg_l2,
            "precision": float(precision),
            "recall": float(recall),
        }
    return summary


def test_single_case(
    case_name,
    training_data_root,
    ori_data_root,
    pred_root_dir,
    dist_thresh=None,          # interpreted in 640×480 pixels
    match_method="hungarian",  # "hungarian" or "greedy"
):
    """
    Key behavior:
      - GT points are in (W_gt,H_gt) from annotation
      - Pred points are in (W_pred,H_pred) from pred json
      - We map BOTH to (640,480), then match & compute errors in (640,480)
    """
    gt_dict = build_kpt_gt(case_name, training_data_root, ori_data_root)
    pred_dict = build_kpt_pred_for_case(pred_root_dir, case_name)

    gt_norm = {normalize_path_key(k): v for k, v in gt_dict.items()}
    pred_norm = {normalize_path_key(k): v for k, v in pred_dict.items()}

    common_keys = sorted(set(gt_norm.keys()) & set(pred_norm.keys()))
    stats = init_stats()

    for rel_img_path in common_keys:
        gt_info = gt_norm[rel_img_path]
        pred_info = pred_norm[rel_img_path]

        W_gt, H_gt = gt_info["W_orig"], gt_info["H_orig"]
        W_pred, H_pred = pred_info["W_pred"], pred_info["H_pred"]

        # --- TIP (both -> 640×480) ---
        gt_tip_640 = to_canon(gt_info["tip"], (W_gt, H_gt))
        pred_tip_640 = to_canon(pred_info["tip"], (W_pred, H_pred))

        tip_matches, tip_un_gt, tip_un_pred = match_points(
            gt_tip_640, pred_tip_640, dist_thresh, method=match_method
        )
        stats["tip"]["TP"] += len(tip_matches)
        stats["tip"]["FN"] += len(tip_un_gt)
        stats["tip"]["FP"] += len(tip_un_pred)

        for gi, pj, dist in tip_matches:
            stats["tip"]["sum_l2"] += float(dist)
            stats["tip"]["sum_l1"] += float(np.linalg.norm(gt_tip_640[gi] - pred_tip_640[pj], ord=1))
            stats["tip"]["num_match"] += 1

        # --- ANCHOR (both -> 640×480) ---
        gt_anchor_640 = to_canon(gt_info["anchor"], (W_gt, H_gt))
        pred_anchor_640 = to_canon(pred_info["anchor"], (W_pred, H_pred))

        anchor_matches, anchor_un_gt, anchor_un_pred = match_points(
            gt_anchor_640, pred_anchor_640, dist_thresh, method=match_method
        )
        stats["anchor"]["TP"] += len(anchor_matches)
        stats["anchor"]["FN"] += len(anchor_un_gt)
        stats["anchor"]["FP"] += len(anchor_un_pred)

        for gi, pj, dist in anchor_matches:
            stats["anchor"]["sum_l2"] += float(dist)
            stats["anchor"]["sum_l1"] += float(np.linalg.norm(gt_anchor_640[gi] - pred_anchor_640[pj], ord=1))
            stats["anchor"]["num_match"] += 1

    # print per-case summary
    def _print_stats(key):
        TP = stats[key]["TP"]
        FP = stats[key]["FP"]
        FN = stats[key]["FN"]
        num_match = stats[key]["num_match"]
        avg_l2 = stats[key]["sum_l2"] / num_match if num_match > 0 else 0.0
        avg_l1 = stats[key]["sum_l1"] / num_match if num_match > 0 else 0.0
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)

        print(f"[{key.upper()}] Avg L2 error (640×480 px): {avg_l2:.4f}, Avg L1 error: {avg_l1:.4f}")
        print(f"[{key.upper()}] TP={TP}, FP={FP}, FN={FN}, precision={precision:.4f}, recall={recall:.4f}")

    print(f"\n=== Single-case evaluation (CANON {CANON_W}×{CANON_H}): {case_name} ===")
    _print_stats("tip")
    _print_stats("anchor")

    return stats


# =========================
# Main
# =========================
if __name__ == "__main__":
    training_data_root = "/data/home/hao/chenyan/data/0923_training_data"
    ori_data_root = "/data/home/hao/chenyan/ori_data/surg_act_09232025"
    pred_root_dir = "/data/home/hao/chenyan/data/0923_test_multiframe_mse_raw/clip"

    case_list = [case for case in sorted(os.listdir(pred_root_dir))]

    global_stats = init_stats()
    per_case_summaries = {}

    for case_name in case_list:
        new_stats = test_single_case(
            case_name=case_name,
            training_data_root=training_data_root,
            ori_data_root=ori_data_root,
            pred_root_dir=pred_root_dir,
            dist_thresh=25.0,          # now in 640×480 pixels
            match_method="hungarian",  # or "greedy"
        )
        global_stats = accumulate_stats(global_stats, new_stats)
        per_case_summaries[case_name] = summarize_stats(new_stats)

    global_summary = summarize_stats(global_stats)
    per_case_summaries["GLOBAL"] = global_summary

    print("\n=== GLOBAL summary (CANON ) ===")
    print(json.dumps(global_summary, indent=2))

    save_json_path = "/data/home/hao/chenyan/data/match_vis_mse_raw/eval_results_clip.json"
    os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(per_case_summaries, f, indent=2)

    print(f"\n[INFO] Saved evaluation summary to: {save_json_path}")
