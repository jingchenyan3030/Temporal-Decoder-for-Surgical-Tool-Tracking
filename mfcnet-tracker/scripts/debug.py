import json
from pathlib import Path
import cv2
import os

# =========================
# Paths
# =========================
img_path = Path(
    "/home/chenyan/fallout_data/data/0923_by_action/clip/"
    "case_059_video_part_001_segment_6/video_001/images/frame_001.png"
)

gt_json_path = Path(
    "/home/chenyan/fallout_data/data/0923_by_action/clip/"
    "case_059_video_part_001_segment_6/video_001/points_detr/frame_001.json"
)

# inference output json (one per video)
pred_json_path = Path(
    "/home/chenyan/fallout_data/data/kpt_test_multiframe_detr_raw/clip/"
    "case_059_video_part_001_segment_6/video_001/pred_points.json"
)

# this must match the key stored in pred_points.json
pred_frame_key = "clip/case_059_video_part_001_segment_6/video_001/images/frame_001.png"

# target size (same as model input / json coordinate space)
H_new, W_new = 256, 320

# save dir
out_dir = Path("~/surg_act_keypoint/mfcnet-tracker/debug_gt_check").expanduser()
out_dir.mkdir(parents=True, exist_ok=True)

# =========================
# Load image
# =========================
img_bgr = cv2.imread(str(img_path))
if img_bgr is None:
    raise FileNotFoundError(f"Cannot read image: {img_path}")

H_old, W_old = img_bgr.shape[:2]
print(f"Original image size: H={H_old}, W={W_old}")
print(f"Target image size  : H={H_new}, W={W_new}")

# resize original image to 320x256
img_resized = cv2.resize(img_bgr, (W_new, H_new), interpolation=cv2.INTER_LINEAR)
vis_resized = img_resized.copy()

# =========================
# Load GT
# =========================
with open(gt_json_path, "r") as f:
    gt = json.load(f)

# =========================
# Load Prediction
# =========================
with open(pred_json_path, "r") as f:
    pred_all = json.load(f)

if pred_frame_key not in pred_all:
    raise KeyError(f"{pred_frame_key} not found in {pred_json_path}")

pred_frame = pred_all[pred_frame_key]

pred_tip = pred_frame.get("tip", [])
pred_anchor = pred_frame.get("anchor", [])

print("Prediction loaded:")
print("tip   =", pred_tip)
print("anchor=", pred_anchor)

# =========================
# Drawing helper
# =========================
def draw_point(img, x, y, color, text=None, radius=4):
    x_i, y_i = int(round(x)), int(round(y))
    cv2.circle(img, (x_i, y_i), radius, color, -1)
    if text is not None:
        cv2.putText(
            img,
            text,
            (x_i + 5, y_i - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

# =========================
# Overlay GT points
# json GT points are already in 320x256 space
# =========================
for i, p in enumerate(gt):
    label = p["label"]
    vis = bool(p.get("visibility", True))
    x = float(p["x"])
    y = float(p["y"])

    tag = f"GT_{label}_{i}_{'V' if vis else 'X'}"

    # GT color
    if label == "tool_tip":
        color = (0, 255, 255)   # yellow
    elif label == "tool_anchor":
        color = (0, 0, 255)     # red
    elif label == "contact":
        color = (0, 255, 0)     # green
    else:
        color = (255, 255, 255) # white

    draw_point(vis_resized, x, y, color, tag, radius=5)

# =========================
# Overlay prediction points
# pred json is also in 320x256 space
# =========================
# predicted tip -> cyan
for i, pt in enumerate(pred_tip):
    if len(pt) < 2:
        continue
    x = float(pt[0])
    y = float(pt[1])
    score = float(pt[2]) if len(pt) > 2 else 1.0
    draw_point(vis_resized, x, y, (255, 255, 0), f"P_tip_{i}_{score:.2f}", radius=6)

# predicted anchor -> magenta
for i, pt in enumerate(pred_anchor):
    if len(pt) < 2:
        continue
    x = float(pt[0])
    y = float(pt[1])
    score = float(pt[2]) if len(pt) > 2 else 1.0
    draw_point(vis_resized, x, y, (255, 0, 255), f"P_anchor_{i}_{score:.2f}", radius=6)

# =========================
# Save
# =========================
out_file = out_dir / f"{img_path.stem}_gt_and_pred_overlay_on_resized_320x256.png"
cv2.imwrite(str(out_file), vis_resized)

print("Saved file:")
print(out_file)