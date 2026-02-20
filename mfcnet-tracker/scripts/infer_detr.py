"""
written by Chenyan 
"""

import os 
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
sys.path.append('.')
sys.path.append('./models/')
import logging, json, random, time, math 
from pathlib import Path

import configargparse
from configs.config_mse import test_config_parser as config_parser

import cv2 
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms
from collections import defaultdict
import matplotlib.pyplot as plt
from src.dataloader_multiframe import get_data_loader
from models import get_multiframe_segmentation_model as get_model
from utils.dataloader_utils import get_MICCAI2017_dataset_filenames, get_JIGSAWS_dataset_filenames, get_MICCAI2015_dataset_filenames, get_ACT_dataset_filenames, get_KPT_dataset_filenames
from utils.log_utils import AverageMeter, ProgressMeter
from utils.model_utils import load_model_weights
from utils.train_utils import add_metrics_meters
from utils.vis_utils import mask_overlay, draw_plus
from utils.localization_utils_v2 import centroid_error_10_classes, centroid_error_4_classes
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from models.PostProcessPoints import PostProcessPointsFixed

# Adding--Sub1: from segmentation to mask to get contact area --> polygon -> 4 points
# sub-function: Contour--4 point
def largest_inscribed_rect_corners(cnt, edge_samples=5, iters=20, tol=1e-3):
    if cnt is None or len(cnt) < 3:
        return None
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.float32)
    C = box.mean(axis=0)

    def inside_after_scale(s: float) -> bool:
        pts = C + s * (box - C)
        for (x, y) in pts:
            if cv2.pointPolygonTest(cnt, (float(x), float(y)), False) < 0:
                return False
        for i in range(4):
            a, b = pts[i], pts[(i + 1) % 4]
            for t in np.linspace(0.0, 1.0, edge_samples):
                x = a[0]*(1-t) + b[0]*t
                y = a[1]*(1-t) + b[1]*t
                if cv2.pointPolygonTest(cnt, (float(x), float(y)), False) < 0:
                    return False
        return True

    low, high = 0.0, 1.0
    if inside_after_scale(1.0):
        s_opt = 1.0
    else:
        s_opt = 0.0
        for _ in range(iters):
            mid = 0.5 * (low + high)
            if inside_after_scale(mid):
                s_opt = mid
                low = mid
            else:
                high = mid
            if high - low < tol:
                break
    pts_opt = C + s_opt * (box - C)
    return np.round(pts_opt).astype(int)


def compute_contact_corners_from_pred_mask(pred_mask, contact_id=3, min_area=10):
    contact_mask = (pred_mask == contact_id).astype(np.uint8) * 255
    contours, _ = cv2.findContours(contact_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < min_area:
        return []
    
    x, y, w, h = cv2.boundingRect(cnt)
    if w <4 and h < 4:
        cx, cy = int(x + w/2), int(y + h/2)
        half_size = 2
        corners = np.array([
            (cx - half_size, cy - half_size),
            (cx + half_size, cy - half_size),
            (cx + half_size, cy + half_size),
            (cx - half_size, cy + half_size)
        ], dtype=int)
    else:
        corners = np.array([
            (x, y),
            (x + w, y),
            (x + w, y + h),
            (x, y + h)
        ], dtype=int)
    
    H, W = pred_mask.shape[:2]
    corners[:, 0] = np.clip(corners[:, 0], 0, W - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, H - 1)
    
    return [(int(x), int(y), 1.0) for x, y in corners]

# denormalize the image for visualization
def postprocess_image(tensor_img):
    mean = np.array([0.485, 0.456, 0.406]).reshape((3,1,1))
    std  = np.array([0.229, 0.224, 0.225]).reshape((3,1,1))
    ori_img = np.clip(tensor_img * std + mean, 0, 1)
    ori_img = (np.transpose(ori_img, (1,2,0)) * 255).astype(np.uint8)
    return ori_img

# integrate 4 points
def prepare_vis_keypoints(results, b, pred_mask, contact_id=3):
    vis_keypoints = {
        'tip': results['tip'][b],
        'anchor': results['anchor'][b],
        'contact': compute_contact_corners_from_pred_mask(pred_mask, contact_id)
    }
    return vis_keypoints




# Adding:
def save_points_prediction(args, img_path, keypoints_dict, input_rgb):
    p = Path(img_path)
    rel = p.relative_to(args.data_dir)

    parts = list(rel.parts)
    if "images" in parts:
        idx = parts.index("images")
        out_rel = Path(*parts[:idx]) / "pred"
    else:
        out_rel = rel.parent / "pred"

    if args.dataset == 'ACT':
        out_dir = Path(args.data_dir).parent / "act_test_multiframe_detr" / out_rel
    else:
        out_dir = Path(args.data_dir).parent / "0923_test_multiframe_detr_raw" / out_rel
    out_dir.mkdir(parents=True, exist_ok=True)

    img_copy = input_rgb.copy()
    for name, keypoints in keypoints_dict.items():
        color = (255, 255, 0) if name == "tip" else (255, 0, 0) if name == "anchor" else (0, 255, 0)
        for (x, y, score) in keypoints:
            cv2.circle(
                img_copy,
                (int(round(x)), int(round(y))),
                radius=3,
                color=color,
                thickness=-1
            )

    out_file = out_dir / f"{p.stem}_points.png"
    print(f"INPUT → {img_path}  ||  OUTPUT → {out_file}")
    cv2.imwrite(str(out_file), cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))


def main(): 
    parser = configargparse.ArgumentParser() 
    parser = config_parser(parser) 
    args = parser.parse_args() 
    main_worker(args) 

def test(dataloader, model, args, file_names, logger, postprocess, writer=None, optflow_model=None, multi_tracker=None):
    if args.add_optflow_inputs: 
        assert optflow_model is not None, "Optical flow model should be provided"
        optflow_model.eval()
    logging.info(f'Testing/Infering on {args.dataset} dataset')
    batch_time = AverageMeter(' Forward Time', ':2.2f')
    data_time = AverageMeter(' Data Time', ':2.2f') 
    progress_meter_list = [batch_time, data_time] 
    progress_meter_list = add_metrics_meters(progress_meter_list, args.metric_fns, args.num_classes) 
    progress = ProgressMeter(len(dataloader), progress_meter_list, prefix='Test: ')
    model.eval()
    data_time_start = time.time()
    step = 0 
    all_pres_gt = []; all_pres = []


    # collect predicted points
    pred_db = {}

    with torch.no_grad():
        for sample in dataloader: 
            data_time.update(time.time() - data_time_start)
            batch_time_start = time.time() 
            if torch.cuda.is_available():
                input = [sample['input'][i].cuda(non_blocking=True)
                     for i in range(len(sample['input']))]
                mask = sample['mask'].type(torch.LongTensor).cuda(non_blocking=True)
                if args.add_depth_inputs:
                    input_depth = [sample['input_depth'][i].cuda(non_blocking=True) for i in range(len(sample['input_depth']))]
            else: 
                mask = sample['mask'].type(torch.LongTensor)
            mask = mask.squeeze(1)
            if args.add_optflow_inputs:
                optflow = [] 
                frame0 = F.interpolate(input[0], scale_factor=1.0, mode='nearest')
                if args.optflow_model == 'FlowFormerPlusPlus':
                    frame0 = frame0 * 0.225 / 0.5 # approximate scaling so as to match the input range of FlowFormerPlusPlus
                for i in range(1,len(input)):
                    frame = F.interpolate(input[i], scale_factor=1.0, mode='nearest')
                    if args.optflow_model == 'FlowFormerPlusPlus':
                        frame = frame * 0.225 / 0.5 # approximate scaling so as to match the input range of FlowFormerPlusPlus
                    if 'Basic' in args.model_type:
                        flow = optflow_model(frame, frame0)[-1]
                    else:
                        flow = optflow_model(frame0, frame)[-1]
                    flow = F.interpolate(flow/1.0, size=(input[0].size(2), input[0].size(3)), mode='bilinear', align_corners=True)
                    optflow.append(flow)
                if args.add_depth_inputs: 
                    output = model(input, optflow=optflow, depth=input_depth)
                else:
                    output = model(input, optflow=optflow)
            elif args.add_depth_inputs: 
                output = model(input, depth=input_depth)
            else: 
                output = model(input)

            # ===== DETR DECODE =====
            # model forward returns (outputs_dict, hs)
            outputs, _ = output

            # outputs["pred_logits"]: (B,Q,3)
            B = outputs["pred_logits"].shape[0]
            H = args.input_height
            W = args.input_width

            pred_items_batch = postprocess(outputs)   # list length B, each is list of dicts

            VIS_IDX = int(getattr(args, "target_pos_from_start", 4)) - 1

            for b in range(B):
                ori_img = postprocess_image(
                    input[VIS_IDX][b, :3, :, :].cpu().numpy()
                )

                img_path = Path(sample['center_path'][b])
                rel_img_path = os.path.relpath(img_path, str(args.data_dir))
                frame_name = img_path.stem

                items = pred_items_batch[b]  # [{'label','x','y',...}, ...]

                # split for visualization function
                tip_list, anchor_list = [], []
                for it in items:
                    x, y = float(it["x"]), float(it["y"])
                    s = float(it.get("score", 1.0))
                    if it["label"] == "tool_tip":
                        tip_list.append([x, y, s])
                    elif it["label"] == "tool_anchor":
                        anchor_list.append([x, y, s])

                # ---- save per-frame json (recommended, matches your GT style) ----
                # put outputs under same video root: .../pred_points_detr/<frame>.json
                rel_path_obj = Path(rel_img_path)
                parts = rel_path_obj.parts
                if "images" in parts:
                    idx = parts.index("images")
                    video_rel = Path(*parts[:idx])
                else:
                    video_rel = rel_path_obj.parent

             
                if args.dataset == "ACT":
                    base_pred_root = Path(args.data_dir).parent / "act_test_multiframe_detr"   
                else:
                    base_pred_root = Path(args.data_dir).parent / "kpt_test_multiframe_detr_raw"

             
                json_dir = base_pred_root / video_rel
                json_dir.mkdir(parents=True, exist_ok=True)
                json_path = json_dir / "pred_points.json"

                if json_path.exists():
                    with open(json_path, "r") as f:
                        video_pred = json.load(f)
                else:
                    video_pred = {}

             
                video_pred[rel_img_path] = {
                    "W_pred": int(args.input_width),
                    "H_pred": int(args.input_height),
                    "tip": tip_list,
                    "anchor": anchor_list
                }

                with open(json_path, "w") as f:
                    json.dump(video_pred, f, indent=2)

                # ---- save overlay points png (reuse your existing function) ----
                keypoints_dict = {
                    "tip": tip_list,
                    "anchor": anchor_list
                }
                save_points_prediction(args, img_path, keypoints_dict, ori_img)
                    
    return 

def main_worker(args): 
    args.mode = 'testing'
    args.data_dir = Path(args.data_dir)
    args.log_dir = Path(os.path.join(args.expt_savedir, args.expt_name, 'logs'))
    args.output_dir = Path(os.path.join(args.expt_savedir, args.expt_name, 'outputs'))
    for dir in [args.log_dir, args.output_dir]:
        if not dir.is_dir():
            print(f"Creating {dir.resolve()} if non-existent")
            dir.mkdir(parents=True, exist_ok=True) 
    
    # set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(os.path.join(args.log_dir, "log.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Set seed
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Seed set to {seed}")

    logger.info(f"Checking dataset under {args.data_dir}")
    # get test dataloader
    if  args.dataset=='ACT':
        test_file_names, _ = get_ACT_dataset_filenames(args)
    elif args.dataset=='KPT':
        test_file_names, _ = get_KPT_dataset_filenames(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    # print(test_file_names)
    _, test_dataloader = get_data_loader(args)
    logger.info(f"Test dataloader: {test_dataloader}")
    if test_dataloader is not None:
        logger.info(f"Test dataset size: {len(test_dataloader.dataset)}")
    else:
        logger.warning("Test dataloader is None!")

    # set up optical flow model if needed
    if args.add_optflow_inputs:
        if args.optflow_model=='RAFT':
            from torchvision.models.optical_flow import raft_large
            optflow_model = raft_large(pretrained=True, progress=False)
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 1:
                    optflow_model = nn.DataParallel(optflow_model)
                optflow_model = optflow_model.cuda()
        elif args.optflow_model=='FlowFormerPlusPlus':
            sys.path.append('./models/optical_flow/flowformerplusplus')
            sys.path.append('./models/optical_flow/flowformerplusplus/PerCostFormer3')
            from models.optical_flow.flowformerplusplus.ffpp_cfg_things import get_cfg
            from models.optical_flow.flowformerplusplus import build_flowformer
            cfg = get_cfg()
            optflow_model = build_flowformer(cfg)
            state_dict = torch.load('./models/optical_flow/flowformerplusplus/ckpts/ffpp_things.pth')
            new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
            optflow_model.load_state_dict(new_state_dict)
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 1:
                    optflow_model = nn.DataParallel(optflow_model)
                optflow_model = optflow_model.cuda()
        else: 
            raise SystemError('GPU device not found! Not configured to train/test.')
        optflow_model.eval()
        logger.info(f"{args.optflow_model} optical flow model loaded")
    else: 
        optflow_model = None

    # set up model 
    model = get_model(args)
    if torch.cuda.is_available():
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
        model = model.cuda(); cudnn.benchmark = True
    else: 
        raise SystemError('GPU device not found! Not configured to train/test.')
    # load pre-trained weights if needed
    model, _, load_flag = load_model_weights(model, args.load_wts_model, args.model_type)
    if load_flag:
        logger.info("Model weights loaded from {}".format(args.load_wts_model))
    else: 
        logger.info("No model weights loaded")
    
    # set up postprocess module
    postprocess = PostProcessPointsFixed(
        H=args.input_height,
        W=args.input_width,
        vis_thresh=getattr(args, "vis_thresh", 0.5),
        cls_thresh=getattr(args, "cls_thresh", 0.0),
        no_object_id=2
    )
    test(test_dataloader, model, args, test_file_names, logger,
     postprocess=postprocess, writer=None, optflow_model=optflow_model, multi_tracker=None)
    return 


if __name__ == '__main__':
    main()
