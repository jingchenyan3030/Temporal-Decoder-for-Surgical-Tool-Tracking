"""
Script for running inference for mse heatmap models 
written by Chenyan 
# Use regression to train heatmap for each keypoint type
"""

import os 
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
sys.path.append('.')
sys.path.append('./models/')
import logging, json, random, time, math 
from pathlib import Path

import configargparse
from configs.config_refine_mid import test_config_parser as config_parser

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
from utils.dataloader_utils import get_MICCAI2017_dataset_filenames, get_JIGSAWS_dataset_filenames, get_MICCAI2015_dataset_filenames, get_ACT_dataset_filenames, get_KPT_dataset_filenames, get_all_image_files
from utils.log_utils import AverageMeter, ProgressMeter
from utils.model_utils import load_model_weights
from utils.train_utils import add_metrics_meters
from utils.vis_utils import mask_overlay, draw_plus
from utils.localization_utils_v2 import centroid_error_10_classes, centroid_error_4_classes
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from src.HeatmapParser import HeatmapParser

def postprocess_image(tensor_img):
    mean = np.array([0.485, 0.456, 0.406]).reshape((3,1,1))
    std  = np.array([0.229, 0.224, 0.225]).reshape((3,1,1))
    ori_img = np.clip(tensor_img * std + mean, 0, 1)
    ori_img = (np.transpose(ori_img, (1,2,0)) * 255).astype(np.uint8)
    return ori_img

def save_points_overlay(video_dir, frame_name, tip_list, anchor_list, input_rgb):
    """
    Save predicted tip/anchor points overlaid on the resized network input image.
    """
    overlay_dir = video_dir / "pred_overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    img = input_rgb.copy()

    for pt in tip_list:
        if len(pt) < 2:
            continue
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(img, (x, y), radius=4, color=(255, 255, 0), thickness=-1)  # RGB yellow

    for pt in anchor_list:
        if len(pt) < 2:
            continue
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(img, (x, y), radius=4, color=(255, 0, 0), thickness=-1)    # RGB red

    out_path = overlay_dir / f"{frame_name}_points.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"[OK] saved overlay: {out_path}")


def main(): 
    parser = configargparse.ArgumentParser() 
    parser = config_parser(parser) 
    args = parser.parse_args() 
    main_worker(args) 

def save_points_json(video_dir, frame_name, tip_list, anchor_list, H, W):
    json_path = video_dir / "pred_points.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            video_pred = json.load(f)
    else:
        video_pred = {}
    video_pred[frame_name] = {
        "W_pred": int(W),
        "H_pred": int(H),
        "tip": tip_list,
        "anchor": anchor_list,
    }
    with open(json_path, "w") as f:
        json.dump(video_pred, f, indent=2)
    print(f"[OK] saved json: {json_path} | frame={frame_name}")

def test(dataloader, model, args, file_names, logger, heatmap_parser , writer=None, optflow_model=None, multi_tracker = None): 
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

            # MSE HEATMAP DECODE:
            output_heatmaps = output.detach().cpu().numpy()  # [B, C, H, W]
            B, C, H, W = output_heatmaps.shape

            end = min(step+B, len(file_names))
            results = heatmap_parser.parse(output.detach())
            for b in range(B):
                VIS_IDX = int(getattr(args, "target_pos_from_start", 4)) - 1
                ori_img = postprocess_image(
                    input[VIS_IDX][b, :3, :, :].cpu().numpy()
                )
                hm = output_heatmaps[b].astype(np.float32)   
                img_path = Path(sample['center_path'][b])
                frame_name = img_path.stem
                video_dir = img_path.parent.parent
                # ===== save heatmaps only =====
                heatmap_coarse_dir = video_dir / "heatmap_coarse"
                heatmap_coarse_dir.mkdir(parents=True, exist_ok=True)

                tip_map = hm[0]         # [H, W]
                anchor_map = hm[1]      # [H, W]
                hm_hw2 = np.stack([tip_map, anchor_map], axis=-1)   # [H, W, 2]
                np.save(heatmap_coarse_dir / f"{frame_name}.npy", hm_hw2) 
                # ===== save predicted points =====
                tip_pts = results["tip"][b] if "tip" in results else []
                anchor_pts = results["anchor"][b] if "anchor" in results else []
                if tip_pts is None or len(tip_pts) == 0:
                    tip_list = []
                else:
                    tip_list = np.asarray(tip_pts, dtype=np.float32).tolist()
                if anchor_pts is None or len(anchor_pts) == 0:
                    anchor_list = []
                else:
                    anchor_list = np.asarray(anchor_pts, dtype=np.float32).tolist()
                save_points_json(
                    video_dir=video_dir,
                    frame_name=frame_name,
                    tip_list=tip_list,
                    anchor_list=anchor_list,
                    H=H,
                    W=W,
                )
                img_path = Path(sample['center_path'][b])
                case_name = img_path.parents[2].name
                if case_name != "case_001_video_part_001_segment_3":
                    continue
                save_points_overlay(
                    video_dir=video_dir,
                    frame_name=frame_name,
                    tip_list=tip_list,
                    anchor_list=anchor_list,
                    input_rgb=ori_img,
                )

            step = end
        
    return 

def main_worker(args): 
    args.mode = 'generate_coarse'
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
    if args.dataset == 'ACT':
        test_file_names, _ = get_ACT_dataset_filenames(args)
    elif args.dataset == 'KPT':
        if args.mode == 'generate_coarse':
            test_file_names = get_all_image_files(args)
        else:
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
    
    if args.dataset == 'ACT':
        if args.action == 'dissect':
            CHANNEL_CONFIGS = {
            0: {"name": "tip", "topk": 5, "max_keep": 1, "threshold": 0.1},
            1: {"name": "anchor", "topk": 5, "max_keep": 2, "threshold":0.1},
             }
        else:
            CHANNEL_CONFIGS = {
            0: {"name": "tip", "topk": 5, "max_keep": 4, "threshold": 0.05},
            1: {"name": "anchor", "topk": 5, "max_keep": 2, "threshold":0.05},
            }
    else: 
        if args.action == 'dissect':
            CHANNEL_CONFIGS = {
            0: {"name": "tip", "topk": 5, "max_keep": 1, "threshold": 0.1},
            1: {"name": "anchor", "topk": 5, "max_keep": 2, "threshold":0.1},
             }
        else:
            CHANNEL_CONFIGS = {
            0: {"name": "tip", "topk": 10, "max_keep": 2, "threshold": 0.1},
            1: {"name": "anchor", "topk": 5, "max_keep": 1, "threshold":0.15},
            }
    heatmap_parser  = HeatmapParser(CHANNEL_CONFIGS, nms_kernel=3, nms_padding=1)

    multi_tracker = None

    test(test_dataloader, model, args, test_file_names, logger, heatmap_parser = heatmap_parser , writer=None, optflow_model=optflow_model, multi_tracker=multi_tracker)
    return


if __name__ == '__main__':
    main()
