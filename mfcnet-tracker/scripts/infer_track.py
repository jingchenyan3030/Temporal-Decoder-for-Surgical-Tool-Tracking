# written by Chenyan, revised for overlapping-window query tracking inference

import os
import sys
import re
import json
import time
import random
import logging
from pathlib import Path

sys.path.append('.')
sys.path.append('./models/')

import configargparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from configs.config_detr import test_config_parser as config_parser
from src.dataloader_multiframe import get_data_loader
from models import get_multiframe_segmentation_model as get_model
from utils.dataloader_utils import get_ACT_dataset_filenames, get_KPT_dataset_filenames
from utils.log_utils import AverageMeter, ProgressMeter
from utils.model_utils import load_model_weights


def postprocess_image(tensor_img):
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    ori_img = np.clip(tensor_img * std + mean, 0, 1)
    ori_img = (np.transpose(ori_img, (1, 2, 0)) * 255).astype(np.uint8)
    return ori_img


def get_video_key(center_path, data_dir):
    img_path = Path(center_path)
    rel_img_path = os.path.relpath(img_path, str(data_dir))
    parts = Path(rel_img_path).parts

    if "images" in parts:
        idx = parts.index("images")
        video_key = str(Path(*parts[:idx]))
    else:
        video_key = str(Path(rel_img_path).parent)
    return video_key


def parse_frame_idx_from_path(img_path):
    stem = Path(img_path).stem
    nums = re.findall(r'\d+', stem)
    if len(nums) == 0:
        return None
    return int(nums[-1])


def get_target_frame_id(sample):
    """
    Try to get the target-frame id used for propagation.
    Priority:
      1) sample['frame_id'] if provided
      2) parse from sample['center_path'][0]
    IMPORTANT:
      This frame_id must correspond to the actual supervised target frame.
    """
    if 'frame_id' in sample:
        x = sample['frame_id']
        if torch.is_tensor(x):
            if x.numel() == 1:
                return int(x.item())
            return int(x[0].item())
        if isinstance(x, (list, tuple)):
            return int(x[0])
        return int(x)

    if 'center_path' in sample:
        return parse_frame_idx_from_path(sample['center_path'][0])

    return None


def is_temporally_continuous(prev_video_key, curr_video_key, prev_frame_id, curr_frame_id, stride=1):
    if prev_video_key is None:
        return False
    if curr_video_key != prev_video_key:
        return False
    if prev_frame_id is None or curr_frame_id is None:
        return False
    return curr_frame_id == prev_frame_id + stride


def get_json_save_root(args):
    if args.dataset == "ACT":
        return Path(args.data_dir).parent / "act_test_multiframe_query_track"
    else:
        return Path(args.data_dir).parent / "kpt_test_multiframe_query_track"


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
        out_dir = Path(args.data_dir).parent / "act_test_multiframe_query_track" / out_rel
    else:
        out_dir = Path(args.data_dir).parent / "kpt_test_multiframe_query_track" / out_rel

    out_dir.mkdir(parents=True, exist_ok=True)

    img_copy = input_rgb.copy()
    color_map = {
        "tip_main": (255, 255, 0),
        "tip_aux": (0, 255, 255),
        "anchor": (255, 0, 0),
    }

    for name, keypoints in keypoints_dict.items():
        color = color_map.get(name, (0, 255, 0))
        for (x, y, score) in keypoints:
            cv2.circle(
                img_copy,
                (int(round(x)), int(round(y))),
                radius=3,
                color=color,
                thickness=-1
            )

    out_file = out_dir / f"{p.stem}_points.png"
    cv2.imwrite(str(out_file), cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))


def save_prediction_json(
    args,
    sample,
    b,
    target_frame_id,
    track_step,
    current_video_key,
    tip_main_list,
    tip_aux_list,
    anchor_list,
    raw_queries
):
    img_path = Path(sample['center_path'][b])
    rel_img_path = os.path.relpath(img_path, str(args.data_dir))

    rel_path_obj = Path(rel_img_path)
    parts = rel_path_obj.parts
    if "images" in parts:
        idx = parts.index("images")
        video_rel = Path(*parts[:idx])
    else:
        video_rel = rel_path_obj.parent

    base_pred_root = get_json_save_root(args)
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
        "video_key": current_video_key,
        "target_frame_id": None if target_frame_id is None else int(target_frame_id),
        "track_step": int(track_step),
        "tip_main": tip_main_list,
        "tip_aux": tip_aux_list,
        "anchor": anchor_list,
        "raw_queries": raw_queries
    }

    with open(json_path, "w") as f:
        json.dump(video_pred, f, indent=2)


def decode_raw_queries(outputs, b):
    pred_points = outputs["pred_points"][b].detach().cpu().numpy()  # (Q,2)
    raw_queries = []

    pred_logits = None
    pred_vis = None

    if "pred_logits" in outputs:
        pred_logits = outputs["pred_logits"][b].detach().softmax(-1).cpu().numpy()
    if "pred_visibility" in outputs:
        pred_vis = outputs["pred_visibility"][b].detach().sigmoid().cpu().numpy()

    for q in range(pred_points.shape[0]):
        item = {
            "query_id": int(q),
            "x_norm": float(pred_points[q, 0]),
            "y_norm": float(pred_points[q, 1]),
        }
        if pred_logits is not None:
            item.update({
                "cls_prob_tip": float(pred_logits[q, 0]),
                "cls_prob_anchor": float(pred_logits[q, 1]),
                "cls_prob_noobj": float(pred_logits[q, 2]),
            })
        if pred_vis is not None:
            item["vis_prob"] = float(pred_vis[q])
        raw_queries.append(item)

    return raw_queries


def decode_fixed_queries(outputs, b, H, W):
    """
    Fixed semantic slots:
      q0 -> tip_main
      q1 -> tip_aux
      q2 -> anchor
    """
    pred_points = outputs["pred_points"][b].detach().cpu()  # (Q,2), normalized
    Q = pred_points.shape[0]
    assert Q >= 3, f"Expected at least 3 queries, got {Q}"

    def to_xy(pt):
        x = float(pt[0]) * W
        y = float(pt[1]) * H
        return [x, y, 1.0]

    tip_main = to_xy(pred_points[0])
    tip_aux = to_xy(pred_points[1])
    anchor = to_xy(pred_points[2])

    return {
        "tip_main": tip_main,
        "tip_aux": tip_aux,
        "anchor": anchor,
    }


def build_optflow_inputs(input_list, args, optflow_model):
    optflow = []
    frame0 = F.interpolate(input_list[0], scale_factor=1.0, mode='nearest')

    if args.optflow_model == 'FlowFormerPlusPlus':
        frame0 = frame0 * 0.225 / 0.5

    for i in range(1, len(input_list)):
        frame = F.interpolate(input_list[i], scale_factor=1.0, mode='nearest')
        if args.optflow_model == 'FlowFormerPlusPlus':
            frame = frame * 0.225 / 0.5

        if 'Basic' in args.model_type:
            flow = optflow_model(frame, frame0)[-1]
        else:
            flow = optflow_model(frame0, frame)[-1]

        flow = F.interpolate(
            flow,
            size=(input_list[0].size(2), input_list[0].size(3)),
            mode='bilinear',
            align_corners=True
        )
        optflow.append(flow)

    return optflow


def forward_with_tracking_state(model, input_list, args, prev_hs, optflow_model=None, input_depth=None):
    if args.add_optflow_inputs:
        assert optflow_model is not None, "Optical flow model should be provided"
        optflow = build_optflow_inputs(input_list, args, optflow_model)

        if args.add_depth_inputs:
            outputs, hs = model(input_list, optflow=optflow, depth=input_depth, prev_hs=prev_hs)
        else:
            outputs, hs = model(input_list, optflow=optflow, prev_hs=prev_hs)

    elif args.add_depth_inputs:
        outputs, hs = model(input_list, depth=input_depth, prev_hs=prev_hs)

    else:
        outputs, hs = model(input_list, prev_hs=prev_hs)

    return outputs, hs


def test(dataloader, model, args, logger, optflow_model=None):
    if args.add_optflow_inputs:
        assert optflow_model is not None, "Optical flow model should be provided"
        optflow_model.eval()

    logger.info(f'Testing on {args.dataset} with overlapping-window query propagation')

    batch_time = AverageMeter('Forward Time', ':2.2f')
    data_time = AverageMeter('Data Time', ':2.2f')
    progress = ProgressMeter(len(dataloader), [batch_time, data_time], prefix='Test: ')

    model.eval()
    data_time_start = time.time()
    step = 0

    prev_hs = None
    prev_video_key = None
    prev_target_frame_id = None
    track_step = -1

    with torch.no_grad():
        for sample in dataloader:
            data_time.update(time.time() - data_time_start)
            batch_time_start = time.time()

            assert len(sample['center_path']) == 1, \
                "Tracking inference currently expects batch_size=1."

            if not torch.cuda.is_available():
                raise SystemError('GPU device not found!')

            input_list = [
                sample['input'][i].cuda(non_blocking=True)
                for i in range(len(sample['input']))
            ]

            if args.add_depth_inputs:
                input_depth = [
                    sample['input_depth'][i].cuda(non_blocking=True)
                    for i in range(len(sample['input_depth']))
                ]
            else:
                input_depth = None

            current_video_key = get_video_key(sample['center_path'][0], args.data_dir)
            current_target_frame_id = get_target_frame_id(sample)

            continuous = is_temporally_continuous(
                prev_video_key=prev_video_key,
                curr_video_key=current_video_key,
                prev_frame_id=prev_target_frame_id,
                curr_frame_id=current_target_frame_id,
                stride=1
            )

            if not continuous:
                logger.info(
                    f"[TRACK RESET] video={current_video_key}, "
                    f"prev_target={prev_target_frame_id}, curr_target={current_target_frame_id}"
                )
                prev_hs = None
                track_step = 0
            else:
                track_step += 1

            outputs, hs = forward_with_tracking_state(
                model=model,
                input_list=input_list,
                args=args,
                prev_hs=prev_hs,
                optflow_model=optflow_model,
                input_depth=input_depth
            )

            # visualize target frame image
            VIS_IDX = int(getattr(args, "target_pos_from_start", 4)) - 1
            B = outputs["pred_points"].shape[0]

            for b in range(B):
                ori_img = postprocess_image(
                    input_list[VIS_IDX][b, :3, :, :].cpu().numpy()
                )

                decoded = decode_fixed_queries(
                    outputs=outputs,
                    b=b,
                    H=args.input_height,
                    W=args.input_width
                )

                tip_main_list = [decoded["tip_main"]]
                tip_aux_list = [decoded["tip_aux"]]
                anchor_list = [decoded["anchor"]]

                raw_queries = decode_raw_queries(outputs, b)

                save_prediction_json(
                    args=args,
                    sample=sample,
                    b=b,
                    target_frame_id=current_target_frame_id,
                    track_step=track_step,
                    current_video_key=current_video_key,
                    tip_main_list=tip_main_list,
                    tip_aux_list=tip_aux_list,
                    anchor_list=anchor_list,
                    raw_queries=raw_queries
                )

                keypoints_dict = {
                    "tip_main": tip_main_list,
                    "tip_aux": tip_aux_list,
                    "anchor": anchor_list,
                }
                save_points_prediction(args, Path(sample['center_path'][b]), keypoints_dict, ori_img)

            prev_hs = hs.detach()
            prev_video_key = current_video_key
            prev_target_frame_id = current_target_frame_id

            batch_time.update(time.time() - batch_time_start)
            data_time_start = time.time()
            step += 1

            if step % args.print_freq == 0:
                progress.display(step, logger=logger)


def main():
    parser = configargparse.ArgumentParser()
    parser = config_parser(parser)
    args = parser.parse_args()
    main_worker(args)


def main_worker(args):
    args.mode = 'testing'
    args.data_dir = Path(args.data_dir)

    args.log_dir = Path(os.path.join(args.expt_savedir, args.expt_name, 'logs'))
    args.output_dir = Path(os.path.join(args.expt_savedir, args.expt_name, 'outputs'))

    for dir_ in [args.log_dir, args.output_dir]:
        if not dir_.is_dir():
            dir_.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(os.path.join(args.log_dir, "log.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Seed set to {seed}")

    if args.dataset == 'ACT':
        _ = get_ACT_dataset_filenames(args)
    elif args.dataset == 'KPT':
        _ = get_KPT_dataset_filenames(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    _, test_dataloader = get_data_loader(args)
    logger.info(f"Test dataset size: {len(test_dataloader.dataset)}")

    if args.add_optflow_inputs:
        if args.optflow_model == 'RAFT':
            from torchvision.models.optical_flow import raft_large
            optflow_model = raft_large(pretrained=True, progress=False)
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 1:
                    optflow_model = nn.DataParallel(optflow_model)
                optflow_model = optflow_model.cuda()

        elif args.optflow_model == 'FlowFormerPlusPlus':
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
            raise ValueError(f"Unknown optical flow model: {args.optflow_model}")

        optflow_model.eval()
        logger.info(f"{args.optflow_model} optical flow model loaded")
    else:
        optflow_model = None

    model = get_model(args)
    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True
    else:
        raise SystemError('GPU device not found!')

    model, _, load_flag = load_model_weights(model, args.load_wts_model, args.model_type)
    if load_flag:
        logger.info(f"Model weights loaded from {args.load_wts_model}")
    else:
        logger.info("No model weights loaded")

    test(
        test_dataloader,
        model,
        args,
        logger,
        optflow_model=optflow_model
    )


if __name__ == '__main__':
    import sys
    main()