# batch_size must be 1
import time, math, random, os, sys
import torch.distributed as dist
import logging
import torch 
import torch.nn.functional as F
from src.loss import get_loss 
from src.detr_loss import compute_tracking_loss as get_compute_detr_loss
from src.metrics import get_metrics 
sys.path.append('../utils/')
from utils.log_utils import AverageMeter, ProgressMeter 
from utils.train_utils import add_loss_meters, add_metrics_meters 
# from models.match import HungarianMatcher

def is_rank0():
    return (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)



def train_one_epoch(dataloader, epoch, model, optimizer, args, logger, writer=None, optflow_model=None):
    m = model.module if hasattr(model,"module") else model
    if args.add_optflow_inputs:
        assert optflow_model is not None, "Optical flow model should be provided"
        optflow_model.eval()
    batch_time = AverageMeter('Time', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    progress_meter_list = [batch_time, data_time]
    total_loss = AverageMeter('Total Loss', ':.3f')

    if args.prediction_task == 'detr_keypoint':
        loss_vis_meter = AverageMeter('Loss_vis', ':.3f')
        loss_vel_meter = AverageMeter('Loss_vel', ':.3f')
        progress_meter_list += [total_loss, loss_vis_meter, loss_vel_meter]
    else:
        progress_meter_list += [total_loss]
        progress_meter_list = add_loss_meters(progress_meter_list, args.loss_fns)
    progress = ProgressMeter(len(dataloader), progress_meter_list, prefix=f"Epoch: [{epoch}]")
    if args.train_base_model:
        m.train()
    else:
        m.base_model.eval()
        m.multiframe_net.train()
    data_time_start = time.time()
    step = 0 

    prev_cache = {}
    # can delete
    cache_hit = 0
    cache_miss = 0
    cache_reset = 0
    #

    for sample in dataloader: 
        sam_weight = None
        data_time.update(time.time() - data_time_start)
        batch_time_start = time.time() 
        if torch.cuda.is_available():
            input = [sample['input'][i].cuda(non_blocking=True) for i in range(len(sample['input']))]

            # Different task:
            if args.prediction_task == 'keypoint_heatmap':
                mask = sample['mask'].float().cuda(non_blocking=True)
                if mask.dim() == 5:
                    mask = mask.squeeze(1)
                sam_weight = sample.get('sam_weight', None)
                if sam_weight is not None:
                    sam_weight = sam_weight.float().cuda(non_blocking=True)
            elif args.prediction_task == 'detr_keypoint':
                targets = []
                B = len(sample["points"])
                for b in range(B):
                    targets.append({
                        "points": sample["points"][b].cuda(non_blocking=True),
                        "labels": sample["labels"][b].cuda(non_blocking=True),
                        "visibility": sample["visibility"][b].cuda(non_blocking=True),  
                    })
            else:
                mask = sample['mask'].long().cuda(non_blocking=True)

            if args.add_depth_inputs: 
                input_depth = [sample['input_depth'][i].cuda(non_blocking=True) for i in range(len(sample['input_depth']))]
        else: 
            # Different task:
            if args.prediction_task == 'keypoint_heatmap':
                mask = sample['mask'].float()
            elif args.prediction_task == 'detr_keypoint':
                targets = []
                B = len(sample["points"])
                for b in range(B):
                    targets.append({
                        "points": sample["points"][b],
                        "labels": sample["labels"][b],
                        "visibility": sample["visibility"][b],
                    })

            else:
                mask = sample['mask'].type(torch.LongTensor)
        # Compute optical flow if required    
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
        optimizer.zero_grad()
        if args.prediction_task not in ['keypoint_heatmap', 'detr_keypoint']:
            mask = mask.squeeze(1)

        # ---- MOTR-style caching mechanism ----
        def _norm_id(x):
            if x is None:
                return None
            if isinstance(x, (str, int)):
                return x
            if torch.is_tensor(x):
                return int(x.item()) if x.numel() == 1 else None
            if isinstance(x, (list, tuple)):
                return _norm_id(x[0])
            return None

        prev_hs = None
        prev_pred_points = None
        prev_vis = None
        prev_gt_points = None
        video_id = frame_id = None

        if args.prediction_task == 'detr_keypoint':
            video_id = _norm_id(sample.get('video_id', None))
            frame_id = _norm_id(sample.get('frame_id', None))

            bs = input[0].size(0)
            if bs == 1 and (video_id is not None) and (frame_id is not None):
                last = prev_cache.get(video_id, None)
                if (last is not None) and (frame_id == last['frame_id'] + 1):
                    prev_hs = last["hs"]                     # (1,Q,D)
                    prev_pred_points = last.get("pred_points", None)  # (1,Q,2)
                    prev_vis = last.get("vis", None)                 # (1,Q)
                    prev_gt_points = last.get("gt_points", None)     # (1,Q,2) optional
                else:
                    prev_hs = None
                    prev_pred_points = None
                    prev_vis = None
                    prev_gt_points = None
            else:
                prev_hs = None

        # can delete
        if args.prediction_task == 'detr_keypoint' and bs == 1 and (video_id is not None) and (frame_id is not None):
            last = prev_cache.get(video_id, None)
            if prev_hs is not None:
                cache_hit += 1
            else:
                if last is None:
                    cache_miss += 1     
                else:
                    cache_reset += 1    

        # Forward pass
        if args.prediction_task == 'detr_keypoint':
            if args.add_optflow_inputs:
                if args.add_depth_inputs:
                    output,hs = model(input, optflow=optflow, depth=input_depth, prev_hs=prev_hs)
                else:
                    output,hs = model(input, optflow=optflow, prev_hs=prev_hs)
            elif args.add_depth_inputs:
                output,hs = model(input, depth=input_depth, prev_hs=prev_hs)
            else: 
                output,hs = model(input, prev_hs=prev_hs) 
        else:
            if args.add_optflow_inputs:
                if args.add_depth_inputs:
                    output = model(input, optflow=optflow, depth=input_depth)
                else:
                    output = model(input, optflow=optflow)
            elif args.add_depth_inputs:
                output = model(input, depth=input_depth)
            else: 
                output = model(input)


        # ---- MOTR-style caching mechanism ----
        if args.prediction_task == 'detr_keypoint':
            bs = input[0].size(0)
            if bs == 1 and (video_id is not None) and (frame_id is not None):
               prev_cache[video_id] = {
                    'frame_id': frame_id,
                    'hs': hs.detach(),
                    'pred_points': output["pred_points"].detach(),                         # (1,Q,2)
                    'vis': targets[0]["visibility"].unsqueeze(0).detach(),                 # (1,Q)
                    'gt_points': targets[0]["points"].unsqueeze(0).detach(),               # (1,Q,2) 
                }
        # Baseed on the task, compute loss
        if args.prediction_task == 'keypoint_heatmap' and 'mse' in args.loss_fns:
            # For keypoint heatmap prediction with MSE loss, do not apply log_softmax
            loss, loss_dict = get_loss(output, mask, args.loss_fns, args.loss_wts, args, sam_weight=sam_weight)
        elif args.prediction_task == 'detr_keypoint':
            pred_points_t = output["pred_points"]
            gt_points_t = targets[0]["points"]
            vis_t = targets[0]["visibility"]
            if gt_points_t.dim() == 2:
                gt_points_t = gt_points_t.unsqueeze(0)
                vis_t = vis_t.unsqueeze(0)
            
            L_t, L_vis_t, L_vel_t =  get_compute_detr_loss(
                        pred_points_t, gt_points_t, vis_t,
                        prev_pred_points=prev_pred_points,   # (1,Q,2) or None
                        prev_vis=prev_vis,                   # (1,Q) or None
                        prev_gt_points=prev_gt_points,       # (1,Q,2) or None
                        alpha=getattr(args, "alpha_vel", 0.01),
                        vel_mode=getattr(args, "vel_mode", "pred_smooth"))
            
            loss = L_t
            loss_dict = {
                "loss_total": L_t.detach().item(),
                "loss_vis": L_vis_t.detach().item(),
                "loss_vel": L_vel_t.detach().item(),
            }

        else:
            output = F.log_softmax(output, dim=1)
            loss, loss_dict = get_loss(output, mask, args.loss_fns, args.loss_wts, args)
        if math.isnan(loss.item()) or math.isinf(loss.item()): 
            logger.debug(f"Loss is {loss.item()}. Exiting...")
            import pdb; pdb.set_trace()
        loss.backward()
        optimizer.step()


        # display progress
        batch_time.update(time.time() - batch_time_start)
        bs = input[0].size(0) 
        total_loss.update(loss.item(), bs)
    
        if args.prediction_task == 'detr_keypoint':
            loss_vis_meter.update(loss_dict["loss_vis"], bs)
            loss_vel_meter.update(loss_dict["loss_vel"], bs)
        else:
            for i, loss_fn in enumerate(args.loss_fns):
                progress_meter_list[i+3].update(loss_dict['loss_'+loss_fn], bs)
        if step % args.print_freq == 0:
            progress.display(step, logger=logger)

        # can delete
        if args.prediction_task == 'detr_keypoint' and is_rank0() and (step % args.print_freq == 0):
            denom = cache_hit + cache_miss + cache_reset
            hit_rate = cache_hit / max(1, denom)
            logger.debug(
                f"[cache] step={step} hit={cache_hit} miss={cache_miss} reset={cache_reset} "
                f"hit_rate={hit_rate:.3f} vid={video_id} fid={frame_id}"
            )
        #
        step += 1
        data_time_start = time.time()

    if writer is not None and is_rank0():
        writer.add_scalar('Training/Loss', total_loss.avg, epoch)
        logger.info(f"Training loss: {total_loss.avg}")
        if args.prediction_task != 'detr_keypoint':
            for i, loss_fn in enumerate(args.loss_fns):
                writer.add_scalar(f'Training/Loss_{loss_fn}', progress_meter_list[i+3].avg, epoch)
                logger.info(f"Training loss {loss_fn}: {progress_meter_list[i+3].avg}")
        else:
            writer.add_scalar('Training/Loss_vis', loss_vis_meter.avg, epoch)
            writer.add_scalar('Training/Loss_vel', loss_vel_meter.avg, epoch)
            logger.info(f"Training loss_vis: {loss_vis_meter.avg}")
            logger.info(f"Training loss_vel: {loss_vel_meter.avg}")
    return model, total_loss.avg

def validate(dataloader, model, args, logger, writer=None, epoch=None, optflow_model=None):
    m = model.module if hasattr(model,"module") else model
    if args.add_optflow_inputs:
        assert optflow_model is not None, "Optical flow model should be provided"
        optflow_model.eval()
    batch_time = AverageMeter(' Forward Time', ':2.2f')
    data_time = AverageMeter(' Data Time', ':2.2f')
    progress_meter_list = [batch_time, data_time]
    total_loss = AverageMeter('Total Loss', ':.3f')
    if args.prediction_task == 'detr_keypoint':
        loss_vis_meter = AverageMeter('Loss_vis', ':.3f')
        loss_vel_meter = AverageMeter('Loss_vel', ':.3f')
        progress_meter_list += [total_loss, loss_vis_meter, loss_vel_meter]
    else:
        progress_meter_list += [total_loss]
        progress_meter_list = add_loss_meters(progress_meter_list, args.loss_fns)
        progress_meter_list = add_metrics_meters(progress_meter_list, args.metric_fns, args.num_classes)
    progress = ProgressMeter(len(dataloader), progress_meter_list, prefix='Epoch: [{}]'.format(epoch))
    m.eval()
    data_time_start = time.time() 
    step = 0
    N = len(args.loss_fns) 
    compute_metrics = (args.prediction_task not in ['keypoint_heatmap','detr_keypoint']) and (len(args.metric_fns) > 0)
    with torch.no_grad(): 
        for sample in dataloader: 
            sam_weight = None
            data_time.update(time.time() - data_time_start)
            batch_time_start = time.time() 
            if torch.cuda.is_available():
                input = [sample['input'][i].cuda(non_blocking=True) for i in range(len(sample['input']))]
                # Different task:
                if args.prediction_task == 'keypoint_heatmap':
                    mask = sample['mask'].float().cuda(non_blocking=True)
                    if mask.dim() == 5:
                        mask = mask.squeeze(1)
                    sam_weight = sample.get('sam_weight', None)
                    if sam_weight is not None:
                        sam_weight = sam_weight.float().cuda(non_blocking=True)
                elif args.prediction_task == 'detr_keypoint':
                    targets = []
                    B = len(sample["points"])
                    for b in range(B):
                        targets.append({
                            "points": sample["points"][b].cuda(non_blocking=True),
                            "labels": sample["labels"][b].cuda(non_blocking=True),
                            "visibility": sample["visibility"][b].cuda(non_blocking=True), 
                        })
                else:
                    mask = sample['mask'].long().cuda(non_blocking=True)

                if args.add_depth_inputs:
                    input_depth = [sample['input_depth'][i].cuda(non_blocking=True) for i in range(len(sample['input_depth']))]
            else: 
                # Different task:
                if args.prediction_task == 'keypoint_heatmap':
                    mask = sample['mask'].float()
                elif args.prediction_task == 'detr_keypoint':
                    targets = []
                    B = len(sample["points"])
                    for b in range(B):
                        targets.append({
                            "points": sample["points"][b],
                            "labels": sample["labels"][b],
                            "visibility": sample["visibility"][b],  
                        })
                else:
                    mask = sample['mask'].long()

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
            if args.prediction_task not in ['keypoint_heatmap', 'detr_keypoint']:
                mask = mask.squeeze(1)

            # Forward pass    
            if args.prediction_task == 'detr_keypoint':
                if args.add_optflow_inputs:
                    if args.add_depth_inputs:
                        output,hs = model(input, optflow=optflow, depth=input_depth, prev_hs=None)
                    else:
                        output,hs = model(input, optflow=optflow, prev_hs=None)
                elif args.add_depth_inputs:
                    output,hs = model(input, depth=input_depth, prev_hs=None)
                else: 
                    output,hs = model(input, prev_hs=None) 
            else:
                if args.add_optflow_inputs:
                    if args.add_depth_inputs:
                        output = model(input, optflow=optflow, depth=input_depth)
                    else:
                        output = model(input, optflow=optflow)
                elif args.add_depth_inputs:
                    output = model(input, depth=input_depth)
                else: 
                    output = model(input)

            # Based on the task, compute loss
            if args.prediction_task == 'keypoint_heatmap' and 'mse' in args.loss_fns:
                # For keypoint heatmap prediction with MSE loss
                loss, loss_dict = get_loss(output, mask, args.loss_fns, args.loss_wts, args, sam_weight=sam_weight)

            elif args.prediction_task == 'detr_keypoint':
                pred_points_t = output["pred_points"]
                gt_points_t = targets[0]["points"]
                vis_t = targets[0]["visibility"]

                if gt_points_t.dim() == 2:
                    gt_points_t = gt_points_t.unsqueeze(0)
                if vis_t.dim() == 1:
                    vis_t = vis_t.unsqueeze(0)

                L_t, L_vis_t, L_vel_t = get_compute_detr_loss(
                    pred_points_t, gt_points_t, vis_t,
                    prev_pred_points=None,
                    prev_vis=None,
                    prev_gt_points=None,
                    alpha=getattr(args, "alpha_vel", 0.01),
                    vel_mode=getattr(args, "vel_mode", "pred_smooth"),
                )
                loss = L_t
                loss_dict = {
                    "loss_total": L_t.detach().item(),
                    "loss_vis": L_vis_t.detach().item(),
                    "loss_vel": L_vel_t.detach().item(),
                }
            else:
                output = F.log_softmax(output, dim=1)
                loss, loss_dict = get_loss(output, mask, args.loss_fns, args.loss_wts, args)
                metrics, metric_dict = get_metrics(output, mask, args.metric_fns, args)
            if math.isnan(loss.item()) or math.isinf(loss.item()):
                logger.debug(f"Loss is NaN/Inf."); import pdb; pdb.set_trace()
            batch_time.update(time.time() - batch_time_start)
            bs = input[0].size(0) 
            total_loss.update(loss.item(), bs)
            if args.prediction_task == 'detr_keypoint':
                loss_vis_meter.update(loss_dict["loss_vis"], bs)
                loss_vel_meter.update(loss_dict["loss_vel"], bs)
            else:
                for i, loss_fn in enumerate(args.loss_fns):
                    progress_meter_list[i+3].update(loss_dict['loss_'+loss_fn], bs)
                if compute_metrics:
                    idx = 0
                    for i, metric_fn in enumerate(args.metric_fns):
                        for cls in range(1, args.num_classes):
                            progress_meter_list[N+3+idx].update(metrics[i][cls-1], bs)
                            idx += 1
            if step % args.print_freq == 0:
                progress.display(step, logger=logger)
            step += 1
            data_time_start = time.time()
    if writer is not None and is_rank0():
        writer.add_scalar('Validation/Loss', total_loss.avg, epoch)
        logger.info(f"Validation loss: {total_loss.avg}")
        if args.prediction_task == 'detr_keypoint':
            writer.add_scalar('Validation/Loss_vis', loss_vis_meter.avg, epoch)
            writer.add_scalar('Validation/Loss_vel', loss_vel_meter.avg, epoch)
            logger.info(f"Validation loss_vis: {loss_vis_meter.avg}")
            logger.info(f"Validation loss_vel: {loss_vel_meter.avg}")
        else:
            for i, loss_fn in enumerate(args.loss_fns):
                writer.add_scalar(f'Validation/Loss_{loss_fn}', progress_meter_list[i+3].avg, epoch)
                logger.info(f"Validation loss {loss_fn}: {progress_meter_list[i+3].avg}")
            if compute_metrics:
                idx = 0
                for i, metric_fn in enumerate(args.metric_fns):
                    for cls in range(1, args.num_classes):
                        writer.add_scalar(f'Validation/{metric_fn} {cls}', progress_meter_list[N+3+idx].avg, epoch)
                        logger.info(f"Validation metric {metric_fn} {cls}: {progress_meter_list[N+3+idx].avg}")
                        idx += 1
    return total_loss.avg
