'''
This code is not available for the pipelines.
'''
import time, math, random, os, sys
import torch.distributed as dist
import logging
import torch 
import torch.nn.functional as F
from src.loss_aux import get_loss 
from src.metrics import get_metrics 
sys.path.append('../utils/')
from utils.log_utils import AverageMeter, ProgressMeter 
from utils.train_utils import add_loss_meters, add_metrics_meters 


def is_rank0():
    return (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)



def train_one_epoch(dataloader, epoch, model, optimizer, args, logger, writer=None, optflow_model=None):
    if args.add_optflow_inputs:
        assert optflow_model is not None, "Optical flow model should be provided"
        optflow_model.eval()
    batch_time = AverageMeter('Time', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    progress_meter_list = [batch_time, data_time] 
    total_loss = AverageMeter('Total Loss', ':.3f')
    progress_meter_list.append(total_loss) 
    progress_meter_list = add_loss_meters(progress_meter_list, args.loss_fns)
    progress = ProgressMeter(len(dataloader), progress_meter_list, prefix=f"Epoch: [{epoch}]")
    m = model.module if hasattr(model, "module") else model
    if args.train_base_model:
        m.train()
    else:
        m.base_model.eval()
        m.multiframe_net.train()
    data_time_start = time.time()
    step = 0 
    for sample in dataloader: 
        data_time.update(time.time() - data_time_start)
        batch_time_start = time.time() 
        if torch.cuda.is_available():
            input = [sample['input'][i].float().cuda(non_blocking=True) for i in range(len(sample['input']))]
            # Different task:
            if args.prediction_task == 'keypoint_heatmap':
                coarse = [sample['coarse'][i].float().cuda(non_blocking=True) for i in range(len(sample['coarse']))]
                mask = sample['mask'].float().cuda(non_blocking=True)
                sam = sample.get('sam', None)
                sam_input = None
                aux_mask_target = None
                if sam is not None:
                    sam = [sam[i].float().cuda(non_blocking=True) for i in range(len(sam))]

                    target_frame_idx = args.target_pos_from_start - 1
                    aux_mask_target = sam[target_frame_idx]

                    sam_input = [x.clone() for x in sam]
                    if not args.use_mask:
                        sam_input = [torch.zeros_like(x) for x in sam]

                    # dropout only on input mask, not GT
                    if getattr(args, 'use_sam_dropout', False):
                        keep_prob = getattr(args, 'sam_keep_prob', 0.8)
                        for i in range(len(sam_input)):
                            keep_mask = (
                                torch.rand(sam_input[i].size(0), 1, 1, 1, device=sam_input[i].device) < keep_prob
                            ).float()
                            sam_input[i] = sam_input[i] * keep_mask

                heatmap_valid = sample.get('heatmap_valid', None)
                if heatmap_valid is not None:
                    heatmap_valid = heatmap_valid.float().cuda(non_blocking=True)
            else:
                mask = sample['mask'].long().cuda(non_blocking=True)

            if args.add_depth_inputs: 
                input_depth = [sample['input_depth'][i].float().cuda(non_blocking=True) for i in range(len(sample['input_depth']))]
        else:
            input = [sample['input'][i].float() for i in range(len(sample['input']))]
            if args.prediction_task == 'keypoint_heatmap':
                coarse = [sample['coarse'][i].float() for i in range(len(sample['coarse']))]
                mask = sample['mask'].float()
                sam = sample.get('sam', None)
                sam_input = None
                aux_mask_target = None
                if sam is not None:
                    sam = [sam[i].float() for i in range(len(sam))]

                    target_frame_idx = args.target_pos_from_start - 1
                    aux_mask_target = sam[target_frame_idx]   # [B,1,H,W]

                    sam_input = [x.clone() for x in sam]

                    if not args.use_mask:
                        sam_input = [torch.zeros_like(x) for x in sam_input]

                    if getattr(args, 'use_sam_dropout', False):
                        keep_prob = getattr(args, 'sam_keep_prob', 0.8)
                        for i in range(len(sam_input)):
                            keep_mask = (
                                torch.rand(sam_input[i].size(0), 1, 1, 1, device=sam_input[i].device) < keep_prob
                            ).float()
                            sam_input[i] = sam_input[i] * keep_mask

                heatmap_valid = sample.get('heatmap_valid', None)
                if heatmap_valid is not None:
                    heatmap_valid = heatmap_valid.float()
            else:
                mask = sample['mask'].type(torch.LongTensor)

            if args.add_depth_inputs:
                input_depth = [sample['input_depth'][i].float() for i in range(len(sample['input_depth']))]
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

        # Forward pass
        if args.add_optflow_inputs:
            if args.add_depth_inputs:
                output = model(input, optflow=optflow, depth=input_depth, coarse=coarse, sam_masks=sam_input)
            else:
                output = model(input, optflow=optflow, coarse=coarse, sam_masks=sam_input)
        elif args.add_depth_inputs:
            output = model(input, depth=input_depth, coarse=coarse, sam_masks=sam_input)
        else: 
            output = model(input, coarse=coarse, sam_masks=sam_input)

        # Baseed on the task, compute loss
        if args.prediction_task == 'keypoint_heatmap' and 'mse' in args.loss_fns:
            # For keypoint heatmap prediction with MSE loss, do not apply log_softmax
            loss, loss_dict = get_loss(output, mask, args.loss_fns, args.loss_wts, args, heatmap_valid=heatmap_valid, aux_mask_target=aux_mask_target, aux_mask_weight=args.aux_mask_weight)
        else:
            output = F.log_softmax(output, dim=1)
            loss, loss_dict = get_loss(output, mask, args.loss_fns, args.loss_wts, args)
        if math.isnan(loss.item()) or math.isinf(loss.item()): 
            logger.debug(f"Loss is {loss.item()}. Exiting...")
            import pdb; pdb.set_trace()
        loss.backward()
        optimizer.step()
        bs = input[0].size(0)
        total_loss.update(loss.item(), bs)
        # display progress
        batch_time.update(time.time() - batch_time_start)
        if args.prediction_task != 'detr_keypoint':
            for i, loss_fn in enumerate(args.loss_fns):
                progress_meter_list[i+3].update(loss_dict['loss_'+loss_fn], bs)
        else:
            pass
        if step % args.print_freq == 0:
            progress.display(step, logger=logger)
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
            pass
    return model, total_loss.avg

def validate(dataloader, model, args, logger, writer=None, epoch=None, optflow_model=None):
    if args.add_optflow_inputs:
        assert optflow_model is not None, "Optical flow model should be provided"
        optflow_model.eval()
    batch_time = AverageMeter(' Forward Time', ':2.2f')
    data_time = AverageMeter(' Data Time', ':2.2f')
    progress_meter_list = [batch_time, data_time]
    total_loss = AverageMeter('Total Loss', ':.3f')
    progress_meter_list.append(total_loss)
    progress_meter_list = add_loss_meters(progress_meter_list, args.loss_fns)
    progress_meter_list = add_metrics_meters(progress_meter_list, args.metric_fns, args.num_classes)
    progress = ProgressMeter(len(dataloader), progress_meter_list, prefix='Epoch: [{}]'.format(epoch))
    model.eval()
    data_time_start = time.time() 
    step = 0
    N = len(args.loss_fns) 
    compute_metrics = (args.prediction_task not in ['keypoint_heatmap','detr_keypoint']) and (len(args.metric_fns) > 0)
    with torch.no_grad(): 
        for sample in dataloader:
            if torch.cuda.is_available():
                batch_time_start = time.time() 
                input = [sample['input'][i].float().cuda(non_blocking=True) for i in range(len(sample['input']))]
                coarse = [sample['coarse'][i].float().cuda(non_blocking=True) for i in range(len(sample['coarse']))]
                mask = sample['mask'].float().cuda(non_blocking=True)
                sam = sample.get('sam', None)
                aux_mask_target = None
                sam_input = None

                if sam is not None:
                    sam = [sam[i].float().cuda(non_blocking=True) for i in range(len(sam))]

                    target_frame_idx = args.target_pos_from_start - 1
                    aux_mask_target = sam[target_frame_idx]

                    sam_input = [x.clone() for x in sam]

                    if not args.use_mask:
                        sam_input = [torch.zeros_like(x) for x in sam_input]

                heatmap_valid = sample.get('heatmap_valid', None)
                if heatmap_valid is not None:
                    heatmap_valid = heatmap_valid.float().cuda(non_blocking=True)

                if args.add_depth_inputs:
                    input_depth = [sample['input_depth'][i].float().cuda(non_blocking=True) for i in range(len(sample['input_depth']))]
            else:
                batch_time_start = time.time()
                input = [sample['input'][i].float() for i in range(len(sample['input']))]
                coarse = [sample['coarse'][i].float() for i in range(len(sample['coarse']))]
                mask = sample['mask'].float()

                sam = sample.get('sam', None)
                sam_input = None
                aux_mask_target = None

                if sam is not None:
                    sam = [sam[i].float() for i in range(len(sam))]

                    target_frame_idx = args.target_pos_from_start - 1
                    aux_mask_target = sam[target_frame_idx]

                    sam_input = [x.clone() for x in sam]

                    if not args.use_mask:
                        sam_input = [torch.zeros_like(x) for x in sam_input]

                heatmap_valid = sample.get('heatmap_valid', None)
                if heatmap_valid is not None:
                    heatmap_valid = heatmap_valid.float()

                if args.add_depth_inputs:
                    input_depth = [sample['input_depth'][i].float() for i in range(len(sample['input_depth']))]

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

            # Forward pass    
            if args.add_optflow_inputs:
                if args.add_depth_inputs:
                    output = model(input, optflow=optflow, depth=input_depth, coarse=coarse, sam_masks=sam_input)
                else:
                    output = model(input, optflow=optflow, coarse=coarse, sam_masks=sam_input)
            elif args.add_depth_inputs:
                output = model(input, depth=input_depth, coarse=coarse, sam_masks=sam_input)
            else:
                output = model(input, coarse=coarse, sam_masks=sam_input)
            # Based on the task, compute loss
            if args.prediction_task == 'keypoint_heatmap' and 'mse' in args.loss_fns:
                # For keypoint heatmap prediction with MSE loss, do not
                loss, loss_dict = get_loss(output, mask, args.loss_fns, args.loss_wts, args, heatmap_valid=heatmap_valid, aux_mask_target=aux_mask_target, aux_mask_weight=args.aux_mask_weight)
            else:
                output = F.log_softmax(output, dim=1)
                loss, loss_dict = get_loss(output, mask, args.loss_fns, args.loss_wts, args)
                metrics, metric_dict = get_metrics(output, mask, args.metric_fns, args)
            if math.isnan(loss.item()) or math.isinf(loss.item()):
                logger.debug(f"Loss is NaN/Inf."); import pdb; pdb.set_trace()
            batch_time.update(time.time() - batch_time_start)
            bs = input[0].size(0)
            total_loss.update(loss.item(), bs)
            if args.prediction_task != 'detr_keypoint':
                for i, loss_fn in enumerate(args.loss_fns):
                    progress_meter_list[i+3].update(loss_dict['loss_'+loss_fn], bs)
                if compute_metrics:
                    idx = 0
                    for i, metric_fn in enumerate(args.metric_fns):
                        for cls in range(1, args.num_classes):
                            progress_meter_list[N+3+idx].update(metrics[i][cls-1], bs)
                            idx += 1
                    # progress_meter_list[N+3+i].update(metric_dict['metric_'+metric_fn], bs)
            if step % args.print_freq == 0:
                progress.display(step, logger=logger)
            step += 1
            data_time_start = time.time()
    if writer is not None and is_rank0():
        writer.add_scalar('Validation/Loss', total_loss.avg, epoch)
        logger.info(f"Validation loss: {total_loss.avg}")
        if args.prediction_task != 'detr_keypoint':
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
