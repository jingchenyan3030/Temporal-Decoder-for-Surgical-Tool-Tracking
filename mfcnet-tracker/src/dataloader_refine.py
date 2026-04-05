import cv2 
import torch 
import numpy as np 
import sys; sys.path.append('./')
from src.dataset_ACT import ACT
from src.dataset_KPT import KPT
from src.dataset_KPT_copy import KPT_test
from src.dataset_KPT_mid import KPT_test_mid
from src.dataset_KPT_track import KPT_track_mid
from src.dataset_KPT_a_test import KPT_test_a
from src.dataset_KPT_refine import KPT_refine
from torch.utils.data import DataLoader 
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import torchvision.transforms.functional as tF

from utils.dataloader_utils import  get_ACT_dataset_filenames, get_KPT_dataset_filenames , get_KPT_refine_dataset_filenames, get_all_image_files

class to_tensor(object): 
    def __call__(self, sample): 
        input = sample['input'] 
        mask = sample['mask'] 
        tensor_dict = {} 

        tensor_dict['input'] = []
        for image in input: 
            tensor_dict['input'].append(
                torch.from_numpy(image.transpose(2,0,1).astype(np.float32) / 255.0)
            )

        if mask.ndim == 2:
            tensor_dict['mask'] = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        else:
            tensor_dict['mask'] = torch.from_numpy(mask.transpose(2,0,1).astype(np.float32))

        if 'coarse' in sample and sample['coarse'] is not None:
            tensor_dict['coarse'] = []
            for coarse in sample['coarse']:
                tensor_dict['coarse'].append(
                    torch.from_numpy(coarse.transpose(2,0,1).astype(np.float32))
                )

        if 'sam' in sample and sample['sam'] is not None:
            tensor_dict['sam'] = []
            for sam in sample['sam']:
                tensor_dict['sam'].append(
                    torch.from_numpy(sam.transpose(2,0,1).astype(np.float32))
                )

        if 'input_depth' in sample: 
            input_depth = sample['input_depth']
            tensor_dict['input_depth'] = []
            for depth in input_depth: 
                if depth.ndim == 2:
                    tensor_dict['input_depth'].append(
                        torch.from_numpy(depth.astype(np.float32)).unsqueeze(0) / 255.0
                    )
                else:
                    tensor_dict['input_depth'].append(
                        torch.from_numpy(depth.transpose(2,0,1).astype(np.float32)) / 255.0
                    )

        if 'points' in sample:
            tensor_dict['points'] = sample['points']

        if 'labels' in sample and sample['labels'] is not None:
            tensor_dict['labels'] = sample['labels']

        if 'visibility' in sample:
            tensor_dict['visibility'] = sample['visibility']

        if 'sam_weight' in sample and sample['sam_weight'] is not None:
            tensor_dict['sam_weight'] = torch.from_numpy(sample['sam_weight'].astype(np.float32))

        if 'valid' in sample:
            tensor_dict['valid'] = torch.as_tensor(sample['valid'], dtype=torch.bool)

        if 'heatmap_valid' in sample:
            tensor_dict['heatmap_valid'] = torch.from_numpy(sample['heatmap_valid'].astype(np.float32))

        return tensor_dict
    
class customResize(object):
    def __init__(self, img_size):
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        elif isinstance(img_size, tuple):
            assert len(img_size) == 2
            self.img_size = img_size
        else:
            raise TypeError

    def __call__(self, sample):
        resized_dict = {}

        resized_dict['input'] = []
        for image in sample['input']:
            resized_dict['input'].append(
                transforms.Resize(
                    self.img_size,
                    interpolation=tF.InterpolationMode.BILINEAR
                )(image)
            )

        resized_dict['mask'] = transforms.Resize(
            self.img_size,
            interpolation=tF.InterpolationMode.BILINEAR
        )(sample['mask'])

        if 'coarse' in sample and sample['coarse'] is not None:
            resized_dict['coarse'] = []
            for coarse in sample['coarse']:
                resized_dict['coarse'].append(
                    transforms.Resize(
                        self.img_size,
                        interpolation=tF.InterpolationMode.BILINEAR
                    )(coarse)
                )

        if 'sam' in sample and sample['sam'] is not None:
            resized_dict['sam'] = []
            for sam in sample['sam']:
                resized_dict['sam'].append(
                    transforms.Resize(
                        self.img_size,
                        interpolation=tF.InterpolationMode.NEAREST
                    )(sam)
                )

        if 'input_depth' in sample:
            resized_dict['input_depth'] = []
            for depth in sample['input_depth']:
                resized_dict['input_depth'].append(
                    transforms.Resize(
                        self.img_size,
                        interpolation=tF.InterpolationMode.NEAREST
                    )(depth)
                )

        if 'sam_weight' in sample and sample['sam_weight'] is not None:
            resized_dict['sam_weight'] = transforms.Resize(
                self.img_size,
                interpolation=tF.InterpolationMode.NEAREST
            )(sample['sam_weight'])

        if 'heatmap_valid' in sample:
            resized_dict['heatmap_valid'] = sample['heatmap_valid']

        if 'points' in sample:
            resized_dict['points'] = sample['points']
        if 'labels' in sample and sample['labels'] is not None:
            resized_dict['labels'] = sample['labels']
        if 'visibility' in sample:
            resized_dict['visibility'] = sample['visibility']
        if 'valid' in sample:
            resized_dict['valid'] = sample['valid']

        return resized_dict

class customRandomRotate(object):
    def __init__(self, angle_range=(-15, 15)):
        self.angle_range = angle_range

    def __call__(self, sample):
        angle = np.random.randint(self.angle_range[0], self.angle_range[1])
        rotated_dict = {}

        rotated_dict['input'] = []
        for image in sample['input']:
            rotated_dict['input'].append(
                tF.rotate(image, angle, interpolation=tF.InterpolationMode.BILINEAR)
            )

        rotated_dict['mask'] = tF.rotate(
            sample['mask'], angle, interpolation=tF.InterpolationMode.BILINEAR
        )

        if 'coarse' in sample and sample['coarse'] is not None:
            rotated_dict['coarse'] = []
            for coarse in sample['coarse']:
                rotated_dict['coarse'].append(
                    tF.rotate(coarse, angle, interpolation=tF.InterpolationMode.BILINEAR)
                )

        if 'sam' in sample and sample['sam'] is not None:
            rotated_dict['sam'] = []
            for sam in sample['sam']:
                rotated_dict['sam'].append(
                    tF.rotate(sam, angle, interpolation=tF.InterpolationMode.NEAREST)
                )

        if 'input_depth' in sample:
            rotated_dict['input_depth'] = []
            for depth in sample['input_depth']:
                rotated_dict['input_depth'].append(
                    tF.rotate(depth, angle, interpolation=tF.InterpolationMode.NEAREST)
                )

        if 'sam_weight' in sample and sample['sam_weight'] is not None:
            rotated_dict['sam_weight'] = tF.rotate(
                sample['sam_weight'], angle, interpolation=tF.InterpolationMode.NEAREST
            )

        if 'heatmap_valid' in sample:
            rotated_dict['heatmap_valid'] = sample['heatmap_valid']
        if 'points' in sample:
            rotated_dict['points'] = sample['points']
        if 'labels' in sample and sample['labels'] is not None:
            rotated_dict['labels'] = sample['labels']
        if 'visibility' in sample:
            rotated_dict['visibility'] = sample['visibility']
        if 'valid' in sample:
            rotated_dict['valid'] = sample['valid']

        return rotated_dict

class customRandomHSVDistortion(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        distorted_dict = {}
        distorted_dict['input'] = []

        if np.random.binomial(size=1, n=1, p=self.p):
            b = np.random.uniform(0.95, 1.05)
            c = np.random.uniform(0.95, 1.05)
            s = np.random.uniform(0.95, 1.05)

            for image in sample['input']:
                image = tF.adjust_brightness(image, b)
                image = tF.adjust_contrast(image, c)
                image = tF.adjust_saturation(image, s)
                distorted_dict['input'].append(image)
        else:
            distorted_dict['input'] = sample['input']

        distorted_dict['mask'] = sample['mask']

        if 'input_depth' in sample:
            distorted_dict['input_depth'] = sample['input_depth']
        if 'sam_weight' in sample and sample['sam_weight'] is not None:
            distorted_dict['sam_weight'] = sample['sam_weight']
        if 'heatmap_valid' in sample and sample['heatmap_valid'] is not None:
            distorted_dict['heatmap_valid'] = sample['heatmap_valid']
        if 'coarse' in sample and sample['coarse'] is not None:
            distorted_dict['coarse'] = sample['coarse']
        if 'sam' in sample and sample['sam'] is not None:
            distorted_dict['sam'] = sample['sam']
        if 'points' in sample:
            distorted_dict['points'] = sample['points']
        if 'labels' in sample and sample['labels'] is not None:
            distorted_dict['labels'] = sample['labels']
        if 'visibility' in sample:
            distorted_dict['visibility'] = sample['visibility']
        if 'valid' in sample:
            distorted_dict['valid'] = sample['valid']

        return distorted_dict

class customHorizontalFlip(object): 
    def __init__(self, prediction_task, p=0.5): 
        self.task = prediction_task
        self.p = p
    
    def __call__(self, sample): 
        input = sample['input']
        mask = sample['mask']
        flipped_dict = {}
        flipped_dict['input'] = []
        if np.random.binomial(size=1, n=1, p=self.p):
            if 'input_depth' in sample:
                flipped_dict['input_depth'] = []
                for depth in sample['input_depth']: 
                    flipped_dict['input_depth'].append(transforms.RandomHorizontalFlip(p=1)(depth))
            if self.task == 'binary': 
                flipped_dict['mask'] = transforms.RandomHorizontalFlip(p=1)(mask)
                for image in input: 
                    flipped_dict['input'].append(transforms.RandomHorizontalFlip(p=1)(image))
                return flipped_dict
            if self.task == 'tooltip_segmentation':
                mask[mask==1] = 3
                mask[mask==2] = 1
                mask[mask==3] = 2
                flipped_dict['mask'] = transforms.RandomHorizontalFlip(p=1)(mask)
                for image in input: 
                    flipped_dict['input'].append(transforms.RandomHorizontalFlip(p=1)(image))
                return flipped_dict
            if self.task == 'toolpose_segmentation':
                mask[mask==1] = 5
                mask[mask==3] = 1
                mask[mask==5] = 3
                mask[mask==2] = 5
                mask[mask==4] = 2
                mask[mask==5] = 4
                flipped_dict['mask'] = transforms.RandomHorizontalFlip(p=1)(mask)
                for image in input:
                    flipped_dict['input'].append(transforms.RandomHorizontalFlip(p=1)(image))
                return flipped_dict
            if self.task == 'endovis15_segmentation':
                mask[mask == 1] = 11
                mask[mask == 6] = 1
                mask[mask == 11] = 6
                mask[mask == 2] = 11
                mask[mask == 7] = 2
                mask[mask == 11] = 7
                mask[mask == 3] = 11
                mask[mask == 8] = 3
                mask[mask == 11] = 8
                mask[mask == 4] = 11
                mask[mask == 10] = 4
                mask[mask == 11] = 10
                mask[mask == 5] = 11
                mask[mask == 9] = 5
                mask[mask == 11] = 9
                flipped_dict['mask'] = transforms.RandomHorizontalFlip(p=1)(mask)
                for image in input:
                    flipped_dict['input'].append(transforms.RandomHorizontalFlip(p=1)(image))
                return flipped_dict
        else:
            return sample

class customVerticalFlip(object):
    def __init__(self, prediction_task, p=0.5): 
        self.task = prediction_task
        self.p = p

    def __call__(self, sample): 
        input = sample['input'] 
        mask = sample['mask'] 
        flipped_dict = {} 
        flipped_dict['input'] = []
        if np.random.binomial(size=1, n=1, p=self.p):
            if self.task == 'endovis15_segmentation':
                mask[mask == 4] = 11
                mask[mask == 5] = 4
                mask[mask == 11] = 5
                mask[mask == 9] = 11
                mask[mask == 10] = 9
                mask[mask == 11] = 10
            flipped_dict['mask'] = transforms.RandomVerticalFlip(p=1)(mask)
            if 'input_depth' in sample:
                flipped_dict['input_depth'] = []
                for depth in sample['input_depth']:
                    flipped_dict['input_depth'].append(transforms.RandomVerticalFlip(p=1)(depth))
            for image in input: 
                flipped_dict['input'].append(transforms.RandomVerticalFlip(p=1)(image))
            return flipped_dict
        else:
            return sample

class customNormalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

    def __call__(self, sample):
        images = sample['input']
        mask = sample['mask']

        normalized_dict = {}
        normalized_dict['input'] = []

        for image in images:
            normalized_dict['input'].append(self.normalize(image))

        normalized_dict['mask'] = mask

        if 'input_depth' in sample:
            normalized_dict['input_depth'] = sample['input_depth']

        if 'sam_weight' in sample and sample['sam_weight'] is not None:
            normalized_dict['sam_weight'] = sample['sam_weight']

        if 'heatmap_valid' in sample:
            normalized_dict['heatmap_valid'] = sample['heatmap_valid']

        if 'sam' in sample and sample['sam'] is not None:
            normalized_dict['sam'] = sample['sam']

        if 'coarse' in sample and sample['coarse'] is not None:
            normalized_dict['coarse'] = sample['coarse']

        if 'points' in sample:
            normalized_dict['points'] = sample['points']

        if 'labels' in sample and sample['labels'] is not None:
            normalized_dict['labels'] = sample['labels']

        if 'visibility' in sample:
            normalized_dict['visibility'] = sample['visibility']

        if 'valid' in sample:
            normalized_dict['valid'] = sample['valid']

        return normalized_dict

# New Adding for DETR:
def get_kpt_detr_transform(mode, args):
    if mode == 'train':
        return transforms.Compose([
            to_tensor(),
            customResize((args.input_height, args.input_width)),
            customNormalize()
        ])
    else:
        return transforms.Compose([
            to_tensor(),
            customResize((args.input_height, args.input_width)),
            customNormalize()
        ])

# New Adding transform for ACT:
def get_act_transform(mode, args):
    if mode == 'train':
        transform_list = [
            to_tensor(),
            customRandomRotate(),         
            customRandomHSVDistortion(p=0.3),  
            customResize((args.input_height, args.input_width)),
            customNormalize()
        ]
    elif mode == 'val' or mode == 'test':
        transform_list = [
            to_tensor(),
            customResize((args.input_height, args.input_width)),
            customNormalize()
        ]
    else:
        raise NotImplementedError
    return transforms.Compose(transform_list)



def get_transform(mode, args):
    if mode == 'train': 
        transform_list = [to_tensor(), customRandomRotate(), customRandomHSVDistortion(), 
                          customResize((args.input_height, args.input_width)), 
                          customVerticalFlip(args.prediction_task), customHorizontalFlip(args.prediction_task), 
                          customNormalize()]
    elif mode == 'val': 
        transform_list = [to_tensor(), customResize((args.input_height, args.input_width)), 
                          customNormalize()]
    elif mode == 'test': 
        transform_list = [to_tensor(), customResize((args.input_height, args.input_width)), 
                          customNormalize()]
    else: 
        raise NotImplementedError
    return transforms.Compose(transform_list)

def get_data_loader(args): 
    if args.dataset == 'ACT':
        if args.mode == 'training': 
            train_file_names, val_file_names = get_ACT_dataset_filenames(args)
            train_transform = get_act_transform('train', args)
            val_transform = get_act_transform('val', args)
            train_dataset = ACT(train_file_names, train_transform,
                                    mode=args.mode, prediction_task=args.prediction_task,
                                    num_input_frames=args.num_input_frames,
                                   # num_frames_per_video=args.num_frames_per_video, 
                                    add_depth_inputs=args.add_depth_inputs)
            val_dataset = ACT(val_file_names, val_transform,
                                    mode=args.mode, prediction_task=args.prediction_task,
                                    num_input_frames=args.num_input_frames,
                                    # num_frames_per_video=args.num_frames_per_video, 
                                    add_depth_inputs=args.add_depth_inputs)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                    shuffle=True, num_workers=args.num_workers, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)
            return train_loader, val_loader
        else: 
            test_file_names, _ = get_ACT_dataset_filenames(args)
            test_transform = get_act_transform('test', args)
            test_dataset = ACT(test_file_names, test_transform, 
                                     mode=args.mode, prediction_task=args.prediction_task,
                                     num_input_frames=args.num_input_frames, 
                                     # num_frames_per_video=args.num_frames_per_video, 
                                     add_depth_inputs=args.add_depth_inputs)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                                    num_workers=args.num_workers, pin_memory=True)
            return None, test_loader
    


    elif args.dataset == "KPT":
        if args.track == 0:
            if args.mode == 'training': 
                train_file_names, val_file_names = get_KPT_dataset_filenames(args)
                if args.prediction_task == 'detr_keypoint':
                    train_transform = get_kpt_detr_transform('train', args)
                    val_transform = get_kpt_detr_transform('val', args)
                else:
                    train_transform = get_act_transform('train', args)
                    val_transform = get_act_transform('val', args)
                train_dataset = KPT_test_mid(train_file_names, train_transform,
                                        mode=args.mode, prediction_task=args.prediction_task,
                                        num_input_frames=args.num_input_frames,
                                    # num_frames_per_video=args.num_frames_per_video, 
                                        add_depth_inputs=args.add_depth_inputs)
                val_dataset = KPT_test_mid(val_file_names, val_transform,
                                        mode=args.mode, prediction_task=args.prediction_task,
                                        num_input_frames=args.num_input_frames,
                                        # num_frames_per_video=args.num_frames_per_video, 
                                        add_depth_inputs=args.add_depth_inputs)
                
                world_size = getattr(args, 'world_size', 1)
                global_rank = getattr(args, 'global_rank', 0)
                if world_size > 1:
                    train_sampler = DistributedSampler(
                        train_dataset,
                        num_replicas=world_size,
                        rank=global_rank,
                        shuffle=True,
                        drop_last = True,
                    )
                    shuffle_train = False

                    val_sampler = None

                else:
                    train_sampler = None
                    val_sampler = None
                    shuffle_train = True

                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle_train,
                        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, drop_last = True, persistent_workers=(args.num_workers > 0),)

                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                        sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)
                return train_loader, val_loader
            
            elif args.mode == 'generate_coarse':
                test_file_names = get_all_image_files(args)
                test_transform = get_act_transform('test', args)
                test_dataset = KPT_test_mid(test_file_names, test_transform, 
                                        mode=args.mode, prediction_task=args.prediction_task,
                                        num_input_frames=args.num_input_frames, 
                                        # num_frames_per_video=args.num_frames_per_video, 
                                        add_depth_inputs=args.add_depth_inputs)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                                        num_workers=args.num_workers, pin_memory=True)
                return None, test_loader                
            else: 
                test_file_names, _ = get_KPT_dataset_filenames(args)
                if args.prediction_task == 'detr_keypoint':
                    test_transform = get_kpt_detr_transform('test', args)
                else:   
                    test_transform = get_act_transform('test', args)
                test_dataset = KPT_test_mid(test_file_names, test_transform, 
                                        mode=args.mode, prediction_task=args.prediction_task,
                                        num_input_frames=args.num_input_frames, 
                                        # num_frames_per_video=args.num_frames_per_video, 
                                        add_depth_inputs=args.add_depth_inputs)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                                        num_workers=args.num_workers, pin_memory=True)
                return None, test_loader
        else:
            if args.mode == 'training': 
                train_file_names, val_file_names = get_KPT_dataset_filenames(args)
                if args.prediction_task == 'detr_keypoint':
                    train_transform = get_kpt_detr_transform('train', args)
                    val_transform = get_kpt_detr_transform('val', args)
                else:
                    train_transform = get_act_transform('train', args)
                    val_transform = get_act_transform('val', args)
                train_dataset = KPT_track_mid(train_file_names, train_transform,
                                        mode=args.mode, prediction_task=args.prediction_task,
                                        num_input_frames=args.num_input_frames,
                                    # num_frames_per_video=args.num_frames_per_video, 
                                        add_depth_inputs=args.add_depth_inputs)
                val_dataset = KPT_track_mid(val_file_names, val_transform,
                                        mode=args.mode, prediction_task=args.prediction_task,
                                        num_input_frames=args.num_input_frames,
                                        # num_frames_per_video=args.num_frames_per_video, 
                                        add_depth_inputs=args.add_depth_inputs)
                
                world_size = getattr(args, 'world_size', 1)
                global_rank = getattr(args, 'global_rank', 0)
                if world_size > 1:
                    train_sampler = DistributedSampler(
                        train_dataset,
                        num_replicas=world_size,
                        rank=global_rank,
                        shuffle=False
                    )
                    shuffle_train = False

                    val_sampler = DistributedSampler(
                        val_dataset,
                        num_replicas=world_size,
                        rank=global_rank,
                        shuffle=False
                    )

                else:
                    train_sampler = None
                    val_sampler = None
                    shuffle_train = False

                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle_train,
                        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)

                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                        sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)
                return train_loader, val_loader
            else: 
                test_file_names, _ = get_KPT_dataset_filenames(args)
                test_transform = get_act_transform('test', args)
                test_dataset = KPT_track_mid(test_file_names, test_transform, 
                                        mode=args.mode, prediction_task=args.prediction_task,
                                        num_input_frames=args.num_input_frames, 
                                        # num_frames_per_video=args.num_frames_per_video, 
                                        add_depth_inputs=args.add_depth_inputs)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                                        num_workers=args.num_workers, pin_memory=True)
                return None, test_loader  
    elif args.dataset == 'KPT_refine':
        if args.mode == 'training': 
            train_file_names, val_file_names = get_KPT_refine_dataset_filenames(args)
            train_transform = get_act_transform('train', args)
            val_transform = get_act_transform('val', args)
            train_dataset = KPT_refine(train_file_names, train_transform,
                                        mode=args.mode, prediction_task=args.prediction_task,
                                        num_input_frames=args.num_input_frames,
                                    # num_frames_per_video=args.num_frames_per_video, 
                                        add_depth_inputs=args.add_depth_inputs,
                                        use_mask=args.use_mask)
            val_dataset = KPT_refine(val_file_names, val_transform,
                                        mode=args.mode, prediction_task=args.prediction_task,
                                        num_input_frames=args.num_input_frames,
                                        # num_frames_per_video=args.num_frames_per_video, 
                                        add_depth_inputs=args.add_depth_inputs,
                                        use_mask=args.use_mask)
                
            world_size = getattr(args, 'world_size', 1)
            global_rank = getattr(args, 'global_rank', 0)
            if world_size > 1:
                    train_sampler = DistributedSampler(
                        train_dataset,
                        num_replicas=world_size,
                        rank=global_rank,
                        shuffle=True,
                        drop_last = True,
                    )
                    shuffle_train = False

                    val_sampler = None

            else:
                    train_sampler = None
                    val_sampler = None
                    shuffle_train = True

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle_train,
                        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, drop_last = True, persistent_workers=(args.num_workers > 0),)

            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                        sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)
            return train_loader, val_loader
        else: 
            test_file_names, _ = get_KPT_refine_dataset_filenames(args)
            test_transform = get_act_transform('test', args)
            test_dataset = KPT_refine(test_file_names, test_transform, 
                                        mode=args.mode, prediction_task=args.prediction_task,
                                        num_input_frames=args.num_input_frames, 
                                        # num_frames_per_video=args.num_frames_per_video, 
                                        add_depth_inputs=args.add_depth_inputs,
                                        use_mask=args.use_mask)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                                        num_workers=args.num_workers, pin_memory=True)
            return None, test_loader         
    else: 
        raise NotImplementedError



if __name__=="__main__":
    from types import SimpleNamespace
    from pathlib import Path
    import matplotlib.pyplot as plt

    # Create a SimpleNamespace object
    args = SimpleNamespace(
        data_dir=Path('/data/home/hao/chenyan/data/training_act_data'),
        dataset='ACT', 
        input_height=256,
        input_width=320,
        mode='training',
        prediction_task='keypoint_segmentation',
        num_input_frames=8,
        num_frames_per_video=225,
        batch_size=16,
        num_workers=8,
        add_depth_inputs=True
        )

    train_loader, val_loader = get_data_loader(args)
    print("Train loader length: ", len(train_loader))
    print("Val loader length: ", len(val_loader))
    for i, sample in enumerate(train_loader): 
        print("Sample keys: ", sample.keys())
        plt.figure(figsize=(12, 4), dpi=300)  # Increase DPI for sharper images
        # Plot the input frames
        plt.subplot(2, 3, 1)
        plt.imshow(sample['input'][0][0].permute(1, 2, 0).numpy() + 0.5)
        plt.title("Input frame 1", fontsize=14)  # Increased font size
        plt.axis('off')
        plt.subplot(2, 3, 2)
        plt.imshow(sample['input'][1][0].permute(1, 2, 0).numpy() + 0.5)
        plt.title("Input frame 2", fontsize=14)
        plt.axis('off')
        plt.subplot(2, 3, 3)
        plt.imshow(sample['input'][2][0].permute(1, 2, 0).numpy() + 0.5)
        plt.title("Input frame 3", fontsize=14)
        plt.axis('off')
        # Plot the mask
        plt.subplot(2, 3, 4)
        # plt.imshow(sample['mask'][0][0].numpy() * 63, cmap='gray')  # Assuming a grayscale mask
        mask = sample['mask'][0][0].numpy()
        mask_vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        mask_vis[mask == 1] = [0, 63, 0]
        mask_vis[mask == 2] = [0, 127, 0]
        mask_vis[mask == 3] = [0, 255, 0]
        mask_vis[mask == 4] = [255, 0, 0]
        mask_vis[mask == 5] = [0, 0, 255]
        mask_vis[mask == 6] = [0, 63, 0]
        mask_vis[mask == 7] = [0, 127, 0]
        mask_vis[mask == 8] = [0, 255, 0]
        mask_vis[mask == 9] = [255, 0, 0]
        mask_vis[mask == 10] = [0, 0, 255]
        plt.imshow(mask_vis)
        plt.title("Mask", fontsize=14)
        plt.axis('off')
        # Adjust layout and save with reduced margins
        plt.tight_layout(pad=0, h_pad=0.5, w_pad=0.5)  # Reduce padding between subplots
        plt.savefig('sample_{:03d}.png'.format(i), bbox_inches='tight')  # Save without excess white space
        plt.close()  # Close the figure to free memory
        if i==5: 
            break

