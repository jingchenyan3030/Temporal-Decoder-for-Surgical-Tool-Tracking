from importlib.metadata import files
from matplotlib.pyplot import step
import numpy as np
import cv2 
import json
import torch
from torchvision import transforms
import torchvision.transforms.functional as tF
from natsort import natsorted
from pathlib import Path
import pandas as pd

# keypoint_heatmap:
#   target: FloatTensor (2,H,W), regression, MSE
# segmentation:
#   target: LongTensor (1,H,W), classification


def load_optflow_map(path, optflow_dir):
    with open(str(path).replace('images', optflow_dir).replace('jpg', 'flo')) as f:
        optflow = np.fromfile(f, dtype=np.float32)
        # optflow = optflow[2:].reshape((1024,1280,2))
        optflow = optflow[2:].reshape((256,350,2))
    return optflow


def load_image(path):
    img = cv2.imread(str(path))
    if img is None: 
        print(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_depthmap(path):
    dmap = cv2.imread(str(path).replace('images','depth_maps'))
    if dmap is None: 
        print(path)
    return cv2.cvtColor(dmap, cv2.COLOR_BGR2GRAY)

def load_mask(path, prediction_task):
    if prediction_task=='tooltip_segmentation':
        maskl = cv2.imread(str(path).replace('images','pose_maps').replace('frame','framel').replace('jpg','png'))
        maskr = cv2.imread(str(path).replace('images','pose_maps').replace('frame','framer').replace('jpg','png'))
        mask = np.zeros((maskl.shape[0], maskl.shape[1]))
        # only try to segment out the tool tips, ignore toolbase
        if np.amax(maskl):
            mask[np.where(maskl[:,:,0]>0)] = 255
            mask[np.where(maskl[:,:,2]>0)] = 255 
        if np.amax(maskr):
            mask[np.where(maskr[:,:,0]>0)] = 127
            mask[np.where(maskr[:,:,2]>0)] = 127
        return (mask / 127).astype(np.uint8)
    elif prediction_task=='endovis15_segmentation':
        maskl = cv2.imread(str(path).replace('images','pose_maps_endovis').replace('frame','framel').replace('jpg','png'))
        maskr = cv2.imread(str(path).replace('images','pose_maps_endovis').replace('frame','framer').replace('jpg','png'))
        mask = np.zeros((maskl.shape[0], maskl.shape[1]))
        if np.amax(maskl):
            mask[np.where(maskl[:,:,0]>0)] = 250
            mask[np.where(maskl[:,:,2]>0)] = 225 
            mask[np.where(maskl[:,:,1]==255)] = 200
            mask[np.where(maskl[:,:,1]==127)] = 175
            mask[np.where(maskl[:,:,1]==63)] = 150
        if np.amax(maskr):
            mask[np.where(maskr[:,:,0]>0)] = 125
            mask[np.where(maskr[:,:,2]>0)] = 100
            mask[np.where(maskr[:,:,1]==255)] = 75
            mask[np.where(maskr[:,:,1]==127)] = 50
            mask[np.where(maskr[:,:,1]==63)] = 25
        return (mask / 25).astype(np.uint8)
    elif prediction_task=='toolpose_segmentation':
        maskl = cv2.imread(str(path).replace('images','pose_maps').replace('frame','framel').replace('jpg','png'))
        maskr = cv2.imread(str(path).replace('images','pose_maps').replace('frame','framer').replace('jpg','png'))
        mask = np.zeros((maskl.shape[0], maskl.shape[1]))
        if np.amax(maskl):
            mask[np.where(maskl[:,:,0]>0)] = 255
            mask[np.where(maskl[:,:,2]>0)] = 255 
            mask[np.where(maskl[:,:,1]>0)] = 191
        if np.amax(maskr):
            mask[np.where(maskr[:,:,0]>0)] = 127
            mask[np.where(maskr[:,:,2]>0)] = 127
            mask[np.where(maskr[:,:,1]>0)] = 63
        return (mask / 63).astype(np.uint8)
    elif prediction_task=='binary':
        binary_factor = 255
        mask_folder = 'binary_masks'
        mask = cv2.imread(str(path).replace('images', mask_folder).replace('jpg', 'png'), 0)
        return (mask / binary_factor).astype(np.uint8)
    
    # NEW_ADDING
    elif prediction_task == "keypoint_segmentation":
        pose_map_path = str(path).replace('images', 'pose_map')  
        pm = cv2.imread(pose_map_path, cv2.IMREAD_COLOR)
        if pm is None:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            H, W = (img.shape[:2] if img is not None else (480, 640))
            return np.zeros((H, W), dtype=np.uint8)
        
        pose_map = cv2.cvtColor(pm, cv2.COLOR_BGR2RGB)
        H, W = pose_map.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        r = pose_map[:, :, 0]
        g = pose_map[:, :, 1]
        b = pose_map[:, :, 2]
        tip_cond     = (r > 0) & (g > 0) & (b == 0)   
        anchor_cond  = (r > 0) & (g == 0) & (b == 0)  
        contact_cond = (r == 0) & (g > 0) & (b == 0)  
        if np.amax(pose_map) > 0:
            mask[tip_cond]     = 1
            mask[anchor_cond]  = 2
            mask[contact_cond] = 3
        return mask
    # NEW ADDING FOR MSE:
    elif prediction_task =="keypoint_heatmap":
        sam2_path = Path(str(path).replace("images", "sam_results")).with_suffix(".npy")
        hm = np.load(sam2_path).astype(np.float32)
        if hm.ndim == 2:
            raise ValueError(f"Heatmap must have 2 channels(tip+anchor)")
        if hm.ndim != 3:
            raise ValueError(f"Unexpected heatmap shape {hm.shape} at {sam2_path}")
        # (2,H,W) -> (H,W,2)
        if hm.shape[0] == 2 and hm.shape[1] != 2:
            hm = np.transpose(hm, (1,2,0))
        if hm.shape[2] != 2:
            raise ValueError(f"UnexpeHeatmap last dim must be 2. Got {hm.shape} at {sam2_path}")
        return hm
    else:
        raise ValueError('Unknown prediction task: {}'.format(prediction_task))

def get_MICCAI2015_dataset_filenames(args): 
    if args.mode=='training': 
        folds = {-1: [], 0: [4], 1: [3], 2: [2], 3: [1]}
        train_path = args.data_dir / 'Tracking_Robotic_Training' / 'Training'
        train_file_names = [] 
        val_file_names = []
        for i in range(1,5): 
            train_file_names += natsorted(list((train_path / ('Dataset' + str(i)) / 'images').glob('*')), key=str)
        val_path = args.data_dir / 'Tracking_Robotic_Testing' / 'Tracking' 
        val_file_names = [] 
        for i in range(1,5): 
            val_file_names += natsorted(list((val_path / ('Dataset' + str(i)) / 'images').glob('*')), key=str)
        return train_file_names, val_file_names
    if args.mode=='testing':
        test_path = args.data_dir / 'Tracking_Robotic_Testing' / 'Tracking' 
        test_file_names = [] 
        for i in range(1,7): 
            test_file_names += natsorted(list((test_path / ('Dataset' + str(i)) / 'images').glob('*')), key=str)
        return test_file_names, None

def get_MICCAI2017_dataset_filenames(args):
    if args.mode=='training':
        folds = {-1: [], 0: [1, 3], 1: [2, 5],
                2: [4, 8], 3: [6, 7]}
        train_path = args.data_dir / 'cropped_train'
        train_file_names = []
        val_file_names = []
        for instrument_id in range(1, 9):
            if instrument_id in folds[args.fold_index]:
                val_file_names += natsorted(list((train_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*')), key=str)
            else:
                train_file_names += natsorted(list((train_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*')), key=str)
        return train_file_names, val_file_names
    if args.mode=='testing':
        test_path = args.data_dir / 'cropped_test'
        test_file_names = []
        for instrument_id in range(1, 11):
            test_file_names += natsorted(list((test_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*')), key=str)
        return test_file_names, None



def get_JIGSAWS_dataset_filenames(args):
    if args.mode=='training': 
        folds = {-1: [], 0: [1], 1: [2], 2: [1,2]}
        train_path = args.data_dir / 'annotations_train'
        val_path = args.data_dir / 'annotations_val'
        train_file_names = []
        val_file_names = [] 
        for i in range(1,7): 
            train_file_names += natsorted(list((train_path / ('video_' + str(i)) / 'images').glob('*')), key=str)
            val_file_names += natsorted(list((val_path / ('video_' + str(i)) / 'images').glob('*')), key=str)
        # train_path = args.data_dir / 'train'
        # train_file_names = []
        # val_file_names = []
        # for i in range(1,7):
        #     if i<6:
        #         train_file_names += natsorted(list((train_path / ('video_' + str(i)) / 'images').glob('*')), key=str)
        #     else:
        #         val_file_names += natsorted(list((train_path / ('video_' + str(i)) / 'images').glob('*')), key=str)
        return train_file_names, val_file_names
    if args.mode=='testing':
        test_path = args.data_dir / 'annotations_val'
        test_file_names = []
        for i in range(1,7):
            test_file_names += natsorted(list((test_path / ('video_' + str(i)) / 'images').glob('*')), key=str)
        # test_path = args.data_dir / 'train' 
        # test_file_names = []
        # for i in range(1,7): 
        #     if i==6:
        #         test_file_names += natsorted(list((test_path / ('video_' + str(i)) / 'images').glob('*')), key=str)
        # for i in range(1,3): 
        #     test_file_names += natsorted(list((test_path / ('random_sample_set_' + str(i)) / 'images').glob('*')), key=str)
        return test_file_names, None


# New Adding
from pathlib import Path
from natsort import natsorted

def get_KPT_dataset_filenames(args):
    root = Path(args.data_dir)
    all_cases = natsorted(
        [d for d in root.iterdir() if d.is_dir() and d.name.lower() != "test" and d.name.lower() != "test_multiframe"],
        key=str
    )

    actions = ['grasp','clip','cut','dissect']
    if hasattr(args, "action") and args.action in actions:
        actions = [args.action] 
    elif hasattr(args, "action") and args.action not in actions and args.action is not None:
        print(f"[Warning] Unknown action '{args.action}', proceeding with all actions.")
       
    for action in actions:
        train_files = []
        val_files = []
        test_files = []
        action_dir = root / action
        if not action_dir.exists():
            raise ValueError(f"Action directory '{action}' does not exist in the dataset root.")
        
        all_cases = natsorted(
            [d for d in action_dir.iterdir() if d.is_dir()],
            key=str
        )

        case_dirs = [d for d in all_cases if d.name.lower().startswith("case_")]
        cholec_dirs = [d for d in all_cases if d.name.lower().startswith("cholec")]
        comp_dirs = [d for d in all_cases if d.name.lower().startswith("comprehensive")]
        heichole_dirs = [d for d in all_cases if d.name.lower().startswith("heichole")]
        youtube_dirs = [d for d in all_cases if d.name.lower().startswith("youtube_cholecystectomy")]

        def split_dirs(dirs, num_test):
            if len(dirs) <= num_test:
                return [], dirs
            return dirs[:-num_test], dirs[-num_test:]

        train_cases, test_cases = [], []

        for group, num_test in [
            (case_dirs, 2),
            (cholec_dirs, 2),
            (comp_dirs, 1),
            (heichole_dirs, 2),
            (youtube_dirs, 1),
        ]:
            train, test = split_dirs(group, num_test)
            train_cases.extend(train)
            test_cases.extend(test)
        
        val_cases = train_cases[-5:] if len(train_cases) > 5 else []
        '''
        if len(train_cases) >= 2:
            val_cases = [train_cases[-1]]     # 1 case for val
            train_cases = train_cases[:-1]    # remaining for train
        else:
            val_cases = []
        '''
        def collect(cases):
            files = []

            for case in cases:
                for video_dir in natsorted(case.iterdir(), key=str):

                    img_dir  = video_dir / "images"
                    pose_dir = video_dir / "pose_map"
                    sam_dir  = video_dir / "sam_results"
                    detr_dir = video_dir / "points_detr"

                    if not img_dir.exists() or not pose_dir.exists() or not sam_dir.exists():
                        continue

                    images = natsorted(
                        [p for p in img_dir.glob("*") if p.suffix.lower() in [".png",".jpg",".jpeg"]],
                        key=str
                    )
                    poses = natsorted(list(pose_dir.glob("*.png")), key=str)
                    sam_masks = natsorted([p for p in sam_dir.glob("frame_*.npy") if not p.stem.endswith("_weight")], key=str)
                    #sam_weights = natsorted(list(sam_dir.glob("frame_*_weight.npy")),key=str)
                    detr_points = natsorted(list(detr_dir.glob("frame_*.json")), key=str)

                    if not (len(images) == len(poses) == len(sam_masks) == len(detr_points)):
                        print(
                            f"[DROP VIDEO] {video_dir} "
                            f"images={len(images)} poses={len(poses)} npy={len(sam_masks)} detr={len(detr_points)} "
                        )
                        continue
                    '''
                    # optional: strict index check
                    ok = True
                    for i in range(len(images)):
                        if not (sam_dir / f"frame_{i+1:03d}_mask.npy").exists():
                            ok = False
                            break
                    if not ok:
                        print(f"[DROP VIDEO] {video_dir} frame index mismatch")
                        continue
                    '''
                    files.extend(images)
                   
            return files


        train_files = collect(train_cases)
        val_files = collect(val_cases)
        test_files = collect(test_cases)


    if args.mode == 'training':
        return train_files, val_files
    elif args.mode == 'testing':
        return test_files, None
    elif args.mode == 'all':
        all_files = collect(all_cases)
        return all_files, None
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


## new adding end
def get_ACT_dataset_filenames(args):
    root = Path(args.data_dir)
    actions = ['grasp','clip','cut','dissect']
    if not hasattr(args, "action") or args.action not in actions:
        raise ValueError(f"Invalid or missing action: {getattr(args, 'action', None)}. "
                         f"Must be one of {actions}.")

    action = args.action
    action_dir = root / action
    if not action_dir.exists():
        raise ValueError(f"Action directory '{action}' does not exist in the dataset root.") 
    all_cases = natsorted(
            [d for d in action_dir.iterdir()
            if d.is_dir() and d.name.lower() not in ("test", "test_multiframe")],
            key=str
        )

    
    def parse_tissue_id(name:str) -> int:
        parse = name.split('_')
        if len(parse) < 2:
            raise ValueError(f"Invalid directory name: {name}")
        return int(parse[1])
        
    val_count = 10
    test_count = 5
    train_cases = all_cases[:-val_count] 
    val_cases = all_cases[-val_count:-test_count]
    test_cases = all_cases[-test_count:]

    def collect(case_dirs):
        files = []
        for case_dir in case_dirs:
            img_dir, pose_dir = case_dir / "images", case_dir / "pose_map"  
            if not(img_dir.exists() and pose_dir.exists()):
                continue
            for p in natsorted(list(img_dir.glob("*")), key=str):
                if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    continue
                if (pose_dir / (p.stem + ".png")).exists():
                    files += [p]
        return files
    if args.mode == 'training':
        train_file_names = collect(train_cases)
        val_file_names   = collect(val_cases)
        return train_file_names, val_file_names
    if args.mode == 'testing':
        test_file_names = collect(test_cases)
        return test_file_names, None


    


class customRandomHSVDistortion(object):
    def __call__(self, sample):
        image = sample['image'] 
        attmap = sample['attmap'] 
        mask = sample['mask'] 
        if np.random.binomial(size=1,n=1,p=0.2):
            image = tF.adjust_brightness(image, np.random.uniform(0.9,1.1))
            image = tF.adjust_contrast(image, np.random.uniform(0.9,1.1))
            image = tF.adjust_saturation(image, np.random.uniform(0.9,1.1))
        return {'image': image, 
                'attmap': attmap, 
                'mask': mask}

class customRandomRotation(object):
    def __call__(self, sample): 
        image = sample['image'] 
        attmap = sample['attmap'] 
        mask = sample['mask'] 
        angle = np.random.randint(-30,30)
        if np.random.binomial(size=1,n=1,p=0.2):
            return {'image': tF.rotate(image, angle),
                'attmap': tF.rotate(attmap, angle),
                'mask': tF.rotate(mask, angle)}
        else:
            return sample

class customVertFlip(object):
    """Flip the image vertically."""
    def __call__(self, sample):
        image = sample['image'] 
        attmap = sample['attmap'] 
        mask = sample['mask'] 
        if np.random.binomial(size=1,n=1,p=0.5):
            return {'image': transforms.RandomVerticalFlip(p=0.5)(image), 
                'attmap': transforms.RandomVerticalFlip(p=0.5)(attmap),
                'mask': transforms.RandomVerticalFlip(p=0.5)(mask)}
        else: 
            return sample

class customHorzFlip(object):
    """Flip the image horizontally."""
    def __call__(self, sample): 
        image = sample['image'] 
        attmap = sample['attmap'] 
        mask = sample['mask'] 
        if np.random.binomial(size=1,n=1,p=0.5):
            # mask[mask==1] = 3
            # mask[mask==2] = 1
            # mask[mask==3] = 2
            return {'image': transforms.RandomHorizontalFlip(p=0.5)(image), 
                'attmap': transforms.RandomHorizontalFlip(p=0.5)(attmap),
                'mask': transforms.RandomHorizontalFlip(p=0.5)(mask)}
        else: 
            return sample

class customHorzFlip_LR(object):
    """Flip the image horizontally."""
    def __call__(self, sample): 
        image = sample['image'] 
        attmap = sample['attmap'] 
        mask = sample['mask'] 
        if np.random.binomial(size=1,n=1,p=0.5):
            mask[mask==1] = 3
            mask[mask==2] = 1
            mask[mask==3] = 2
            return {'image': transforms.RandomHorizontalFlip(p=0.5)(image), 
                'attmap': transforms.RandomHorizontalFlip(p=0.5)(attmap),
                'mask': transforms.RandomHorizontalFlip(p=0.5)(mask)}
        else: 
            return sample

class customNormalize(object):
    """Normalize the image."""
    def __call__(self, sample):
        image = sample['image'].type(torch.float32)/255.0
        attmap = sample['attmap'].type(torch.float32)
        mask = sample['mask'] 
        return {'image': transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image), 
                'attmap': attmap,
                'mask': mask}    
