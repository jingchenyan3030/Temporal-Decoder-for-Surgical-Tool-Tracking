"""
Configuration file stating all the args for toolpose segmentation task
"""

def train_config_parser(parser):
    # dataset related arguments
    parser.add_argument('--data_dir', type=str, default='/data/home/hao/chenyan/data/training_act_data', 
                        help='Path to data directory. Default: ''/data/home/hao/chenyan/data/training_act_data')#change
    parser.add_argument('--dataset', type=str, default='ACT', choices=['MICCAI2015', 'MICCAI2017', 'JIGSAWS', 'KPT','ACT'], #change
                        help='Dataset name. Default: ACT')
    parser.add_argument('--fold_index', type=int, default=-1, choices=[-1,0,1,2,3], 
                        help='Fold index for cross validation. Default: -1, no cross validation') # ?
    parser.add_argument('--prediction_task', type=str, default='keypoint_segmentation',
                        help='Prediction task. Default: keypoint_segmentation')
    parser.add_argument('--mode', type=str, default='training', choices=['training', 'testing'], 
                        help='Mode of operation. Default: training')
    parser.add_argument('--num_frames_per_video', type=int, default=225, 
                        help='Number of frames per video/folder in the dataset. Default: 225')
    #Addung new
    parser.add_argument('--val_cases_count', type=int, default=9,
                        help='How many top-level cases (folders under data_dir) to use for validation (deterministic split).')
    
    # I/O related arguments
    parser.add_argument('--expt_savedir', type=str, default='/data/home/hao/chenyan/checkpoint', 
                        help='Path to save experiment results. Default:/data/home/hao/chenyan/checkpoint')
    parser.add_argument('--expt_name', type=str, default='toolpose_segmentation_expt',
                        help='Experiment name. Default: toolpose_segmentation_expt')
    parser.add_argument('--print_freq', type=int, default=10, 
                        help='Print frequency. Default: 10')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save frequency. Default: 10')
    parser.add_argument('--debug', type=bool, default=False, help='Debug mode')

    # optimizer related arguments   
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size. Default: 32')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader. Default: 8')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes (incl. background). Default: 4') #change
    parser.add_argument('--metric_fns', type=str, nargs='+', default=['iou', 'dice'], choices=['iou', 'dice'], 
                        help='List of metric functions. Default: iou, dice')
    parser.add_argument('--loss_fns', type=str, nargs='+', default=['nll'], choices=['mse', 'nll', 'soft_jaccard'],  
                        help='List of loss functions. Default: nll')
    parser.add_argument('--loss_wts', type=float, nargs='+', default=[1.0], 
                        help='List of loss weights. Default: 1.0')
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Learning rate. Default: 1e-4')
    parser.add_argument('--scheduler', type=str, default='StepDecay', choices=['StepDecay', 'Constant'], 
                        help='Learning rate scheduler. Default: StepDecay at halfway point')
    parser.add_argument('--num_epochs', type=int, default=10, 
                        help='Number of epochs. Default: 10')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Seed for random number generator. Default: 42')
    parser.add_argument('--resume', type=bool, default=False, 
                        help='Resume training')
    parser.add_argument('--starting_epoch', type=int, default=0, 
                        help='Starting epoch. Default: 0')
    parser.add_argument('--class_weights', type=float, nargs='+', default=[1,100,100,100], # change[1,100,100,100,100]
                        help='Class weights for NLL loss function. Default: [1,100,100,100]')
    
    # model related arguments
    parser.add_argument('--model_type', type=str, default='TernausNet16', 
                        choices=['TernausNet11', 'TernausNet16', 'TAPNet11', 'TAPNet16', 'DeepLab_v3', 'FCN', 'HRNet', 'SegFormer'], 
                        help='Model name')
    parser.add_argument('--pretrained', type=bool, default=False, 
                        help='Use pre-trained weights. Default: False')
    parser.add_argument('--load_wts_model', type=str, default=None, 
                        help='Path to model weights. Default: None')
    parser.add_argument('--input_height', type=int, default=256, help='NN input image height')
    parser.add_argument('--input_width', type=int, default=320, help='NN input image width')
    parser.add_argument('--add_optflow_inputs', type=bool, default=False, help='Add optical flow inputs')
    parser.add_argument('--optflow_dir', type=str, default=None, 
                        choices=['optflows_unflow', 'optf]=defsflows_raft'])
    parser.add_argument('--update_attmaps', type=bool, default=False, help='Update attention maps')
    return parser

def test_config_parser(parser):
    # dataset related arguments
    parser.add_argument('--data_dir', type=str, default='/data/home/hao/chenyan/data/training_act_data', 
                        help='Path to data directory. Default:/data/home/hao/chenyan/data/training_act_data')
    parser.add_argument('--dataset', type=str, default='ACT', choices=['MICCAI2015', 'MICCAI2017', 'JIGSAWS', 'KPT', 'ACT'], # should change but not change
                        help='Dataset name. Default: ACT')
    parser.add_argument('--prediction_task', type=str, default='keypoint_segmentation', 
                        choices=['tooltip_segmentation', 'toolpose_segmentation', 'endovis15_segmentation', 'binary', 'keypoint_segmentation'], 
                        help='Prediction task. Default: toolpose_segmentation')
    parser.add_argument('--num_frames_per_video', type=int, default=75, 
                        help='Number of frames per video/folder in the dataset. Default: 75')
    #Addung new
    parser.add_argument('--val_cases_count', type=int, default=2,
                        help='How many top-level cases (folders under data_dir) to use for validation (deterministic split).')
    
    # I/O related arguments
    parser.add_argument('--expt_savedir', type=str, default='./', 
                        help='Path to save experiment results. Default: ./')
    parser.add_argument('--expt_name', type=str, default='toolpose_segmentation_expt',
                        help='Experiment name. Default: toolpose_segmentation_expt')
    parser.add_argument('--print_freq', type=int, default=10, 
                        help='Print frequency. Default: 10')
    parser.add_argument('--save_output_freq', type=int, default=10,
                        help='Save output frequency. Default: 10')

    # optimizer related arguments   
    parser.add_argument('--num_classes', type=int, default=4 , help='Number of classes (incl. background). Default: 4') 
    parser.add_argument('--num_workers', type=int, default=12, help='Number of workers for dataloader. Default: 12')
    parser.add_argument('--metric_fns', type=str, nargs='+', default=['iou', 'dice'], choices=['iou', 'dice'], 
                        help='List of metric functions. Default: iou, dice')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Seed for random number generator. Default: 42')
    parser.add_argument('--resume', type=bool, default=False, 
                        help='Resume training. Default: False')
    
    # model related arguments
    parser.add_argument('--model_type', type=str, default='TernausNet16', 
                        choices=['TernausNet11', 'TernausNet16', 'TAPNet11', 'TAPNet16', 'DeepLab_v3', 'FCN', 'HRNet', 'SegFormer'],  
                        help='Model name')
    parser.add_argument('--pretrained', type=bool, default=False, 
                        help='Use pre-trained weights. Default: False')
    parser.add_argument('--load_wts_model', type=str, default='/data/home/hao/chenyan/checkpoint/toolpose_segmentation_expt_full/ckpts/model_010.pth', 
                        help='Path to model weights. Default: None')
    parser.add_argument('--input_height', type=int, default=256, help='NN input image height')
    parser.add_argument('--input_width', type=int, default=320, help='NN input image width')
    parser.add_argument('--add_optflow_inputs', type=bool, default=False, help='Add optical flow inputs')
    parser.add_argument('--optflow_dir', type=str, default=None, 
                        choices=['optflows_unflow', 'optflows_raft'])
    parser.add_argument('--update_attmaps', type=bool, default=False, help='Update attention maps')
    return parser

