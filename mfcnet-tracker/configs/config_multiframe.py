"""
Configuration file stating all the args for multi-frame segmentation task
"""

def train_config_parser(parser):
    # dataset related arguments
    parser.add_argument('--data_dir', type=str, default='/data/home/hao/chenyan/data/act_training_data_by_action', 
                        help='Path to data directory. Default: /data/home/hao/chenyan/data/act_training_data_by_action')
    parser.add_argument('--dataset', type=str, default='ACT', choices=['MICCAI2015', 'MICCAI2017', 'JIGSAWS', 'KPT','ACT'],
                        help='Dataset name. Default: ACT')
    parser.add_argument('--fold_index', type=int, default=-1, choices=[-1,0,1,2,3], 
                        help='Fold index for cross validation. Default: -1, no cross validation')
    parser.add_argument('--prediction_task', type=str, default='keypoint_heatmap', 
                        choices=['tooltip_segmentation', 'toolpose_segmentation', 'endovis15_segmentation', 'binary', 'keypoint_segmentation','keypoint_heatmap'],
                        help='Prediction task. Default: keypoint_heatmap')
    parser.add_argument('--mode', type=str, default='training', choices=['training', 'testing'], 
                        help='Mode of operation. Default: training')
    parser.add_argument('--num_frames_per_video', type=int, default=1000,    
                        help='Number of frames per video/folder in the dataset. Default: 225')  # Do not want to use
    parser.add_argument('--action', type=str, default='clip', choices=['grasp', 'clip','dissect', 'cut'], 
                        help='Mode of operation. Default: clip')

    # I/O related arguments
    parser.add_argument('--expt_savedir', type=str, default='/data/home/hao/chenyan/checkpoint', 
                        help='Path to save experiment results. Default: /data/home/hao/chenyan/checkpoint')
    parser.add_argument('--expt_name', type=str, default='multiframe_segmentation_expt_act_clip_mse_full',
                        help='Experiment name. Default: multiframe_segmentation_expt_act_clip_mse_full')
    parser.add_argument('--print_freq', type=int, default=1, 
                        help='Print frequency. Default: 1')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='Save frequency. Default: 1')
    parser.add_argument('--debug', type=bool, default=False, help='Debug mode')
    parser.add_argument('--save_output_freq', type=int, default=1, 
                        help='Save output frequency. Default: 1')
    parser.add_argument('--num_input_frames', type=int, default=8,
                    help='number of frames used as input in multiframe model')

    # optimizer related arguments
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size. Default: 4')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for dataloader. Default: 8')
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

    
    #segmentation-nll
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes (incl. background). Default: 4')
    parser.add_argument('--class_weights', type=float, nargs='+', default=[1,100,100,100],
                        help='Class weights for NLL loss function. Default: [1,100,100,100]')
    parser.add_argument('--metric_fns', type=str, nargs='+', default=['iou', 'dice'], choices=['iou', 'dice'], 
                        help='List of metric functions. Default: iou, dice')
    parser.add_argument('--loss_fns', type=str, nargs='+', default=['nll'], choices=['mse', 'nll', 'soft_jaccard'],  
                        help='List of loss functions. Default: nll')
    parser.add_argument('--loss_wts', type=float, nargs='+', default=[1.0], 
                        help='List of loss weights. Default: 1.0')
    '''
    # mse
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--class_weights', type=float, nargs='+', default=[1, 1])  
    parser.add_argument('--metric_fns', type=str, nargs='+', default=[],
                    help='Heatmap task: keep empty.')
    parser.add_argument('--loss_fns', type=str, nargs='+', default=['mse'], choices=['mse', 'nll', 'soft_jaccard'],  
                        help='List of loss functions. Default: mse')
    parser.add_argument('--loss_wts', type=float, nargs='+', default=[1.0], 
                        help='List of loss weights. Default: 1.0')
    ''' 

    
    # model related arguments
    parser.add_argument('--model_type', type=str, default='TernausNetMulti-Basic', 
                        choices=['TernausNetMulti-Basic', 'TernausNetMulti-Large', 'DeepLabMulti-Basic', 'DeepLabMulti-Large', 
                                 'FCNMulti-Basic', 'FCNMulti-Large', 'SegFormerMulti-Basic', 'SegFormerMulti-Large', 'HRNetMulti-Basic', 'HRNetMulti-Large'], 
                        help='Model name')
    parser.add_argument('--pretrained', type=bool, default=False, 
                        help='Use pre-trained weights. Default: False')
    parser.add_argument('--train_base_model', type=bool, default=True,
                        help='Train base model. Default: True')
    parser.add_argument('--load_wts_base_model', type=str, default=None,
                        help='Path to base model weights from a pretrained per-frame model. Default: None')
    parser.add_argument('--load_wts_model', type=str, default=None, 
                        help='Path to model weights. Default: None')
    parser.add_argument('--input_height', type=int, default=256, help='NN input image height')
    parser.add_argument('--input_width', type=int, default=320, help='NN input image width')
    parser.add_argument('--add_optflow_inputs', type=bool, default=False, help='Add optical flow inputs')
    parser.add_argument('--optflow_model', type=str, default='RAFT', choices=['RAFT', 'FlowFormerPlusPlus'],)
    parser.add_argument('--add_depth_inputs', type=bool, default=True, help='Add monocular depth inputs')
    return parser

def test_config_parser(parser):
    # dataset related arguments
    parser.add_argument('--action', type=str, default='cut', choices=['grasp', 'clip','dissect', 'cut'], 
                        help='Mode of operation. Default: cut')
    parser.add_argument('--data_dir', type=str, default='/data/home/hao/chenyan/data/act_training_data_by_action', 
                        help='Path to data directory. Default: /data/home/hao/chenyan/data/act_training_data_by_action')
    parser.add_argument('--dataset', type=str, default='ACT', choices=['MICCAI2015', 'MICCAI2017', 'JIGSAWS', 'KPT', 'ACT'], # should change but not change
                        help='Dataset name. Default: ACT')
    parser.add_argument('--prediction_task', type=str, default='keypoint_heatmap', 
                        choices=['tooltip_segmentation', 'toolpose_segmentation', 'endovis15_segmentation', 'binary', 'keypoint_segmentation','keypoint_heatmap'], 
                        help='Prediction task. Default: keypoint_heatmap')
    parser.add_argument('--num_frames_per_video', type=int, default=1000, 
                        help='Number of frames per video/folder in the dataset. Default: 225')
    parser.add_argument('--num_input_frames', type=int, default=8,
                        help='Number of input frames for the model. Default: 8')
    
    # I/O related arguments
    parser.add_argument('--expt_savedir', type=str, default='/data/home/hao/chenyan/checkpoint/testing_multiframe_act', 
                        help='Path to save experiment results. Default: /data/home/hao/chenyan/checkpoint/testing_multiframe_act')
    parser.add_argument('--expt_name', type=str, default='multiframe_expt',
                        help='Experiment name. Default: multiframe_expt')
    parser.add_argument('--print_freq', type=int, default=1, 
                        help='Print frequency. Default: 1')
    parser.add_argument('--save_output_freq', type=int, default=1, 
                        help='Save output frequency. Default: 1')

    # optimizer related arguments
    parser.add_argument('--num_workers', type=int, default=12, help='Number of workers for dataloader. Default: 12')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Seed for random number generator. Default: 42')
    parser.add_argument('--resume', type=bool, default=False, 
                        help='Resume training. Default: False')
    
    # model related arguments   
    parser.add_argument('--model_type', type=str, default='TernausNetMulti-Basic', 
                        choices=['TernausNetMulti-Basic', 'TernausNetMulti-Large', 'DeepLabMulti-Basic', 'DeepLabMulti-Large', 
                                 'FCNMulti-Basic', 'FCNMulti-Large', 'SegFormerMulti-Basic', 'SegFormerMulti-Large', 'HRNetMulti-Basic', 'HRNetMulti-Large'],  
                        help='Model name')
    parser.add_argument('--pretrained', type=bool, default=False, 
                        help='Use pre-trained weights. Default: False')
    parser.add_argument('--load_wts_base_model', type=str, default= None,
                        help='Path to base model weights from a pretrained per-frame model. Default: None')
    parser.add_argument('--load_wts_model', type=str, default='/data/home/hao/chenyan/checkpoint/multiframe_segmentation_expt_act_cut_final/ckpts/model_010.pth', 
                        help='Path to model weights. Default: None')
    parser.add_argument('--input_height', type=int, default=256, help='NN input image height')
    parser.add_argument('--input_width', type=int, default=320, help='NN input image width')
    parser.add_argument('--add_optflow_inputs', type=bool, default=False, help='Add optical flow inputs')
    parser.add_argument('--optflow_model', type=str, default='RAFT', choices=['RAFT', 'FlowFormerPlusPlus'],)
    parser.add_argument('--add_depth_inputs', type=bool, default=True, help='Add monocular depth inputs')
    parser.add_argument('--target_pos_from_start', type=int, default=4,
                    help='1-based index of target frame within input window (KPT_test_mid).')
    
    
    # nll
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes (incl. background). Default: 4')
    parser.add_argument('--metric_fns', type=str, nargs='+', default=['iou', 'dice'], choices=['iou', 'dice'], 
                        help='List of metric functions. Default: iou, dice')
    '''
    # mse
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--metric_fns', type=str, nargs='+', default=[],
                    help='Heatmap task: keep empty.') 
    '''
    return parser

