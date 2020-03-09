import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset_video_level import get_training_set, get_validation_set, get_test_set
from utils import Logger
from video_level_accuracy import val_epoch
#import test

import test 


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False

    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    ######################early exit add############
    if "ee_" in opt.model:
        opt.num_exits = len(opt.earlyexit_thresholds) + 1
        opt.loss_exits = [0] * opt.num_exits
        opt.exit_taken = [0] * opt.num_exits
        opt.true_positive = [0] * opt.num_exits
        opt.false_positive = [0] * opt.num_exits
        opt.true_negative = [0] * opt.num_exits 
        opt.false_negative = [0] * opt.num_exits
        opt.exit0correctexit1no = 0
    ################################################
    
    

    model, parameters = generate_model(opt)
    

    
    
    #####How many parameters in the model############
   # print(model)
    print ('Number of GPUs:', torch.cuda.device_count())

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("num of params::", pytorch_total_params)
    print("num of trainable params::", trainable)
    #################################################
    

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    

    spatial_transform = Compose([
        Scale(int(opt.frame_size / opt.scale_in_test)),
        CornerCrop(opt.frame_size, opt.crop_position_in_test),
        ToTensor(opt.norm_value), norm_method
    ])
    temporal_transform = LoopPadding(opt.frames_sequence)
    target_transform = ClassLabel()

    test_data = get_test_set(opt, spatial_transform, temporal_transform,
                             target_transform)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True)
        
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        validation_loss = val_epoch(i, test_loader, model,opt,criterion)
