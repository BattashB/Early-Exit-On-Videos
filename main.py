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
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger
from train import train_epoch
from train_selflearn import train_epoch as train_epoch_selflearn

from validation import val_epoch


import test 


if __name__ == '__main__':
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


    ######################early exit add############
    if opt.earlyexit_thresholds:
        opt.num_exits = len(opt.earlyexit_lossweights) + 1
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
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Num of params:", pytorch_total_params)
    print("Num of trainable params:", trainable)
    if opt.earlyexit_thresholds:
        exit_params = 0
        if opt.exit1:
            exit_params = exit_params + sum(p.numel() for p in model.module.exit1.parameters())
        if opt.exit2:
            exit_params = exit_params + sum(p.numel() for p in model.module.exit2.parameters())
        print("Added params:", exit_params)
    
    
    #################################################
    
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.frame_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.frame_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.frame_size, crop_positions=['c'])
        spatial_transform = Compose([
            crop_method,
            RandomHorizontalFlip(),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = TemporalRandomCrop(opt.frames_sequence)
        target_transform = ClassLabel()
        training_data = get_training_set(opt, spatial_transform,
                                         temporal_transform, target_transform)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True,
            drop_last=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        if opt.optimizer == 'adam':
            optimizer = optim.Adam(parameters, lr=0.1, betas=(0.99, 0.999), eps=1e-09, weight_decay=0, amsgrad=False)
        else:
            optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                dampening=dampening,
                weight_decay=opt.weight_decay,
                nesterov=opt.nesterov)
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=opt.lr_patience)
    if not opt.no_val:
        spatial_transform = Compose([
            Scale(opt.frame_size),
            CenterCrop(opt.frame_size),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(opt.frames_sequence)
        target_transform = ClassLabel()
        validation_data = get_validation_set(
            opt, spatial_transform, temporal_transform, target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True,
            drop_last=True)
            

        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])

    if opt.resume_path:
        checkpoint = torch.load(opt.resume_path)

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    opt.avg = 0
    if "_th" in opt.model:
        th_loss = torch.nn.MSELoss()      
        for i in range(opt.begin_epoch, 10):
            init_th_process(i, train_loader, model,th_loss , optimizer, opt,train_logger, train_batch_logger)
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            if  "selflearn" in opt.model :
                train_epoch_selflearn(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)
            else:
                train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)
        if not opt.no_val:
            validation_loss = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)
                  
        if not opt.no_train and not opt.no_val:
            scheduler.step(validation_loss)
            
    if opt.evaluate_clip_level:
        score = opt.avg/opt.n_epochs
        print("Validation Score = ",score)

    if opt.test:
        spatial_transform = Compose([
            Scale(int(opt.frame_size / opt.scale_in_test)),
            CornerCrop(opt.frame_size, opt.crop_position_in_test),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(opt.frames_sequence)
        target_transform = VideoID()

        test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                 target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test.test(test_loader, model, opt, test_data.class_names)
