import torch
from torch.autograd import Variable
import time
import sys
from train import build_string
from torch import nn
from utils import AverageMeter, calc_top1, calc_top5, calculate_accuracy, calc_total_top1,calculate_accuracy_video_level, probabily_accumolator,ee_calculate_accuracy_video_level
import math


def val_earlyexit_loss(output, target, opt):
    loss = 0
    sum_lossweights = 0
    criterion = nn.CrossEntropyLoss(reduction ='none')
    this_batch_size = target.size()[0]

    for exitnum in range(opt.num_exits - 1):
        if len(output[exitnum].shape) == 1:
            out = output[exitnum].view(1,output[exitnum].shape[0])
        else: 
            out = output[exitnum]
        current_loss = criterion(out, target)
        weighted_current_loss = (opt.earlyexit_lossweights[exitnum] * current_loss)
        loss += torch.sum(weighted_current_loss) / this_batch_size
        sum_lossweights += opt.earlyexit_lossweights[exitnum]
        opt.loss_exits[exitnum] = current_loss
    #
    current_loss = criterion(output[opt.num_exits - 1], target)
    weighted_current_loss = ((1.0 - sum_lossweights) * current_loss)
    loss += torch.sum(weighted_current_loss) / this_batch_size
    opt.loss_exits[opt.num_exits - 1] = current_loss
    return loss


def earlyexit_validate_loss(output, target, criterion, opt):
    # We need to go through each sample in the batch itself - in other words, we are
    # not doing batch processing for exit criteria - we do this as though it were batchsize of 1
    # but with a grouping of samples equal to the batch size.
    # Note that final group might not be a full batch - so determine actual size.
    binary_taken = []
    this_batch_size = target.size()[0]
    loss = val_earlyexit_loss(output, target, opt)  # updating  opt.loss_exits

    for batch_index in range(this_batch_size):
        earlyexit_taken = False
        # take the exit using CrossEntropyLoss as confidence measure (lower is more confident)
        for exitnum in range(opt.num_exits-1):
            if opt.loss_exits[exitnum][batch_index] < opt.earlyexit_thresholds[exitnum]:
                opt.exit_taken[exitnum] += 1
                earlyexit_taken = True
                binary_taken.append(1)
                break
            else:
                binary_taken.append(0)

        if not earlyexit_taken:
            exitnum = opt.num_exits - 1
            opt.exit_taken[exitnum] += 1

    return loss, binary_taken


def val_epoch(epoch, data_loader, model,opt,criterion):
    print('validation at epoch {}'.format(epoch))

    model.eval()
    batch_size = opt.batch_size
    batch_time = AverageMeter()
    data_time = AverageMeter()

    if (opt.earlyexit_lossweights):
        exits_top1 = [0] * opt.num_exits
        exits_top5 = [0] * opt.num_exits
        acc_top_1 = [0] * opt.num_exits
        acc_top_5 = [0] * opt.num_exits

        for i in range(opt.num_exits):
            exits_top1[i] = AverageMeter()
            exits_top5[i] = AverageMeter()
    end_time = time.time()
    #####################
    results_dict = {}
    probability_dict = {}
    #####################
    predictions=0
    exits={}
    exits['0']     = 0
    exits['final'] = 0
    if opt.num_exits == 3:
        exits['1']     = 0
    for batch, (inputs, targets,video_name) in enumerate(data_loader):
        temp_total_top1 = 0
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda()
            inputs = inputs.cuda()
        
        with torch.no_grad():
            inputs = Variable(inputs)
            targets = Variable(targets)
        
        
        if opt.model == "ee_resnext_eval" or opt.model == "ee_resnext3d_eval_reuse":
            outputs, num_exit = model(inputs)
            predictions = predictions + 1
            if opt.num_exits == 2:
                if num_exit == 0:
                    exits['0'] = exits['0'] + 1   
                    outputs = outputs.view(1,opt.n_classes)
                elif num_exit == 1:
                    exits['final'] = exits['final'] + 1
                
            else:
                if num_exit == 0:
                    exits['0'] = exits['0'] + 1   
                    outputs = outputs.view(1,opt.n_classes)
                elif num_exit == 1:
                    exits['1'] = exits['1'] + 1   
                    outputs = outputs.view(1,opt.n_classes)
                else:
                    exits['final'] = exits['final'] + 1

        else:
            outputs = model(inputs)
        if  opt.model == "ee_resnext" :
            loss, binary_taken = earlyexit_validate_loss(outputs, targets, criterion, opt)
            ee_calculate_accuracy_video_level(outputs, targets, video_name,results_dict,binary_taken,opt)
        else:
            probabily_accumolator(outputs, targets, video_name,probability_dict,num_classes=opt.n_classes)
            calculate_accuracy_video_level(outputs, targets, video_name,results_dict)
        batch_time.update(time.time() - end_time)
        end_time = time.time()
   # print(probability_dict)
    
    final_result_list=[]
    
    for (video,values) in probability_dict.items():
        #print(values)
        listOfProbabilities =  torch.FloatTensor(values['probabilities'])
        target_label = values['target_label']
        
        _, pred = listOfProbabilities.topk(1, 0, True)
        
        if pred.item() == target_label.item():
            final_result_list.append(1)
        else:   
            final_result_list.append(0)
    video_acc = sum(final_result_list) / len(final_result_list)
    if opt.num_exits == 2:
        print("Toatal number of predictions:",predictions,"earlyexit taken:",exits['0']," times")
        print("Top1 (accumolating clip level scores) is:",video_acc*100,"%")
    else:
        print("Toatal number of predictions:",predictions,"First earlyexit taken:",exits['0']," times","Second earlyexit taken:",exits['1']," times")
        print("Top1 (accumolating clip level scores) is:",video_acc*100,"%")

    
    avg_list = []
    for (video, values) in results_dict.items():
        # print(values)
        video_acc = sum(values) / len(values)
        # print("video:",video,", got :",video_acc*100,"%") 
        avg_list.append(video_acc * 100)

    video_acc = sum(avg_list) / len(avg_list)
    print("TOP1 (avareging clip level top1) :", video_acc)
    
   
###notice the changes
   # return losses.avg, binary_taken, corrects