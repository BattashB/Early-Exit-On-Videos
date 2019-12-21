import torch
from torch.autograd import Variable
import time
import sys
from train import build_string
from torch import nn
from utils import AverageMeter, calc_top1,calc_top5, calculate_accuracy,calc_total_top1
import math


def val_earlyexit_loss(output, target, opt):
    
    loss = 0
    sum_lossweights = 0
    criterion = nn.CrossEntropyLoss(reduce=False)
    this_batch_size = target.size()[0]
    for exitnum in range(opt.num_exits - 1):

        
        current_loss = criterion(output[exitnum], target)
        if  not opt.evaluate_clip_level:
            weighted_current_loss = (opt.earlyexit_lossweights[exitnum] * current_loss )
            loss += torch.sum(weighted_current_loss)/this_batch_size
            sum_lossweights += opt.earlyexit_lossweights[exitnum]
        opt.loss_exits[exitnum] = current_loss
        
    current_loss = criterion(output[opt.num_exits - 1], target)
    if  not opt.evaluate_clip_level:
        weighted_current_loss = ((1.0 - sum_lossweights) * current_loss)
        loss += torch.sum(weighted_current_loss)/this_batch_size
    opt.loss_exits[opt.num_exits - 1] = current_loss
    return loss


def earlyexit_validate_loss(output, target, criterion, opt):
    # We need to go through each sample in the batch itself - in other words, we are
    # not doing batch processing for exit criteria - we do this as though it were batchsize of 1
    # but with a grouping of samples equal to the batch size. 
    # Note that final group might not be a full batch - so determine actual size.
    binary_taken = []
    this_batch_size = target.size()[0]
    loss = val_earlyexit_loss(output, target, opt) # updating  opt.loss_exits 
    for batch_index in range(this_batch_size):
        earlyexit_taken = False
        # take the exit using CrossEntropyLoss as confidence measure (lower is more confident)
        for exitnum in range(opt.num_exits - 1):
            if opt.loss_exits[exitnum][batch_index] < opt.earlyexit_thresholds[exitnum] and not earlyexit_taken: ## if an exit already taken, I still want to fill the binary taken list
                earlyexit_taken = True    
                binary_taken.append(1)
            else:
                binary_taken.append(0)
                    
           

    return loss,binary_taken




def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()
    batch_size = opt.batch_size
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    total_top1 = AverageMeter()
    if "ee_" in opt.model:  
        exits_top1 = [0] * opt.num_exits
        exits_top5 = [0] * opt.num_exits
        acc_top_1  = [0] * opt.num_exits
        acc_top_5  = [0] * opt.num_exits
        
        for i in range(opt.num_exits):
            exits_top1[i] = AverageMeter()
            exits_top5[i] = AverageMeter()
    end_time = time.time()
    for batch, (inputs, targets) in enumerate(data_loader):
        temp_total_top1 = 0
        
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda()
            inputs = inputs.cuda()
        with torch.no_grad():
            inputs = Variable(inputs)
            targets = Variable(targets)

        if "_th" in opt.model:
            outputs,threshold = model(inputs)
        else:
            outputs = model(inputs)
            
        if "ee_" in opt.model:
            ####################calc loss + accuracies################
            loss,binary_taken = earlyexit_validate_loss(outputs, targets, criterion, opt)

            
            corrects = []
            for i in range(opt.num_exits):
                acc_top_1[i],correct = calc_top1(outputs[i], targets,opt)
                acc_top_5[i] = calc_top5(outputs[i], targets, opt)
                if i == 0:
                     corrects = correct.t()
                if i > 0:
                    corrects = torch.cat((corrects,correct.t()),1)

            for m in range(opt.batch_size):
                if(corrects.data[m,0] > corrects.data[m,1]):
                    opt.exit0correctexit1no += 1
            temp_total_top1 = calc_total_top1(outputs,targets,opt,binary_taken,corrects)
            ############################################################
            
            ##########update loss + accuracies#####################
            if  not opt.evaluate_clip_level:
                losses.update(loss.item(), inputs.size(0))
            total_top1.update(temp_total_top1, inputs.size(0))
            for i in range(opt.num_exits):
                exits_top1[i].update(acc_top_1[i], inputs.size(0))
                exits_top5[i].update(acc_top_5[i], inputs.size(0))
            #######################################################
        
        else:
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))
        

        batch_time.update(time.time() - end_time)
        end_time = time.time()
           
        if "ee_" in opt.model:
            """
            batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i + 1),
                'loss': losses.val,
                'acc':'0',
                #    'top1': classerr.value(1),
                #             #    'top5': classerr.value(5),
                'lr': optimizer.param_groups[0]['lr']
            })
            """
            top1_string = build_string(opt,'Top1','top_1')
            top5_string = build_string(opt,'Top5','top_5')

            print('Epoch: [{0}][{1}/{2}]\t'.format(epoch,batch + 1,len(data_loader)), end =" ")
            print('Total_top1 {acc_top.val:.4f} ({acc_top.avg:.4f})\t'.format(acc_top=total_top1), end =" ")
           # if "_th" in opt.model:
           #     print('Threshold : {0} \t'.format(threshold[0]), end=" ")
           # print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(batch_time=batch_time), end =" ")
           # print('Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(data_time=data_time), end =" ")
            print('Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss=losses), end =" ")
            print(top1_string.format(top_1 = exits_top1),end =" ")
            print(top5_string.format(top_5 = exits_top5))
                
        else:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch,
                      batch + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      acc=accuracies))

                      
    if opt.evaluate_clip_level:
        if "ee_" in opt.model:
            opt.avg += total_top1.avg
        else:
            opt.avg += accuracies.avg

                
    if "ee_" in opt.model:
        print("\n")
        for i in range(opt.num_exits-1):
            print("Exit: ", i,"  has been taken:",opt.exit_taken[i],"times, so far")
            print("TP in exit:", i, " is:", opt.true_positive[i])
            #opt.true_positive[i] = 0
            print("FP  in exit:", i, "is:", opt.false_positive[i])
            #opt.false_positive[i] = 0
            print("TN  in exit:", i, "is:", opt.true_negative[i])
            #opt.true_negative[i] = 0
        
        print("Final Exit been taken:",opt.exit_taken[i+1])

        

    return losses.avg