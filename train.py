import torch
from torch.autograd import Variable
import time
import os
import sys
import numpy as np
from utils import AverageMeter, calc_top1, calc_top5,calculate_accuracy

def build_string(opt,prefix,var_name):
    string = prefix
    string = string + ' ['
    for i in range(opt.num_exits):
        string = string + '{'
        if (i == opt.num_exits -1):
            string = string + str(var_name) + '[' + str(i) + ']' + '.' + 'val:.4f}] '
        else:
            string = string + str(var_name) + '[' + str(i) + ']' + '.' + 'val:.4f}, '
    string = string + '('
    for i in range(opt.num_exits):
        string = string + '{'
        if (i == opt.num_exits -1):
            string = string + str(var_name) + '[' + str(i) + ']' + '.' + 'avg:.4f})\t'
        else:
            string = string + str(var_name) + '[' + str(i) + ']' + '.' + 'avg:.4f}, '
    return string

    
def earlyexit_loss(output, target, criterion, opt):
    loss = 0
    sum_lossweights = 0
    #opt.num_exits = 2
    for exitnum in range(opt.num_exits - 1):
        #print("targets shape:",target.shape)
        #print("output shape:",output[0].shape)

        current_loss = (opt.earlyexit_lossweights[exitnum] * criterion(output[exitnum], target))
        loss += current_loss
        sum_lossweights += opt.earlyexit_lossweights[exitnum]
        opt.loss_exits[exitnum] = current_loss
      #  opt.exiterrors[exitnum].add(output[exitnum].data, target)

    current_loss = (1.0 - sum_lossweights) * criterion(output[opt.num_exits - 1], target)
    loss += current_loss
    opt.loss_exits[exitnum] = current_loss
   # opt.exiterrors[opt.num_exits - 1].add(output[opt.num_exits - 1].data, target)
    return loss

    
def ee_train_method(batch,epoch,opt,inputs,optimizer, outputs, targets,criterion,losses,end_time,batch_time,data_time,batch_logger,len_data_loader,exits_top1,exits_top5):        

    acc_top_1 = [0] * opt.num_exits
    acc_top_5 = [0] * opt.num_exits
    
     



    ####################calc loss + accuracies################
    loss = earlyexit_loss(outputs, targets, criterion, opt)
    
    for i in range(opt.num_exits):
        acc_top_1[i],correct = calc_top1(outputs[i], targets,opt)
        acc_top_5[i] = calc_top5(outputs[i], targets, opt)
    ############################################################


    ##########update loss + accuracies#####################
    losses.update(loss.item(), inputs.size(0))
    for i in range(opt.num_exits):
        exits_top1[i].update(acc_top_1[i], inputs.size(0))
        exits_top5[i].update(acc_top_5[i], inputs.size(0))
    ################################################
    

    
    
    optimizer.zero_grad()
    loss.backward()
    if opt.optimizer == 'adam':
        optimizer.step()
    if opt.optimizer == 'SGD':
        optimizer.step()
    elif opt.optimizer == 'noam':
        optimizer.step_and_update_lr()

    batch_time.update(time.time() - end_time)
    end_time = time.time()
       
    batch_logger.log({
        'epoch': epoch,
        'batch': batch + 1,
        'iter': (epoch - 1) * len_data_loader + (i + 1),
        'loss': losses.val,
        'acc':'0',
        'lr': optimizer.param_groups[0]['lr']
    })

    top1_string = build_string(opt,'Top1','top_1')

    top5_string = build_string(opt,'Top5','top_5')

    print('Epoch: [{0}][{1}/{2}]\t'.format(epoch,batch + 1,len_data_loader), end =" ")
    print('Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss=losses), end =" ")
    print(top1_string.format(top_1 = exits_top1),end =" ")
    print(top5_string.format(top_5 = exits_top5))
    

def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    if "ee_" in opt.model:
        exits_top1 = [0] * opt.num_exits
        exits_top5 = [0] * opt.num_exits
        for i in range(opt.num_exits):
            exits_top1[i] = AverageMeter()
            exits_top5[i] = AverageMeter()
    
    end_time = time.time()
    len_data_loader =len(data_loader)
    for batch, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda()
            inputs = Variable(inputs).cuda() ##changed on 2.7.19 17:43
        targets = Variable(targets)
        if "_th" in opt.model:
            outputs,threshold = model(inputs)
        else:
            outputs = model(inputs)
        #print("outputs.shape",outputs[0].shape)
        if "ee_" in opt.model:
            ee_train_method(batch,epoch,opt,inputs,optimizer, outputs, targets,criterion,losses,end_time,batch_time,data_time,batch_logger,len_data_loader,exits_top1,exits_top5)
        else:
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.data[0], inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            batch_logger.log({
                'epoch': epoch,
                'batch': batch + 1,
                'iter': (epoch - 1) * len(data_loader) + (batch + 1),
                'loss': losses.val,
                'acc': accuracies.val,
                'lr': optimizer.param_groups[0]['lr']
            })

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

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[0]['lr']
    })

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
