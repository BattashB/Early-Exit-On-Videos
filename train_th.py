import torch
from torch.autograd import Variable
import time
import os
import sys
from torch import nn
import numpy as np
import loss as loss_file
from utils import AverageMeter, calc_top1, calc_top5, calculate_accuracy


def build_string(opt, prefix, var_name):
    string = prefix
    string = string + ' ['
    for i in range(opt.num_exits):
        string = string + '{'
        if (i == opt.num_exits - 1):
            string = string + str(var_name) + '[' + str(i) + ']' + '.' + 'val:.4f}] '
        else:
            string = string + str(var_name) + '[' + str(i) + ']' + '.' + 'val:.4f}, '
    string = string + '('
    for i in range(opt.num_exits):
        string = string + '{'
        if (i == opt.num_exits - 1):
            string = string + str(var_name) + '[' + str(i) + ']' + '.' + 'avg:.4f})\t'
        else:
            string = string + str(var_name) + '[' + str(i) + ']' + '.' + 'avg:.4f}, '
    return string
def train_th_earlyexit_loss(output, target, opt):
    
    loss = 0
    sum_lossweights = 0
    criterion = nn.CrossEntropyLoss(reduce=False)
    this_batch_size = target.size()[0]
    for exitnum in range(opt.num_exits - 1):

        
        current_loss = criterion(output[exitnum], target)
        weighted_current_loss = (opt.earlyexit_lossweights[exitnum] * current_loss )
        loss += torch.sum(weighted_current_loss)/this_batch_size
        sum_lossweights += opt.earlyexit_lossweights[exitnum]
        opt.loss_exits[exitnum] = current_loss
        
    current_loss = criterion(output[opt.num_exits - 1], target)
    weighted_current_loss = ((1.0 - sum_lossweights) * current_loss)
    loss += torch.sum(weighted_current_loss)/this_batch_size
    opt.loss_exits[opt.num_exits - 1] = current_loss
    return loss


def earlyexit_train_th_loss(output, target, criterion, opt,thresholds):
    # We need to go through each sample in the batch itself - in other words, we are
    # not doing batch processing for exit criteria - we do this as though it were batchsize of 1
    # but with a grouping of samples equal to the batch size.
    # Note that final group might not be a full batch - so determine actual size.
    binary_taken = []
    this_batch_size = target.size()[0]
    loss = train_th_earlyexit_loss(output, target, opt) # updating  opt.loss_exits 
    for batch_index in range(this_batch_size):
        earlyexit_taken = False
        # take the exit using CrossEntropyLoss as confidence measure (lower is more confident)
        for exitnum in range(opt.num_exits - 1):
            if opt.loss_exits[exitnum][batch_index] < thresholds[exitnum][batch_index]:
                opt.exit_taken[exitnum] += 1
                earlyexit_taken = True    
                binary_taken.append(1)
                break
            else:
                binary_taken.append(0)
                    
            
        if not earlyexit_taken:
            exitnum = opt.num_exits - 1
            opt.exit_taken[exitnum] += 1
    return loss,binary_taken




def train_epoch_th(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train threshold at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()


    end_time = time.time()
    len_data_loader =len(data_loader)
    for batch, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda()
            inputs = Variable(inputs).cuda()  ##changed on 2.7.19 17:43
        targets = Variable(targets)

        outputs, thresholds = model(inputs)
        
        print("thresholds: ",thresholds)
        loss, binary_taken = earlyexit_train_th_loss(outputs, targets, criterion, opt,thresholds)

        ####################calc loss + accuracies################
        acc_top_1 = [0] * opt.num_exits
        corrects = []
        for i in range(opt.num_exits-1):
            acc_top_1[i],correct = calc_top1(outputs[i], targets,opt)
            if i == 0:
                 corrects = correct.t()
            if i > 0:
                corrects = torch.cat((corrects,correct.t()),1)
        print("corrects:",corrects.t())
        print("binary_taken:",binary_taken)
        pred = Variable(torch.cuda.ByteTensor(binary_taken))
        targets = Variable(corrects)
        loss = loss_file.dice_loss(pred,targets)
        if loss == 0:
            continue
        loss = Variable(loss,requires_grad = True)


        ##########update loss + accuracies#####################
        losses.update(loss.item(), inputs.size(0))
        """for i in range(opt.num_exits):############we will need this once we will have more exits 
            exits_top1[i].update(acc_top_1[i], inputs.size(0))
            exits_top5[i].update(acc_top_5[i], inputs.size(0))"""
        ################################################

        optimizer.zero_grad()
        loss.backward()
        if opt.optimizer == 'adam':
            optimizer.step()
        if opt.optimizer == 'SGD':
            optimizer.step()
        elif opt.optimizer == 'noam':
            optimizer.step_and_update_lr()

        if opt.model == "ee_model_th":
            dice_loss = diceloss()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': batch + 1,
            'iter': (epoch - 1) * len_data_loader + (i + 1),
            'loss': losses.val,
            'acc': '0',
            'lr': optimizer.param_groups[0]['lr']
        })

      

        print('Epoch: [{0}][{1}/{2}]\t'.format(epoch, batch + 1, len_data_loader), end=" ")
        print('Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss=losses))


    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': 0,
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
