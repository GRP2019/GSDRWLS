from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable
import torch.nn as nn

from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss
from .utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self,model,criterion,alpha,grp_num,num_classes=0,num_instances=4,walkstep='combine', epsilon=0.1):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon
        self.num_classes = num_classes
        self.num_instances = num_instances
        self.walkstep = walkstep
        self.alpha = alpha
        self.grp_num = grp_num       
        self.total_grp_num = self.grp_num*self.grp_num        
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def train(self, epoch, data_loader, optimizer, base_lr, warm_up=True, print_freq=1, warm_up_epoch = 20):
        self.model.train() 

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        
        warm_up_ep = warm_up_epoch
        warm_iters = float(len(data_loader) * warm_up_ep)

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs) 
            loss, prec1 = self._forward(inputs, targets) 

            losses.update(loss.item(), targets.size(0)) 
            precisions.update(prec1, targets.size(0)) 
            if warm_up: 
                if epoch <= (warm_up_ep):
                    lr = (base_lr / warm_iters) + (epoch*len(data_loader) +(i+1))*(base_lr / warm_iters)
                    print(lr)
                    for g in optimizer.param_groups:
                        g['lr'] = lr * g.get('lr_mult', 1)
                else:
                    lr = base_lr
                    for g in optimizer.param_groups:
                        g['lr'] = lr * g.get('lr_mult', 1)

            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 

            batch_time.update(time.time() - end) 
            end = time.time() 

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        num_classes = self.num_classes
        Y = torch.zeros(targets.size(0), num_classes)
        for i in range(targets.size(0)):
            Y[i][targets[i].cpu().data] = 1.0
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, OIMLoss):
            loss, outputs = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        elif isinstance(self.criterion, SupervisedClusteringLoss):
            loss = self.criterion(outputs, Y)
            prec = 0.0
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec


class GSDRWLSTrainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        
        pairwise_targets = Variable(torch.zeros(targets.size(0),targets.size(0)).cuda()) 
        targets = targets.view(-1)    
        for i in range(targets.size(0)): 
            pairwise_targets[i] = (targets[i] == targets).long()
        pairwise_targets = pairwise_targets.view(-1).long() 

       
        if self.walkstep == 'combine':
            pairwise_targets = pairwise_targets.repeat(2 * self.total_grp_num) 
        else:
            pairwise_targets = pairwise_targets.repeat(self.total_grp_num) 
             
        outputs = self.model(*inputs) 
        
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):

            binary_targets = torch.zeros(outputs.size()).scatter_(1, pairwise_targets.unsqueeze(1).data.cpu(), 1)
            binary_targets = binary_targets.cuda()
            binary_targets = (1 - self.epsilon) * binary_targets + self.epsilon / (targets.size(0) / self.num_instances)
            log_probs = self.logsoftmax(outputs.cuda())
            loss = (- binary_targets * log_probs).mean(0).sum()
            prec, = accuracy(outputs.data.cuda(), pairwise_targets.data.cuda())
            prec = prec[0]
           
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec
