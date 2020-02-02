#! /home/ruopeiguo/anaconda3/envs python35
# -*- coding: utf-8 -*-    
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import time

import numpy as np
import sys
sys.path.append('./')

import torch
import torch.nn.functional as F
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable


from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.loss import TripletLoss
from reid.trainers_GSDRWLS import Trainer,  GSDRWLSTrainer
from reid.evaluators_GSDRWLS import Evaluator, CascadeEvaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler, RandomMultipleGallerySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.models.embedding_GSDRWLS import RandomWalkEmbed, EltwiseSubEmbed
from reid.models.multi_branch_GSDRWLS import  GSDRWLSNet
from reid.utils.tools import collect_env_info
from bisect import bisect_right




def get_data(name, split_id, data_dir, height, width, batch_size, num_instances,
             workers, combine_trainval):
    root = osp.join(data_dir, name) 
    dataset = datasets.create(name, root, split_id=split_id) 

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) 

    train_set = dataset.trainval if combine_trainval else dataset.train 
    num_classes = (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)

    train_transformer = T.Compose([
        T.RectScale(height, width),
        T.RandomSizedEarser(),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ]) 

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=RandomMultipleGallerySampler(train_set, num_instances), 
        pin_memory=True, drop_last=True) 

    val_loader = DataLoader(
        Preprocessor(dataset.val, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer), 
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, val_loader, test_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    
    
    log_name = 'test.log' if args.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    print('==========\nArgs:{}\n=========='.format(args))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))
    
    

    # Create data loaders
    assert args.num_instances > 1, "num_instances should be greater than 1" 
    assert args.batch_size % args.num_instances == 0, \
        'num_instances should divide batch_size' 
        #这里args.batch_size=64
    if args.height is None or args.width is None: 
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (256, 128)
            
    
    dataset, num_classes, train_loader, val_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, args.batch_size, args.num_instances, args.workers,
                 args.combine_trainval) 
        

    # Create model
    # Hacking here to let the classifier be the last feature embedding layer
    # Net structure: avgpool -> FC(1024) -> FC(args.features)
   
    base_model = models.create(args.arch, num_features=1024, cut_at_pooling=True,
                          dropout=args.dropout, num_classes=args.features) 
    
    grp_num = args.grp_num 
    embed_model = [EltwiseSubEmbed(nonlinearity='square',  use_batch_norm=True,
                            use_classifier=True, num_features=(2048 // grp_num),
                            num_classes=2).cuda() for i in range(grp_num)] 

   
    base_model = nn.DataParallel(base_model).cuda() 

    model = GSDRWLSNet(instances_num=args.num_instances,base_model=base_model, embed_model=embed_model, alpha=args.alpha, walkstep=args.walkstep) 
    

    if args.retrain:
        if args.evaluate_from:
            print('loading trained model...')
            checkpoint = load_checkpoint(args.evaluate_from)
            print(load_state_dict(checkpoint['state_dict']))
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('loading base part of pretrained model...')
            checkpoint = load_checkpoint(args.retrain)
            copy_state_dict(checkpoint['state_dict'], base_model, strip='base.module.', replace='module.')
            print('loading embed part of pretrained model...')
            if not args.onlybackbone:
                if grp_num > 1:
                    for i in range(grp_num):
                        copy_state_dict(checkpoint['state_dict'], embed_model[i], strip='embed_model.bn_'+str(i)+'.', replace='bn.')
                        copy_state_dict(checkpoint['state_dict'], embed_model[i], strip='embed_model.classifier_'+str(i)+'.', replace='classifier.')
                else:
                    copy_state_dict(checkpoint['state_dict'], embed_model[0], strip='embed_'+str(0)+'.', replace='')

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric) 

    # Load from checkpoint
    start_epoch = best_top1 = 0
    best_mAP = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))

    evaluator = CascadeEvaluator(
                            base_model,
                            embed_model,
                            embed_dist_fn=lambda x: F.softmax(x, dim=1).data[:, 0])

    if args.evaluate:
        metric.train(model, train_loader)
        if args.evaluate_from:
            print('loading trained model...')
            checkpoint = load_checkpoint(args.evaluate_from)
            model.load_state_dict(checkpoint['state_dict'])
        print("Test:")
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery, args.alpha, metric, rerank_topk=args.rerank, dataset=args.dataset,
                          save_dir=args.logs_dir)
                  
        return

    # Criterion 
    criterion = nn.CrossEntropyLoss().cuda()

    # base lr rate and embed lr rate
    new_params = [z for z in model.embed] 
    param_groups = [
        {'params': model.base.module.base.parameters(), 'lr_mult': 1.0}] + \
        [{'params': new_params[i].parameters(), 'lr_mult': 10.0} for i in range(grp_num)] 

    # Optimizer
    optimizer = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)     

    # Trainer
    trainer = GSDRWLSTrainer(model, criterion, args.alpha, grp_num, walkstep=args.walkstep, epsilon=args.epsilon) 
  

    # Schedule learning rate
    def adjust_lr(epoch):
        multiple_step = [40,70]
        if args.multi_ss:
            lr = args.lr * (0.1 ** bisect_right(multiple_step, epoch))
        else:          
            step_size = args.ss
            lr = args.lr * (0.1 ** (epoch // step_size))
        
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
        return lr
    

    # Start training
    for epoch in range(start_epoch, args.epochs):
        print('\n  grp_num: {:3d}  walkstep: {:s} \n'.
                  format( args.grp_num, args.walkstep))
        lr = adjust_lr(epoch)
        
        trainer.train(epoch, train_loader, optimizer, lr, warm_up=args.warm_up, warm_up_epoch =args.warm_up_ep)
        if not args.economic:
            top1, mAP = evaluator.evaluate(val_loader, dataset.val, dataset.val, args.alpha, second_stage=True, dataset=args.dataset)
            
            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_top1': best_top1,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint_rw_whole.pth.tar'))
        
            print('\n * Finished epoch {:3d}  mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))
            
        else:
            if (epoch < args.recordpoint)and(epoch%args.print_freq == 0):
                top1, mAP = evaluator.evaluate(val_loader, dataset.val, dataset.val, args.alpha, second_stage=True, dataset=args.dataset)
                
                is_best = mAP > best_mAP
                best_mAP = max(mAP, best_mAP)
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_top1': best_top1,
                }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint_rw_whole.pth.tar'))
        
                print('\n * Finished epoch {:3d}  mAP: {:5.1%}  best: {:5.1%}{}\n'.
                      format(epoch, mAP, best_mAP, ' *' if is_best else ''))
            
            if (epoch >= args.recordpoint):
                top1, mAP = evaluator.evaluate(val_loader, dataset.val, dataset.val, args.alpha, second_stage=True, dataset=args.dataset)
                
                is_best = mAP > best_mAP
                best_mAP = max(mAP, best_mAP)
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_top1': best_top1,
                }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint_rw_whole.pth.tar'))
        
                print('\n * Finished epoch {:3d}  mAP: {:5.1%}  best: {:5.1%}{}\n'.
                      format(epoch, mAP, best_mAP, ' *' if is_best else ''))

            
    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    metric.train(model, train_loader)
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, args.alpha, metric, rerank_topk=args.rerank, dataset=args.dataset,
                      visrank=args.visrank, visrank_topk=args.topk, save_dir=args.logs_dir, zhongrerank=args.zhongrerank)
                    #加上rank结果的可视化以及zhong's的reranking方法


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--grp-num', type=int, default=1)
    parser.add_argument('--rerank', type=int, default=75)
    parser.add_argument( '--walkstep', type=str, default='combine')
    parser.add_argument('--warm-up', action='store_true') 
    parser.add_argument('--warm-up-ep', type=int, default=20) 
    # loss
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    parser.add_argument('-e','--epsilon', type=float, default=0.1,
                        help="epsilon of the label-smooth, default: 0.1")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.0002,
                        help="learning rate of all parameters")
    parser.add_argument('--ss', type=int, default=40,
                        help="step size for adjusting learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--multi-ss', action='store_true',
                        help="multiple step size for adjusting learning rate")
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--retrain', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--evaluate-from', type=str, default='', metavar='PATH')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10) #测试的间隔
    parser.add_argument('--economic', action='store_true') #开启经济模式，每个多个epoch测试一次并保存模型
    parser.add_argument('--recordpoint', type=int, default=50) #从该epoch开始，每次进行模型测试和保存
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--onlybackbone', action='store_true')
    
    main(parser.parse_args())
