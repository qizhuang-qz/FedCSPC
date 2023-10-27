import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import random
from sklearn.metrics import confusion_matrix
import torch.optim as optim
# from model import *
from datasets import CIFAR10_truncated, CIFAR100_truncated, ImageFolder_custom, MNIST_truncated, FashionMNIST_truncated, TinyImageNet_load, Vireo172_truncated, Food101_truncated
import ipdb
import copy
from RGA import *
from loss import *
from utils import *

def exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay, decay_rate):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    # 每四次epoch调整一下lr，将lr减半
    lr = init_lr * (decay_rate ** (epoch // lr_decay))  # *是乘法，**是乘方，/是浮点除法，//是整数除法，%是取余数

    if epoch % lr_decay == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # 返回改变了学习率的optimizer
    return optimizer


def retrain_cls_final(global_dnn, prototypes, proto_labels, client_ids, n_classes, args, round, device):
    
    if round <= 5:
        init_lr = 1e-1
    elif round <= 40:
        init_lr = 1e-2
    else:
        init_lr = 1e-3

    lr_decay = 30
    decay_rate = 0.1
    batch_size = 100
    global_dnn.to(device)
    cuda = 1
    prototypes = prototypes.cpu().numpy()
    proto_labels = proto_labels.cpu().numpy()
    client_ids = client_ids.cpu().numpy()
    
    kwargs = {'num_workers': 2, 'pin_memory': True}


    dataset_c = Data_for_Retraining_final(prototypes, proto_labels, client_ids)
    data_loader = torch.utils.data.DataLoader(dataset_c, batch_size=batch_size, shuffle=True, **kwargs)    
        
#     prototypes = prototypes.to(device)
#     proto_labels = proto_labels.to(device)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, global_dnn.parameters()), lr=init_lr, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)
    
    SCLoss = SupervisedContrastiveLoss()  
    ACLoss = TripletLoss()
    
    idx_list = np.array(np.arange(len(proto_labels)))
#     ipdb.set_trace()
    with torch.no_grad():
        prototypes = torch.tensor(prototypes).to(device)
        proto_labels = torch.tensor(proto_labels).to(device)
        h2, out = global_dnn(prototypes)
        pred_label = torch.argmax(out.data, 1)
        total = prototypes.data.size()[0]
        
        correct = (pred_label == proto_labels.data).sum().item()
        print('before', correct)
 
    print('proto_labels', proto_labels.shape)
    

    for epoch in range(100):
        optimizer = exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay, decay_rate)
#         random.shuffle(idx_list)
#         batch_size = 100
        epoch_loss_collector=[] 
        for batch_idx, (x, posi_x, nega_x, target) in enumerate(data_loader):
#         for i in range(len(proto_labels)//batch_size):
            x, posi_x, nega_x, target = x.to(device), posi_x.to(device), nega_x.to(device), target.reshape(-1).to(device)
        
            epoch_loss_collector = []

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            feats, out = global_dnn(x)

            if args.re_version == 'v1':
                loss = criterion(out, target) 
            elif args.re_version == 'v2':
                
                mix_posi = (posi_x - x) * args.posi_lambda + posi_x
                mix_nega = (nega_x - x) * args.nega_lambda + x
                
                feats_posi, _ = global_dnn(mix_posi)
                feats_nega, _ = global_dnn(mix_nega)
                
                loss1 = criterion(out, target)
                loss2 = args.re_mu * SCLoss(feats, target)
                loss3 = ACLoss(feats, feats_posi, feats_nega)
                                
                mixed_x, y_a, y_b, lam = mixup_data(x, target, args.posi_lambda)

                _, out_mix = global_dnn(mixed_x)

#                 loss_mix = mixup_criterion(criterion, out_mix, y_a, y_b, lam)
                
                loss = loss1 + loss2 + args.re_beta * loss3 # + 0.1 * loss_mix
              
            epoch_loss_collector.append(loss.data)

            loss.backward()
            optimizer.step()
        print(epoch, sum(epoch_loss_collector)/len(epoch_loss_collector))
    
    with torch.no_grad():
        feats, out = global_dnn(prototypes)
      
        pred_label = torch.argmax(out.data, 1)
        total = prototypes.data.size()[0]
        correct = (pred_label == proto_labels.data).sum().item()
        correct_id = torch.nonzero(pred_label == proto_labels.data).reshape(-1)
        
        protos, labels = gen_proto_global(feats[correct_id], proto_labels[correct_id], n_classes) 
        print('after', correct)

    return global_dnn, protos, labels



