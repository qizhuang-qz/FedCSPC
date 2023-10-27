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
from torchvision import datasets

import ipdb


import torch
import torch.nn.functional as F

def supervised_contrastive_loss(features, labels, temperature=0.07):
    """
    带标签的对比损失函数的实现。

    参数：
    - features：形状为 (batch_size, embedding_size) 的张量，表示输入的特征向量。
    - labels：形状为 (batch_size,) 的张量，表示输入的样本标签。
    - temperature：温度参数。

    返回值：
    - loss：对比损失。
    """

    # 将特征向量 L2 归一化
    features = F.normalize(features, dim=1)

    # 对所有样本计算相似度矩阵
    similarity_matrix = torch.matmul(features, features.T) / temperature

    # 将对角线的值排除在外，避免同一样本与自身比较
    mask = torch.eye(labels.size(0), dtype=torch.bool).cuda()
    similarity_matrix = similarity_matrix.masked_fill(mask, 1)
    ipdb.set_trace()
    # 计算每个样本的正样本对的对比损失和负样本对的对比损失
    pos_pairs_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).bool()
    neg_pairs_mask = ~pos_pairs_mask

    # 计算正样本对的对比损失
    pos_pairs_similarity = similarity_matrix[pos_pairs_mask]
    pos_pairs_loss = -torch.log(pos_pairs_similarity / torch.sum(similarity_matrix))

    # 计算负样本对的对比损失
    neg_pairs_similarity = similarity_matrix[neg_pairs_mask].view(labels.size(0), -1)
    neg_pairs_loss = -torch.log(torch.sum(torch.exp(neg_pairs_similarity), dim=1) / torch.sum(neg_pairs_mask, dim=1))

    # 对所有样本的对比损失取平均
    loss = torch.mean(torch.cat([pos_pairs_loss, neg_pairs_loss]))

    return loss




    
class SupervisedContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x, y):
        # x: the feature representations of the samples
        # y: the ground truth labels

        # normalize the feature vectors
        x = F.normalize(x, dim=1)

        # compute the similarity matrix
        sim_matrix = torch.matmul(x, x.t()) / self.temperature

        # generate the mask for positive and negative pairs
        mask = torch.eq(y.unsqueeze(0), y.unsqueeze(1)).float()
        mask = mask / mask.sum(dim=1, keepdim=True)

        # calculate the contrastive loss
        loss = (-torch.log_softmax(sim_matrix, dim=1) * mask).sum(dim=1).mean()

        return loss

    
def nt_xent(x1, x2, t=0.07):
    """Contrastive loss objective function"""
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)
    batch_size = x1.size(0)
    out = torch.cat([x1, x2], dim=0)
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / t)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
    pos_sim = torch.exp(torch.sum(x1 * x2, dim=-1) / t)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss

    
def PCLoss(features, f_labels, prototypes, p_labels, t=0.5):
    
    a_norm = features / features.norm(dim=1)[:, None]
    b_norm = prototypes / prototypes.norm(dim=1)[:, None]
    sim_matrix = torch.exp(torch.mm(a_norm, b_norm.transpose(0,1)) / t)
    
    pos_sim = torch.exp(torch.diag(torch.mm(a_norm, b_norm[f_labels].transpose(0,1))) / t)
    
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    
    return loss

def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).to(logits.device)
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)

    logits = torch.gather(logits, 1, nt_positions)

    return logits

class NTD_Loss(nn.Module):
    """Not-true Distillation Loss"""

    def __init__(self, num_classes=10, tau=3, lamb=1):
        super(NTD_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.num_classes = num_classes
        self.tau = tau
        self.beta = lamb

    def forward(self, logits, targets, dg_logits):
        ce_loss = self.CE(logits, targets)
        ntd_loss = self._ntd_loss(logits, dg_logits, targets)

        loss = ce_loss + self.beta * ntd_loss

        return loss

    def _ntd_loss(self, logits, dg_logits, targets):
        """Not-tue Distillation Loss"""

        # Get smoothed local model prediction
        logits = refine_as_not_true(logits, targets, self.num_classes)
        pred_probs = F.log_softmax(logits / self.tau, dim=1)

        # Get smoothed global model prediction
        with torch.no_grad():
            dg_logits = refine_as_not_true(dg_logits, targets, self.num_classes)
            dg_probs = torch.softmax(dg_logits / self.tau, dim=1)

        loss = (self.tau ** 2) * self.KLDiv(pred_probs, dg_probs)

        return loss

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, target, weights):
        # logits: [N, C, H, W], target: [N, H, W]
        # loss = sum(-y_i * log(c_i))

        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # [N, C, HW]
            logits = logits.transpose(1, 2)  # [N, HW, C]
            logits = logits.contiguous().view(-1, logits.size(2))  # [NHW, C]
        target = target.view(-1, 1)  # [NHW，1]

        logits = F.log_softmax(logits, 1)

        # import ipdb; ipdb.set_trace()

        logits = logits.gather(1, target).reshape(-1)  # [NHW, 1]

        loss = -1 * (torch.mul(logits, weights))

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return torch.sum(loss, 0)
        
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        dist_pos = torch.norm(anchor - positive, 2, dim=1)
        dist_neg = torch.norm(anchor - negative, 2, dim=1)
        loss = torch.mean(torch.clamp(dist_pos - dist_neg + self.margin, min=0.0))
        return loss
