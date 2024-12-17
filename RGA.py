import torch
import torch.nn.functional as F
from torch import nn
# import torch.tensor as tensor
"Embedding Graph Alignment Loss"
import ipdb
def PCC(m):
    '''Compute the Pearson’s correlation coefficients.'''
    fact = 1.0 / (m.size(1) - 1)
    m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t() 
    c = fact * m.matmul(mt).squeeze()    
    d = torch.diag(c, 0)
    std = torch.sqrt(d)
    c /= std[:, None]
    c /= std[None, :]
    return c



# def pdist(a,dim=2, p=2):
#     dist_matrix = torch.norm(a[:, None]-a, dim, p) / a.shape[1]
#     return dist_matrix 

def cosinematrix(A):
    prod = torch.mm(A, A.t())#分子
    norm = torch.norm(A,p=2,dim=1).unsqueeze(0)#分母
    cos = prod.div(torch.mm(norm.t(),norm))
    return cos
 
    
def RKdNode(features, f_labels, prototypes, p_labels, t=0.5):
    
    a_norm = features / features.norm(dim=1)[:, None]
    b_norm = prototypes / prototypes.norm(dim=1)[:, None]
    sim_matrix = torch.exp(torch.mm(a_norm, b_norm.transpose(0,1)) / t)
    c_norm = prototypes[f_labels] / prototypes[f_labels].norm(dim=1)[:, None]
    pos_sim = torch.exp(torch.diag(torch.mm(a_norm, c_norm.transpose(0,1))) / t)
    
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    
    return loss




def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


def RKdAngle(student, teacher):
    # N x C
    # N x N x C

    with torch.no_grad():
        td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
        norm_td = F.normalize(td, p=2, dim=2)
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

    sd = (student.unsqueeze(0) - student.unsqueeze(1))
    norm_sd = F.normalize(sd, p=2, dim=2)
    s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

    loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
    return loss



def RkdEdge(student, teacher):
    with torch.no_grad():
        t_d = pdist(teacher, squared=False)
        mean_td = t_d[t_d>0].mean()
        t_d = t_d / mean_td

    d = pdist(student, squared=False)
    mean_d = d[d>0].mean()
    d = d / mean_d

    loss = F.smooth_l1_loss(d, t_d, reduction='mean')
    return loss

class RGA_loss(torch.nn.Module):
    def __init__(self, node_weight=1, edge_weight=0.3, angle_weight=0.1, t=0.5):
        super(RGA_loss, self).__init__()
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.angle_weight = angle_weight
        self.t = t
        
    def forward(self, student, student_labels, teacher, teacher_label, mode='N'):
        REloss = RkdEdge(student, teacher[student_labels])
        RAloss = RKdAngle(student, teacher[student_labels])
        RNloss = RKdNode(student, student_labels, teacher, teacher_label, self.t)    
        if mode == 'N' or mode == 'N*':
            RGAloss = RNloss
        elif mode == 'E':
            RGAloss = REloss
        elif mode == 'A':
            RGAloss = RAloss
        elif mode == 'N+E':
            RGAloss = self.node_weight * RNloss + self.edge_weight * REloss#     ipdb.set_trace()
        elif mode == 'N+A':
            RGAloss = self.node_weight * RNloss + self.angle_weight * RAloss# 
        elif mode == 'A+E':
            RGAloss = self.angle_weight * RAloss + self.edge_weight * REloss# 
        elif mode == 'N+E+A':
            RGAloss = self.node_weight * RNloss + self.angle_weight * RAloss + self.edge_weight * REloss#         

        return RGAloss

    
# class RGA(torch.nn.Module):

#     def __init__(self, node_weight=1, edge_weight=0.3, t=0.5):
#        
        # super(RGA, self).__init__()

#         self.node_weight = node_weight
#         self.edge_weight = edge_weight
#         self.t = t

#     def forward(self, feats, feats_label, prototype, proto_label):

#         X = torch.cat((feats, prototype[feats_label]), 0)
# #         C = PCC(X) 
#         C = pdist(X)
#         n = C.shape[0]//2

#         Et = C[0:n, 0:n] # compute teacher edge matrix
#         Es = C[n:, n:] # compute student edge matrix
#         Nts= C[0:n, n:] # compute node matrix

        
#         loss_edge = torch.norm((Et-Es), 2) 
#         loss_node = PCLoss(feats, feats_label, prototype, proto_label, self.t)

#         RGA_loss = self.node_weight * loss_node + self.edge_weight * loss_edge 

#         return RGA_loss

