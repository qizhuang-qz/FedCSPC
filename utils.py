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
from kmeans import *
from loss import *
from clustering import *
from sklearn import metrics

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def get_updateModel_before(model, cal_dnn):
    model_dict = model.state_dict()
    dnn_dict = cal_dnn.state_dict()
#     for k, v in dnn_dict.items():
#         if k in dnn_dict:
#             print(k)
#     print('********************************')   
#     import ipdb; ipdb.set_trace()    
    shared_dict = {k: v for k, v in dnn_dict.items() if (k in model_dict)}

    model_dict.update(shared_dict)
    model.load_state_dict(model_dict)
    return model


def get_updateModel_after(model, cal_dnn):
    model_dict = model.state_dict()
    dnn_dict = cal_dnn.state_dict()
#     for k, v in dnn_dict.items():
#         if k in dnn_dict:
#             print(k)
#     print('********************************')   
#     import ipdb; ipdb.set_trace()    
    shared_dict = {k: v for k, v in dnn_dict.items() if (k in model_dict)}

    model_dict.update(shared_dict)
    model.load_state_dict(model_dict)
    return model




def sim_mat(features, prototypes, p_labels, t=0.1):
    
    a_norm = features / features.norm(dim=1)[:, None]
    b_norm = prototypes / prototypes.norm(dim=1)[:, None]
#     sim_matrix = torch.mm(a_norm, b_norm.transpose(0,1))
    sim_matrix = torch.exp(torch.mm(a_norm, b_norm.transpose(0,1)) / t)
    
    return sim_matrix




def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def load_mnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def load_fmnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    fmnist_train_ds = FashionMNIST_truncated(datadir, train=True, download=True, transform=transform)
    fmnist_test_ds = FashionMNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = fmnist_train_ds.data, fmnist_train_ds.target
    X_test, y_test = fmnist_test_ds.data, fmnist_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_tinyimagenet_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = TinyImageNet_load('../datasets/tiny-imagenet-200/', train=True, transform=transform)
    xray_test_ds = TinyImageNet_load('../datasets/tiny-imagenet-200/', train=False, transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)

def load_vireo_data():
    transform = transforms.Compose([transforms.ToTensor()])

    vireo_train_ds = Vireo172_truncated(transform=transform, mode='train')
    vireo_test_ds = Vireo172_truncated(transform=transform, mode='test')

    X_train, y_train = vireo_train_ds.path_to_images, vireo_train_ds.labels
    X_test, y_test = vireo_test_ds.path_to_images, vireo_test_ds.labels

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def load_food_data():
    transform = transforms.Compose([transforms.ToTensor()])

    vireo_train_ds = Food101_truncated(transform=transform, mode='train')
    vireo_test_ds = Food101_truncated(transform=transform, mode='test')

    X_train, y_train = vireo_train_ds.path_to_images, vireo_train_ds.labels
    X_test, y_test = vireo_test_ds.path_to_images, vireo_test_ds.labels

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def record_net_data_stats(y_train, net_dataidx_map, logdir):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list=[]
    for net_id, data in net_cls_counts.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(dataset, datadir, logdir, partition, n_parties, beta=0.4):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    elif dataset == 'fmnist':
        X_train, y_train, X_test, y_test = load_fmnist_data(datadir)        
        
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'tinyimagenet':
        X_train, y_train, X_test, y_test = load_tinyimagenet_data(datadir)
    elif dataset == 'vireo172':
        X_train, y_train, X_test, y_test = load_vireo_data()
    elif dataset == 'food101':
        X_train, y_train, X_test, y_test = load_food_data()        
        
    n_train = y_train.shape[0]

    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}


    elif partition == "noniid-labeldir" or partition == "noniid":
        min_size = 0
        min_size_test = 0
        min_require_size = 10
        K = 10
        if dataset == 'cifar100':
            K = 100
        elif dataset == 'tinyimagenet':
            K = 200
            # min_require_size = 100
        elif dataset == 'vireo172':
            K = 172
        elif dataset == 'food101':
            K = 101    
            
        N_train = y_train.shape[0]
        N_test = y_test.shape[0]
        print('mmm', np.unique(y_train))
        
        net_dataidx_map = {}
        net_dataidx_map_test = {}
        while min_size < min_require_size and min_size_test < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            idx_batch_test = [[] for _ in range(n_parties)]
            for k in range(K):
                
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N_train / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])


        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]           

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)






def get_trainable_parameters(net, device='cpu'):
    'return trainable parameter values as a vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    # print("net.parameter.data:", list(net.parameters()))
    paramlist = list(trainable)
    #print("paramlist:", paramlist)
    N = 0
    for params in paramlist:
        N += params.numel()
        # print("params.data:", params.data)
    X = torch.empty(N, dtype=torch.float64, device=device)
    X.fill_(0.0)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            X[offset:offset + numel].copy_(params.data.view_as(X[offset:offset + numel].data))
        offset += numel
    # print("get trainable x:", X)
    return X


def put_trainable_parameters(net, X):
    'replace trainable parameter values by the given vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    paramlist = list(trainable)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset + numel].data.view_as(params.data))
        offset += numel


def compute_accuracy_v6(model, glo_proto, glo_proto_label, dataloader, args, get_confusion_matrix=False, device="cpu", multiloader=False):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    correct_out, correct_sim = 0, 0
    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device.type:
        criterion = nn.CrossEntropyLoss().to(device)
    loss_collector = []

    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            if device != 'cpu':
                x, target = x.to(device), target.to(dtype=torch.int64).to(device)
            _, _, feats, out = model(x)
            loss = criterion(out, target)            
            sim_matrix = sim_mat(feats, glo_proto, glo_proto_label, args.temp_final)
            
            final_out = args.final_weights * out + (1-args.final_weights) * sim_matrix
#             ipdb.set_trace()

            _, pred_out = torch.max(out.data, 1)
            _, pred_sim = torch.max(sim_matrix.data, 1)
            _, pred_label = torch.max(final_out.data, 1)
    
            loss_collector.append(loss.item())
            total += x.data.size()[0]
            correct_out += (pred_out == target.data).sum().item()
            correct_sim += (pred_sim == target.data).sum().item()
            correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())                             
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
        avg_loss = sum(loss_collector) / len(loss_collector)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct_out / float(total), correct_sim / float(total), correct / float(total), conf_matrix, avg_loss

    return correct_out / float(total), correct_sim / float(total), correct / float(total), avg_loss



def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, dataidxs_test=None, noise_level=0):
    if dataset in ('cifar10', 'cifar100'):
        if dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=noise_level),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])

        elif dataset == 'cifar100':
            dl_obj = CIFAR100_truncated

            normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            transform_train = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])

            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])



        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)


    elif dataset == 'tinyimagenet':
        dl_obj = TinyImageNet_load
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                Variable(x.unsqueeze(0), requires_grad=False),
                (4, 4, 4, 4), mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=noise_level),
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        train_ds = dl_obj('../datasets/tiny-imagenet-200/', train=True, dataidxs=dataidxs, transform=transform_train)
        test_ds = dl_obj('../datasets/tiny-imagenet-200/', train=False, transform=transform_test)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)
    elif dataset == 'mnist':
        dl_obj = MNIST_truncated

        normalize = transforms.Normalize((0.1307,), (0.3081,))

        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()])

        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)
    
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)
    elif dataset == 'fmnist':
        dl_obj = FashionMNIST_truncated

        normalize = transforms.Normalize((0.1307,), (0.3081,))

        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()])
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)

        
    elif dataset == 'vireo172':
        dl_obj = Vireo172_truncated
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
#         transform_train = transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
#          ])       
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
         ])

        
        train_ds = dl_obj(dataidxs, transform_train, mode='train')
        test_ds = dl_obj(None, transform_test, mode='test')
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=True)          
        
    elif dataset == 'food101':
        dl_obj = Food101_truncated
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
#         transform_train = transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
#          ])       
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
         ])

        
        train_ds = dl_obj(dataidxs, transform_train, mode='train')
        test_ds = dl_obj(None, transform_test, mode='test')
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=2, pin_memory=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=2, pin_memory=True)          
                
    return train_dl, test_dl, train_ds, test_ds

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval() 



def dropout_proto_local_v2(net, dataloader, args, n_class=10, device='cuda:0'):
    feats = []
    labels = []
    net.eval()
    net.apply(fix_bn)
    net.to(device)
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.to(device), target.to(device)
            _, feat, _ = net(x)

            feats.append(feat)
            labels.extend(target)

    feats = torch.cat(feats)
    labels = torch.tensor(labels)
#     ipdb.set_trace()
    prototype = []
    proto_label = []
    class_label = []
    class_idx = []
    for i in range(n_class):
        index = torch.nonzero(labels == i).reshape(-1)
        if len(index) > 0:
            class_idx.append(index)
            class_label.append(int(i))
        else:  
            class_idx.append([-1])
            class_label.append(-1)
            
    for i in range(n_class):
        if i in class_label: 
            if len(class_idx[i])>=5:
                for j in range(args.number): #len(class_idx[i])
                    idx = np.random.choice(np.arange(len(class_idx[i])), int(len(class_idx[i])*args.ratio))
                    feature_classwise = feats[class_idx[i][idx]]
                    prototype.append(torch.mean(feature_classwise, axis=0).reshape((1, -1)))
                    proto_label.append(int(i))
            else:
                proto_label.append(int(i))
                feature_classwise = feats[class_idx[i]]
                prototype.append(torch.mean(feature_classwise, axis=0).reshape((1, -1)))            

    return torch.cat(prototype, dim=0), torch.tensor(proto_label)




def gen_proto_global(feats, labels, n_classes):
    local_proto = []
    local_labels = []
    for i in range(n_classes):
#         ipdb.set_trace()
        c_i = torch.nonzero(labels == i).reshape(-1)
        proto_i = torch.sum(feats[c_i], dim=0) / len(c_i)
        local_proto.append(proto_i.reshape(1, -1))
        local_labels.append(i)
    
    return torch.cat(local_proto, dim=0), torch.tensor(local_labels)


def aug_protos(local_protos, local_labels, posi_lambda, nega_lambda, n_classes=10):
    
    aug_protos = []
    aug_labels = []
    for p_id, proto in enumerate(local_protos):
        for x_id, x in enumerate(local_protos):
            if x_id != p_id:
                if local_labels[p_id] == local_labels[x_id]:
                    aug_protos.append((1+posi_lambda)*proto-posi_lambda*x)
                    aug_labels.append(local_labels[p_id])
                else:
                    aug_protos.append((1-posi_lambda)*proto-posi_lambda*x)
                    aug_labels.append(local_labels[x_id])
#     ipdb.set_trace()
    aug_protos = torch.stack(aug_protos).cuda()
    aug_labels = torch.tensor(aug_labels).cuda()
    
    final_protos = torch.cat([local_protos, aug_protos]).cuda()
    final_labels = torch.cat([local_labels, aug_labels]).cuda()
    
    return final_protos, final_labels

def mixup_data(x, y, alpha):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

