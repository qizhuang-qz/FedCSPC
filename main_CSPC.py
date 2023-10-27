import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random
# os.environ['CUDA_VISIBLE_DEVICES']='0'
from resnet import resnet18, DNN
from model import ModelFedCon
from utils import *
from loss import *
from re_training import *
import torch.nn.functional as F
from RGA import *
import ipdb

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=100, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/N+E/cifar10/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    
    parser.add_argument('--mu', type=float, default=0.5,
                        help='The parameter for the weight of RGA loss')   
    parser.add_argument('--node_weight', default=1.0, type=float, help='The weight of node-based loss')
    parser.add_argument('--edge_weight', default=0.2, type=float, help='The weight of edge-based loss')
    parser.add_argument('--angle_weight', default=0.2, type=float, help='The weight of angle-based loss')
    parser.add_argument('--mode', default='N', type=str, help='The loss mode')   
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--local_max_epoch', type=int, default=100,
                        help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1,
                        help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=None, help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss', type=str, default='contrastive')
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--use_project_head', type=int, default=1)
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--number', type=int, default=5)
    parser.add_argument('--re_mu', type=float, default=0.1)    
    parser.add_argument('--re_beta', type=float, default=0.1)  
    parser.add_argument('--temp_final', type=float, default=0.5) 
    parser.add_argument('--re_version', type=str, default='v1')    
    parser.add_argument('--final_weights', type=float, default=0.1)  
    parser.add_argument('--posi_lambda', type=float, default=0.5)
    parser.add_argument('--nega_lambda', type=float, default=0.5)
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    args = parser.parse_args()
    return args

  


def init_nets(net_configs, n_parties, args, n_classes, device='cuda:0'):
    nets = {net_i: None for net_i in range(n_parties)}
        
    for net_i in range(n_parties):
        if 'cifar' in args.dataset: 
            net = ModelFedCon(args.model, args.out_dim, n_classes, net_configs)
            net.to(device)
            nets[net_i] = net
            if args.model == 'simple-cnn':    
                dnn = DNN(input_dim=84, hidden_dims=[84, 256], n_classes=n_classes).to(device)
            elif args.model == 'resnet18':    
                dnn = DNN(input_dim=512, hidden_dims=[512, 256], n_classes=n_classes).to(device)
        else:        
#             ipdb.set_trace()
            if args.dataset == 'vireo172' or args.dataset == 'food101':
                net = resnet18(args.dataset, kernel_size=7, pretrained=False)
                nets[net_i] = net
            else:
                net = resnet18(args.dataset, kernel_size=3, pretrained=False)   
                nets[net_i] = net
            dnn = DNN(input_dim=512, hidden_dims=[512, 512], n_classes=n_classes).to(device)
    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, dnn, model_meta_data, layer_type





def train_net_CSPC(net_id, net, train_dataloader, test_dataloader, anchors, epochs, lr, args_optimizer, args, n_classes=200, device="cuda:0"):

    net.to(device)   
    
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
        
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    net.train()
    
    anchors_label = torch.tensor(list(range(n_classes))).to(device)
    RGAloss = RGA_loss(node_weight=args.node_weight, edge_weight=args.edge_weight, angle_weight=args.angle_weight,
                        t=args.temperature)

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
#             ipdb.set_trace()
            x, target = x.to(device), target.to(device)
            if args.dataset == 'pmnist':
                target = target.reshape(-1)
            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()
            h, h_out, _, out = net(x) 
            if round == 0:   
                loss = criterion(out, target) + criterion(h_out, target)
            else:
#                 ipdb.set_trace()
                loss = criterion(out, target) + criterion(h_out, target) + args.mu * RGAloss(h, target, anchors, anchors_label, 'N')
            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
    logger.info(' ** Training complete **') 
    
    net.to('cpu')
    
    return 0, 0


def local_train_net(nets, args, net_dataidx_map, local_proto_list, local_proto_label_list, net_id_list, train_dl=None, test_dl=None, round=None, global_model=None, prev_model_pool=None, anchors=None, n_classes=None, device="cuda:0"):
    avg_acc = 0.0
    nets_global = copy.deepcopy(nets)
    k = 0
    for net_id, net in nets.items():
        print(net_id)
        
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)

        n_epoch = args.epochs
        
        if args.alg == 'CSPC':
            _, _ = train_net_CSPC(net_id, net, train_dl_local, test_dl, anchors, n_epoch, args.lr, args.optimizer, args, n_classes=n_classes, device=device)  
            
        local_protos, local_labels = dropout_proto_local_v2(net, train_dl_local, args, n_class=n_classes)

        if k == 0:
#             ipdb.set_trace()
            clients_ids = torch.tensor([net_id] * local_labels.shape[0])
        else:
            clients_ids = torch.cat([clients_ids, torch.tensor([net_id] * local_labels.shape[0])])
#         print(local_proto_list)
        local_proto_list[net_id] = local_protos
        local_proto_label_list[net_id] = local_labels
        net_id_list[net_id] = 1
        k+=1
    return nets, local_proto_list, local_proto_label_list, net_id_list, clients_ids


if __name__ == '__main__':
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)


    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    global_party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset, args.datadir,
                                                                               args.batch_size, 32)

    print("len train_dl_global:", len(train_ds_global))
    train_dl = None
    data_size = len(test_ds_global)

    logger.info("Initializing nets")
    nets, _, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, n_classes, device=device)

    global_models, global_dnn, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, n_classes, device=device)
    global_model = global_models[0]

    n_comm_rounds = args.comm_round


    if args.load_model_file and args.alg != 'plot_visual':
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0

    
    local_proto_list = [[]]*args.n_parties
    local_proto_label_list = [[]]*args.n_parties
    net_id_list = torch.tensor([0]*args.n_parties)
    if args.alg == 'CSPC':
        anchors=0
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]
            global_w = global_model.state_dict()

            nets_this_round = {k: nets[k] for k in party_list_this_round}
        

            for net in nets_this_round.values():
                net.load_state_dict(global_w)           

            nets_this_round, local_proto_list, local_proto_label_list, net_id_list, clients_ids = local_train_net(nets_this_round, args, net_dataidx_map, local_proto_list, local_proto_label_list, net_id_list, round=round, train_dl=train_dl, anchors=anchors, test_dl=test_dl, n_classes=n_classes, device=device)
            global_model.to('cpu')      
            
            # update global model
            
            global_nets_this_round = {k: nets[k] for k in global_party_list}
            
            total_data_points = sum([len(net_dataidx_map[r]) for r in global_party_list])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in global_party_list]

            for net_id, net in enumerate(global_nets_this_round.values()):
                net.to('cpu')
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            global_model.load_state_dict(global_w)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            
            
            # re_training
            id_list = torch.nonzero(net_id_list == 1).reshape(-1)
#             ipdb.set_trace()
            for net_id, idx in enumerate(id_list):
#                 ipdb.set_trace() 
                if net_id == 0:
                    local_protos = local_proto_list[idx]
                    local_labels = local_proto_label_list[idx]
                else:
                    local_protos = torch.cat([local_protos, local_proto_list[idx]])
                    local_labels = torch.cat([local_labels, local_proto_label_list[idx]])
        
            anchors, labels = gen_proto_global(local_protos, local_labels, n_classes)
            
            global_dnn = get_updateModel_before(global_dnn, global_model)
#             clients_ids = clients_ids.cpu().numpy()
            global_dnn, glo_proto, glo_proto_label = retrain_cls_final(global_dnn, local_protos, local_labels, clients_ids, n_classes, args, round, device)   
            global_model.to(device)
            global_model = get_updateModel_after(global_model, global_dnn)
            

#             train_acc, train_loss = compute_accuracy_v6(global_model, train_dl_global, device=device)
            acc_out, acc_sim, acc_final, conf_matrix, _ = compute_accuracy_v6(global_model, glo_proto, glo_proto_label, test_dl, args, get_confusion_matrix=True, device=device)


            logger.info('>> Global Model Test_out accuracy: %f' % acc_out)
            logger.info('>> Global Model Test_sim accuracy: %f' % acc_sim)
            logger.info('>> Global Model Test_final accuracy: %f' % acc_final)




