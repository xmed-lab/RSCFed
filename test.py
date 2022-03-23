CUDA_VISIBLE_DEVICES="0"
from validation import epochVal_metrics_test
import numpy as np
import torch.backends.cudnn as cudnn
import random
from cifar_load import get_dataloader, partition_data_allnoniid, record_net_data_stats
import copy
import torch
import torch.optim
import torch.nn.functional as F
from options import args_parser
from networks.models import ModelFedCon
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
# import json

args = args_parser()
save_mode_path='final_model/SVHN_best.pth'

if __name__ == "__main__":
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.dataset == 'SVHN':
        partition = torch.load('partition_strategy/SVHN_noniid_10%labeled_ordered.pth')
    elif args.dataset == 'cifar100':
        partition = torch.load('partition_strategy/cifar100_noniid_10%labeled.pth')
    elif args.dataset == 'skin':
        partition = torch.load('partition_strategy/skin_1:9_b0.8.pth')
        # dict_users = partition['data_partition']
        train_list = partition['train_list']
        test_list = partition['test_list']
    net_dataidx_map = partition['data_partition']

    if args.dataset == 'skin':
        X_train, y_train, X_test, y_test, _, traindata_cls_counts = partition_data_allnoniid(
            args.dataset, 'skin/skin/', train_idxs=train_list, test_idxs=test_list,
            n_parties=args.num_users,
            beta=args.beta)
    else:
        X_train, y_train, X_test, y_test, _, traindata_cls_counts = partition_data_allnoniid(
            args.dataset, args.datadir, partition=args.partition, n_parties=args.num_users, beta=args.beta)

    if args.dataset == 'SVHN':
        X_train = X_train.transpose([0, 2, 3, 1])
        X_test = X_test.transpose([0, 2, 3, 1])

    if args.dataset == 'SVHN':
        n_classes = 10
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'skin':
        n_classes = 7
    all_client = np.zeros([n_classes, n_classes], dtype=int)
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    checkpoint_path = save_mode_path
    checkpoint = torch.load(checkpoint_path)

    net = ModelFedCon(args.model, args.out_dim, n_classes=n_classes)
    model = net.cuda()
    model.load_state_dict(checkpoint['state_dict'])

    if args.dataset == 'SVHN' or args.dataset =='cifar100':
        test_dl, test_ds = get_dataloader(args, X_test, y_test,
                                          args.dataset, args.datadir, args.batch_size,
                                          is_labeled=True, is_testing=True)
    elif args.dataset == 'skin':
        test_dl, test_ds = get_dataloader(args, X_test, y_test,
                                          args.dataset, args.datadir, args.batch_size,
                                          is_labeled=True, is_testing=True,pre_sz = args.pre_sz,input_sz = args.input_sz)

    AUROCs, Accus, Pre, Recall = epochVal_metrics_test(model, test_dl,args.model, n_classes=n_classes)
    AUROC_avg = np.array(AUROCs).mean()
    Accus_avg = np.array(Accus).mean()
