import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
import logging
from dataloaders import dataset

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
from datasets import CIFAR10_truncated, SVHN_truncated, CIFAR100_truncated
import pandas as pd


def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

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


def load_SVHN_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    SVHN_train_ds = SVHN_truncated(datadir, split='train', download=True, transform=transform)
    SVHN_test_ds = SVHN_truncated(datadir, split='test', download=True, transform=transform)

    X_train, y_train = SVHN_train_ds.data, SVHN_train_ds.target
    X_test, y_test = SVHN_test_ds.data, SVHN_test_ds.target

    return (X_train, y_train, X_test, y_test)


def load_skin_data(datadir, train_idxs, test_idxs):  # idxs相对所有data
    CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    all_data_path = 'med_classify_dataset/HAM10000_metadata'
    all_data_df = pd.read_csv(all_data_path)
    all_data_df = pd.concat([all_data_df['image_id'], all_data_df['dx']], axis=1)

    X_train, y_train, X_test, y_test = [], [], [], []
    train_df = all_data_df.iloc[train_idxs]
    test_df = all_data_df.iloc[test_idxs]

    train_names = all_data_df.iloc[train_idxs]['image_id'].values.astype(str).tolist()
    train_lab = all_data_df.iloc[train_idxs]['dx'].values.astype(str)
    test_names = all_data_df.iloc[test_idxs]['image_id'].values.astype(str).tolist()
    test_lab = all_data_df.iloc[test_idxs]['dx'].values.astype(str)

    for idx in range(len(train_idxs)):
        X_train.append(datadir + train_names[idx] + '.jpg')
        y_train.append(CLASS_NAMES.index(train_lab[idx]))

    for idx in range(len(test_idxs)):
        X_test.append(datadir + test_names[idx] + '.jpg')
        y_test.append(CLASS_NAMES.index(test_lab[idx]))
    return X_train, y_train, X_test, y_test


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list = []
    for net_id, data in net_cls_counts.items():
        n_total = 0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(dataset, datadir, logdir, partition, n_parties, labeled_num, beta=0.4):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)

    state = np.random.get_state()
    np.random.shuffle(X_train)
    # print(a)
    # result:[6 4 5 3 7 2 0 1 8 9]
    np.random.set_state(state)
    np.random.shuffle(y_train)
    n_train = y_train.shape[0]

    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}


    elif partition == "noniid-labeldir" or partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = 10
        # min_require_size = 100
        sup_size = int(labeled_num *len(y_train) / 10)
        N = y_train.shape[0] - sup_size
        net_dataidx_map = {}
        for sup_i in range(labeled_num):
            net_dataidx_map[sup_i] = [i for i in range(sup_i * sup_size, (sup_i + 1) * sup_size)]

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties - labeled_num)]
            for k in range(K):
                idx_k = np.where(y_train[int(labeled_num * len(y_train) / 10):] == k)[0] + sup_size
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array(
                    [p * (len(idx_j) < N / (n_parties - labeled_num)) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties - labeled_num):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j + labeled_num] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


def partition_data_allnoniid(dataset, datadir, train_idxs=None, test_idxs=None, partition="noniid", n_parties=10,
                             beta=0.4):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'SVHN':
        X_train, y_train, X_test, y_test = load_SVHN_data(datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'skin':
        X_train, y_train, X_test, y_test = load_skin_data(datadir, train_idxs, test_idxs)

    if dataset != 'skin':
        n_train = y_train.shape[0]
        if partition == "homo" or partition == "iid":
            idxs = np.random.permutation(n_train)
            batch_idxs = np.array_split(idxs, n_parties)
            net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}


        elif partition == "noniid-labeldir" or partition == "noniid":
            min_size = 0
            min_require_size = 10
            K = 10

            N = y_train.shape[0]
            net_dataidx_map = {}

            while min_size < min_require_size:
                idx_batch = [[] for _ in range(n_parties)]
                for k in range(K):
                    idx_k = np.where(y_train == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                    proportions = np.array(
                        [p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            for j in range(n_parties):
                np.random.shuffle(idx_batch[j])
                net_dataidx_map[j] = idx_batch[j]

            traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
        return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts
    else:
        return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def get_dataloader(args, data_np, label_np, dataset_type, datadir, train_bs, is_labeled=None, data_idxs=None,
                   is_testing=False, pre_sz=40, input_sz=32):
    if dataset_type == 'SVHN':
        normalize = transforms.Normalize(mean=[0.4376821, 0.4437697, 0.47280442],
                                         std=[0.19803012, 0.20101562, 0.19703614])
        assert pre_sz == 40 and input_sz == 32, 'Error: Wrong input size for 32*32 dataset'
    elif dataset_type == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                         std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        assert pre_sz == 40 and input_sz == 32, 'Error: Wrong input size for 32*32 dataset'
    elif dataset_type == 'skin':
        normalize = transforms.Normalize(mean=[0.7630332, 0.5456457, 0.57004654],
                                         std=[0.14092809, 0.15261231, 0.16997086])

    if not is_testing:
        if is_labeled:
            trans = transforms.Compose(
                [transforms.RandomCrop(size=(input_sz, input_sz)),
                 transforms.RandomHorizontalFlip(p=0.5),
                 transforms.ToTensor(),
                 normalize
                 ])
            ds = dataset.CheXpertDataset(dataset_type, data_np, label_np, pre_sz, pre_sz, lab_trans=trans,
                                         is_labeled=True, is_testing=False)
        else:
            weak_trans = transforms.Compose([
                transforms.RandomCrop(size=(input_sz, input_sz)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                normalize
            ])

            ds = dataset.CheXpertDataset(dataset_type, data_np, label_np, pre_sz, pre_sz,
                                         un_trans_wk=dataset.TransformTwice(weak_trans),
                                         data_idxs=data_idxs,
                                         is_labeled=False,
                                         is_testing=False)
        dl = data.DataLoader(dataset=ds, batch_size=train_bs, drop_last=False, shuffle=True, num_workers=8)
    else:
        ds = dataset.CheXpertDataset(dataset_type, data_np, label_np, input_sz, input_sz, lab_trans=transforms.Compose([
            # K.RandomCrop((224, 224)),
            transforms.ToTensor(),
            normalize
        ]), is_labeled=True, is_testing=True)
        dl = data.DataLoader(dataset=ds, batch_size=train_bs, drop_last=False, shuffle=False, num_workers=8)
    return dl, ds
