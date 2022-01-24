from validation import epochVal_metrics_test
import numpy as np
import torch.backends.cudnn as cudnn
import random
from cifar_load import get_dataloader, partition_data_allnoniid, record_net_data_stats
import torch
import torch.optim
import torch.nn.functional as F
from options import args_parser
from networks.models import ModelFedCon
import torch.nn as nn
from utils_SimPLE import label_guessing, sharpen
from ramp import LinearRampUp
from torchvision import transforms
from utils import losses
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn import manifold

args = args_parser()
# save_mode_path = '/home/xliangak/CPS_cifar/model/glob_w+unlabeled_w+raw/epoch_199.pth'
# save_mode_path='/home/xliangak/SimPLE_FFT_cifar/model/log-08-20-1309-21/epoch_200.pth'#带warmup的SimPLE+FFT
# save_mode_path='/home/xliangak/SimPLE-cifar/model/log-08-18-0353-52/epoch_197.pth' #带warmup的原始SimPLE
# save_mode_path = '/home/xliangak/SimPLE_FFT_cifar/model/strong+3+7_labCE_cifar/epoch_115.pth'
# save_mode_path ='/home/xliangak/SimPLE_FFT_cifar/model/FFT_UnLossNoTh_7+7/epoch_20.pth'
# save_mode_path='/home/xliangak/SimPLE_FFT_cifar/model/09-03-2341-33/epoch_41.pth'
save_mode_path = '/home/xliangak/CVPR_FSSL/baseline_noFFT/model/OnlySup_lr0.03/epoch_13.pth'


# save_mode_path="/home/xliangak/SimPLE_FFT_cifar/model/strong+3+7_labCE_cifar/epoch_54.pth"
# save_mode_path = '/home/xliangak/SimPLE_FFT_cifar/model/08-31-0535-49/epoch_37.pth'


# without FFT:197

class Unsupervised_test(object):
    def __init__(self, args, idxs):
        # self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.batch_size, shuffle=True)

        net = ModelFedCon(args.model, args.out_dim, n_classes=10)
        if len(args.gpu.split(',')) > 1:
            net = torch.nn.DataParallel(net, device_ids=[i for i in range(round(len(args.gpu) / 2))])
            # net = torch.nn.DataParallel(net, device_ids=[6,7])
        self.ema_model = net.cuda()
        for param in self.ema_model.parameters():
            param.detach_()
        self.data_idxs = idxs
        self.epoch = 0
        self.iter_num = 0
        self.flag = True
        self.base_lr = 5e-4
        self.softmax = nn.Softmax()
        # self.unsupervised_loss = UnsupervisedLoss(
        #     loss_type=args.u_loss_type,
        #     loss_thresholded=args.u_loss_thresholded,
        #     confidence_threshold=args.confidence_threshold,
        #     reduction="mean")
        # self.pairloss = build_pair_loss(args)
        self.max_grad_norm = args.max_grad_norm
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.max_step = args.rounds * round(len(idxs) / args.batch_size)

        # SimPLE original paper: lr=0.002, weight_decay=0.02
        self.max_warmup_step = round(len(idxs) / args.batch_size) * args.num_warmup_epochs
        self.ramp_up = LinearRampUp(length=self.max_warmup_step)

    def train(self, args, net, epoch, train_dl_local):
        net.eval()

        # for param_group in self.optimizer.param_groups:
        #    param_group['lr'] = self.base_lr

        self.epoch = epoch
        if self.flag:
            self.ema_model.load_state_dict(net.state_dict())
            self.flag = False
            # logging.info('EMA model initialized')

        epoch_loss = []
        # logging.info('Unlabeled client %d begin unsupervised training' % unlabeled_idx)
        # unloader = transforms.ToPILImage()
        #with torch.no_grad():
        batch_loss = []
        # iter_max = len(self.ldr_train)
        # pair_mat_str = np.zeros([10, 10])
        # pair_mat_fft = np.zeros([10, 10])
        pseu_mat = np.zeros([10, 10])
        pseu_pred_mat = np.zeros([10, 10])
        same_pred_num=0
        for i, (_, weak_aug_batch, label_batch) in enumerate(train_dl_local):
            guessed = label_guessing(self.ema_model, weak_aug_batch)
            sharpened = sharpen(guessed)

            # probs_str = [F.softmax(net(batch)[1], dim=1) for batch in strong_aug_batch]

            logits_str = net(weak_aug_batch[1])[2]
            probs_str = F.softmax(logits_str, dim=1)
            if i == 0:
                all_pseu = torch.argmax(sharpened, dim=1)
                all_pred = torch.argmax(probs_str, dim=1)
                all_label = label_batch.squeeze()
            else:
                all_pseu = torch.cat([all_pseu, torch.argmax(sharpened, dim=1)])
                all_pred = torch.cat([all_pred, torch.argmax(probs_str, dim=1)])
                all_label = torch.cat([all_label, label_batch.squeeze()])

            loss_u = torch.sum(losses.softmax_mse_loss(probs_str, sharpened)) / args.batch_size
            ramp_up_value = self.ramp_up(current=self.epoch)
            # ramp_up_value = 1
            loss = ramp_up_value * args.lambda_u * loss_u  # + ramp_up_value * args.lambda_p * loss_pair

            batch_loss.append(loss.item())

            self.iter_num = self.iter_num + 1

        self.epoch = self.epoch + 1
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
        assert len(all_pseu) == len(all_label) == len(all_pred)
        for i in range(len(all_pseu)):
            pseu_mat[all_label[i].int().item(), all_pseu[i].int().item()] += 1
            pseu_pred_mat[all_pseu[i].int().item(), all_pred[i].int().item()] += 1
            if all_pseu[i].int().item()==all_pred[i].int().item():
                same_pred_num+=1

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss),same_pred_num

supervised_user_id = [0]
# unsupervised_user_id = [0, 1, 2, 3, 4, 5, 6, 7, 8]
unsupervised_user_id = [1, 2, 3, 4, 5, 6, 7, 8, 9]

if __name__ == "__main__":
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print('==> Reloading data partitioning strategy..')
    # assert os.path.isdir('partition_strategy'), 'Error: no partition_strategy directory found!'
    if len(supervised_user_id) == 1:
        if args.dataset == 'cifar10':
            partition = torch.load('partition_strategy/cifar_noniid_10%labeled.pth')
        elif args.dataset == 'SVHN':
            partition = torch.load('partition_strategy/SVHN_noniid_10%labeled.pth')
        net_dataidx_map = partition['data_partition']
    elif len(supervised_user_id) == 2:
        if args.dataset == 'cifar10':
            partition = torch.load('partition_strategy/cifar_noniid_20%labeled.pth')
        elif args.dataset == 'SVHN':
            partition = torch.load('partition_strategy/SVHN_noniid_20%labeled.pth')
        net_dataidx_map = partition['data_partition']

    all_client = np.zeros([10, 10], dtype=int)

    X_train, y_train, X_test, y_test, _, traindata_cls_counts = partition_data_allnoniid(
        args.dataset, args.datadir, args.logdir, args.partition, args.num_users, args.num_labeled, beta=args.beta)

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    for client_idx in range(10):
        for class_idx in range(10):
            if class_idx not in traindata_cls_counts[client_idx]:
                traindata_cls_counts[client_idx][class_idx] = 0
            all_client[client_idx][class_idx] = int(traindata_cls_counts[client_idx][class_idx])
    checkpoint_path = save_mode_path
    checkpoint = torch.load(checkpoint_path)
    net = ModelFedCon(args.model, args.out_dim, n_classes=10)
    model = net.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    # normalize = transforms.Normalize([0.485, 0.456, 0.406],
    #                                 [0.229, 0.224, 0.225])
    # test_idx = [i for i in range(len(test_imgID))]

    #############################################################################################################
    #test_client_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    same_pred_num=[]
    for client_idx in unsupervised_user_id:
        train_dl_local, train_ds_local = get_dataloader(args,
                                                        X_train[net_dataidx_map[client_idx]],
                                                        y_train[net_dataidx_map[client_idx]],
                                                        args.dataset,
                                                        args.datadir, args.batch_size, is_labeled=False,data_idxs=net_dataidx_map[client_idx])
        trainer = Unsupervised_test(args, net_dataidx_map[client_idx])
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.base_lr, momentum=0.9,
                                    weight_decay=5e-4)
        w, loss,same_pred_this = trainer.train(args, model, 0, train_dl_local)
        same_pred_num.append(same_pred_this)
    #############################################################################################################

    test_dl, test_ds = get_dataloader(args, X_test, y_test,
                                      args.dataset, args.datadir, args.batch_size,
                                      is_labeled=True, is_testing=True)
    # trainer = Unsupervised_test(args, 10000)
    # w, loss, op, ratio = trainer.train(args, model, 0, test_dl)

    # AUROCs, Accus, Senss, Specs, _, _ = epochVal_metrics_test(model, test_dl, thresh=0.4)

    AUROCs, Accus, all_features, all_labels = epochVal_metrics_test(model, test_dl, thresh=0.4)
    draw_tsne(all_features, all_labels)
    AUROC_avg = np.array(AUROCs).mean()
    Accus_avg = np.array(Accus).mean()
