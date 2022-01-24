from torch.utils.data import Dataset
import copy
import torch
import torch.optim
import torch.nn.functional as F
from options import args_parser
from networks.models import ModelFedCon
from utils import losses, ramps
import torch.nn as nn
from utils_SimPLE import label_guessing, sharpen
from loss.loss import UnsupervisedLoss  # , build_pair_loss
import logging
from torchvision import transforms
from ramp import LinearRampUp

args = args_parser()


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


# alpha=0.999
def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        items, index, weak_aug, strong_aug, label = self.dataset[self.idxs[item]]
        return items, index, weak_aug, strong_aug, label


class UnsupervisedLocalUpdate(object):
    def __init__(self, args, idxs, n_classes):
        net = ModelFedCon(args.model, args.out_dim, n_classes=n_classes)
        net_ema = ModelFedCon(args.model, args.out_dim, n_classes=n_classes)

        if len(args.gpu.split(',')) > 1:
            net = torch.nn.DataParallel(net, device_ids=[i for i in range(round(len(args.gpu) / 2))])
            net_ema = torch.nn.DataParallel(net_ema, device_ids=[i for i in range(round(len(args.gpu) / 2))])
            # net = torch.nn.DataParallel(net, device_ids=[6,7])
        self.ema_model = net_ema.cuda()
        self.model = net.cuda()

        for param in self.ema_model.parameters():
            param.detach_()
        self.data_idxs = idxs
        self.epoch = 0
        self.iter_num = 0
        self.flag = True
        self.unsup_lr = args.unsup_lr
        self.softmax = nn.Softmax()
        self.unsupervised_loss = UnsupervisedLoss(
            loss_type=args.u_loss_type,
            loss_thresholded=args.u_loss_thresholded,
            confidence_threshold=args.confidence_threshold,
            reduction="mean")
        # self.pairloss = build_pair_loss(args)
        self.max_grad_norm = args.max_grad_norm
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.max_step = args.rounds * round(len(self.data_idxs) / args.batch_size)
        if args.opt == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.unsup_lr,
                                              betas=(0.9, 0.999), weight_decay=5e-4)
        elif args.opt == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.unsup_lr, momentum=0.9,
                                             weight_decay=5e-4)
        elif args.opt == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.unsup_lr,
                                               weight_decay=0.02)
            # SimPLE original paper: lr=0.002, weight_decay=0.02
        self.max_warmup_step = round(len(self.data_idxs) / args.batch_size) * args.num_warmup_epochs
        self.ramp_up = LinearRampUp(length=self.max_warmup_step)

    def train(self, args, net_w, op_dict, epoch, unlabeled_idx, train_dl_local, n_classes):
        self.model.load_state_dict(copy.deepcopy(net_w))
        self.model.train()
        self.ema_model.eval()

        self.model.cuda()
        self.ema_model.cuda()

        self.optimizer.load_state_dict(op_dict)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.unsup_lr

        self.epoch = epoch
        if self.flag:
            self.ema_model.load_state_dict(copy.deepcopy(net_w))
            self.flag = False
            logging.info('EMA model initialized')

        epoch_loss = []
        logging.info('Unlabeled client %d begin unsupervised training' % unlabeled_idx)
        correct_pseu = 0
        all_pseu = 0
        test_right = 0
        test_right_ema = 0
        train_right = 0
        same_total = 0
        for epoch in range(args.local_ep):
            batch_loss = []
            # iter_max = len(self.ldr_train)

            for i, (_, weak_aug_batch, label_batch) in enumerate(train_dl_local):
                weak_aug_batch = [weak_aug_batch[version].cuda() for version in range(len(weak_aug_batch))]
                with torch.no_grad():
                    guessed = label_guessing(self.ema_model, [weak_aug_batch[0]], args.model)
                    sharpened = sharpen(guessed)

                pseu = torch.argmax(sharpened, dim=1)
                label = label_batch.squeeze()
                if len(label.shape) == 0:
                    label = label.unsqueeze(dim=0)
                correct_pseu += torch.sum(label[torch.max(sharpened, dim=1)[0] > args.confidence_threshold] == pseu[
                    torch.max(sharpened, dim=1)[0] > args.confidence_threshold].cpu()).item()
                all_pseu += len(pseu[torch.max(sharpened, dim=1)[0] > args.confidence_threshold])
                train_right += sum([pseu[i].cpu() == label_batch[i].int() for i in range(label_batch.shape[0])])

                logits_str = self.model(weak_aug_batch[1], model=args.model)[2]
                probs_str = F.softmax(logits_str, dim=1)
                pred_label = torch.argmax(logits_str, dim=1)

                same_total += sum([pred_label[sam] == pseu[sam] for sam in range(logits_str.shape[0])])

                loss_u = torch.sum(losses.softmax_mse_loss(probs_str, sharpened)) / args.batch_size

                ramp_up_value = self.ramp_up(current=self.epoch)

                loss = ramp_up_value * args.lambda_u * loss_u
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=self.max_grad_norm)
                self.optimizer.step()

                update_ema_variables(self.model, self.ema_model, args.ema_decay, self.iter_num)

                batch_loss.append(loss.item())

                self.iter_num = self.iter_num + 1

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.epoch = self.epoch + 1
        self.model.cpu()
        self.ema_model.cpu()
        return self.model.state_dict(), self.ema_model.state_dict(), sum(epoch_loss) / len(epoch_loss), copy.deepcopy(
            self.optimizer.state_dict()), ramp_up_value, correct_pseu, all_pseu, test_right, train_right.cpu().item(), test_right_ema, same_total.cpu().item()
