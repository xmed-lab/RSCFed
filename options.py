import argparse


# from networks.models import DenseNet121
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/home/xliangak/med_classify_dataset/skin',
                        help='dataset root dir')
    parser.add_argument('--csv_file_train', type=str, default='train_samples.csv', help='training set csv file')
    parser.add_argument('--csv_file_test', type=str, default='test_samples.csv', help='testing set csv file')
    parser.add_argument('--batch_size', type=int, default=48, help='batch_size per gpu')
    parser.add_argument('--drop_rate', type=int, default=0.2, help='dropout rate')
    parser.add_argument('--ema_consistency', type=int, default=1, help='whether train baseline model')
    parser.add_argument('--base_lr', type=float, default=2e-4,
                        help='maximum epoch number to train')  # adam:2e-4 sgd:2e-3 adamw:2e-3?
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--gpu', type=str, default='0,1,2', help='GPU to use')
    parser.add_argument('--local_ep', type=int, default=1, help='local epoch')
    parser.add_argument('--num_users', type=int, default=10, help='local epoch')
    parser.add_argument('--num_labeled', type=int, default=1, help='local epoch')
    # parser.add_argument('--num_unlabeled', type=int, default=9, help='local epoch')
    parser.add_argument('--rounds', type=int, default=200, help='local epoch')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--logdir', type=str, default='logs/', help='The log file name')
    parser.add_argument('--opt', type=str, default='sgd', help='sgd or adam or adamw')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'skin','SVHN','cifar100'], default='cifar10',
                        help='dataset used for training')
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--model', type=str, default='simple-cnn', help='neural network used in training')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    ### tune
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--pre_trained', type=str, default='labeled_trained_20r_model', help='model to resume')
    parser.add_argument('--start_epoch', type=int, default=0, help='start_epoch')
    parser.add_argument('--global_step', type=int, default=0, help='global_step')
    parser.add_argument('--weight_decay', dest="weight_decay", default=0.02, type=float, help='weight decay')

    ### costs
    parser.add_argument('--label_uncertainty', type=str, default='U-Ones', help='label type')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='ema_decay')
    parser.add_argument('--consistency', type=float, default=1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float, default=30, help='consistency_rampup')

    parser.add_argument('--lambda_u', type=float, default=75, help='start_epoch')
    parser.add_argument('--lambda_p', type=float, default=75, help='start_epoch')
    ### unlabeled client training parameters
    parser.add_argument('--num-warmup-epochs',
                        '--num-warm-up-epochs',
                        dest="num_warmup_epochs",
                        default=0,
                        type=int,
                        help='number of warm-up epochs for unsupervised loss ramp-up during training'
                             'set to 0 to disable ramp-up')
    # parser.add_argument('--lr-scheduler',
    #                     '--lr-scheduler-type',
    #                     dest="lr_scheduler_type",
    #                     choices=["nop", "cosine_decay", "step_decay"],
    #                     default="nop",
    #                     type=str,
    #                     help=f"learning rate scheduler type")
    parser.add_argument('--lr-cosine-factor',
                        '--learning-rate-cosine-factor',
                        dest="lr_cosine_factor",
                        default=0.49951171875,
                        type=float,
                        help='factor for cosine learning rate decay')
    parser.add_argument('--lr-step-size',
                        '--learning-rate-step-size',
                        dest="lr_step_size",
                        default=5,
                        type=int,
                        help='step size for step learning rate decay')
    parser.add_argument('--lr-gamma',
                        '--learning-rate-gamma',
                        dest="lr_gamma",
                        default=0.1,
                        type=float,
                        help='factor for step learning rate decay')
    parser.add_argument("--max_grad_norm",
                        dest="max_grad_norm",
                        type=float,
                        default=5,
                        help="max gradient norm allowed (used for gradient clipping)")
    ### unsupervised loss
    parser.add_argument('--u-loss-thresholded',
                        '--unsupervised-loss-thresholded',
                        dest="u_loss_thresholded",
                        # default=True,
                        action='store_true',
                        help='Apply threshold to unsupervised loss')
    parser.add_argument('--conf-threshold',
                        '--confidence-threshold',
                        dest="confidence_threshold",
                        default=0.,
                        type=float,
                        help="confidence threshold for pair loss and unsupervised loss")
    ### pair loss
    parser.add_argument('--strong_trans_times', type=int, default=7, help='start_epoch')
    parser.add_argument('--u-loss-type',
                        '--unsupervised-loss-type',
                        dest="u_loss_type",
                        choices=["mse", "entropy"],
                        default="mse",
                        type=str,
                        help=f"Unsupervised loss type")
    parser.add_argument('--test', action='store_true', help='resume from checkpoint')

    ### meta
    parser.add_argument('--meta_round', type=int, default=3, help='start_epoch')
    parser.add_argument('--meta_client_num', type=int, default=5, help='start_epoch')
    parser.add_argument('--agg_per_meta', action='store_true', help='resume from checkpoint')
    #parser.add_argument('--label_each_meta', action='store_true', help='resume from checkpoint')
    # parser.add_argument('--double_lab_w', action='store_true', help='resume from checkpoint')
    # parser.add_argument('--five_lab_w', action='store_true', help='resume from checkpoint')
    parser.add_argument('--from_labeled', action='store_true', help='resume from checkpoint')
    parser.add_argument('--w_mul_times', type=int, default=1, help='start_epoch')
    #parser.add_argument('--meta_plus', action='store_true', help='resume from checkpoint')
    #parser.add_argument('--guarant_1sup', action='store_true', help='resume from checkpoint')

    parser.add_argument('--unsup_lr', type=float, default=0.021,
                        help='maximum epoch number to train')
    #parser.add_argument('--add_drop', action='store_true', help='resume from checkpoint')
    #parser.add_argument('--cos_labw', action='store_true', help='resume from checkpoint')
    #parser.add_argument('--un_dist',default='',type=str,choices=["avg", "prev","mix"], help='resume from checkpoint')
    #parser.add_argument('--un_dist_onlyunsup', action='store_true', help='resume from checkpoint')
    #parser.add_argument('--same_pred_unsup', action='store_true', help='resume from checkpoint')
    #parser.add_argument('--inverse', action='store_true', help='resume from checkpoint')
    parser.add_argument('--dist_scale', type=float or int, default=1e4, help='start_epoch')

    parser.add_argument('--input_sz', type=int, default=32, help='start_epoch')
    parser.add_argument('--pre_sz', type=int, default=40, help='start_epoch')
    parser.add_argument('--unsup_num', type=int, default=9, help='start_epoch')
    #parser.add_argument('--record_dist', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()
    return args
