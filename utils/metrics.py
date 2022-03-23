# encoding: utf-8
import numpy as np
# from sklearn.metrics._ranking import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # , sensitivity_score
from imblearn.metrics import sensitivity_score, specificity_score
import pdb
from sklearn.metrics._ranking import roc_auc_score

N_CLASSES = 10


# CLASS_NAMES = [ 'Melanoma', 'Melanocytic nevus', 'Basal cell carcinoma', 'Actinic keratosis', 'Benign keratosis']

def compute_metrics_test(gt, pred, n_classes=10):
    """
    Computes accuracy, precision, recall and F1-score from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
        competition: whether to use competition tasks. If False,
          use all tasks
    Returns:
        List of AUROCs of all classes.
    """

    gt_np = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()
    indexes = range(n_classes)

    AUROCs = roc_auc_score(gt_np, pred_np, multi_class='ovr')
    Accus = accuracy_score(gt_np, np.argmax(pred_np, axis=1))
    Pre = precision_score(gt_np, np.argmax(pred_np, axis=1), average='macro')
    Recall = recall_score(gt_np, np.argmax(pred_np, axis=1), average='macro')
    return AUROCs, Accus, Pre, Recall  # , Senss, Specs, Pre, F1


def compute_pred_matrix(gt, pred, n_classes):
    matrix = np.zeros([n_classes, n_classes])
    for idx_gt in range(len(gt)):
        matrix[int(gt[idx_gt])][pred[idx_gt]] += 1
    return matrix
