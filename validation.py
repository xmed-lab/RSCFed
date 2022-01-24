import os
import sys

import shutil
import argparse
import logging
import time
import random
import numpy as np
import pandas as pd


import torch
from torch.nn import functional as F

from utils.metrics import  compute_metrics_test



def epochVal_metrics_test(model, dataLoader,model_type, thresh,n_classes):
    training = model.training
    model.eval()

    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    
    gt_study   = {}
    pred_study = {}
    studies    = []

    with torch.no_grad():
        for i, (study, image, label) in enumerate(dataLoader):
            image, label = image.cuda(), label.cuda()
            _,feature, output = model(image, model=model_type)
            study=study.tolist()
            output = F.softmax(output, dim=1)

            # if i==0:
            #     all_features = feature
            #     all_labels=label
            # else:
            #     all_features=torch.cat((all_features,feature),dim=0)
            #     all_labels=torch.cat((all_labels,label),dim=0)

            for i in range(len(study)):
                if study[i] in pred_study:
                    assert torch.equal(gt_study[study[i]], label[i])
                    pred_study[study[i]] = torch.max(pred_study[study[i]], output[i])
                else:
                    gt_study[study[i]] = label[i]
                    pred_study[study[i]] = output[i]
                    studies.append(study[i])

          
        
        for study in studies:
            gt = torch.cat((gt, gt_study[study].view(1, -1)), 0)
            pred = torch.cat((pred, pred_study[study].view(1, -1)), 0)
        #gt=F.one_hot(gt.to(torch.int64).squeeze())
        #AUROCs, Accus, Senss, Specs, pre, F1 = compute_metrics_test(gt, pred,  thresh=thresh, competition=True)
        AUROCs, Accus = compute_metrics_test(gt, pred, thresh=thresh, competition=True,n_classes=n_classes)

    model.train(training)

    return AUROCs, Accus#,all_features.cpu(),all_labels.cpu()#, Senss, Specs, pre,F1
