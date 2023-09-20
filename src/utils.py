import os
import random

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def writelog(file, line):
    """Define the function to print and write log"""
    file.write(line + '\n')
    print(line)

def seed_setting(args):
    """GPU connection and seed setting"""
    # devices = args.gpu
    devices = "5"
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    device = torch.device("cpu" if torch.cuda.is_available() else "cuda:0")
    if torch.cuda.is_available() == 1:
        print('GPU is working.')
    else:
        print('GPU is not working...')

    return device

def preprocessing():
    """fMRI preprocessing procedure"""
    import glob
    from scipy import io
    from nilearn.connectome import ConnectivityMeasure

    tmp_data_pth = glob.glob(os.path.join(os.getcwd(),'ADNI/**/*.mat'), recursive=True)
    corr_data, corr_label = [], []
    for _, pth_ in enumerate(tmp_data_pth):
        if pth_.split('/')[-1].split('_')[0] == 'NC':
            corr_label.append(0)
        elif pth_.split('/')[-1].split('_')[0] == 'EMCI':
            corr_label.append(1)
        elif pth_.split('/')[-1].split('_')[0] == 'LMCI':
            corr_label.append(2)
        elif pth_.split('/')[-1].split('_')[0] == 'MCI':
            corr_label.append(3)
        else: #AD
            corr_label.append(4)

        data_ = io.loadmat(pth_)['time_series']
        correlation_measure = ConnectivityMeasure(kind='correlation')
        correlation_matrix = correlation_measure.fit_transform([data_])[0]
        corr_data.append(correlation_matrix)
    corr_data = np.array(corr_data)
    corr_label = np.array(corr_label)
    np.savez('./data.npz', data=corr_data, label=corr_label)

def Evaluate_Binary(label, pred, pred_prob, num_classes=2):
    from sklearn import metrics
    label = label.cpu()
    pred = pred.cpu()
    if num_classes == 1:
        pred_prob = pred_prob.cpu()
    else:
        pred_prob = pred_prob[:, 1].cpu()
    # confusion = metrics.confusion_matrix(label.cpu(), pred.cpu())
    confusion = metrics.confusion_matrix(label, pred)

    TP = confusion[1,1]
    TN = confusion[0,0]
    FP = confusion[0,1]
    FN = confusion[1,0]

    fpr, tpr, thresholds = metrics.roc_curve(label, pred_prob)

    Acc =  (TP+TN) / float(TP+TN+FP+FN) * 100  # metrics.accuracy_score(label, pred)
    Sen =  TP / float(TP+FN) * 100             # metrics.recall_score(label, pred)##
    Spe =  TN / float(TN+FP) * 100
    AUC =  metrics.roc_auc_score(label,pred_prob)

    precision = TP / float(TP+FP)              # metrics.precision_score(label, pred)

    return (Acc, Sen, Spe, AUC), (TP,TN,FP,FN)
