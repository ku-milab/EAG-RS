import os
import numpy as np
from src.utils import *

import deepdish as ddish

def setting(opt_, st_Flag=False):
    """GPU & seed setting"""
    device = seed_setting(opt_)

    data_pth = os.path.join('/home/Dataset/') #TODO: define data path
    model_save_dir = os.path.join('/home/Exp') #TODO: define save directory

    total_data_dir = os.path.join(data_pth, 'ABIDE_HO_data.npz') #TODO: define data file
    fold_dir = os.path.join(data_pth, 'CV_5/rp1_f{}.npz'.format(opt_.tmp_fold+1)) #TODO: define fold index file

    data = np.load(total_data_dir, allow_pickle=True)
    cur_fold_idx = np.load(fold_dir, allow_pickle=True)
    final_data = data['fc']
    final_label = data['label']
    final_label[final_label == -1] = 0
    final_id = data['subject']

    final_tr_idx, final_val_idx, final_te_idx = cur_fold_idx['trn_idx'], cur_fold_idx['val_idx'], cur_fold_idx['tst_idx']


    dataType = ['NC', 'MDD']
    exp = {'dataset': 'fMRI',
           'Model_store_dir': model_save_dir,
           'dataType1': dataType[0],  ## TD
           'dataType2': dataType[1],  ## ASD
           }
    exp['kfold'] = opt_.kfold
    exp['tmp_fold'] = opt_.tmp_fold
    exp['seed'] = opt_.seed
    exp['lr'] = opt_.lr
    exp['wd'] = opt_.weight_decay
    exp['lr_decay_step'] = opt_.lr_decay_step
    exp['lr_decay'] = opt_.lr_decay
    exp['epoch'] = opt_.num_epochs
    exp['batch'] = opt_.batch_size
    exp['remove_roi'] = opt_.remove_roi
    exp['device'] = device
    exp['site'] = opt_.site
    exp['atlas'] = opt_.atlas
    exp['repeat'] = opt_.repeat
    # exp['t1'] = opt_.t1
    # exp['t2'] = opt_.t2

    exp['fold_indices'] = [final_tr_idx, final_val_idx, final_te_idx]
    exp['data'] = final_data
    exp['label'] = final_label
    exp['id'] = final_id

    exp['exp_dir'] = os.path.join(model_save_dir, 'b{}_r{}_repeat{}'.format(exp['batch'], exp['remove_roi'],exp['repeat']))
    exp['site'] = opt_.site

    return exp