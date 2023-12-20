import os
import torch
import nibabel as nib
import glob
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from torch.backends import cudnn

from src.losses import *
from src.data_loader import *
from src.pretrain import *
from src.fine_tune import *
from utils import writelog, cluster_coef_from_FC_sync
import matplotlib.pyplot as plt

# matplotlib.use('Agg')
plt.switch_backend('agg')
'''

'''

class Auto_Encoder_v2(nn.Module):
    def __init__(self, config, input_size, hidden_size, flag=None):
        super(Auto_Encoder_v2, self).__init__()
        if flag == '1':
            self.encoder = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.SELU(),
            )
            self.decoder1 = nn.Sequential(
                nn.Linear(hidden_size, input_size),
                nn.Tanh(),
            )
        elif flag == '2':
            # self.encoder = nn.Linear(input_size, int(np.ceil(config.input_size*1.5)))
            # self.Tanh = nn.Tanh()
            # self.dropout = nn.Dropout(p=0.5)
            self.encoder2 = nn.Sequential(
                # nn.Linear(int(np.ceil(config.input_size*1.5)), hidden_size),
                nn.Linear(int(np.ceil(input_size)), hidden_size),
                nn.Tanh(),
            )
            self.decoder1 = nn.Sequential(
                # nn.Linear(hidden_size, int(np.ceil(config.input_size*1.5))),
                nn.Linear(hidden_size, int(np.ceil(input_size))),
                nn.Tanh()
            )
        elif flag == '3':
            # self.encoder = nn.Linear(input_size, int(np.ceil(config.input_size*1.5)))
            # self.Tanh = nn.Tanh()
            # self.dropout = nn.Dropout(p=0.5)
            self.encoder3 = nn.Sequential(
                # nn.Linear(int(np.ceil(config.input_size*1.5)), hidden_size),
                nn.Linear(int(np.ceil(input_size)), hidden_size),
                nn.Tanh(),
            )
            self.decoder3 = nn.Sequential(
                # nn.Linear(hidden_size, int(np.ceil(config.input_size*1.5))),
                nn.Linear(hidden_size, int(np.ceil(input_size))),
                nn.Tanh()
            )
            # self.decoder2 = nn.Sequential(
            #     nn.Linear(int(np.ceil(config.input_size*1.5)), input_size),
            #     nn.Sigmoid(),
            # )
    #
    def forward(self, x, noisy=False, training = False, flag=None):
        # x = x.detach()
        # Add noise, but use the original lossless input as the target.
        if flag == '1':
            embeded_1 = self.encoder(x) #([50, 200]) -> ([50, 300])
            Reconstruct = self.decoder1(embeded_1) #([50, 300]) -> ([50, 200])
        elif flag == '2':
            # embeded_1 = self.dropout(self.Tanh(self.encoder(x)))  # ([50, 200]) -> ([50, 300])
            embeded_1 = self.encoder2(x)  #
            Reconstruct = self.decoder1(embeded_1)  #
            # Reconstruct = self.decoder2(Reconstruct)  #
        elif flag == '3':
            embeded_1 = self.encoder3(x)  #
            Reconstruct = self.decoder3(embeded_1)  #

        return Reconstruct, embeded_1

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.bn1 = nn.BatchNorm1d(dim_hidden)
        self.relu = nn.ReLU(dim_hidden)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # return F.log_softmax(x, dim=-1)
        return x

def Evaluate_LRP(opt_, Flag_sw=True):
    # import model.MLP as NN
    if opt_['site'] != 'total':
        data = opt_['data']['data']
        label = opt_['data']['label']
        age = opt_['data']['age']
        gender = opt_['data']['gender']
        id = opt_['data']['id']
    else:
        data = opt_['data']
        label = opt_['label']
        id = opt_['id']

    if Flag_sw == True:
        import deepdish as ddish
        import pandas as pd

        data_sw = ddish.io.load(os.path.join(os.getcwd(), opt_['pre_pth'], opt_['site'] + '_correlation_matrix.h5'))
        tmp_id = []
        for idx in list(data_sw['id']):
            tmp_id.append(idx.split('.')[0])
        tmp_id = pd.DataFrame(tmp_id)

        cur_tr_id = id[opt_['fold_indices'][0]]
        cur_tmp_id = []
        for cur_idx in list(cur_tr_id):
            cur_tmp_id.append(cur_idx.split('.')[0])

        sw_tr_idx = np.where(tmp_id[0].isin(cur_tmp_id) == 1)[0]
        sw_tr_idx_rand = np.random.permutation(sw_tr_idx)
        x_trn = data_sw['data'][sw_tr_idx_rand, ...]
        y_trn = data_sw['label'][sw_tr_idx_rand, ...]
        y_trn = np.expand_dims(y_trn, 1)

        cur_tr_id = id[opt_['fold_indices'][1]]
        cur_tmp_id = []
        for cur_idx in list(cur_tr_id):
            cur_tmp_id.append(cur_idx.split('.')[0])

        sw_val_idx = np.where(tmp_id[0].isin(cur_tmp_id) == 1)[0]
        x_val = data_sw['data'][sw_val_idx, ...]
        y_val = data_sw['label'][sw_val_idx, ...]
        y_val = np.expand_dims(y_val, 1)

        cur_tr_id = id[opt_['fold_indices'][2]]
        cur_tmp_id = []
        for cur_idx in list(cur_tr_id):
            cur_tmp_id.append(cur_idx.split('.')[0])

        sw_tst_idx = np.where(tmp_id[0].isin(cur_tmp_id) == 1)[0]
        x_tst = data_sw['data'][sw_tst_idx, ...]
        y_tst = data_sw['label'][sw_tst_idx, ...]
        y_tst = np.expand_dims(y_tst, 1)

        tmp_sample_idx = np.random.permutation(70)
        x_trn = x_trn[tmp_sample_idx, ...]
        y_trn = y_trn[tmp_sample_idx, ...]

    else:
        x_trn, y_trn = data[opt_['fold_indices'][0], ...], label[[opt_['fold_indices'][0]], ...]
        y_trn = np.transpose(y_trn, (1, 0))

    x_tst, y_tst = data[opt_['fold_indices'][2], ...], label[[opt_['fold_indices'][2]], ...]
    x_val, y_val = data[opt_['fold_indices'][1], ...], label[[opt_['fold_indices'][1]], ...]
    y_val = np.transpose(y_val, (1, 0))
    y_tst = np.transpose(y_tst, (1, 0))

    if opt_['atlas'] == 'HO':
        input_size = 6670 #5995

    ae_input_size = [input_size, int(np.ceil(input_size * 1.5))]
    ae_output_size = [int(np.ceil(input_size * 1.5)), int(np.ceil(input_size * 0.3))]

    AE_1 = nn.DataParallel(Auto_Encoder_v2(opt_, ae_input_size[0], ae_output_size[0], flag='1')).cuda()
    AE_1prime = nn.DataParallel(Auto_Encoder_v2(opt_, ae_input_size[0], ae_output_size[0], flag='1')).cuda()
    AE_2 = nn.DataParallel(Auto_Encoder_v2(opt_, ae_input_size[1], ae_output_size[1], flag='2')).cuda()
    # NN = nn.DataParallel(NN.Neural_Net(ae_output_size[1], 8)).cuda()
    NN = nn.DataParallel(MLP(ae_output_size[1], 5, 2)).cuda()

    ##TODO: change the objective function (MAE -> MSE)
    optimizer1 = optim.Adam(list(AE_1.parameters()), lr=opt_['lr'], weight_decay=opt_['wd'])
    optimizer2 = optim.Adam(list(AE_2.parameters()), lr=opt_['lr'], weight_decay=opt_['wd'])
    # optimizer3 = optim.Adam(list(NN.parameters()), lr=exp['lr'], weight_decay=exp['wd'])
    optimizer3 = optim.Adam(list(NN.parameters()), lr=0.001, weight_decay=opt_['wd'])
    # optimizer3 = optim.Adam(list(NN.parameters()), lr=0.0001, weight_decay=5e-6)
    # optimizer3 = optim.SGD(list(NN.parameters()), lr=0.0001, weight_decay=exp['wd'])
    # optimizer3 = optim.SGD([{'params': AE_1.parameters(), 'lr': 1e-4},
    #                         {'params': AE_2.parameters(), 'lr': 1e-4},
    #                         {'params': NN.parameters()},
    #                         ], lr=1e-3, weight_decay=exp['wd'])
    # optimizer3 = optim.Adam(list(NN.parameters()), lr=0.0003, weight_decay=exp['wd'])

    criterion1 = Recon_Loss(opt_).cuda()
    criterion2 = Recon_Loss(opt_).cuda()
    criterion3 = Recon_Loss(opt_).cuda()

    criterion = Classifi_Loss(opt_).cuda()

    fold = opt_['tmp_fold']
    if not os.path.exists(os.path.join('/lustre/external/milab/2023_Project/TMI_revision/LRP_feature')):
        for f in range(5):
            os.makedirs(os.path.join('/lustre/external/milab/2023_Project/TMI_revision/LRP_feature/{}/train').format(f))
            os.makedirs(os.path.join('/lustre/external/milab/2023_Project/TMI_revision/LRP_feature/{}/val').format(f))
            os.makedirs(os.path.join('/lustre/external/milab/2023_Project/TMI_revision/LRP_feature/{}/test').format(f))

    print('Fold {} operation'.format(fold))
    print('Train')
    save_dir = os.path.join('/lustre/external/milab/2023_Project/TMI_revision/LRP_feature/{}/train').format(fold)
    x = torch.from_numpy(x_trn).float()

    # print('Valid')
    # save_dir = os.path.join('/lustre/external/milab/2023_Project/TMI_revision/LRP_feature/{}/val').format(fold)
    # x = torch.from_numpy(x_val).float()
    # print('Test')
    # save_dir = os.path.join('/lustre/external/milab/2023_Project/TMI_revision/LRP_feature/{}/test').format(fold)
    # x = torch.from_numpy(x_tst).float()

    train_loader = convert_Dloader(opt_['batch'], x_trn, y_trn, num_workers=0, shuffle= True)
    val_loader = convert_Dloader(x_val.shape[0], x_val, y_val, num_workers=0, shuffle=False)
    test_loader = convert_Dloader(x_tst.shape[0], x_tst, y_tst, num_workers=0, shuffle=False)

    valid = {'epoch':0, 'loss': 100000,}
    import natsort

    def mask_rois(x, num_roi, mode=True):
        mask = np.ones_like(x)
        re_x = []

        upper_indices = np.mask_indices(x.shape[1], np.triu, 1)

        for idx in range(x.shape[0]):
            #         roi_select = np.random.choice(x.shape[1], np.int(num_roi), replace=False)
            mask[idx, num_roi, :] = 0
            mask[idx, :, num_roi] = 0
            mask[idx, num_roi, num_roi] = 1

        if mode == True:
            masked_x = np.array(x) * mask
        else:
            masked_x = np.array(x)

        for i in range(masked_x.shape[0]):
            re_x.append(masked_x[i][upper_indices])

        return np.array(re_x), upper_indices, mask

    def rho(w, l):
        return w + [None, 0.1, 0.0, 0.0][l] * np.maximum(0, w)

    def incr(z, l):
        return z + [None, 0.0, 0.1, 0.0][l] * (z ** 2).mean() ** .5 + 1e-9

    m1_dict_file = natsort.natsorted(glob.glob(os.path.join(opt_['exp_dir'], '{}'.format(opt_['tmp_fold']), '*.ckpt')))[-1]
    m2_dict_file = natsort.natsorted(glob.glob(os.path.join(opt_['exp_dir'], '{}/fine_tune/'.format(opt_['tmp_fold']), '*.ckpt')))[-1]
    pre_model_dict = torch.load(m1_dict_file)
    AE_1prime.load_state_dict(pre_model_dict)
    model_dict = torch.load(m2_dict_file)
    AE_2.load_state_dict(model_dict)


    layers = nn.Sequential(AE_1prime.module.encoder, AE_2.module.encoder2, AE_2.module.decoder1, AE_1prime.module.decoder1)
    L = len(layers)

    W = [AE_1prime.module.encoder.state_dict()['0.weight'], AE_2.module.encoder2.state_dict()['0.weight'],
         AE_2.module.decoder1.state_dict()['0.weight'], AE_1prime.module.decoder1.state_dict()['0.weight']]
    B = [AE_1prime.module.encoder.state_dict()['0.bias'], AE_2.module.encoder2.state_dict()['0.bias'],
         AE_2.module.decoder1.state_dict()['0.bias'], AE_1prime.module.decoder1.state_dict()['0.bias']]

    # num_ROI = 110
    num_ROI = 116
    for num_roi in range(112,num_ROI):
        print('ROI {}'.format(num_roi))
        mask_x, upper_indices, mask = mask_rois(x, num_roi=num_roi, mode=True)
        non_mask_x, _, _ = mask_rois(x, num_roi=num_roi, mode=False)

        A = [torch.from_numpy(mask_x).cuda()] + [None] * L
        for l in range(L): A[l + 1] = layers[l].forward(A[l])

        roi_idx = np.zeros((1,num_ROI,num_ROI)) -1
        roi_val = np.arange(input_size)
        upper_indices = np.mask_indices(num_ROI, np.triu, 1)
        roi_idx[0][upper_indices] = roi_val

        if num_roi == 0:
            ROI = np.arange(num_ROI-1).reshape(1, -1)
        else:
            ROI = roi_idx[0][num_roi:num_roi + 1, :]
        if num_roi == num_ROI-1:
            ROI_ = roi_idx[0][:, num_roi:num_roi + 1][:-1]
        else:
            ROI_ = roi_idx[0][:, num_roi:num_roi + 1]
        t_ROI = np.unique(np.concatenate((ROI, ROI_.T), axis=1))[1:]

        for i, idx in enumerate(t_ROI):
            print('*' * (i % 10))
            T = torch.FloatTensor((1.0 * np.arange(mask_x.shape[1]) == idx).reshape([1, -1]))
            # R = [None] * L + [(A[-1].cpu() * T).data]
            # R = [None] * L + [(torch.exp(-torch.sqrt(torch.abs(torch.FloatTensor(non_mask_x).cuda() - A[-1])) / 0.05).max().cpu() * T).data]
            # R = [None] * L + [(torch.exp(-torch.sqrt(torch.abs(torch.FloatTensor(non_mask_x).cuda() - A[-1])) / 0.05).cpu() * T).data]
            R = [None] * L + [(torch.exp(-torch.pow((torch.FloatTensor(non_mask_x).cuda() - A[-1]), 2)).cpu() * T).data]

            for l in range(1, L)[::-1]:
                w = rho(W[l].cpu(), l)
                b = rho(B[l].cpu(), l)
                z = incr(torch.matmul(A[l].cpu(), w.T) + b, l)
                s = R[l + 1] / z
                c = torch.matmul(s, w).cpu()
                R[l] = A[l].cpu() * c

            w = W[0]
            wp = np.maximum(0, w.cpu())
            wm = np.minimum(0, w.cpu())

            lb = A[0].cpu() * 0 - 1
            hb = A[0].cpu() * 0 + 1

            z = torch.matmul(A[0], w.T).cpu() - torch.matmul(lb, wp.T).cpu() - torch.matmul(hb, wm.T).cpu() + 1e-9
            s = R[1] / z
            c, cp, cm = torch.matmul(s.cuda(), w), torch.matmul(s.cuda(), wp.cuda()), torch.matmul(s.cuda(),
                                                                                                   wm.cuda())
            R[0] = A[0] * c - lb.cuda() * cp - hb.cuda() * cm
            relev_ = R[0].detach().cpu()

            if i == 0:
                tmp_x = relev_.unsqueeze(1)
            else:
                tmp_x = np.concatenate((tmp_x, relev_.unsqueeze(1)), axis=1)

        np.save('{}/ROI_{}'.format(save_dir, num_roi), tmp_x)

    return print('cc')

def matrix_return(please):
    please_mat = []
    for p in range(please.shape[0]):
        matrix = np.eye(please.shape[1])
        for i in range(please.shape[1]):
            x_ = please[p]
            matrix[i, :i] = x_[i, :i]
            matrix[i, i + 1:] = x_[i, i:]
        np.fill_diagonal(matrix,1)
        please_mat.append(matrix)
    return np.array(please_mat)

def Extract_LRP_graph_feat(opt_, Flag_sw=True):
    from sklearn.svm import SVC
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.metrics import confusion_matrix, roc_auc_score
    import natsort
    import networkx as nx
    from tqdm import tqdm

    if opt_['atlas'] == 'HO':
        input_size = 6670 #5995

    """ ABIDE """
    # fold = opt_['tmp_fold']
    # if not os.path.exists(os.path.join('/lustre/external/milab/LRP_feature/Graph_LRP_feature')):
    #     for f in range(5):
    #         os.makedirs(os.path.join('/lustre/external/milab/LRP_feature/Graph_LRP_feature/{}').format(f))

    """ REST-meta MDD """
    fold = opt_['tmp_fold']
    if not os.path.exists(os.path.join('/lustre/external/milab/2023_Project/TMI_revision/LRP_feature/Graph_LRP_feature')):
        for f in range(5):
            os.makedirs(os.path.join('/lustre/external/milab/2023_Project/TMI_revision/LRP_feature/Graph_LRP_feature/{}').format(f))

    print('Fold {} operation'.format(fold))
    print('Train')
    for st_setting in list(['train','val','test']):
        # load_dir = natsort.natsorted(glob.glob(os.path.join('/lustre/external/milab/LRP_feature/{}/{}/ROI_*.npy').format(fold,st_setting)))
        load_dir = natsort.natsorted(glob.glob(os.path.join('/lustre/external/milab/2023_Project/TMI_revision/LRP_feature/{}/{}/ROI_*.npy').format(fold, st_setting)))
        print('Setting: {}'.format(st_setting))

        _ROI = 116 # 116 (AAL) 110 (HO)

        total_LRP_feature = []
        # """ ROI check """
        for file_roi_num in range(_ROI):
            print('ROI {}'.format(file_roi_num))
            data_roi = np.load(load_dir[file_roi_num])

            """ Sample check """
            total_sample_ = []
            for sample_num in range(data_roi.shape[0]):
                #     for sample_num in range(1):
                infi_i = 0
                """ ROI components """
                num_roi = file_roi_num
                for _, roi_name in enumerate(tqdm(np.arange(_ROI))):
                    if roi_name == num_roi:
                        pass
                    else:
                        roi_idx = np.zeros((1, _ROI, _ROI)) - 1
                        roi_val = np.arange(_ROI * (_ROI - 1) / 2)
                        upper_indices = np.mask_indices(_ROI, np.triu, 1)
                        roi_idx[0][upper_indices] = roi_val
                        a = np.array(roi_idx[0][roi_name, :], dtype=np.int)
                        a_ = np.array(roi_idx[0][:, roi_name], dtype=np.int)
                        tmp_ROI_indices = np.unique(np.concatenate((a, a_)))
                        sample = np.abs(data_roi[sample_num, :, np.array(tmp_ROI_indices[1:], dtype=np.int)])
                        if infi_i == 0:
                            total_sample = np.expand_dims(sample, 0)
                            infi_i += 1
                        else:
                            total_sample = np.concatenate((total_sample, np.expand_dims(sample, 0)), axis=0)
                            infi_i += 1

                total_sample_.append(np.sum(total_sample, 0))
            total_sample_ = np.sum(np.array(total_sample_), 1)
            total_LRP_feature.append(total_sample_)
        total_LRP_feature = np.array(total_LRP_feature)

        total_LRP_feature_ = np.transpose(total_LRP_feature, (1, 0, 2))

        norm_lrp_feature = []
        print('Normalized features')
        for i in range(total_LRP_feature_.shape[0]):
            min_ = matrix_return(total_LRP_feature_)[i].min()
            max_ = matrix_return(total_LRP_feature_)[i].max()
            tmp_norm_feat = (matrix_return(total_LRP_feature_)[i] - min_) / (max_ - min_)
            norm_lrp_feature.append(tmp_norm_feat)
        norm_lrp_feature = np.array(norm_lrp_feature)

        tmp_networkx = []
        print('Extract graph features')
        for idx in range(len(norm_lrp_feature)):
            tmp_a = norm_lrp_feature[idx, ...]
            np.fill_diagonal(tmp_a, 0)
            G = nx.from_numpy_array(tmp_a)
            tmp_b = nx.clustering(G, weight='weight')
            tmp_c = list(tmp_b.values())
            tmp_networkx.append(tmp_c)
        norm_lrp_graph_feature = np.array(tmp_networkx)

        # np.savez(os.path.join('/lustre/external/milab/LRP_feature/Graph_LRP_feature/{}/LRP_feature_{}_{}'.format(fold,st_setting,fold)),
        #          orig=matrix_return(total_LRP_feature_), norm=norm_lrp_feature, norm_graph=norm_lrp_graph_feature)
        np.savez(os.path.join('/lustre/external/milab/2023_Project/TMI_revision/LRP_feature/Graph_LRP_feature/{}/LRP_feature_{}_{}'.format(fold, st_setting, fold)),
                 orig=matrix_return(total_LRP_feature_),
                 norm=norm_lrp_feature,
                 norm_graph=norm_lrp_graph_feature)


