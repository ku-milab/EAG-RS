import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.losses import *
from src.data_loader import *
from src.pretrain import *
from src.fine_tune import *
from src.utils import writelog, cluster_coef_from_FC_sync
plt.switch_backend('agg')
# from gumbel import *

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

    def forward(self, c, y, z):
        x = self.dropout(c)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # return F.log_softmax(x, dim=-1)
        return x


def First_step(opt_, Flag_sw=True):

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

    else:
        x_trn, y_trn = data[opt_['fold_indices'][0], ...], label[[opt_['fold_indices'][0]], ...]
        y_trn = np.transpose(y_trn, (1, 0))

    # x_trn = data_sw['data'][opt_['fold_indices'][0], ...]
    # y_trn = data_sw['label'][opt_['fold_indices'][0], ...]
    x_tst, y_tst = data[opt_['fold_indices'][2], ...], label[[opt_['fold_indices'][2]], ...]
    x_val, y_val = data[opt_['fold_indices'][1], ...], label[[opt_['fold_indices'][1]], ...]
    # y_trn = np.expand_dims(y_trn,1)
    y_val = np.transpose(y_val, (1, 0))
    y_tst = np.transpose(y_tst, (1, 0))

    if opt_['atlas'] == 'HO':
        input_size = 5995

    ae_input_size = [input_size, int(np.ceil(input_size * 1.5))]
    ae_output_size = [int(np.ceil(input_size * 1.5)), int(np.ceil(input_size * 0.3))]

    if torch.cuda.is_available() == True:
        print('Cuda')
        AE_1 = nn.DataParallel(Auto_Encoder_v2(opt_, ae_input_size[0], ae_output_size[0], flag='1')).cuda()
        AE_1prime = nn.DataParallel(Auto_Encoder_v2(opt_, ae_input_size[0], ae_output_size[0], flag='1')).cuda()
        AE_2 = nn.DataParallel(Auto_Encoder_v2(opt_, ae_input_size[1], ae_output_size[1], flag='2')).cuda()

        criterion1 = Recon_Loss(opt_).cuda()
        criterion2 = Recon_Loss(opt_).cuda()
        criterion3 = Recon_Loss(opt_).cuda()
        criterion = Classifi_Loss(opt_).cuda()
    else:
        AE_1 = nn.DataParallel(Auto_Encoder_v2(opt_, ae_input_size[0], ae_output_size[0], flag='1'))
        AE_1prime = nn.DataParallel(Auto_Encoder_v2(opt_, ae_input_size[0], ae_output_size[0], flag='1'))
        AE_2 = nn.DataParallel(Auto_Encoder_v2(opt_, ae_input_size[1], ae_output_size[1], flag='2'))

        criterion1 = Recon_Loss(opt_)
        criterion2 = Recon_Loss(opt_)
        criterion3 = Recon_Loss(opt_)
        criterion = Classifi_Loss(opt_)

    ##TODO: change the objective function (MAE -> MSE)
    optimizer1 = optim.Adam(list(AE_1.parameters()), lr=opt_['lr'], weight_decay=opt_['wd'])
    optimizer2 = optim.Adam(list(AE_2.parameters()), lr=opt_['lr'], weight_decay=opt_['wd'])

    fold = opt_['tmp_fold']
    if not os.path.exists(os.path.join(opt_['exp_dir'],'{}'.format(fold))):
        os.makedirs(os.path.join(opt_['exp_dir'],'{}'.format(fold)))
        os.makedirs(os.path.join(opt_['exp_dir'], '{}/v_recon_img'.format(fold)))
        os.makedirs(os.path.join(opt_['exp_dir'], '{}/v_orig_img'.format(fold)))
        os.makedirs(os.path.join(opt_['exp_dir'], '{}/v_mask'.format(fold)))
        os.makedirs(os.path.join(opt_['exp_dir'], '{}/t_recon_img'.format(fold)))
        os.makedirs(os.path.join(opt_['exp_dir'], '{}/t_recon_img/TD'.format(fold)))
        os.makedirs(os.path.join(opt_['exp_dir'], '{}/t_recon_img/ASD'.format(fold)))
        os.makedirs(os.path.join(opt_['exp_dir'], '{}/t_orig_img'.format(fold)))
        os.makedirs(os.path.join(opt_['exp_dir'], '{}/t_mask'.format(fold)))
    f = open(os.path.join(opt_['exp_dir'],'{}'.format(fold),'fold{}_setting.log'.format(fold)), 'a')
    writelog(f, '----------------------')
    writelog(f, 'Model: %s' % AE_1)
    # ut2.writelog(f, 'Model Depth: {}'.format(exp['model_depth']))
    writelog(f, 'Weight Decay: {}'.format(opt_['wd']))
    writelog(f, '----------------------')
    writelog(f, 'Fold: {}'.format(opt_['tmp_fold']))
    writelog(f, 'Learning Rate: {}'.format(opt_['lr']))
    writelog(f, 'Learning Rate Decay: {}'.format(opt_['lr_decay']))
    writelog(f, 'Learning Rate Decay Step: {}'.format(opt_['lr_decay_step']))
    writelog(f, 'Epoch: {}'.format(opt_['epoch']))
    writelog(f, 'objective func.: {}'.format(criterion1))
    writelog(f, '----------------------')
    f.close()

    f = open(os.path.join(opt_['exp_dir'],'{}'.format(fold),'fold{}_log.log'.format(fold)), 'a')
    opt_['check_ckpt'] = os.path.join(opt_.get('exp_dir') + '/Network/Fold_{}/'.format(fold + 1))
    print('Fold {} operation'.format(fold))

    train_loader = convert_Dloader(opt_['batch'], x_trn, y_trn, num_workers=0, shuffle= True)
    val_loader = convert_Dloader(x_val.shape[0], x_val, y_val, num_workers=0, shuffle=False)
    test_loader = convert_Dloader(x_tst.shape[0], x_tst, y_tst, num_workers=0, shuffle=False)

    valid = {'epoch':0, 'loss': 100000,}
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=exp['lr_decay_step'], gamma=exp['lr_decay'])
    for epoch in range(opt_['epoch']):
        writelog(f, '- Epoch {} -'.format(epoch))
        writelog(f, '--- Training ---')
        train_(AE_1, train_loader, optimizer1, criterion1, opt_, f)
        writelog(f, '--- Valid. ---')
        v_loss, v_recon, v_orig, v_mask = valid_(AE_1, val_loader, criterion1, f, opt_, flag='valid')

        if v_loss < valid['loss']:
            if len(glob.glob(os.path.join(opt_['exp_dir'],'{}'.format(fold),'fold{}_*.ckpt'.format(fold)))) ==0:
                pass
            else:
                [os.remove(f) for f in glob.glob(os.path.join(opt_['exp_dir'],'{}'.format(fold),'fold{}_*.ckpt'.format(fold)))]
            torch.save(AE_1.state_dict(), os.path.join(opt_['exp_dir'],'{}'.format(fold),'fold{}_{}.ckpt'.format(fold, epoch)))
            ut2.writelog(f, 'Lowest Validation loss!: {}'.format(v_loss))
            ut2.writelog(f, 'Saved the model at Epoch: {}'.format(epoch))
            valid['loss'] = v_loss
            valid['epoch'] = epoch
            f2 = open(os.path.join(opt_['exp_dir'],'{}'.format(fold),'fold{}_result.log'.format(fold)), 'a')
            ut2.writelog(f2, 'Best epoch: {}'.format(epoch))
            t_loss, t_recon, t_orig, t_mask, t_label = valid_(AE_1, test_loader, criterion1, f2, opt_, flag='test')

    return print('cc')

def Second_step(opt_, Flag_sw=True):
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

    else:
        x_trn, y_trn = data[opt_['fold_indices'][0], ...], label[[opt_['fold_indices'][0]], ...]
        y_trn = np.transpose(y_trn, (1, 0))

    x_tst, y_tst = data[opt_['fold_indices'][2], ...], label[[opt_['fold_indices'][2]], ...]
    x_val, y_val = data[opt_['fold_indices'][1], ...], label[[opt_['fold_indices'][1]], ...]
    # y_trn = np.expand_dims(y_trn,1)
    y_val = np.transpose(y_val, (1, 0))
    y_tst = np.transpose(y_tst, (1, 0))

    if opt_['atlas'] == 'HO':
        input_size = 5995

    ae_input_size = [input_size, int(np.ceil(input_size * 1.5))]
    ae_output_size = [int(np.ceil(input_size * 1.5)), int(np.ceil(input_size * 0.3))]

    if torch.cuda.is_available() == True:
        AE_1 = nn.DataParallel(Auto_Encoder_v2(opt_, ae_input_size[0], ae_output_size[0], flag='1')).cuda()
        AE_1prime = nn.DataParallel(Auto_Encoder_v2(opt_, ae_input_size[0], ae_output_size[0], flag='1')).cuda()
        AE_2 = nn.DataParallel(Auto_Encoder_v2(opt_, ae_input_size[1], ae_output_size[1], flag='2')).cuda()

        criterion1 = Recon_Loss(opt_).cuda()
        criterion2 = Recon_Loss(opt_).cuda()
        criterion3 = Recon_Loss(opt_).cuda()
        criterion = Classifi_Loss(opt_).cuda()
    else:
        AE_1 = Auto_Encoder_v2(opt_, ae_input_size[0], ae_output_size[0], flag='1')
        AE_1prime = Auto_Encoder_v2(opt_, ae_input_size[0], ae_output_size[0], flag='1')
        AE_2 = Auto_Encoder_v2(opt_, ae_input_size[1], ae_output_size[1], flag='2')

        criterion1 = Recon_Loss(opt_)
        criterion2 = Recon_Loss(opt_)
        criterion3 = Recon_Loss(opt_)
        criterion = Classifi_Loss(opt_)

    ##TODO: change the objective function (MAE -> MSE)
    optimizer1 = optim.Adam(list(AE_1.parameters()), lr=opt_['lr'], weight_decay=opt_['wd'])
    optimizer2 = optim.Adam(list(AE_2.parameters()), lr=opt_['lr'], weight_decay=opt_['wd'])

    fold = opt_['tmp_fold']
    if not os.path.exists(os.path.join(opt_['exp_dir'], '{}/fine_tune'.format(fold))):
        os.makedirs(os.path.join(opt_['exp_dir'], '{}/fine_tune'.format(fold)))
        os.makedirs(os.path.join(opt_['exp_dir'], '{}/fine_tune/v_recon_img'.format(fold)))
        os.makedirs(os.path.join(opt_['exp_dir'], '{}/fine_tune/v_orig_img'.format(fold)))
        os.makedirs(os.path.join(opt_['exp_dir'], '{}/fine_tune/v_mask'.format(fold)))
        os.makedirs(os.path.join(opt_['exp_dir'], '{}/fine_tune/t_recon_img'.format(fold)))
        os.makedirs(os.path.join(opt_['exp_dir'], '{}/fine_tune/t_recon_img/TD'.format(fold)))
        os.makedirs(os.path.join(opt_['exp_dir'], '{}/fine_tune/t_recon_img/ASD'.format(fold)))
        os.makedirs(os.path.join(opt_['exp_dir'], '{}/fine_tune/t_orig_img'.format(fold)))
        os.makedirs(os.path.join(opt_['exp_dir'], '{}/fine_tune/t_mask'.format(fold)))
    f = open(os.path.join(opt_['exp_dir'],'{}/fine_tune'.format(fold),'fold{}_setting.log'.format(fold)), 'a')
    writelog(f, '----------------------')
    writelog(f, 'Model: %s' % AE_1)
    # ut2.writelog(f, 'Model Depth: {}'.format(exp['model_depth']))
    writelog(f, 'Weight Decay: {}'.format(opt_['wd']))
    writelog(f, '----------------------')
    writelog(f, 'Fold: {}'.format(opt_['tmp_fold']))
    writelog(f, 'Learning Rate: {}'.format(opt_['lr']))
    writelog(f, 'Learning Rate Decay: {}'.format(opt_['lr_decay']))
    writelog(f, 'Learning Rate Decay Step: {}'.format(opt_['lr_decay_step']))
    writelog(f, 'Epoch: {}'.format(opt_['epoch']))
    writelog(f, 'objective func.: {}'.format(criterion1))
    writelog(f, '----------------------')
    f.close()

    f = open(os.path.join(opt_['exp_dir'],'{}'.format(fold),'fold{}_log.log'.format(fold)), 'a')
    opt_['check_ckpt'] = os.path.join(opt_.get('exp_dir') + '/Network/Fold_{}/'.format(fold + 1))
    print('Fold {} operation'.format(fold))

    train_loader = convert_Dloader(opt_['batch'], x_trn, y_trn, num_workers=0, shuffle= True)
    val_loader = convert_Dloader(x_val.shape[0], x_val, y_val, num_workers=0, shuffle=False)
    test_loader = convert_Dloader(x_tst.shape[0], x_tst, y_tst, num_workers=0, shuffle=False)

    valid = {'epoch':0, 'loss': 100000,}
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=exp['lr_decay_step'], gamma=exp['lr_decay'])
    for epoch in range(opt_['epoch']):
        writelog(f, '- Epoch {} -'.format(epoch))
        writelog(f, '--- Training ---')
        fine_tuning_(AE_2, AE_1prime, train_loader, optimizer2, criterion2, criterion3, f, opt_)
        writelog(f, '--- Valid. ---')
        v_loss, v_recon, v_orig, v_mask = valid_fine_tune(AE_2, AE_1prime, val_loader, criterion2, criterion3, f, opt_, flag='valid')

        # for img_idx in range (v_recon.shape[0]):
        #     plt.imshow(v_mask[img_idx])
        #     plt.clim(-.6, .6)
        #     plt.colorbar()
        #     plt.savefig(os.path.join(opt_['exp_dir'], '{}'.format(fold), 'v_mask/{}.png'.format(epoch)))
        #     plt.close()
        #     # plt.imshow((v_orig-v_recon)[img_idx])
        #     plt.imshow((v_recon)[img_idx])
        #     plt.clim()
        #     plt.colorbar()
        #     plt.savefig(os.path.join(opt_['exp_dir'], '{}'.format(fold), 'v_recon_img/{}_{}.png'.format(epoch,img_idx)))
        #     plt.close()
        #     if epoch == 0:
        #         plt.imshow(v_orig[img_idx])
        #         plt.clim(-.6, .6)
        #         plt.colorbar()
        #         plt.savefig(os.path.join(opt_['exp_dir'], '{}'.format(fold), 'v_orig_img/{}.png'.format(img_idx)))
        #         plt.close()
        #     else:
        #         pass
        if v_loss < valid['loss']:
            if len(glob.glob(os.path.join(opt_['exp_dir'],'{}/fine_tune'.format(fold),'fold{}_*.ckpt'.format(fold)))) ==0:
                pass
            else:
                [os.remove(f) for f in glob.glob(os.path.join(opt_['exp_dir'],'{}/fine_tune'.format(fold),'fold{}_*.ckpt'.format(fold)))]
            torch.save(AE_2.state_dict(), os.path.join(opt_['exp_dir'],'{}/fine_tune'.format(fold),'fold{}_{}.ckpt'.format(fold, epoch)))
            ut2.writelog(f, 'Lowest Validation loss!: {}'.format(v_loss))
            ut2.writelog(f, 'Saved the model at Epoch: {}'.format(epoch))
            valid['loss'] = v_loss
            valid['epoch'] = epoch
            f2 = open(os.path.join(opt_['exp_dir'],'{}/fine_tune'.format(fold),'fold{}_result.log'.format(fold)), 'a')
            ut2.writelog(f2, 'Best epoch: {}'.format(epoch))
            t_loss, t_recon, t_orig, t_mask, t_label = valid_fine_tune(AE_2, AE_1prime, test_loader, criterion2, criterion3, f2, opt_, flag='test')

            cnt = 0
            for t_img_idx in range(t_recon.shape[0]):
                plt.imshow(t_mask[t_img_idx])
                # plt.clim(-.6, .6)
                plt.colorbar()
                plt.savefig(os.path.join(opt_['exp_dir'], '{}/fine_tune'.format(fold), 't_mask/{}.png'.format(epoch)))
                plt.close()
                # plt.imshow((t_orig-t_recon)[t_img_idx])
                plt.imshow((t_recon)[t_img_idx])
                plt.clim()
                plt.colorbar()
                plt.savefig(os.path.join(opt_['exp_dir'], '{}/fine_tune'.format(fold), 't_recon_img/{}/{}_{}.png'.format(t_label[t_img_idx], epoch, t_img_idx)))
                plt.close()
                if os.path.isfile(os.path.join(opt_['exp_dir'], '{}/fine_tune'.format(fold), 't_orig_img')):
                    pass
                else:
                    plt.imshow(t_orig[t_img_idx])
                    # plt.clim(-.6, .6)
                    plt.colorbar()
                    plt.savefig(os.path.join(opt_['exp_dir'], '{}/fine_tune'.format(fold), 't_orig_img/{}.png'.format(t_img_idx)))
                    plt.close()
                cnt +=1
                if cnt == 15:
                    break
            f2.close()
    return print('cc')


def Final_step(opt_, Flag_sw=True):
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

    trn_features = np.load('/lustre/external/milab/2023_Project/TMI_revision/LRP_feature/Graph_LRP_feature/{}/LRP_feature_train_{}.npz'.format(opt_['tmp_fold'], opt_['tmp_fold']))['orig']
    val_features = np.load('/lustre/external/milab/2023_Project/TMI_revision/LRP_feature/Graph_LRP_feature/{}/LRP_feature_val_{}.npz'.format(opt_['tmp_fold'], opt_['tmp_fold']))['orig']
    te_features = np.load('/lustre/external/milab/2023_Project/TMI_revision/LRP_feature/Graph_LRP_feature/{}/LRP_feature_test_{}.npz'.format(opt_['tmp_fold'], opt_['tmp_fold']))['orig']

    trn_mask = np.ones_like(trn_features)
    val_mask = np.ones_like(val_features)
    te_mask = np.ones_like(te_features)

    for idx in range(trn_features.shape[0]):
        for roi in range(trn_features.shape[1]):
            trn_mask[idx][roi][trn_features[idx][roi] < trn_features[idx][roi].mean()] = 0

    for idx in range(val_features.shape[0]):
        for roi in range(val_features.shape[1]):
            val_mask[idx][roi][val_features[idx][roi] < val_features[idx][roi].mean()] = 0

    for idx in range(te_features.shape[0]):
        for roi in range(te_features.shape[1]):
            te_mask[idx][roi][te_features[idx][roi] < te_features[idx][roi].mean()] = 0

    trn_tmp = (trn_features * trn_mask)
    val_tmp = (val_features * val_mask)
    te_tmp = (te_features * te_mask)

    scaler_ = MinMaxScaler().fit(trn_tmp.reshape(-1,116*116))
    trn_tmp = scaler_.transform(trn_tmp.reshape(-1,116*116)).reshape(-1,116,116)
    val_tmp = scaler_.transform(val_tmp.reshape(-1,116*116)).reshape(-1,116,116)
    te_tmp = scaler_.transform(te_tmp.reshape(-1,116*116)).reshape(-1,116,116)

    scaler_2 = MinMaxScaler().fit(trn_features.reshape(-1,116*116))
    trn_features = scaler_2.transform(trn_features.reshape(-1,116*116)).reshape(-1,116,116)
    val_features = scaler_2.transform(val_features.reshape(-1,116*116)).reshape(-1,116,116)
    te_features = scaler_2.transform(te_features.reshape(-1,116*116)).reshape(-1,116,116)

    trn_hist_feat, val_hist_feat, te_hist_feat = [], [], []
    for idx in range(trn_tmp.shape[0]):
        trn_hist_feat.append(trn_tmp[idx].sum(0))
    trn_hist_feat = np.array(trn_hist_feat)

    for idx in range(val_tmp.shape[0]):
        val_hist_feat.append(val_tmp[idx].sum(0))
    val_hist_feat = np.array(val_hist_feat)

    for idx in range(te_tmp.shape[0]):
        te_hist_feat.append(te_tmp[idx].sum(0))
    te_hist_feat = np.array(te_hist_feat)

    trn_hist_feat2, val_hist_feat2, te_hist_feat2 = [], [], []
    for idx in range(trn_mask.shape[0]):
        trn_hist_feat2.append(trn_mask[idx].sum(0))
    trn_hist_feat2 = np.array(trn_hist_feat2)

    for idx in range(val_mask.shape[0]):
        val_hist_feat2.append(val_mask[idx].sum(0))
    val_hist_feat2 = np.array(val_hist_feat2)

    for idx in range(te_mask.shape[0]):
        te_hist_feat2.append(te_mask[idx].sum(0))
    te_hist_feat2 = np.array(te_hist_feat2)


    if opt_['atlas'] == 'HO':
        input_size = 6670  #5995

    ae_input_size = [input_size, int(np.ceil(input_size * 1.5))]
    ae_output_size = [int(np.ceil(input_size * 1.5)), int(np.ceil(input_size * 0.3))]

    if torch.cuda.is_available() == True:
        AE_1 = nn.DataParallel(Auto_Encoder_v2(opt_, ae_input_size[0], ae_output_size[0], flag='1')).cuda()
        AE_1prime = nn.DataParallel(Auto_Encoder_v2(opt_, ae_input_size[0], ae_output_size[0], flag='1')).cuda()
        AE_2 = nn.DataParallel(Auto_Encoder_v2(opt_, ae_input_size[1], ae_output_size[1], flag='2')).cuda()
        # NN = nn.DataParallel(NN.Neural_Net(ae_output_size[1], 8)).cuda()
        NN = nn.DataParallel(MLP(ae_output_size[1], 10, 2)).cuda()

        temp = 1.
        gumbel_model = VAE_gumbel(temp)
        criterion = Classifi_Loss(opt_).cuda()
    else:
        AE_1 = nn.DataParallel(Auto_Encoder_v2(opt_, ae_input_size[0], ae_output_size[0], flag='1'))
        AE_1prime = nn.DataParallel(Auto_Encoder_v2(opt_, ae_input_size[0], ae_output_size[0], flag='1'))
        AE_2 = nn.DataParallel(Auto_Encoder_v2(opt_, ae_input_size[1], ae_output_size[1], flag='2'))
        NN = nn.DataParallel(MLP(ae_output_size[1], 10, 2))

        temp = 1.
        gumbel_model = VAE_gumbel(temp)

        criterion = Classifi_Loss(opt_)

    ##TODO: change the objective function (MAE -> MSE)
    optimizer1 = optim.Adam(list(AE_1.parameters()), lr=opt_['lr'], weight_decay=opt_['wd'])
    optimizer2 = optim.Adam(list(AE_2.parameters()), lr=opt_['lr'], weight_decay=opt_['wd'])
    # optimizer3 = optim.Adam(list(NN.parameters()), lr=1e-6, weight_decay=opt_['wd'])
    optimizer3 = optim.Adam([{'params': NN.parameters()},
                            {'params': gumbel_model.parameters(), 'lr': 1e-5},
                            {'params': AE_1.parameters(), 'lr': 1e-5},
                            {'params': AE_2.parameters(), 'lr': 1e-5},
                             ], lr=1e-4, weight_decay=opt_['wd'])

    fold = opt_['tmp_fold']
    if not os.path.exists(os.path.join(opt_['exp_dir'], '{}/fine_tune'.format(fold))):
        os.makedirs(os.path.join(opt_['exp_dir'], '{}/fine_tune'.format(fold)))
        os.makedirs(os.path.join(opt_['exp_dir'], '{}/fine_tune/v_recon_img'.format(fold)))
        os.makedirs(os.path.join(opt_['exp_dir'], '{}/fine_tune/v_orig_img'.format(fold)))
        os.makedirs(os.path.join(opt_['exp_dir'], '{}/fine_tune/v_mask'.format(fold)))
        os.makedirs(os.path.join(opt_['exp_dir'], '{}/fine_tune/t_recon_img'.format(fold)))
        os.makedirs(os.path.join(opt_['exp_dir'], '{}/fine_tune/t_recon_img/TD'.format(fold)))
        os.makedirs(os.path.join(opt_['exp_dir'], '{}/fine_tune/t_recon_img/ASD'.format(fold)))
        os.makedirs(os.path.join(opt_['exp_dir'], '{}/fine_tune/t_orig_img'.format(fold)))
        os.makedirs(os.path.join(opt_['exp_dir'], '{}/fine_tune/t_mask'.format(fold)))

    f = open(os.path.join(opt_['exp_dir'],'{}'.format(fold),'fold{}_log_cls.log'.format(fold)), 'a')
    opt_['check_ckpt'] = os.path.join(opt_.get('exp_dir') + '/Network/Fold_{}/'.format(fold + 1))
    print('Fold {} operation'.format(fold))

    train_loader = convert_Dloader_fusion(opt_['batch'], x_trn, y_trn, trn_features, trn_hist_feat, trn_hist_feat2, num_workers=0, shuffle= True)
    # train_loader = convert_Dloader(50, x_trn, y_trn, num_workers=0, shuffle=True)
    val_loader = convert_Dloader_fusion(x_val.shape[0], x_val, y_val, val_features, val_hist_feat, val_hist_feat2, num_workers=0, shuffle=False)
    test_loader = convert_Dloader_fusion(x_tst.shape[0], x_tst, y_tst, te_features, te_hist_feat, te_hist_feat2, num_workers=0, shuffle=False)

    valid = {'epoch':0, 'loss': 100000, 'auc':0}
    # for epoch in range(opt_['epoch']):
    for epoch in range(300):
        # if not os.path.exists(os.path.join(opt_['exp_dir'], '{}/fine_tune_MLP8_05_b50/NN_final_with_mask.{}'.format(fold, opt_['remove_roi']))):
        if not os.path.exists(os.path.join(opt_['exp_dir'], '{}/0721_gumbel_MLP10/NN_final_with_mask.{}'.format(fold, opt_['remove_roi']))):
            os.makedirs(os.path.join(opt_['exp_dir'], '{}/0721_gumbel_MLP10/NN_final_with_mask.{}'.format(fold, opt_['remove_roi'])))

        ut2.writelog(f, '--- Training {}---'.format(epoch))

        fine_tuning_NN_22(AE_2, AE_1prime, NN, gumbel_model, train_loader, optimizer3, criterion, f, opt_)
        ut2.writelog(f, '--- Valid. ---')
        v_auc, v_loss, v_recon, v_orig, v_mask = valid_fine_tune_NN_22(AE_2, AE_1prime, NN, gumbel_model, val_loader, criterion, f, opt_, flag='valid')

        if v_loss < valid['loss']:
        # if v_auc > valid['auc']:
            torch.save(NN.state_dict(), os.path.join(opt_['exp_dir'], '{}/0721_gumbel_MLP10/NN_final_with_mask.{}'.format(fold, opt_['remove_roi']), 'fold{}_{}.ckpt'.format(fold, epoch)))
            # torch.save(NN.state_dict(), os.path.join(opt_['exp_dir'], '{}/fine_tune_MLP5_05/NN_final_with_mask.{}'.format(fold, opt_['remove_roi']), 'fold{}_{}.ckpt'.format(fold, epoch)))
            # torch.save(NN.state_dict(), os.path.join(opt_['exp_dir'], '{}/fine_tune_MLP5_05/NN_final_with_mask.{}'.format(fold, opt_['remove_roi']), 'fold{}_{}.ckpt'.format(fold, epoch)))

            ut2.writelog(f, 'Lowest Validation loss!: {}'.format(v_loss))
            ut2.writelog(f, 'Saved the model at Epoch: {}'.format(epoch))
            valid['loss'] = v_loss
            valid['auc'] = v_auc
            valid['epoch'] = epoch
            f2 = open( os.path.join(opt_['exp_dir'], '{}/0721_gumbel_MLP10/NN_final_with_mask.{}'.format(fold, opt_['remove_roi']), 'mlp10_lr554_latent10_fold{}_result.log'.format(fold)), 'a')
            ut2.writelog(f2, 'Best epoch: {}'.format(epoch))
            _, v_loss, v_recon, v_orig, v_mask = valid_fine_tune_NN_22(AE_2, AE_1prime, NN, gumbel_model, val_loader, criterion, f2, opt_, flag='valid')
            t_loss, t_recon, t_orig, t_mask, t_label, t_embed = valid_fine_tune_NN_22(AE_2, AE_1prime, NN, gumbel_model, test_loader, criterion, f2, opt_, flag='test')

            # cnt = 0
            # for t_img_idx in range(t_recon.shape[0]):
            #     plt.imshow(t_mask[t_img_idx])
            #     # plt.clim(-.6, .6)
            #     plt.colorbar()
            #     plt.savefig(os.path.join(opt_['exp_dir'], '{}/gumbel_MLP5'.format(fold), 't_mask/{}.png'.format(epoch)))
            #     plt.close()
            #     # plt.imshow((t_orig-t_recon)[t_img_idx])
            #     # plt.imshow((t_recon)[t_img_idx])
            #     # plt.clim()
            #     # plt.colorbar()
            #     # plt.savefig(os.path.join(opt_['exp_dir'], '{}/gumbel_MLP5'.format(fold),
            #     #                          't_recon_img/{}/{}_{}.png'.format(t_label[t_img_idx], epoch, t_img_idx)))
            #     # plt.close()
            #     if os.path.isfile(os.path.join(opt_['exp_dir'], '{}/gumbel_MLP5'.format(fold), 't_orig_img')):
            #         pass
            #     else:
            #         plt.imshow(t_orig[t_img_idx])
            #         # plt.clim(-.6, .6)
            #         plt.colorbar()
            #         plt.savefig(
            #             os.path.join(opt_['exp_dir'], '{}/gumbel_MLP5'.format(fold), 't_orig_img/{}.png'.format(t_img_idx)))
            #         plt.close()
            #     cnt += 1
            #     if cnt == 15:
            #         break
            f2.close()