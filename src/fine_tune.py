from tqdm import tqdm
from src.mask_vector import *

import torch
import sys, os
import glob

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src import utils as ut2
from gumbel import *


def fine_tuning_(model, pre_model, dataloader, optimizer, object_func, object_func2, f, exp):
    import natsort
    dict_file = natsort.natsorted(glob.glob(os.path.join(exp['exp_dir'], '{}'.format(exp['tmp_fold']), '*.ckpt')))[-1]
    # print(dict_file)
    pre_model_dict = torch.load(dict_file)
    # print(pre_model_dict)
    pre_model.load_state_dict(pre_model_dict, strict=False)
    model.train()
    loss_model = 0
    n_samples = 0
    for _, (x, label) in enumerate(tqdm(dataloader)):
        mask_x, upper_indices, _ = mask_rois(x, num_roi=np.round(x.shape[1]*exp['remove_roi']))

        if torch.cuda.is_available() == True:
            y = x.cuda()
            batch_size = x.shape[0]
            with torch.no_grad():
                # print(pre_model.module)
                h1 = pre_model.module.encoder(torch.from_numpy(mask_x).cuda())
            recon_h2, _ = model(h1.cuda(), flag='2')
            with torch.no_grad():
                recon_h1 = pre_model.module.decoder1(recon_h2.cuda())

            tmp_recon = torch.zeros(batch_size, x.shape[1], x.shape[1])
            for i in range(batch_size):
                tmp_recon[i][upper_indices] = recon_h1[i].cpu()
            recon_x = tmp_recon + torch.transpose(tmp_recon,2,1)

            loss_ = object_func(recon_h2.cuda(), h1.cuda())
            loss_2 = object_func2(recon_x.cuda(), y)

            loss_t = (loss_+loss_2) / (2 * batch_size)
            # print('{:4f}'.format(loss_t))
            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()

            loss_model += (loss_+loss_2).item()
            n_samples += batch_size
            del loss_
        else:
            y = x
            batch_size = x.shape[0]
            with torch.no_grad():
                # h1 = pre_model.module.encoder(torch.from_numpy(mask_x))
                h1 = pre_model.encoder(torch.from_numpy(mask_x))
            recon_h2, _ = model(h1, flag='2')
            with torch.no_grad():
                # recon_h1 = pre_model.module.decoder1(recon_h2)
                recon_h1 = pre_model.decoder1(recon_h2)

            tmp_recon = torch.zeros(batch_size, x.shape[1], x.shape[1])
            for i in range(batch_size):
                tmp_recon[i][upper_indices] = recon_h1[i].cpu()
            recon_x = tmp_recon + torch.transpose(tmp_recon, 2, 1)

            loss_ = object_func(recon_h2, h1)
            loss_2 = object_func2(recon_x, y)

            loss_t = (loss_ + loss_2) / (2 * batch_size)
            # print('{:4f}'.format(loss_t))
            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()

            loss_model += (loss_ + loss_2).item()
            n_samples += batch_size
            del loss_

    loss_resnet_total = loss_model / (2 * n_samples)
    #     if i == 0:
    #         all_prob = soft_prob
    #         all_prediction = prediction
    #         all_labels = y
    #     else:
    #         all_prob = torch.cat((all_prob, soft_prob), dim=0)
    #         all_prediction = torch.cat((all_prediction, prediction), dim=0)
    #         all_labels = torch.cat((all_labels, y), dim=0)
    # # (acc, sen, spec, auc), _ = ut2.Evaluate_Binary(all_labels.detach(), all_prediction.detach(), all_prob.detach(), config.num_classes)
    ut2.writelog(f, '--- Fine tuning ---')
    ut2.writelog(f,'Loss: {:.4f}'.format(loss_resnet_total))

def valid_fine_tune(model, pre_model, dataloader, object_func, object_func2, f, exp, flag):
    model.eval()
    pre_model.eval()

    val_loss_resnet = 0
    n_samples = 0
    v_recon_output = []
    v_label_output = []
    v_orig_output = []
    with torch.no_grad():
        for _, (v_x, v_label) in enumerate(tqdm(dataloader)):
            v_mask_x, v_upper_indices, v_mask = mask_rois(v_x, num_roi=np.round(v_x.shape[1]*exp['remove_roi']),mode=False)
            v_y = v_x.cuda()
            batch_size = v_x.shape[0]

            v_h1 = pre_model.module.encoder(torch.from_numpy(v_mask_x).cuda())
            # v_h1 = pre_model.encoder(torch.from_numpy(v_mask_x).cuda())
            v_recon_h2, _ = model(v_h1, flag='2')
            recon_h1 = pre_model.module.decoder1(v_recon_h2)
            # recon_h1 = pre_model.decoder1(v_recon_h2)

            tmp_recon = torch.zeros(batch_size, v_x.shape[1], v_x.shape[1])
            for i in range(batch_size):
                tmp_recon[i][v_upper_indices] = recon_h1[i].cpu()
                if flag == 'test':
                    if v_label[i] == 1:
                        v_label_output.append('TD')
                    else:
                        v_label_output.append('ASD')
                else:
                    pass
            v_recon_ = tmp_recon + torch.transpose(tmp_recon, 2, 1)


            # v_loss_ = object_func(v_recon_.cuda(), v_y)
            v_loss_ = object_func(v_recon_h2.cuda(), v_h1.cuda())
            v_loss_2 = object_func2(v_recon_.cuda(), v_y.cuda())

            # v_loss_t = (v_loss_ + v_loss_2) / (2 * batch_size)
            val_loss_resnet += (v_loss_ + v_loss_2).item()
            n_samples += batch_size
            v_recon_output.append(v_recon_.detach().cpu().numpy())
            v_orig_output.append(v_x.cpu().numpy())
            del v_loss_

        v_loss_resnet_total = val_loss_resnet / (2 * n_samples)
        if flag == 'test':
            ut2.writelog(f, '(Te)Loss: {:.4f}'.format(v_loss_resnet_total))
            return v_loss_resnet_total, np.array(v_recon_output).squeeze(), np.array(v_orig_output).squeeze(), v_mask, v_label_output
        else:
            ut2.writelog(f, '(V)Loss: {:.4f}'.format(v_loss_resnet_total))
            return v_loss_resnet_total, np.array(v_recon_output).squeeze(), np.array(v_orig_output).squeeze(), v_mask

def fine_tuning_NN(model, pre_model, NN, dataloader, optimizer, object_func, f, exp):
    import natsort
    m1_dict_file = natsort.natsorted(glob.glob(os.path.join(exp['exp_dir'], '{}'.format(exp['tmp_fold']), '*.ckpt')))[-1]
    m2_dict_file = natsort.natsorted(glob.glob(os.path.join(exp['exp_dir'], '{}/fine_tune/'.format(exp['tmp_fold']), '*.ckpt')))[-1]
    pre_model_dict = torch.load(m1_dict_file)
    pre_model.load_state_dict(pre_model_dict)
    model_dict = torch.load(m2_dict_file)
    model.load_state_dict(model_dict, strict=False)
    NN.train()
    loss_model = 0
    n_samples = 0
    for i, (x, label) in enumerate(tqdm(dataloader)):
        mask_x, upper_indices, _ = mask_rois(x, num_roi=np.round(x.shape[1] * exp['remove_roi']), mode=False)

        y = label.long().cuda()

        batch_size = x.shape[0]
        with torch.no_grad():
            # h1 = pre_model.module.encoder(torch.from_numpy(mask_x).cuda())
            h1 = pre_model.module.encoder(torch.from_numpy(mask_x).cuda().float()) #dAE
            _, embedded = model(h1, flag='2')

        # print(model.module.decoder1)
        prob = NN(embedded)
        # import matplotlib.pyplot as plt
        # for k in range(10):
        #     plt.imshow(embedded[k].detach().cpu().reshape(-1, 229))
        #     plt.savefig('{}/{}/tmp_result/{}_{}.png'.format(exp['exp_dir'], exp['tmp_fold'],i,k))
        #     plt.close()
        soft_prob = torch.nn.Softmax(dim=1)(prob)
        prediction = torch.argmax(soft_prob, 1)

        loss_ = object_func(prob, y.squeeze())

        loss_t = loss_ * batch_size

        optimizer.zero_grad()
        loss_t.backward()
        optimizer.step()

        loss_model += loss_.item()
        n_samples += batch_size
        del loss_

        loss_resnet_total = loss_model / n_samples
        if i == 0:
            all_prob = soft_prob
            all_prediction = prediction
            all_labels = y
        else:
            all_prob = torch.cat((all_prob, soft_prob), dim=0)
            all_prediction = torch.cat((all_prediction, prediction), dim=0)
            all_labels = torch.cat((all_labels, y), dim=0)
    (acc, sen, spec, auc), _ = ut2.Evaluate_Binary(all_labels.detach(), all_prediction.detach(), all_prob.detach(), 2)
    ut2.writelog(f, '--- Fine tuning ---')
    ut2.writelog(f,'Loss: {:.4f}'.format(loss_resnet_total))
    ut2.writelog(f,'(T)ACC {:.4f} | SEN {:.4f} | SPC {:.4f} | AUC {:.4f}'.format(acc, sen, spec, auc))

def fine_tuning_NN_22(model, pre_model, NN, gumber_model, dataloader, optimizer, object_func, f, exp):
    import natsort
    m1_dict_file = natsort.natsorted(glob.glob(os.path.join(exp['exp_dir'], '{}'.format(exp['tmp_fold']), '*.ckpt')))[-1]
    m2_dict_file = natsort.natsorted(glob.glob(os.path.join(exp['exp_dir'], '{}/fine_tune/'.format(exp['tmp_fold']), '*.ckpt')))[-1]
    if torch.cuda.is_available() == True:
        pre_model_dict = torch.load(m1_dict_file)
        pre_model.load_state_dict(pre_model_dict)
        model_dict = torch.load(m2_dict_file)
        model.load_state_dict(model_dict)
    else:
        pre_model_dict = torch.load(m1_dict_file, map_location=torch.device('cpu'))
        pre_model.load_state_dict(pre_model_dict)
        model_dict = torch.load(m2_dict_file, map_location=torch.device('cpu'))
        model.load_state_dict(model_dict)

    pre_model.train()
    model.train()
    NN.train()
    gumber_model.train()
    loss_model = 0
    n_samples = 0
    for i, (x, l_lrp_x, u_lrp_x, hist_x, hist_x2, label) in enumerate(tqdm(dataloader)):
        # mask_x, upper_indices, _ = mask_rois(x, num_roi=np.round(x.shape[1] * exp['remove_roi']), mode=False)
        # mask_l_lrp_x, upper_indices, _ = mask_rois(l_lrp_x, num_roi=np.round(x.shape[1] * exp['remove_roi']), mode=False)
        # mask_u_lrp_x, upper_indices, _ = mask_rois(u_lrp_x, num_roi=np.round(x.shape[1] * exp['remove_roi']), mode=False)

        y = label.long()
        hard = False
        recon_batch, qy = gumber_model(hist_x, hist_x2, .01, hard)
        loss_gumbel = loss_function(recon_batch, hist_x, hist_x2, qy)

        # selected_ROIs = torch.topk(recon_batch, int(recon_batch.shape[1] * .1), largest=False)
        selected_ROIs = recon_batch > 0.5
        # selected_ROI_indices = exp['roi_indices'][selected_ROIs[1]].reshape(-1, int(len(recon_batch) * .1) * 109)
        mask_x, _, mask = mask_rois_v2(x, selected_ROIs, mode=True)
        mask_l_lrp_x, _, _ = mask_rois_v2(l_lrp_x, selected_ROIs, mode=True)
        mask_u_lrp_x, _, _ = mask_rois_v2(u_lrp_x, selected_ROIs, mode=True)

        batch_size = x.shape[0]
        # with torch.no_grad():
        h1 = pre_model.module.encoder(torch.from_numpy(mask_x).cuda())
        _, embedded = model(h1, flag='2')

        l_h1 = pre_model.module.encoder(torch.from_numpy(mask_l_lrp_x).cuda())
        _, embedded1 = model(l_h1, flag='2')

        u_h1 = pre_model.module.encoder(torch.from_numpy(mask_u_lrp_x).cuda())
        _, embedded2 = model(u_h1, flag='2')

        # print(model.module.decoder1)
        # final_embedded = torch.cat((embedded,embedded*embedded1), dim=1)
        prob = NN(embedded, embedded1, embedded2)
        # import matplotlib.pyplot as plt
        # for k in range(10):
        #     plt.imshow(embedded[k].detach().cpu().reshape(-1, 229))
        #     plt.savefig('{}/{}/tmp_result/{}_{}.png'.format(exp['exp_dir'], exp['tmp_fold'],i,k))
        #     plt.close()
        soft_prob = torch.nn.Softmax(dim=1)(prob)
        prediction = torch.argmax(soft_prob, 1)

        loss_ = object_func(prob, y.squeeze().cuda())

        loss_t = (loss_ + loss_gumbel) * batch_size
        # loss_t = (loss_) * batch_size

        optimizer.zero_grad()
        loss_t.backward()
        optimizer.step()

        loss_model += loss_t.item()
        n_samples += batch_size
        del loss_

        loss_resnet_total = loss_model / n_samples
        if i == 0:
            all_prob = soft_prob
            all_prediction = prediction
            all_labels = y
        else:
            all_prob = torch.cat((all_prob, soft_prob), dim=0)
            all_prediction = torch.cat((all_prediction, prediction), dim=0)
            all_labels = torch.cat((all_labels, y), dim=0)
    (acc, sen, spec, auc), _ = ut2.Evaluate_Binary(all_labels.detach(), all_prediction.detach(), all_prob.detach(), 2)
    ut2.writelog(f, '--- Fine tuning ---')
    ut2.writelog(f,'Loss: {:.4f}'.format(loss_resnet_total))
    ut2.writelog(f,'(T)ACC {:.4f} | SEN {:.4f} | SPC {:.4f} | AUC {:.4f}'.format(acc, sen, spec, auc))

def valid_fine_tune_NN(model, pre_model, NN, dataloader, object_func, f, exp, flag):
    model.eval()
    pre_model.eval()
    NN.eval()

    val_loss_resnet = 0
    n_samples = 0
    v_recon_output = []
    v_label_output = []
    v_orig_output = []
    v_embed = []
    with torch.no_grad():
        for j, (v_x, v_label) in enumerate(tqdm(dataloader)):
            v_mask_x, v_upper_indices, v_mask = mask_rois(v_x, num_roi=np.round(v_x.shape[1]*exp['remove_roi']),mode=False)
            v_y = v_label.long().cuda()

            batch_size = v_x.shape[0]

            v_h1 = pre_model.module.encoder(torch.from_numpy(v_mask_x).cuda())
            v_r1, v_embedded = model(v_h1, flag='2')
            v_r2 = pre_model.module.decoder1(v_r1.cuda())
            v_prob = NN(v_embedded)
            v_soft_prob = torch.nn.Softmax(dim=1)(v_prob)
            v_prediction = torch.argmax(v_soft_prob, 1)

            v_loss_ = object_func(v_prob, v_y.squeeze().cuda())

            val_loss_resnet += v_loss_.item()
            n_samples += batch_size
            v_tmp_recon = torch.zeros(batch_size, v_x.shape[1], v_x.shape[1])
            for i in range(batch_size):
                v_tmp_recon[i][v_upper_indices] = v_r2[i].cpu()
            v_recon_hat = v_tmp_recon + torch.transpose(v_tmp_recon, 2, 1)
            # v_recon_output.append(v_recon_hat.detach().cpu().numpy())
            # v_embed.append(v_embedded.detach().cpu().numpy())
            # v_orig_output.append(v_x.cpu().numpy())
            del v_loss_

            if j == 0:
                all_recon_output = v_recon_hat
                all_orig_output = v_x
                all_embed = v_embedded
                all_prob = v_soft_prob
                all_prediction = v_prediction
                all_labels = v_y
            else:
                all_recon_output = torch.cat((all_recon_output, v_recon_hat), dim=0)
                all_orig_output = torch.cat((all_orig_output, v_x), dim=0)
                all_embed = torch.cat((all_embed, v_embedded), dim=0)
                all_prob = torch.cat((all_prob, v_soft_prob), dim=0)
                all_prediction = torch.cat((all_prediction, v_prediction), dim=0)
                all_labels = torch.cat((all_labels, v_y), dim=0)

        v_loss_resnet_total = val_loss_resnet / n_samples
        (acc, sen, spec, auc), _ = ut2.Evaluate_Binary(all_labels.detach(), all_prediction.detach(), all_prob.detach(), 2)
        if flag == 'test':
            ut2.writelog(f,'(Te)ACC {:.4f} | SEN {:.4f} | SPC {:.4f} | AUC {:.4f}'.format(acc, sen, spec, auc))
            return v_loss_resnet_total, np.array(all_recon_output.detach().cpu()).squeeze(), np.array(all_orig_output.detach().cpu()).squeeze(), v_mask, v_label_output, np.array(all_embed.detach().cpu()).squeeze()
        else:
            ut2.writelog(f, '(V)ACC {:.4f} | SEN {:.4f} | SPC {:.4f} | AUC {:.4f}'.format(acc, sen, spec, auc))
            # return v_loss_resnet_total, np.array(v_recon_output).squeeze(), np.array(v_orig_output).squeeze(), v_mask
        return v_loss_resnet_total, np.array(all_recon_output.detach().cpu()).squeeze(), np.array(all_orig_output.detach().cpu()).squeeze(), v_mask

def valid_fine_tune_NN_22(model, pre_model, NN, gumber_model, dataloader, object_func, f, exp, flag):
    model.eval()
    pre_model.eval()
    NN.eval()
    gumber_model.eval()

    val_loss_resnet = 0
    n_samples = 0
    v_recon_output = []
    v_label_output = []
    v_orig_output = []
    v_embed = []
    with torch.no_grad():
        for j, (v_x, v_l_lrp_x, v_u_lrp_x, v_hist_x, v_hist_x2, v_label) in enumerate(tqdm(dataloader)):
            # v_mask_x, v_upper_indices, v_mask = mask_rois(v_x, num_roi=np.round(v_x.shape[1]*exp['remove_roi']),mode=False)
            # v_mask_l_lrp_x, upper_indices, _ = mask_rois(v_l_lrp_x, num_roi=np.round(v_x.shape[1] * exp['remove_roi']), mode=False)
            # v_mask_u_lrp_x, upper_indices, _ = mask_rois(v_u_lrp_x, num_roi=np.round(v_x.shape[1] * exp['remove_roi']), mode=False)

            v_y = v_label.long()
            hard = False
            recon_batch, qy = gumber_model(v_hist_x, v_hist_x2, .01, hard)
            loss_gumbel = loss_function(recon_batch, v_hist_x, v_hist_x2, qy)

            # selected_ROIs = torch.topk(recon_batch, int(recon_batch.shape[1] * .1), largest=False)
            selected_ROIs = recon_batch>0.5
            # selected_ROI_indices = exp['roi_indices'][selected_ROIs[1]].reshape(-1, int(len(recon_batch) * .1) * 109)
            v_mask_x, v_upper_indices, v_mask = mask_rois_v2(v_x, selected_ROIs, mode=True)
            v_mask_l_lrp_x, _, _ = mask_rois_v2(v_l_lrp_x, selected_ROIs, mode=True)
            v_mask_u_lrp_x, _, _ = mask_rois_v2(v_u_lrp_x, selected_ROIs, mode=True)

            batch_size = v_x.shape[0]

            v_h1 = pre_model.module.encoder(torch.from_numpy(v_mask_x).cuda())
            v_r1, v_embedded = model(v_h1, flag='2')

            v_l_h1 = pre_model.module.encoder(torch.from_numpy(v_mask_l_lrp_x).cuda())
            v_r1, v_embedded1 = model(v_l_h1, flag='2')

            v_u_h1 = pre_model.module.encoder(torch.from_numpy(v_mask_u_lrp_x).cuda())
            v_r1, v_embedded2 = model(v_u_h1, flag='2')

            v_r2 = pre_model.module.decoder1(v_r1)

            # v_final_embedded = torch.cat((v_embedded,v_embedded*v_embedded1),dim=1)
            v_prob = NN(v_embedded, v_embedded1, v_embedded2)
            v_soft_prob = torch.nn.Softmax(dim=1)(v_prob)
            v_prediction = torch.argmax(v_soft_prob, 1)

            v_loss_ = object_func(v_prob, v_y.squeeze().cuda())

            v_loss_t = (v_loss_ + loss_gumbel) * batch_size
            # v_loss_t = (v_loss_) * batch_size

            val_loss_resnet += v_loss_t.item()
            n_samples += batch_size
            v_tmp_recon = torch.zeros(batch_size, v_x.shape[1], v_x.shape[1])
            for i in range(batch_size):
                v_tmp_recon[i][v_upper_indices] = v_r2[i].cpu()
            v_recon_hat = v_tmp_recon + torch.transpose(v_tmp_recon, 2, 1)
            # v_recon_output.append(v_recon_hat.detach().cpu().numpy())
            # v_embed.append(v_embedded.detach().cpu().numpy())
            # v_orig_output.append(v_x.cpu().numpy())
            del v_loss_

            if j == 0:
                all_recon_output = v_recon_hat
                all_orig_output = v_x
                all_embed = v_embedded
                all_prob = v_soft_prob
                all_prediction = v_prediction
                all_labels = v_y
            else:
                all_recon_output = torch.cat((all_recon_output, v_recon_hat), dim=0)
                all_orig_output = torch.cat((all_orig_output, v_x), dim=0)
                all_embed = torch.cat((all_embed, v_embedded), dim=0)
                all_prob = torch.cat((all_prob, v_soft_prob), dim=0)
                all_prediction = torch.cat((all_prediction, v_prediction), dim=0)
                all_labels = torch.cat((all_labels, v_y), dim=0)

        v_loss_resnet_total = val_loss_resnet / n_samples
        (acc, sen, spec, auc), _ = ut2.Evaluate_Binary(all_labels.detach(), all_prediction.detach(), all_prob.detach(), 2)
        if flag == 'test':
            ut2.writelog(f,'(Te)ACC {:.4f} | SEN {:.4f} | SPC {:.4f} | AUC {:.4f}'.format(acc, sen, spec, auc))
            print(recon_batch[0])
            return v_loss_resnet_total, np.array(all_recon_output.detach().cpu()).squeeze(), np.array(all_orig_output.detach().cpu()).squeeze(), v_mask, v_label, np.array(all_embed.detach().cpu()).squeeze()
        else:
            ut2.writelog(f, '(V)ACC {:.4f} | SEN {:.4f} | SPC {:.4f} | AUC {:.4f}'.format(acc, sen, spec, auc))
            # return v_loss_resnet_total, np.array(v_recon_output).squeeze(), np.array(v_orig_output).squeeze(), v_mask
        return auc, v_loss_resnet_total, np.array(all_recon_output.detach().cpu()).squeeze(), np.array(all_orig_output.detach().cpu()).squeeze(), v_mask

def fine_tuning_NN_MLP(NN, dataloader, optimizer, object_func, f, exp):
    import natsort
    NN.train()
    loss_model = 0
    n_samples = 0
    for i, (x, label) in enumerate(tqdm(dataloader)):
        mask_x, upper_indices, _ = mask_rois(x, num_roi=np.round(x.shape[1] * exp['remove_roi']), mode=False)

        y = label.long().cuda()

        batch_size = x.shape[0]
        prob = NN(torch.from_numpy(mask_x).cuda())

        soft_prob = torch.nn.Softmax(dim=1)(prob)
        prediction = torch.argmax(soft_prob, 1)

        loss_ = object_func(prob, y.squeeze())

        loss_t = loss_ * batch_size

        optimizer.zero_grad()
        loss_t.backward()
        optimizer.step()

        loss_model += loss_.item()
        n_samples += batch_size
        del loss_

        loss_resnet_total = loss_model / n_samples
        if i == 0:
            all_prob = soft_prob
            all_prediction = prediction
            all_labels = y
        else:
            all_prob = torch.cat((all_prob, soft_prob), dim=0)
            all_prediction = torch.cat((all_prediction, prediction), dim=0)
            all_labels = torch.cat((all_labels, y), dim=0)
    (acc, sen, spec, auc), _ = ut2.Evaluate_Binary(all_labels.detach(), all_prediction.detach(), all_prob.detach(), 2)
    ut2.writelog(f, '--- Fine tuning ---')
    ut2.writelog(f,'Loss: {:.4f}'.format(loss_resnet_total))
    ut2.writelog(f,'(T)ACC {:.4f} | SEN {:.4f} | SPC {:.4f} | AUC {:.4f}'.format(acc, sen, spec, auc))

def valid_fine_tune_NN_MLP(NN, dataloader, object_func, f, exp, flag):
    NN.eval()

    val_loss_resnet = 0
    n_samples = 0
    v_recon_output = []
    v_label_output = []
    v_orig_output = []
    v_embed = []
    with torch.no_grad():
        for j, (v_x, v_label) in enumerate(tqdm(dataloader)):
            v_mask_x, v_upper_indices, v_mask = mask_rois(v_x, num_roi=np.round(v_x.shape[1]*exp['remove_roi']),mode=False)
            v_y = v_label.long().cuda()

            batch_size = v_x.shape[0]

            v_prob = NN(torch.from_numpy(v_mask_x).cuda())
            v_soft_prob = torch.nn.Softmax(dim=1)(v_prob)
            v_prediction = torch.argmax(v_soft_prob, 1)

            v_loss_ = object_func(v_prob, v_y.squeeze())

            val_loss_resnet += v_loss_.item()
            n_samples += batch_size
            del v_loss_

            if j == 0:
                all_prob = v_soft_prob
                all_prediction = v_prediction
                all_labels = v_y
            else:
                all_prob = torch.cat((all_prob, v_soft_prob), dim=0)
                all_prediction = torch.cat((all_prediction, v_prediction), dim=0)
                all_labels = torch.cat((all_labels, v_y), dim=0)

        v_loss_resnet_total = val_loss_resnet / n_samples
        (acc, sen, spec, auc), _ = ut2.Evaluate_Binary(all_labels.detach(), all_prediction.detach(), all_prob.detach(), 2)
        if flag == 'test':
            ut2.writelog(f,'(Te)ACC {:.4f} | SEN {:.4f} | SPC {:.4f} | AUC {:.4f}'.format(acc, sen, spec, auc))
            return v_loss_resnet_total, v_label_output
        else:
            ut2.writelog(f, '(V)ACC {:.4f} | SEN {:.4f} | SPC {:.4f} | AUC {:.4f}'.format(acc, sen, spec, auc))
            # return v_loss_resnet_total, np.array(v_recon_output).squeeze(), np.array(v_orig_output).squeeze(), v_mask
        return v_loss_resnet_total
