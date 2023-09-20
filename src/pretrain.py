from tqdm import tqdm
from src.mask_vector import *

import torch
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src import utils as ut2


def train_(model, dataloader, optimizer, object_func, exp, f):
    model.train()
    loss_model = 0
    n_samples = 0
    for _, (x, label) in enumerate(tqdm(dataloader)):
        mask_x, upper_indices, a = mask_rois(x, num_roi=np.round(x.shape[1]*exp['remove_roi']))

        if torch.cuda.is_available() == True:
            y = x.cuda()
            batch_size = x.shape[0]
            recon, _ = model(torch.from_numpy(mask_x).cuda().float(), flag='1')

            tmp_recon = torch.zeros(batch_size, x.shape[1], x.shape[1])
            for i in range(batch_size):
                tmp_recon[i][upper_indices] = recon[i].cpu()
            recon_ = tmp_recon + torch.transpose(tmp_recon, 2, 1)
            # soft_prob = nn.Softmax(dim=1)(prob)
            loss_ = object_func(recon_.cuda(), y.cuda())
            # prediction = torch.argmax(soft_prob, 1)
            loss_t = loss_ / batch_size

            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()

            loss_model += loss_.item()
            n_samples += batch_size
            del loss_
        else:
            y = x
            batch_size = x.shape[0]
            recon, _ = model(torch.from_numpy(mask_x).float(), flag='1')

            tmp_recon = torch.zeros(batch_size, x.shape[1], x.shape[1])
            for i in range(batch_size):
                tmp_recon[i][upper_indices] = recon[i]
            recon_ = tmp_recon + torch.transpose(tmp_recon, 2, 1)
            # soft_prob = nn.Softmax(dim=1)(prob)
            loss_ = object_func(recon_, y)
            # prediction = torch.argmax(soft_prob, 1)
            loss_t = loss_ / batch_size

            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()

            loss_model += loss_.item()
            n_samples += batch_size
            del loss_

    loss_resnet_total = loss_model / n_samples

    ut2.writelog(f,'Loss: {:.4f}'.format(loss_resnet_total))
    # ut2.writelog(f,'(T)ACC {:.4f} | SEN {:.4f} | SPC {:.4f} | AUC {:.4f}'.format(acc, sen, spec, auc))

def valid_(model, dataloader, object_func, f, exp, flag):
    model.eval()

    val_loss_resnet = 0
    n_samples = 0
    v_recon_output = []
    v_label_output = []
    v_orig_output = []
    with torch.no_grad():
        for _, (v_x, v_label) in enumerate(tqdm(dataloader)):
            v_mask_x, v_upper_indices, v_mask = mask_rois(v_x, num_roi=np.round(v_x.shape[1]*exp['remove_roi']), mode=False)
            if torch.cuda.is_available() == True:
                v_y = v_x.cuda()

                batch_size = v_x.shape[0]

                v_recon, _ = model(torch.from_numpy(v_mask_x).cuda().float(), flag='1')
                tmp_recon = torch.zeros(batch_size, v_x.shape[1], v_x.shape[1])
                for i in range(batch_size):
                    tmp_recon[i][v_upper_indices] = v_recon[i].cpu()
                    if flag == 'test':
                        if v_label[i] == 1:
                            v_label_output.append('TD')
                        else:
                            v_label_output.append('ASD')
                    else:
                        pass
                v_recon_ = tmp_recon + torch.transpose(tmp_recon, 2, 1)

                v_loss_ = object_func(v_recon_.cuda(), v_y)

                val_loss_resnet += v_loss_.item()
                n_samples += batch_size
                v_recon_output.append(v_recon_.detach().cpu().numpy())
                v_orig_output.append(v_x.cpu().numpy())
                del v_loss_

            else:
                v_y = v_x

                batch_size = v_x.shape[0]

                v_recon, _ = model(torch.from_numpy(v_mask_x).float(), flag='1')
                tmp_recon = torch.zeros(batch_size, v_x.shape[1], v_x.shape[1])
                for i in range(batch_size):
                    tmp_recon[i][v_upper_indices] = v_recon[i]
                    if flag == 'test':
                        if v_label[i] == 1:
                            v_label_output.append('TD')
                        else:
                            v_label_output.append('ASD')
                    else:
                        pass
                v_recon_ = tmp_recon + torch.transpose(tmp_recon, 2, 1)

                v_loss_ = object_func(v_recon_, v_y)

                val_loss_resnet += v_loss_.item()
                n_samples += batch_size
                v_recon_output.append(v_recon_.detach().numpy())
                v_orig_output.append(v_x.numpy())
                del v_loss_

        v_loss_resnet_total = val_loss_resnet / n_samples
        if flag == 'test':
            ut2.writelog(f, '(Te)Loss: {:.4f}'.format(v_loss_resnet_total))
            return v_loss_resnet_total, np.array(v_recon_output).squeeze(), np.array(v_orig_output).squeeze(), v_mask, v_label_output
        else:
            ut2.writelog(f, '(V)Loss: {:.4f}'.format(v_loss_resnet_total))
            return v_loss_resnet_total, np.array(v_recon_output).squeeze(), np.array(v_orig_output).squeeze(), v_mask
