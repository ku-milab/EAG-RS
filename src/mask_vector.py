import numpy as np

def mask_rois(x, num_roi, mode=True):
    mask = np.ones_like(x)
    re_x = []

    upper_indices = np.mask_indices(x.shape[1], np.triu, 1)

    for idx in range(x.shape[0]):
        roi_select = np.random.choice(x.shape[1], np.int(num_roi), replace=False)
        mask[idx, roi_select, :] = 0
        mask[idx, :, roi_select] = 0
        mask[idx, roi_select, roi_select] = 1

    if mode == True:
        masked_x = np.array(x) * mask
    else:
        masked_x = np.array(x)

    for i in range(masked_x.shape[0]):
        re_x.append(masked_x[i][upper_indices])

    return np.array(re_x), upper_indices, mask

def mask_rois_v2(x, selected_rois, mode=True):
    mask = np.zeros_like(x.cpu())
    re_x = []
    re_x2 = []

    upper_indices = np.mask_indices(x.shape[1], np.triu, 1)

    for idx in range(x.shape[0]):
        tmp_k = np.where(selected_rois[idx].cpu() == True)[0]
        mask[idx, tmp_k, :] = 1
        mask[idx, :, tmp_k] = 1
        mask[idx, tmp_k, tmp_k] = 0

        # mask[idx, selected_rois[1][idx], :] = 0
        # mask[idx, :, selected_rois[1][idx]] = 0
        # mask[idx, selected_rois[1][idx], selected_rois[1][idx]] = 1

    if mode == True:
        masked_x = np.array(x.cpu()) * mask
    else:
        masked_x = np.array(x)

    for i in range(masked_x.shape[0]):
        re_x.append(masked_x[i][upper_indices])

    for i in range(mask.shape[0]):
        re_x2.append(mask[i][upper_indices])

    return np.array(re_x), upper_indices, np.array(re_x2)
