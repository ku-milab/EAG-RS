import argparse

def get_args(opt):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='ABIDE', help='ABIDE or Rest-meta MDD (MDD)')
    parser.add_argument('--site', type=str, default=opt['site'], help='site information')
    # parser.add_argument('--atlas', type=str, default=opt['atlas'], help='We can choose AAL, CC, Dosenbach, HO, and Yeo17 atlases.')
    parser.add_argument('--atlas', type=str, default='HO', help='We can choose AAL, CC, Dosenbach, HO, and Yeo17 atlases.')
    parser.add_argument('--gpu', type=str, default='5')
    parser.add_argument('--seed', type=int, default=5930)
    parser.add_argument('--remove_roi', type=float, default=opt['ROI_mask'], help='percentage of the removing ROIs')
    # parser.add_argument('--remove_roi', type=float, default=0.1, help='percentage of the removing ROIs')
    # parser.add_argument('--t1', type=float, default=opt['t1'], help='')
    # parser.add_argument('--t2', type=float, default=opt['t2'], help='')

    # parser.add_argument('--class_remove_roi', type=float, default=opt['ROI_mask'], help='percentage of the removing ROIs')

    # parser.add_argument('--sw', type=float, default=opt['sw'], help='sliding window size')
    # parser.add_argument('--stride', type=float, default=opt['stride'], help='sliding window size')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay_step', type=int, default=1)
    parser.add_argument("--lr_decay", type=float, default=0.98)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--tmp_fold', type=int, default=opt['tmp_fold'])
    parser.add_argument('--kfold', type=int, default=5)
    # parser.add_argument('--num_classes', type=int, default=2)
    # parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--batch_size', type=int, default=50)
    # parser.add_argument('--AE_dropout', type=float, default=0.5)
    args = parser.parse_args()
    return args