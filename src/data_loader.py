import torch
import torch.utils.data as data

def convert_Dloader(batch_size, data, label, num_workers = 0, shuffle = True):

    dataset = Dataset(data, label)
    Data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              num_workers=num_workers,
                                              batch_size=batch_size,
                                              shuffle=shuffle)
    return Data_loader

class Dataset(data.Dataset):

    def __init__(self, in_data, label):
        super(Dataset, self).__init__()
        self.data = in_data
        self.label = label
        # self.shape = np.shape(self.Data)
        # self.is_training = is_training

    def __getitem__(self, idx):
        # if self.is_training:
            return torch.from_numpy(self.data[idx,...]).float(), torch.from_numpy(self.label[idx,...]).float()
        # else:
        #     idx = self.test_nonzero_idx[index]
        #     return torch.from_numpy(np.expand_dims(self.test_input_mri[idx[0]-self.margin:idx[0]+self.margin+1,
        #                                                 idx[1]-self.margin:idx[1]+self.margin+1,
        #                                                 idx[2]-self.margin:idx[2]+self.margin+1],axis=0)), idx

    def __len__(self):
        return self.data.shape[0]

def convert_Dloader_fusion(batch_size, data, label, lrp_data, hist_data, hist_data2, num_workers=0, shuffle=True):
    dataset = Dataset_fusion(data, lrp_data, hist_data, hist_data2, label)
    Data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              num_workers=num_workers,
                                              batch_size=batch_size,
                                              shuffle=shuffle)
    return Data_loader

class Dataset_fusion(data.Dataset):

    def __init__(self, in_data, lrp_feature, hist_data, hist_data2, label):
        super(Dataset_fusion, self).__init__()
        self.data = in_data
        self.label = label
        self.lrp_data = lrp_feature
        self.hist_data = hist_data
        self.hist_data2 = hist_data2
        # self.shape = np.shape(self.Data)
        # self.is_training = is_training

    def __getitem__(self, idx):
        import numpy as np
        # if self.is_training:
        l_lrp_ = np.tril(self.lrp_data[idx]) + np.tril(self.lrp_data[idx]).T
        u_lrp_ = np.triu(self.lrp_data[idx]) + np.triu(self.lrp_data[idx]).T
        return torch.from_numpy(self.data[idx, ...]).float(), torch.from_numpy(u_lrp_).float(), torch.from_numpy(l_lrp_).float(), torch.from_numpy(self.hist_data[idx, ...]).float(), torch.from_numpy(self.hist_data2[idx, ...]).float(), torch.from_numpy(self.label[idx, ...]).float()

    # else:
    #     idx = self.test_nonzero_idx[index]
    #     return torch.from_numpy(np.expand_dims(self.test_input_mri[idx[0]-self.margin:idx[0]+self.margin+1,
    #                                                 idx[1]-self.margin:idx[1]+self.margin+1,
    #                                                 idx[2]-self.margin:idx[2]+self.margin+1],axis=0)), idx

    def __len__(self):
        return self.data.shape[0]