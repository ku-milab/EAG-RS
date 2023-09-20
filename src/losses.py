import torch
from torch.distributions.normal import Normal
from torch.nn import functional as F

import torch.nn as nn
from torch.autograd import Variable

class Classifi_Loss(nn.Module):
    def __init__(self, args):
        super(Classifi_Loss, self).__init__()
        self.args = args
        self.loss = nn.CrossEntropyLoss()

    def forward(self, y_pred, y):
        pred_loss = self.loss(y_pred, y)

        return pred_loss

class Classifi_Loss_comp(nn.Module):
    def __init__(self, args):
        super(Classifi_Loss_comp, self).__init__()
        self.args = args
        self.loss = nn.CrossEntropyLoss(size_average=False, reduction='sum')

    def forward(self, y_pred, y):
        pred_loss = self.loss(y_pred, y)

        return pred_loss

class Recon_Loss(nn.Module):
    def __init__(self, args):
        super(Recon_Loss, self).__init__()
        self.args = args
        self.loss = nn.MSELoss(size_average=False, reduction='sum')

    def forward(self, y_pred, y):
        pred_loss = self.loss(y_pred, y)

        return pred_loss

class Recon_Loss_v2(nn.Module):
    def __init__(self, args):
        super(Recon_Loss_v2, self).__init__()
        self.args = args
        self.loss = nn.L1Loss(size_average=False,reduction='sum')

    def forward(self, y_pred, y):
        pred_loss = self.loss(y_pred, y)

        return pred_loss