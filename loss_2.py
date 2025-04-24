import numpy as np
import torch
import torch.nn as nn
import os

class SedDoaLoss(nn.Module):
    def __init__(self, loss_weight=[1.0, 10.0]):
        super().__init__()
        self.criterion_sed = nn.BCELoss()
        self.criterion_doa = nn.MSELoss()
        self.loss_weight = loss_weight
    
    def forward(self, output, target):
        sed_out = output[:,:,:13]
        doa_out = output[:,:,13:] # torch.Size([32, 100, 52])
        sed_label = target[:,:,:13]
        doa_label = target[:,:,13:52]
        loss_sed = self.criterion_sed(sed_out, sed_label)
        sed_label_repeat = sed_label.repeat(1,1,3) # torch.Size([32, 100, 39])
        # sed_label_repeat = sed_label.repeat(1,1,4)
        #pdb.set_trace()
        loss_doa = self.criterion_doa(doa_out * sed_label_repeat, doa_label) # why multiply with sed_label_repeat? be
        loss = self.loss_weight[0] * loss_sed + self.loss_weight[1] * loss_doa
        return loss

class MSPELoss(torch.nn.Module):
    def __init__(self):
        super(MSPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        absolute_error = torch.abs(y_true - y_pred)
        percentage_error = absolute_error / torch.abs(y_true)
        mspe = torch.mean(percentage_error ** 2)
        return mspe
    
class SedLoss_2024(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.criterion_sed = nn.BCELoss()
    
    def forward(self, output, target):
        sed_out = output[:,:,:13]
        sed_label = target[:,:,:13]
        loss_sed = self.criterion_sed(sed_out, sed_label)
        #sed_label_repeat = sed_label.repeat(1,1,3) # torch.Size([32, 100, 39])
        return loss_sed
    
class SedDistLoss_2024_MSPE(nn.Module):
    def __init__(self, loss_weight=[1.0, 0.1]):
        super().__init__()
        self.criterion_sed = nn.BCELoss()
        # self.criterion_doa = nn.MSELoss()
        self.criterion_dist = nn.MSELoss()
        # self.criterion_dist = MSPELoss()
        self.loss_weight = loss_weight
    
    def forward(self, output, target):
        sed_out = output[:,:,:13]
        doa_out = output[:,:,13:52] # torch.Size([32, 100, 65])
        dist_out = output[:,:,52:]
        sed_label = target[:,:,:13]
        doa_label = target[:,:,13:52]
        dist_label = target[:,:,52:]
        dist_label += 1e-8
        loss_sed = self.criterion_sed(sed_out, sed_label)
        #sed_label_repeat = sed_label.repeat(1,1,3) # torch.Size([32, 100, 39])
        # sed_label_repeat = sed_label.repeat(1,1,3)
        #pdb.set_trace()
        # loss_doa = self.criterion_doa(doa_out * sed_label_repeat, doa_label)
        # loss_dist = self.criterion_dist(dist_out * sed_label, dist_label)
        loss_dist = self.criterion_dist(dist_out* sed_label/ dist_label, dist_label* sed_label/ dist_label)
        loss = self.loss_weight[0] * loss_sed  + self.loss_weight[1] * loss_dist
        return loss_sed, loss_dist, loss
    
class SedDistLoss_2024_MAPE(nn.Module):
    def __init__(self, loss_weight=[1.0, 0.1]):
        super().__init__()
        self.criterion_sed = nn.BCELoss()
        # self.criterion_doa = nn.MSELoss()
        # self.criterion_dist = nn.MSELoss()
        self.criterion_dist = nn.L1Loss()
        self.loss_weight = loss_weight
    
    def forward(self, output, target):
        sed_out = output[:,:,:13]
        doa_out = output[:,:,13:52] # torch.Size([32, 100, 65])
        dist_out = output[:,:,52:]
        sed_label = target[:,:,:13]
        doa_label = target[:,:,13:52]
        dist_label = target[:,:,52:]
        dist_label += 1e-8
        loss_sed = self.criterion_sed(sed_out, sed_label)
        loss_dist = self.criterion_dist(dist_out* sed_label/ dist_label, dist_label* sed_label/ dist_label)
        loss = self.loss_weight[0] * loss_sed  + self.loss_weight[1] * loss_dist
        return loss_sed, loss_dist, loss
    
class SedDistLoss_2024_MSE(nn.Module):
    def __init__(self, loss_weight=[1.0, 0.1]):
        super().__init__()
        self.criterion_sed = nn.BCELoss()
        # self.criterion_doa = nn.MSELoss()
        self.criterion_dist = nn.MSELoss()
        # self.criterion_dist = MSPELoss()
        self.loss_weight = loss_weight
    
    def forward(self, output, target):
        sed_out = output[:,:,:13]
        doa_out = output[:,:,13:52] # torch.Size([32, 100, 65])
        dist_out = output[:,:,52:]
        sed_label = target[:,:,:13]
        doa_label = target[:,:,13:52]
        dist_label = target[:,:,52:]
        loss_sed = self.criterion_sed(sed_out, sed_label)
        #sed_label_repeat = sed_label.repeat(1,1,3) # torch.Size([32, 100, 39])
        # sed_label_repeat = sed_label.repeat(1,1,3)
        #pdb.set_trace()
        # loss_doa = self.criterion_doa(doa_out * sed_label_repeat, doa_label)
        loss_dist = self.criterion_dist(dist_out * sed_label, dist_label)
        loss = self.loss_weight[0] * loss_sed  + self.loss_weight[1] * loss_dist
        return loss_sed, loss_dist, loss

class SedDoaLoss_2024(nn.Module):
    def __init__(self, loss_weight=[1.0, 10.0, 0.1]):
        super().__init__()
        self.criterion_sed = nn.BCELoss()
        self.criterion_doa = nn.MSELoss()
        self.criterion_dist = nn.MSELoss()
        # self.criterion_dist = MSPELoss()
        self.loss_weight = loss_weight
    
    def forward(self, output, target):
        sed_out = output[:,:,:13]
        doa_out = output[:,:,13:52] # torch.Size([32, 100, 65])
        dist_out = output[:,:,52:]
        sed_label = target[:,:,:13]
        doa_label = target[:,:,13:52]
        dist_label = target[:,:,52:]
        loss_sed = self.criterion_sed(sed_out, sed_label)
        #sed_label_repeat = sed_label.repeat(1,1,3) # torch.Size([32, 100, 39])
        sed_label_repeat = sed_label.repeat(1,1,3)
        #pdb.set_trace()
        loss_doa = self.criterion_doa(doa_out * sed_label_repeat, doa_label)
        loss_dist = self.criterion_dist(dist_out * sed_label, dist_label)
        loss = self.loss_weight[0] * loss_sed + self.loss_weight[1] * loss_doa + self.loss_weight[2] * loss_dist
        return loss
    

    

class SedDoaKLLoss(nn.Module):
    def __init__(self, loss_weight=[1.0, 10.0]):
        super().__init__()
        self.criterion_doa = nn.MSELoss()
        self.loss_weight = loss_weight
    
    def forward(self, output, target):
        sed_out = output[:,:,:13]
        doa_out = output[:,:,13:]
        sed_label = target[:,:,:13]
        doa_label = target[:,:,13:]
        sed_label_repeat = sed_label.repeat(1,1,3)
        loss_doa = self.criterion_doa(doa_out * sed_label_repeat, doa_label)
        loss_kl_sub1 = (sed_label*torch.log(1e-7+sed_label/(1e-7+sed_out))).mean()
        loss_kl_sub2 = ((1-sed_label)*torch.log(1e-7+(1-sed_label)/(1e-7+1-sed_out))).mean()
        loss_sed = loss_kl_sub1 + loss_kl_sub2
        loss = self.loss_weight[0] * loss_sed + self.loss_weight[1] * loss_doa
        return loss

class SedDoaKLLoss_2(nn.Module):
    def __init__(self, loss_weight=[1.0, 10.0]):
        super().__init__()
        self.criterion_doa = nn.MSELoss()
        self.loss_weight = loss_weight
    
    def forward(self, output, target):
        sed_out = output[:,:,:13]
        doa_out = output[:,:,13:]
        sed_label = target[:,:,:13]
        doa_label = target[:,:,13:]
        sed_label_repeat = sed_label.repeat(1,1,3)
        loss_doa = self.criterion_doa(doa_out * sed_label_repeat, doa_label * sed_label_repeat)
        loss_kl_sub1 = (sed_label*torch.log(1e-7+sed_label/(1e-7+sed_out))).mean()
        loss_kl_sub2 = ((1-sed_label)*torch.log(1e-7+(1-sed_label)/(1e-7+1-sed_out))).mean()
        loss_sed = loss_kl_sub1 + loss_kl_sub2
        loss = self.loss_weight[0] * loss_sed + self.loss_weight[1] * loss_doa
        return loss

class SedDoaKLLoss_3(nn.Module):
    def __init__(self, loss_weight=[1.0, 10.0]):
        super().__init__()
        self.criterion_doa = nn.MSELoss()
        self.loss_weight = loss_weight
    
    def forward(self, output, target):
        sed_out = output[:,:,:13]
        doa_out = output[:,:,13:]
        sed_label = (target[:,:,:13] > 0.5) * 1.0
        doa_label = target[:,:,13:]
        sed_label_repeat = sed_label.repeat(1,1,3)
        loss_doa = self.criterion_doa(doa_out * sed_label_repeat, doa_label * sed_label_repeat)
        loss_kl_sub1 = (sed_label*torch.log(1e-7+sed_label/(1e-7+sed_out))).mean()
        loss_kl_sub2 = ((1-sed_label)*torch.log(1e-7+(1-sed_label)/(1e-7+1-sed_out))).mean()
        loss_sed = loss_kl_sub1 + loss_kl_sub2
        loss = self.loss_weight[0] * loss_sed + self.loss_weight[1] * loss_doa
        return loss