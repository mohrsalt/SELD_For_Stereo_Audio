
import torch.nn as nn


class SedDoaLoss(nn.Module):
    def __init__(self, loss_weight=[0.1, 1]):
        super().__init__()
        self.criterion_sed = nn.BCELoss()
        self.criterion_doa = nn.MSELoss()
        self.loss_weight = loss_weight
    
    def forward(self, output, target):
        sed_out = output[:,:,:13]
        doa_out = output[:,:,13:] 
        sed_label = target[:,:,:13]
        doa_label = target[:,:,13:39]
        loss_sed = self.criterion_sed(sed_out, sed_label)
        sed_label_repeat = sed_label.repeat(1,1,2) 
 
        loss_doa = self.criterion_doa(doa_out * sed_label_repeat, doa_label) 
        loss = self.loss_weight[0] * loss_sed + self.loss_weight[1] * loss_doa
        return loss

    
class SedSdeLoss(nn.Module):
    def __init__(self, loss_weight=[0.1, 1]):
        super().__init__()
        self.criterion_sed = nn.BCELoss()      
        self.criterion_dist = nn.MSELoss()
        self.loss_weight = loss_weight
    
    def forward(self, output, target):
        sed_out = output[:,:,:13]   
        dist_out = output[:,:,39:52]
        sed_label = target[:,:,:13]   
        dist_label = target[:,:,39:52]
        dist_label += 1e-8
        loss_sed = self.criterion_sed(sed_out, sed_label)

        loss_dist = self.criterion_dist(dist_out* sed_label/ dist_label, dist_label* sed_label/ dist_label)
        loss = self.loss_weight[0] * loss_sed  + self.loss_weight[1] * loss_dist
        return loss
    

    
################################################################################################################################################################################################################################
    

