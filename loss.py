
import torch.nn as nn

import torch
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

    
# class SedSdeLoss(nn.Module):
#     def __init__(self, loss_weight=[0.1, 1]):
#         super().__init__()
#         self.criterion_sed = nn.BCELoss()      
#         self.criterion_dist = nn.MSELoss()
#         self.loss_weight = loss_weight
    
#     def forward(self, output, target):
#         sed_out = output[:,:,:13]   
#         dist_out = output[:,:,39:52]
#         sed_label = target[:,:,:13]   
#         dist_label = target[:,:,39:52]
#         dist_label += 1e-8
#         loss_sed = self.criterion_sed(sed_out, sed_label)

#         loss_dist = self.criterion_dist(dist_out* sed_label/ dist_label, dist_label* sed_label/ dist_label)
#         loss = self.loss_weight[0] * loss_sed  + self.loss_weight[1] * loss_dist
#         return loss
    

class SedSdeLoss(nn.Module):
    def __init__(self, loss_weight=[0.1, 1]):
        super().__init__()
        self.criterion_sed = nn.BCELoss()      
        self.criterion_dist = nn.MSELoss()
        self.loss_weight = loss_weight
    
    def forward(self, output, target):
        dist_out = output[:,:,:13]   
       
        sed_label = target[:,:,:13]   
        dist_label = target[:,:,39:52]
        dist_label += 1e-8
        

        loss_dist = self.criterion_dist(dist_out* sed_label/ dist_label, dist_label* sed_label/ dist_label)
        loss = self.loss_weight[1] * loss_dist
        return loss
    
################################################################################################################################################################################################################################
    




class SedSdeLossRDE(nn.Module):
    def __init__(self, loss_weight=[0.1, 1], sed_threshold=0.5):
        super().__init__()
        self.criterion_sed = nn.BCELoss()
        self.loss_weight = loss_weight
        self.sed_threshold = sed_threshold

    def forward(self, sed_out, output, target):
        # Assume output: (B, T, 26); first 13: distance preds, next 13: SED preds
        dist_out = output[:, :, :13]            # shape: (B, T, C)
                 

        sed_label = target[:, :, :13]           # shape: (B, T, C)
        dist_label = target[:, :, 39:52] + 1e-8 # shape: (B, T, C), avoid division by 0

        # Binarize sed prediction
        sed_pred = (sed_out >= self.sed_threshold).float()
       
        # TP mask: where both pred and label are active
        tp_mask = (sed_pred * sed_label) > 0

        # Relative error
        rel_error = torch.abs(dist_out - dist_label) / dist_label

        # Per-class sums
        delta_sum = (rel_error * tp_mask).sum(dim=(0, 1))  # shape: (C,)
        tp_counts = tp_mask.sum(dim=(0, 1))                # shape: (C,)

        # RDE per class
        rde_c = torch.where(tp_counts > 0, delta_sum / tp_counts, torch.zeros_like(tp_counts))

        # Macro-average RDE across valid classes
        valid_classes = (tp_counts > 0)
        if valid_classes.sum() > 0:
            loss_dist = rde_c[valid_classes].mean()
        else:
            loss_dist = torch.tensor(0.0, device=output.device)

        # Final loss (only RDE used)
        loss = self.loss_weight[1] * loss_dist
        return loss
