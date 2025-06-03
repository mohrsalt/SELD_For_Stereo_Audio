import os.path
import torch
from parameters_sedsde import params
from models.model import SED_SDE
from loss import SedSdeLoss
from metrics_sedsde import ComputeSELDResults
from data_generator import DataGenerator
from torch.utils.data import DataLoader
from lr_scheduler.tri_stage_lr_scheduler import TriStageLRScheduler
from extract_features import SELDFeatureExtractor
import utils
from tqdm import tqdm
print('Loading model weights and optimizer state dict from initial checkpoint...')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_ckpt = torch.load(os.path.join("/home/var/Desktop/Mohor/DCASE2025_Nercslip_op/SDE_Pre", 'best_model.pth'), map_location=device, weights_only=False)
# print(model_ckpt['seld_model']["resnet.bn1.weight"].shape)
seld_model = SED_SDE(in_channel=6, in_dim=64).to(device)
filtered_state_dict = {
    k: v for k, v in model_ckpt['seld_model'].items()
    if 'resnet' in k and k != 'resnet.conv1.weight'
}


# # # Filtered state_dict
# filtered_state_dict = {k: v for k, v in model_ckpt['seld_model'].items() if k not in excluded_keys}
print(len(filtered_state_dict.keys()))
