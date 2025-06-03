"""
evaluate.py

This script evaluates the trained models on the DCASE2025 evaluation dataset

Author: Parthasaarathy Sudarsanam, Audio Research Group, Tampere University
Date: May 2025
"""

import utils
from models.model import SED_DOA
from models.model import SED_SDE
from parameters import params
import parameters_seddoa
import parameters_sedsde
import pickle
import os
import torch
from torch.utils.data import DataLoader
from extract_features import SELDFeatureExtractor
from torch.utils.data import Dataset
import glob


class EvalDataGenerator(Dataset):
    def __init__(self, params):
        """
        Initializes the EvalDataGenerator instance.
        Args:
            params (dict): Parameters for data generation.
        """
        super().__init__()
        self.params = params
        self.root_dir = params['root_dir']
        self.feat_dir = params['feat_dir']
        self.modality = params['modality']

        # self.video_files will be an empty [] if self.modality == 'audio'
        self.audio_files, self.video_files = self.get_feature_files()

    def __getitem__(self, item):
        """
        Returns the data for a given index.
        Args:
            item (int): Index of the data.
        Returns:
            tuple: A tuple containing audio features, video_features (for audio_visual modality)
        """
        audio_file = self.audio_files[item]
        audio_features_raw = torch.load(audio_file)
        logmel_feat = audio_features_raw["logmel"]
        onepeace_feat = audio_features_raw["onepeace"]


        return logmel_feat,onepeace_feat

    def __len__(self):
        """
        Returns the number of data points.
        Returns:
            int: Number of data points.
        """
        return len(self.audio_files)

    def get_feature_files(self):
        """
        Collects the paths to the feature files based on the modality.
        Returns:
            tuple: A tuple containing lists of paths to audio feature files, video feature files.
        """
        audio_files, video_files = [], []

        # Loop through each fold and collect files

        audio_files += glob.glob(os.path.join(self.feat_dir, f'stereo_eval/*.pt'))

        # Only collect video files if modality is 'audio_video'
        if self.modality == 'audio_visual':
            video_files += glob.glob(os.path.join(self.feat_dir, f'video_eval/*.pt'))

        # Sort files to ensure corresponding audio, video, and label files are in the same order
        audio_files = sorted(audio_files, key=lambda x: x.split('/')[-1])

        # Sort video files only if modality is 'audio_visual'
        if self.modality == 'audio_visual':
            video_files = sorted(video_files, key=lambda x: x.split('/')[-1])

        # Return the appropriate files based on modality
        if self.modality == 'audio':
            return audio_files, []
        elif self.modality == 'audio_visual':
            return audio_files, video_files
        else:
            raise ValueError(f"Invalid modality: {self.modality}. Choose from ['audio', 'audio_visual'].")


def evaluate():

    reference_sde = model_dir_sde.split('/')[-1]
    reference_doa = model_dir_doa.split('/')[-1]
    output_dir_sde = os.path.join(parameters_sedsde.params['output_dir'], reference_sde)
    output_dir_doa = os.path.join(parameters_seddoa.params['output_dir'], reference_doa)
    os.makedirs(parameters_sedsde.params['output_dir'], exist_ok=True)
    os.makedirs(output_dir_sde, exist_ok=True)
    os.makedirs(parameters_seddoa.params['output_dir'], exist_ok=True)
    os.makedirs(output_dir_doa, exist_ok=True)
    os.makedirs(params['output_dir'], exist_ok=True)

    # Feature extraction code.
    feature_extractor = SELDFeatureExtractor(params)
    feature_extractor.extract_features(split='eval')

    doa_model = SED_DOA(in_channel=6, in_dim=64).to(device)
    sed_model = SED_SDE(in_channel=6, in_dim=64).to(device)
    model_ckpt_sde = torch.load(os.path.join(model_dir_sde, 'best_model.pth'), map_location=device, weights_only=False)
    model_ckpt_doa = torch.load(os.path.join(model_dir_doa, 'best_model.pth'), map_location=device, weights_only=False)
    doa_model.load_state_dict(model_ckpt_doa['seld_model'])
    sed_model.load_state_dict(model_ckpt_sde['seld_model'])

    eval_dataset = EvalDataGenerator(params=params)
    eval_iterator = DataLoader(dataset=eval_dataset, batch_size=params['batch_size'], num_workers=params['nb_workers'],
                                shuffle=False, drop_last=False)

    sed_model.eval()
    doa_model.eval()
    with torch.no_grad():
        for j, (logmel_feat,onepeace_feat) in enumerate(eval_iterator):
            
            # Handling modalities
            if params['modality'] == 'audio':
                logmel_feat, onepeace_feat,video_features = logmel_feat.to(device),onepeace_feat.to(device), None
            elif params['modality'] == 'audio_visual':
                logmel_feat, onepeace_feat, video_features = logmel_feat.to(device),onepeace_feat.to(device), None
            else:
                raise AssertionError("Modality should be one of 'audio' or 'audio_visual'.")

            # Forward pass
            logits_sde = sed_model(logmel_feat)
            logits_doa = doa_model(logmel_feat, onepeace_feat)
            # save predictions to csv files for metric calculations
            nb_classes=params['nb_classes']
            sed_1 = logits_sde[:, :, :nb_classes]
            sed_2 = logits_doa[:, :, :nb_classes]
            sed=(sed_1 + sed_2) /2
            #sed=sed_2
            x, y = logits_doa[:, :, nb_classes:2 * nb_classes], logits_doa[:, :, 2 * nb_classes: 3 * nb_classes]
            distance = logits_sde[:, :, 3*nb_classes: 4 * nb_classes]
            pred = torch.cat((sed, x,y,distance), dim=-1)
            utils.write_logits_to_dcase_format(pred, params, params["output_dir"], eval_iterator.dataset.label_files[j * params['batch_size']: (j + 1) * params['batch_size']], split='eval')


if __name__ == '__main__':
    model_dir_sde = "/home/var/Desktop/Mohor/15_DCASE2025_Nercslip_op/checkpoints_sde/BestSdeModel"
    model_dir_doa = "/home/var/Desktop/Mohor/15_DCASE2025_Nercslip_op/checkpoints_doa/OneP8_2_48_2_Best"
    device =torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    evaluate()