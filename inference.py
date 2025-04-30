"""
inference.py

This module provides utilities for running inference using the trained model,

Author: Parthasaarathy Sudarsanam, Audio Research Group, Tampere University
Date: March 2025
"""

import utils
from models.model import SED_DOA,SED_SDE
import pickle
import os
from parameters import params
import parameters_seddoa
import parameters_sedsde
import torch
from metrics import ComputeSELDResults
from data_generator import DataGenerator
from torch.utils.data import DataLoader
from extract_features import SELDFeatureExtractor


def run_inference():

    # params_file = os.path.join(model_dir, 'config.pkl')
    # f = open(params_file, "rb")
    # params = pickle.load(f)
    # print(params)
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
    feature_extractor.extract_features(split='dev')
    feature_extractor.extract_labels(split='dev')

    doa_model = SED_DOA(in_channel=6, in_dim=64).to(device)
    sed_model = SED_SDE(in_channel=6, in_dim=64).to(device)
    model_ckpt_sde = torch.load(os.path.join(model_dir_sde, 'best_model.pth'), map_location=device, weights_only=False)
    model_ckpt_doa = torch.load(os.path.join(model_dir_doa, 'best_model.pth'), map_location=device, weights_only=False)
    sed_model.load_state_dict(model_ckpt_sde['seld_model'])
    doa_model.load_state_dict(model_ckpt_doa['seld_model'])
    print(params['root_dir'])
    seld_metrics = ComputeSELDResults(params=params, ref_files_folder=os.path.join(params['root_dir'], 'metadata_dev'))

    test_dataset = DataGenerator(params=params, mode='dev_test')
    test_iterator = DataLoader(dataset=test_dataset, batch_size=params['batch_size'], num_workers=params['nb_workers'], shuffle=False, drop_last=False)

    sed_model.eval()
    doa_model.eval()
    with torch.no_grad():
        for j, (input_features, labels) in enumerate(test_iterator):
            labels = labels.to(device)
            # Handling modalities
            if params['modality'] == 'audio':
                audio_features, video_features = input_features.to(device), None
            elif params['modality'] == 'audio_visual':
                audio_features, video_features = input_features[0].to(device), input_features[1].to(device)
            else:
                raise AssertionError("Modality should be one of 'audio' or 'audio_visual'.")

            # Forward pass
            logits_sde = sed_model(audio_features)
            logits_doa = doa_model(audio_features)
            # save predictions to csv files for metric calculations
            nb_classes=params['nb_classes']
            sed_1 = logits_sde[:, :, :nb_classes]
            sed_2 = logits_doa[:, :, :nb_classes]
            sed=(sed_1 + sed_2) /2
            #sed=sed_2
            x, y = logits_doa[:, :, nb_classes:2 * nb_classes], logits_doa[:, :, 2 * nb_classes: 3 * nb_classes]
            distance = logits_sde[:, :, 3*nb_classes: 4 * nb_classes]
            pred = torch.cat((sed, x,y,distance), dim=-1)
            utils.write_logits_to_dcase_format(pred, params, params["output_dir"], test_iterator.dataset.label_files[j * params['batch_size']: (j + 1) * params['batch_size']])

        test_metric_scores = seld_metrics.get_SELD_Results(pred_files_path=os.path.join(params["output_dir"], 'dev-test'), is_jackknife=False)
        test_f, test_ang_error, test_dist_error, test_rel_dist_error, test_onscreen_acc, class_wise_scr = test_metric_scores
        utils.print_results(test_f, test_ang_error, test_dist_error, test_rel_dist_error, test_onscreen_acc, class_wise_scr, params)


if __name__ == '__main__':
    model_dir_sde = "/home/var/Desktop/Mohor/DCASE2025_Nercslip/checkpoints_sde/SELDnet_audio_singleACCDOA_20250430_102157"
    model_dir_doa = "/home/var/Desktop/Mohor/DCASE2025_Nercslip/checkpoints_doa/SELDnet_audio_singleACCDOA_20250429_235448"
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    run_inference()
