"""
main.py

This is the entry point for the project. It orchestrates the training pipeline,
including data preparation, model training, and evaluation.

Author: Parthasaarathy Sudarsanam, Audio Research Group, Tampere University
Date: January 2025
"""

import os.path
import torch
from parameters_sedsde import params
from models.model import SED_SDE_Post_3, SED_DOA
from loss import SedSdeLoss
from metrics import ComputeSELDResults
from data_generator import DataGenerator
from torch.utils.data import DataLoader
from lr_scheduler.tri_stage_lr_scheduler import TriStageLRScheduler
from extract_features import SELDFeatureExtractor
import utils
from tqdm import tqdm


def train_epoch(doa_model,seld_model, dev_train_iterator, optimizer, scheduler,seld_loss,nb_classes=13):

    seld_model.train()
    train_loss_per_epoch = 0  # Track loss per iteration to average over the epoch.
    doa_model.eval()
    for i, (input_features, labels) in enumerate(dev_train_iterator):
        labels = labels.to(device)
        # Handling modalities
        if params['modality'] == 'audio':
            audio_features, video_features = input_features.to(device), None
        elif params['modality'] == 'audio_visual':
            audio_features, video_features = input_features[0].to(device), input_features[1].to(device)
        else:
            raise AssertionError("Modality should be one of 'audio' or 'audio_visual'.")
        with torch.no_grad():
            logits_doa = doa_model(audio_features)
        sed = logits_doa[:, :, :nb_classes]
        x, y = logits_doa[:, :, nb_classes:2 * nb_classes], logits_doa[:, :, 2 * nb_classes: 3 * nb_classes]
        #########
        optimizer.zero_grad()
        
        sed=sed.view(sed.shape[0], 1,sed.shape[1], nb_classes)
        x=x.view(x.shape[0], 1, x.shape[1], nb_classes)
        y=y.view(y.shape[0], 1, y.shape[1], nb_classes)
        new_input = torch.cat((sed, x, y), dim=1)
        # Forward pass
        logits = seld_model(new_input)

        # Compute loss and back propagate
        loss = seld_loss(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Track loss
        train_loss_per_epoch += loss.item()

    avg_train_loss = train_loss_per_epoch / len(dev_train_iterator)
    return avg_train_loss


def val_epoch(doa_model,seld_model, dev_test_iterator, seld_loss, seld_metrics, output_dir, is_jackknife=False,nb_classes=13):
    doa_model.eval()
    seld_model.eval()
    val_loss_per_epoch = 0  # Track loss per iteration to average over the epoch.
    with torch.no_grad():
        for j, (input_features, labels) in enumerate(dev_test_iterator):
            labels = labels.to(device)

            # Handling modalities
            if params['modality'] == 'audio':
                audio_features, video_features = input_features.to(device), None
            elif params['modality'] == 'audio_visual':
                audio_features, video_features = input_features[0].to(device), input_features[1].to(device)
            else:
                raise AssertionError("Modality should be one of 'audio' or 'audio_visual'.")

            # Forward pass
            logits_doa = doa_model(audio_features)
            sed = logits_doa[:, :, :nb_classes]
            x, y = logits_doa[:, :, nb_classes:2 * nb_classes], logits_doa[:, :, 2 * nb_classes: 3 * nb_classes]
            #########
            
            
            sed=sed.view(sed.shape[0], 1,sed.shape[1], nb_classes)
            x=x.view(x.shape[0], 1, x.shape[1], nb_classes)
            y=y.view(y.shape[0], 1, y.shape[1], nb_classes)
            new_input = torch.cat((sed, x, y), dim=1)
            # Forward pass
            logits = seld_model(new_input)
            # Compute loss
            loss = seld_loss(logits, labels)
            val_loss_per_epoch += loss.item()
            logits=torch.cat((logits_doa, logits), dim=-1)
            # save predictions to csv files for metric calculations
            utils.write_logits_to_dcase_format(logits, params, output_dir, dev_test_iterator.dataset.label_files[j * params['batch_size']: (j + 1) * params['batch_size']])
        avg_val_loss = val_loss_per_epoch / len(dev_test_iterator)

        metric_scores = seld_metrics.get_SELD_Results(pred_files_path=os.path.join(output_dir, 'dev-test'), is_jackknife=is_jackknife)

        return avg_val_loss, metric_scores


def main():

    # Set up directories for storing model checkpoints, predictions(output_dir), and create a summary writer
    checkpoints_folder, output_dir, summary_writer = utils.setup(params)

    # Feature extraction code.
    feature_extractor = SELDFeatureExtractor(params)
    feature_extractor.extract_features(split='dev')
    feature_extractor.extract_labels(split='dev')

    # Set up dev_train and dev_test data iterator
    dev_train_dataset = DataGenerator(params=params, mode='dev_train')
    dev_train_iterator = DataLoader(dataset=dev_train_dataset, batch_size=params['batch_size'], num_workers=params['nb_workers'], shuffle=params['shuffle'], drop_last=True) ###change here

    dev_test_dataset = DataGenerator(params=params, mode='dev_test')
    dev_test_iterator = DataLoader(dataset=dev_test_dataset, batch_size=params['batch_size'], num_workers=params['nb_workers'], shuffle=False, drop_last=False)

    # create model, optimizer, loss and metrics
    seld_model = SED_SDE_Post_3().to(device)
    doa_model = SED_DOA(in_channel=6, in_dim=64).to(device)
    model_ckpt_doa = torch.load(os.path.join(model_dir_doa, 'best_model.pth'), map_location=device, weights_only=False)
    doa_model.load_state_dict(model_ckpt_doa['seld_model'])
    optimizer = torch.optim.Adam(params=seld_model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

    seld_loss = SedSdeLoss().to(device)
    total_steps = len(dev_train_dataset)// params['batch_size'] * params['nb_epochs']
   
    warmup_steps = int(total_steps*0.1)
    hold_steps = int(total_steps*0.6)
    decay_steps = int(total_steps*0.3)
    scheduler = TriStageLRScheduler(optimizer, peak_lr=params['learning_rate'], init_lr_scale=0.01, final_lr_scale=0.05, 
                                    warmup_steps=warmup_steps, hold_steps=hold_steps, decay_steps=decay_steps)
    seld_metrics = ComputeSELDResults(params=params, ref_files_folder=os.path.join(params['root_dir'], 'metadata_dev'))

    start_epoch = 0
    best_f_score = float('-inf')

    for epoch in tqdm(range(start_epoch, params['nb_epochs'])):
        # ------------- Training -------------- #
        avg_train_loss = train_epoch(doa_model,seld_model, dev_train_iterator, optimizer, scheduler,seld_loss)
        # -------------  Validation -------------- #
        avg_val_loss, metric_scores = val_epoch(doa_model,seld_model, dev_test_iterator, seld_loss, seld_metrics, output_dir)
        val_f, val_doa_error,val_dist_error, val_rel_dist_error, val_onscreen_acc, class_wise_scr = metric_scores
        # ------------- Log losses and metrics ------------- #

        print(
            f"Epoch {epoch + 1}/{params['nb_epochs']} | "
            f"Train Loss: {avg_train_loss:.2f} | "
            f"Val Loss: {avg_val_loss:.2f} | "
            f"F-score: {val_f * 100:.2f} | "
            f"DOA Err: {val_doa_error:.2f} | "
            f"Dist Err: {val_dist_error:.2f} | "
            f"Rel Dist Err: {val_rel_dist_error:.2f}" +
            (f" | On-Screen Acc: {val_onscreen_acc:.2f}" if params['modality'] == 'audio_visual' else "")
        )
        # ------------- Save model if validation f score improves -------------#
        if val_f >= best_f_score:
            best_f_score = val_f
            net_save = {'seld_model': seld_model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': epoch,
                        'best_f_score': best_f_score, 'best_rel_dist_err': val_rel_dist_error}
            if params['modality'] == 'audio_visual':
                net_save['best_onscreen_acc'] = val_onscreen_acc
            torch.save(net_save, checkpoints_folder + "/best_model.pth")

    # Evaluate the best model on dev-test.
    best_model_ckpt = torch.load(os.path.join(checkpoints_folder, 'best_model.pth'), map_location=device, weights_only=False)
    seld_model.load_state_dict(best_model_ckpt['seld_model'])
    use_jackknife = params['use_jackknife']
    test_loss, test_metric_scores = val_epoch(doa_model,seld_model, dev_test_iterator, seld_loss, seld_metrics, output_dir, is_jackknife=use_jackknife)
    test_f, test_doa,test_dist_error, test_rel_dist_error, test_onscreen_acc, class_wise_scr = test_metric_scores
    utils.print_results(test_f, test_doa,test_dist_error, test_rel_dist_error, test_onscreen_acc, class_wise_scr, params)


if __name__ == '__main__':
    model_dir_doa = "/home/var/Desktop/Mohor/DCASE2025_Nercslip/checkpoints_doa/SELDnet_audio_singleACCDOA_20250501_081031"
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    main()

