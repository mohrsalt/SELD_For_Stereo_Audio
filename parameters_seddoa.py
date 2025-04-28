"""
parameters.py

This module stores all the configurable parameters and hyperparameters used
across the project, ensuring easy tuning and reproducibility.

Author: Parthasaarathy Sudarsanam, Audio Research Group, Tampere University
Date: January 2025
"""

params = {

    # choose task
    'modality': 'audio',  # 'audio' or 'audio_visual'
    'net_type': 'SELDnet',

    # data params
    'root_dir': '/home/var/Desktop/Mohor/DCASE2025_seld_baseline/DCASE2025_SELD_dataset',  # parent directory containing the audio, video and labels directory
    'feat_dir': '/home/var/Desktop/Mohor/DCASE2025_seld_baseline/DCASE2025_SELD_dataset/features',  # store extracted features here

    'log_dir': 'logs_doa',  # save all logs here like loss and metrics
    'checkpoints_dir': 'checkpoints_doa',  # save trained model checkpoints and config
    'output_dir': 'outputs_doa',  # save the predicted files here.


    # audio feature extraction params
    'sampling_rate': 24000,
    'hop_length_s': 0.02,
    'nb_mels': 64,

    # video feature extraction params
    'fps': 10,
    'resnet_feature_size': 49,  # (7,7) feature_map for every frame from resnet

   
    'nb_classes': 13,
    'label_sequence_length': 50,  # 5 seconds with 100ms frames

    # loss params
    'multiACCDOA': False,
    'thresh_unify': 15,

    # training params
    'nb_epochs': 300,
    'batch_size': 32,
    'nb_workers': 0,
    'shuffle': True,

    # optimizer params
    'learning_rate': 1e-3,
    'weight_decay': 0,

    # folds for training, testing
    'dev_train_folds': ['fold1', 'fold3'],  # 'fold1' is the synthetic training data. You can skip that if you do not use the synthetic data to train.
    'dev_test_folds': ['fold4'],

    # metric params
    'average': 'macro',                  # Supports 'micro': sample-wise average and 'macro': class-wise average.
    'segment_based_metrics': False,      # If True, uses segment-based metrics, else uses event-based metrics.
    'lad_doa_thresh': 20,                # DOA error threshold for computing the detection metrics.
    'lad_dist_thresh': float('inf'),     # Absolute distance error threshold for computing the detection metrics.
    'lad_reldist_thresh': float('1.0'),  # Relative distance error threshold for computing the detection metrics.
    'lad_req_onscreen': False,           # Require correct on-screen estimation when computing the detection metrics.

    'use_jackknife': False,               # If True, uses jackknife to calc results of the best model on test/eval set.
                                          # CAUTION: Too slow to use jackknife

}
