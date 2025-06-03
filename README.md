1. Create the following directories inside root directory-:
``` bash
a. outputs
b. outputs_doa
c. outputs_sde
e. checkpoints
f. checkpoints_doa
g. checkpoints_sde
h. logs
i. logs_doa
j. logs_sde
```
2. Download dataset folders in following directory hierarchy

```bash  
DCASE2025_SELD_dataset/
├── stereo_dev/
│   ├── dev-train-tau/*.wav
│   ├── dev-train-sony/*.wav
│   ├── dev-test-tau/*.wav
│   ├── dev-test-sony/*.wav
├── metadata_dev/
│   ├── dev-train-tau/*.csv
│   ├── dev-train-sony/*.csv
│   ├── dev-test-tau/*.csv
│   ├── dev-test-sony/*.csv
├── video_dev/
│   ├── dev-train-tau/*.mp4
│   ├── dev-train-sony/*.mp4
│   ├── dev-test-tau/*.mp4
│   ├── dev-test-sony/*.mp4
 ``` 


If you generate synthetic data, place it into the respective folders under the name dev-train-synth.

3. In parameters.py file, change the file paths. Include absolute file paths
4. Create environment from env2025.yml file and activate it by conda activate newein
5. In one terminal, run python main_seddoa.py
6. In second terminal, run python main_sedsde.py
7. For inference, change the file paths in main functions of inference files to model paths which you can find under respective model outputs directories

This repository hosts the full experimental framework for the Nercslip architecture variants developed for the DCASE 2025 SELD challenge. This includes multiple configurations of the ResNet-Conformer backbone designed to process stereo log-mel and spatial features, with options for mid-fusion of high-dimensional contextual embeddings such as those from the One-Peace model. The repository is structured modularly to accommodate both SED-DOA and SED-SDE objectives, with shared components for feature extraction, loss computation, and evaluation.

Key scripts include main_seddoa.py and main_sedsde.py, which handle the training loops, checkpointing logic, and data loading workflows. Custom loss functions tailored to each task variant are implemented in loss.py, supporting compound objectives like class-wise BCE, directional MSE, and distance regression. Feature extraction is centralized in feature.py, which handles stereo spectral-spatial composition (log-mel, ILD, IPD, GCC-PHAT) and alignment with pre-extracted One-Peace embeddings. Evaluation metrics are computed using metrics_seddoa.py and metrics_sedsde.py, with logging via both Tensorboard and CSV formats.

The repository also includes experiment-specific configuration files, scheduler logic, and directory management tools. Outputs such as logs, metrics, and checkpoints are saved in organized subfolders under logs/, outputs/, and checkpoints/. The design supports distributed execution and is compatible with SLURM-based GPU clusters, ensuring scalability for hyperparameter tuning and cross-validation across multiple folds.