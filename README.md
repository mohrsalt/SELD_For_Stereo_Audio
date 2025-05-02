1. Create the following directories inside root directory-:
a. outputs
b. outputs_doa
c. outputs_sde
e. checkpoints
f. checkpoints_doa
g. checkpoints_sde
h. logs
i. logs_doa
j. logs_sde

2. Download dataset folders in following directory hierarchy

<pre> ```text DCASE2025_SELD_dataset/ ├── stereo_dev/ │ ├── dev-train-tau/*.wav │ ├── dev-train-sony/*.wav │ ├── dev-test-tau/*.wav │ ├── dev-test-sony/*.wav ├── metadata_dev/ │ ├── dev-train-tau/*.csv │ ├── dev-train-sony/*.csv │ ├── dev-test-tau/*.csv │ ├── dev-test-sony/*.csv ├── video_dev/ │ ├── dev-train-tau/*.mp4 │ ├── dev-train-sony/*.mp4 │ ├── dev-test-tau/*.mp4 │ ├── dev-test-sony/*.mp4 ``` </pre>


If you generate synthetic data, place it into the respective folders under the name dev-train-synth.

3. In parameters.py file, change the file paths. Include absolute file paths
4. Create environment from env2025.yml file and activate it by conda activate newein
5. In one terminal, run python main_seddoa.py
6. In second terminal, run python main_sedsde.py
7. For inference, change the file paths in main functions of inference files to model paths which you can find under respective model outputs directories