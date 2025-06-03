# 🎼 DCASE2025 SELD Evaluation – Inference Instructions

This guide provides step-by-step instructions to set up the repository and perform inference on the 10K evaluation dataset using pre-trained DOA and SDE models.

---

## 🛠️ Repository Setup

To prepare the environment and directory structure:

### 1. Create Required Folders

Open a terminal and run the following command:

```bash
mkdir checkpoints checkpoints_doa checkpoints_sde \
      logs logs_doa logs_sde \
      outputs outputs_doa outputs_sde \
      DCASE2025_SELD_dataset
```

---

### 2. Add Evaluation Dataset

Place your `stereo_eval` folder — containing the 10K evaluation dataset `.wav` files — into the following directory:

```
DCASE2025_SELD_dataset/
```

---

### 3. Update Configuration Files

Edit the following Python files to match your local system's directory structure:

- `parameters.py`
- `parameters_seddoa.py`
- `parameters_sedsde.py`

Inside each file, update the values of the following keys:

```python
root_dir = "your/local/system/path"
feat_dir = "your/local/system/path"
```

---

## 🧪 Running Inference on the 10K Evaluation Dataset

To perform inference and generate predictions:

### 1. Add Model Checkpoints

- Place your **DOA model checkpoint subfolder** inside:
  ```
  checkpoints_doa/
  ```

- Place your **SDE model checkpoint subfolder** inside:
  ```
  checkpoints_sde/
  ```

---

### 2. Update Model Paths

Open the file `inference_evaldataset_submission.py` and locate the global variables:

```python
model_dir_sde = ...
model_dir_doa = ...
```

Update them to point to the corresponding checkpoint subfolders.

---

### 3. Activate Environment

In your terminal, run:
```bash
conda activate newein
```

---

### 4. Select GPU (Optional)

If your available GPU index is not 0, set it using:
```bash
export CUDA_VISIBLE_DEVICES=<your_gpu_index>
```

---

### 5. Clear Output Directory

To avoid conflicts with previous output files:
```bash
cd outputs
rm -rf *
```

---

### 6. Run Inference

Start the evaluation by running:
```bash
python inference_evaldataset_submission.py
```

This will generate **10,000 prediction CSV files** in the `outputs/` directory.

---

## 📂 Feature Generation Note

A `features/` subdirectory will be automatically created within:

```
DCASE2025_SELD_dataset/
```

This will contain both **logmel** and **One-Peace** features.  
⏱️ Feature extraction for all 10K files takes approximately **5 hours**.
