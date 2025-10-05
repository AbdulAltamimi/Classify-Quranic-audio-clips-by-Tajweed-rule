# 🕌 Tajweed Rule Classification from Quran Recitations  
🎧 *Deep Learning with EfficientNet on Log-Mel Spectrograms*

---

## 📌 Overview

This repository presents a full **end-to-end pipeline** for classifying Tajweed rules from Quranic recitations using deep learning.  
The model leverages **EfficientNet-B0** trained on **log-Mel spectrograms** derived from audio clips.

The project includes:

- ✅ Audio preprocessing & exploratory data analysis
- ✅ Custom PyTorch Dataset class
- ✅ Waveform + spectrogram augmentations
- ✅ Stratified K-Fold CV training with Focal Loss
- ✅ Test-time augmentation (TTA) for inference
- ✅ Ensemble predictions and final submission

---


## 🎛️ Configuration & Hyperparameters

Below we define key constants used throughout the pipeline:

```python
# Audio parameters
SAMPLE_RATE = 16_000       # Hz
MAX_SEC     = 6.0          # seconds
MAX_LEN     = int(SAMPLE_RATE * MAX_SEC)

# Training parameters
BATCH   = 64
EPOCHS  = 10
LR      = 3e-4
K_FOLDS = 5
```
## 📊 Exploratory Data Analysis (EDA)

We explore:

- Distribution of `sheikh_name` in train and test sets
- <img width="631" height="203" alt="image" src="https://github.com/user-attachments/assets/6112dd7f-b1c2-42dc-b2a9-f32ff6fb3185" />

- Histogram of `label_name` (Tajweed rule distribution)
- <img width="534" height="425" alt="image" src="https://github.com/user-attachments/assets/2354d92a-027b-44a2-a878-eda2ac6dc0e6" />

- Audio duration histogram
- <img width="633" height="285" alt="image" src="https://github.com/user-attachments/assets/15ef27ef-cb45-486c-af0f-80305cd2e077" />

- RMS energy and waveform mean energy
- <img width="566" height="447" alt="image" src="https://github.com/user-attachments/assets/46bf8bca-5079-4c5c-9f99-56510c69c9f4" />

- Detection of short (< 0.5s) or leaked audio samples

---

## 🧹 Data Cleaning

We perform the following:

- ✅ Remove duplicated audio (using SHA-256 waveform hash)
- ✅ Remove very short audios in the train set (< 0.5 sec)
- ℹ️ Short audios in the test set are logged but **not removed**

---

## 🎶 Feature Extraction: Log-Mel Spectrograms

Each `.wav` file is:

- Resampled to 16 kHz
- Converted to mono
- Padded or trimmed to 6 seconds
- Transformed into a **log-Mel spectrogram** (`3 × 96 × T`)

> 🔄 Normalization is applied using `GLOBAL_MEAN` and `GLOBAL_STD` computed over the full training set.

---

## 🧾 Dataset Class (`TajweedSpecDataset`)

This custom PyTorch dataset class handles:

- Log-Mel spectrogram transformation
- Waveform augmentations:
  - Gaussian noise
  - Time stretching (SoX-based)
  - Volume gain
- Spectrogram augmentations:
  - Time masking
  - Frequency masking

---

## 🧼 Performance Summary

| Metric       | Value                                |
|--------------|--------------------------------------|
| CV F1 Score  | **~0.88** 🔥                         |
| Test TTA     | ✅ Enabled                            |
| Inference    | Ensemble of 5 folds + softmax voting |
