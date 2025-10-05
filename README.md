🎧 Tajweed Rule Classification — ConvNeXt + Log-Mel Spectrograms

End-to-end audio classification pipeline for recognizing Tajweed rules from Quran recitation audio clips using log-Mel spectrograms and ConvNeXt. Developed as part of the KAUST vs KKU ML Tournament — Round 5.

⸻

📘 Overview

This project builds a multi-class classifier to identify the Tajweed rule applied in a short audio clip of Quran recitation.
	•	Input: .wav files sampled at 16kHz
	•	Output: One of 4 Tajweed rule labels (e.g., “Ikhfa”, “Idgham”)
	•	Model: ConvNeXt-Small pretrained on ImageNet, fine-tuned on log-Mel spectrograms
	•	Cross-validation: 5-fold StratifiedKFold with macro-F1 metric

⸻

📊 Dataset
	•	Train CSV: train.csv — contains id, label_name, and sheikh_name
	•	Test CSV: test.csv — contains id, sheikh_name (no labels)
	•	Audio: train/ and test/ folders with .wav files

Each audio file corresponds to a short recitation clip. Class balance and reciter diversity are visualized during EDA.

⸻

🔍 EDA Highlights
	•	Label distribution and reciter balance between train and test
	•	Duration histogram: Most audios ~3–6 seconds
	•	RMS and Energy plots show dynamic range of clips
	•	Leaked samples removed by hashing waveforms
	•	Very short audios (<0.5s) removed from train set

⸻

🎛️ Configuration
	•	SAMPLE_RATE = 16000
	•	MAX_SEC = 6.0
	•	MAX_LEN = 96000 samples
	•	BATCH = 64, EPOCHS = 10, LR = 3e-4
	•	K_FOLDS = 5
	•	DEVICE = 'cuda' if available

⸻

🧪 Input Pipeline: Log-Mel Spectrograms

Custom Dataset class handles:
	•	Resampling to 16kHz
	•	Padding or trimming to fixed duration
	•	Computing MelSpectrogram → log-Mel (dB)
	•	Normalization using global mean/std
	•	Repeating to 3 channels for image models

🎛️ Augmentations include Gaussian noise, time-stretching, random gain, SpecAugment (Time/Frequency masking)

⸻

🏗️ Model: ConvNeXt-Small

from torchvision.models import convnext_small, ConvNeXt_Small_Weights
model = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
model.classifier[2] = nn.Linear(in_features, n_classes)

	•	Final classifier head is swapped to match n_classes=4
	•	Pretrained backbone fine-tuned end-to-end

⸻

🧠 Loss Function: Focal Loss

To handle class imbalance, we use Focal Loss with label smoothing:

FocalLoss(gamma=2.0, smoothing=0.1, weight=class_weights)

Class weights are computed from the training labels.

⸻

🔁 Training Procedure
	•	5-fold StratifiedKFold on label_name
	•	Each fold trains for 10 epochs
	•	Model checkpoint (.pt) saved per fold based on best val_F1
	•	Metrics: Macro F1 Score (torchmetrics)

Example results:

Fold1 best val_F1 = 0.9430
Fold2 best val_F1 = 0.9535
Fold3 best val_F1 = 0.9440
Fold4 best val_F1 = 0.9510
Fold5 best val_F1 = 0.9717
→ Mean CV F1 = 0.9527


⸻

🔎 Inference & Submission
	•	Load all 5 fold checkpoints
	•	Apply Test-Time Augmentation (TTA): 5 rounds of SpecAugment
	•	Average softmax predictions across folds & TTA rounds
	•	Decode label_id → label_name using LabelEncoder
	•	Save submission.csv

⸻

📊 Test Distribution Visualization

Final predicted class distribution on test set is plotted to inspect balance and model bias.

⸻

✅ Summary

This solution achieved 95.2% macro-F1 average across 5 folds using:
	•	Robust log-Mel spectrogram preprocessing
	•	ConvNeXt-Small pretrained backbone
	•	Focal loss with label smoothing and class weights
	•	5-fold stratified CV + TTA during inference

🏆 Strong baseline with room for improvements like:
	•	Using larger models (e.g., ConvNeXt-Base, Swin)
	•	Pseudo-labeling test data
	•	Advanced audio augmentations (SpecMix, MixUp)

⸻

📁 Deliverables:
	•	effb0_fold{1–5}.pt (checkpoints)
	•	submission.csv
	•	Optional: training logs & sample predictions
