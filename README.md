# EEG Motor Imagery Classification

An EEG-based Motor Imagery Classification project that uses signal preprocessing, frequency-domain feature extraction, and Machine Learning to classify four motor imagery tasks.

## Project Overview

This project focuses on classifying EEG signals into four motor imagery classes:

- Foot
- Left Hand
- Right Hand
- Tongue

The project uses EEG recordings from the **BCI Competition IV – Dataset 2a**, containing recordings from nine subjects and 22 EEG channels.

## Dataset Information

| Property | Value |
|---|---|
| Dataset | BCI Competition IV – Dataset 2a |
| Subjects | 9 |
| EEG Channels | 22 |
| Sampling Frequency | 250 Hz |
| Number of Classes | 4 |
| Epoch Duration | 201 samples |
| Total Epochs | 2,448 |

**Note:** The original dataset is not included in this repository due to its large file size.

## EEG Preprocessing

The following preprocessing steps were applied to the EEG signals:

1. **Common Average Reference (CAR):** Re-referencing EEG channels to reduce common noise.
2. **Detrending:** Removing linear trends from the signals.
3. **Bandpass Filtering:** Applying a 4th-order Butterworth filter between 1 and 40 Hz.
4. **Zero-Phase Filtering:** Using `sosfiltfilt` to avoid phase distortion.

## Feature Extraction

Frequency-domain features were extracted using Welch's Power Spectral Density (PSD).

Three frequency bands were selected:

| Frequency Band | Range |
|---|---|
| Theta | 4–8 Hz |
| Mu | 8–13 Hz |
| Beta | 13–30 Hz |

For each of the 22 EEG channels, three frequency-band power features were calculated.

**Total Features: 22 × 3 = 66 features per epoch.**

## Machine Learning Model

The final selected model is **Logistic Regression**.

The model is implemented using a Scikit-learn Pipeline containing:

- StandardScaler
- LogisticRegression

The model was evaluated using **Leave-One-Subject-Out (LOSO) Cross-Validation**, where each subject is held out for testing while the remaining subjects are used for training.

## Evaluation Results

| Metric | Result |
|---|---:|
| Mean Accuracy | 31.37% |
| Macro F1-Score | 28.81% |

These results represent the average performance across the nine subject-level test folds.

## Project Structure

```text
EEG/
│
├── final_eeg_model.joblib
├── inference.py
├── gui.py
├── task_7.ipynb
└── README.md
```

### File Description

| File | Description |
|---|---|
| `task_7.ipynb` | Complete notebook containing preprocessing, feature extraction, experiments, and evaluation. |
| `final_eeg_model.joblib` | Saved trained Logistic Regression model and metadata. |
| `inference.py` | Feature extraction and prediction functions. |
| `gui.py` | Graphical user interface for EEG classification. |

## Installation

Clone the repository:

```bash
git clone https://github.com/mo7amed3ali/EEG.git
cd EEG
```

Install the required libraries:

```bash
pip install numpy pandas scipy scikit-learn==1.6.1 joblib matplotlib
```

## Usage

Run the graphical interface:

```bash
python gui.py
```

You can also use the inference module directly:

```python
import numpy as np
from inference import predict_eeg

# Example input: one preprocessed EEG epoch
# Shape: (22 channels, 201 samples)
eeg_epoch = np.random.randn(22, 201)

result = predict_eeg(eeg_epoch)

print("Prediction:", result["prediction"])
print("Probabilities:", result["probabilities"])
```

**Important:** The input epoch must already be preprocessed using the same preprocessing pipeline as the training data and must follow the same EEG channel order. The inference module does not apply CAR, detrending, or bandpass filtering automatically.

The random signal above is only an example of the required input shape and must not be considered a meaningful EEG prediction.

## Technologies Used

- Python
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Joblib
- Matplotlib
- Tkinter
- Jupyter Notebook

## Limitations

- Classification performance is still limited and leaves room for improvement.
- EEG signals can vary significantly between subjects.
- The inference pipeline requires preprocessed EEG input.
- The original dataset is not included in the repository.

## Author

**Mohamed Ali**

- GitHub: [@mo7amed3ali](https://github.com/mo7amed3ali)
- LinkedIn: [Mohamed Ali](https://www.linkedin.com/in/mohamed-ali-cs/)
