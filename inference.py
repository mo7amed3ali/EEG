import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.signal import welch
from scipy.integrate import trapezoid


# ==========================================
# 1. Load Saved Model
# ==========================================

MODEL_PATH = Path(__file__).parent / "final_eeg_model.joblib"

model_artifact = joblib.load(MODEL_PATH)

model = model_artifact["model"]
feature_names = model_artifact["feature_names"]
classes = model_artifact["classes"]


# ==========================================
# 2. EEG Configuration
# ==========================================

EEG_CHANNELS = [
    "EEG-Fz", "EEG-0", "EEG-1", "EEG-2", "EEG-3",
    "EEG-4", "EEG-5", "EEG-C3", "EEG-6", "EEG-Cz",
    "EEG-7", "EEG-C4", "EEG-8", "EEG-9", "EEG-10",
    "EEG-11", "EEG-12", "EEG-13", "EEG-14", "EEG-Pz",
    "EEG-15", "EEG-16"
]

FS = 250

BANDS = {
    "Theta": (4, 8),
    "Mu": (8, 13),
    "Beta": (13, 30)
}


# ==========================================
# 3. Extract Frequency Features
# ==========================================

def extract_features(eeg_epoch):

    eeg_epoch = np.asarray(eeg_epoch)

    if eeg_epoch.shape != (22, 201):
        raise ValueError(
            f"Expected EEG shape (22, 201), got {eeg_epoch.shape}"
        )

    new_features = {}

    for channel_idx, channel in enumerate(EEG_CHANNELS):

        signal = eeg_epoch[channel_idx]

        freqs, psd = welch(
            signal,
            fs=FS,
            nperseg=200
        )

        for band, (low, high) in BANDS.items():

            mask = (freqs >= low) & (freqs <= high)

            power = trapezoid(
                psd[mask],
                x=freqs[mask]
            )

            new_features[f"{channel}_{band}_Power"] = power

    X_new = pd.DataFrame([new_features])

    X_new = X_new[feature_names]

    return X_new


# ==========================================
# 4. Make Prediction
# ==========================================

def predict_eeg(eeg_epoch):

    X_new = extract_features(eeg_epoch)

    prediction = model.predict(X_new)[0]

    probabilities = model.predict_proba(X_new)[0]

    probability_dict = {
        class_name: float(probability)
        for class_name, probability in zip(classes, probabilities)
    }

    return {
        "prediction": prediction,
        "probabilities": probability_dict
    }