import numpy as np
import pandas as pd
import joblib
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import os

print("📡 Chargement des artefacts...")
model = joblib.load("models/model.joblib")
scaler = joblib.load("models/scaler.joblib")
feature_names = joblib.load("models/feature_names.joblib")

# Reconstruction du jeu de référence (même seed)
np.random.seed(42)
N_SAMPLES = 1000
data = {
    'vibration_x': np.random.normal(0.5, 0.1, N_SAMPLES),
    'vibration_y': np.random.normal(0.5, 0.1, N_SAMPLES),
    'vibration_z': np.random.normal(0.5, 0.1, N_SAMPLES),
    'temperature_moteur': np.random.normal(65, 5, N_SAMPLES),
    'temperature_roulement': np.random.normal(70, 6, N_SAMPLES),
    'pression_hydraulique': np.random.normal(150, 10, N_SAMPLES),
    'courant_electrique': np.random.normal(12, 2, N_SAMPLES),
    'vitesse_rotation': np.random.normal(1500, 100, N_SAMPLES),
    'humidite': np.random.normal(45, 5, N_SAMPLES),
    'bruit_acoustique': np.random.normal(60, 8, N_SAMPLES),
}
df_ref = pd.DataFrame(data)
df_ref['defaillance'] = (
    (df_ref['vibration_x'] > 0.7) |
    (df_ref['temperature_moteur'] > 75) |
    (df_ref['pression_hydraulique'] < 130) |
    (df_ref['courant_electrique'] > 16)
).astype(int)

X = df_ref[feature_names].values
y = df_ref['defaillance'].values
X_scaled = scaler.transform(X)

# Paramètres simulation
batch_size = 100
total_batches = 20
drift_start_batch = 5

def apply_industrial_drift(X_batch_scaled, intensity):
    """Simule dérive réaliste sur données normalisées."""
    X_drift = X_batch_scaled.copy()
    # Vibrations (décalibration)
    X_drift[:, 0] += intensity * 0.5
    X_drift[:, 1] += intensity * 0.4
    X_drift[:, 2] += intensity * 0.3
    # Température (biais thermique)
    X_drift[:, 3] += intensity * 1.2
    # Pression (fuite/usure)
    X_drift[:, 5] -= intensity * 0.8
    return X_drift

print(f"🚀 Simulation de dérive ({total_batches} batches)...")
for batch_idx in range(total_batches):
    sample_indices = np.random.choice(N_SAMPLES, batch_size, replace=False)
    X_batch = X_scaled[sample_indices]
    y_batch = y[sample_indices]

    if batch_idx >= drift_start_batch:
        intensity = (batch_idx - drift_start_batch + 1) / (total_batches - drift_start_batch)
        intensity = min(intensity, 1.0)
    else:
        intensity = 0.0

    X_drifted = apply_industrial_drift(X_batch, intensity)
    y_pred = model.predict(X_drifted)

    acc = accuracy_score(y_batch, y_pred)
    prec = precision_score(y_batch, y_pred, zero_division=0)
    rec = recall_score(y_batch, y_pred, zero_division=0)
    f1 = f1_score(y_batch, y_pred, zero_division=0)

    timestamp = datetime.datetime.now()
    new_row = pd.DataFrame([{
        "timestamp": timestamp, "accuracy": acc, "precision": prec,
        "recall": rec, "f1_score": f1, "num_predictions": batch_size,
        "num_labels_received": batch_size
    }])

    try:
        df_hist = pd.read_csv("data/metrics_history.csv")
        df_hist = pd.concat([df_hist, new_row], ignore_index=True)
    except FileNotFoundError:
        df_hist = new_row

    os.makedirs("data", exist_ok=True)
    df_hist.to_csv("data/metrics_history.csv", index=False)

    print(f"📦 Batch {batch_idx+1:02d}/{total_batches} | Drift: {intensity:.2f} | Acc: {acc:.3f} | Rec: {rec:.3f}")
    time.sleep(2)

print("\n✅ Simulation terminée. CSV mis à jour.")