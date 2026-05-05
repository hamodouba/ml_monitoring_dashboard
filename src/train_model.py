import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import os

# Paramètres
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SAMPLES = 1000

print(" Génération des données IoT synthétiques...")
np.random.seed(RANDOM_STATE)

# Simulation de capteurs industriels
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
df = pd.DataFrame(data)

# Création de la cible : défaillance imminente (1) ou normal (0)
df['defaillance'] = (
    (df['vibration_x'] > 0.7) |
    (df['temperature_moteur'] > 75) |
    (df['pression_hydraulique'] < 130) |
    (df['courant_electrique'] > 16)
).astype(int)

print(f"📊 Taux de défaillance: {df['defaillance'].mean():.2%}")

# Séparation
feature_names = [col for col in df.columns if col != 'defaillance']
X = df[feature_names].values
y = df['defaillance'].values

# Normalisation (standard IoT)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Entraînement
print("🤖 Entraînement Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced')
model.fit(X_train, y_train)

# Évaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"✅ Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")

# Sauvegarde
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(feature_names, "models/feature_names.joblib")

print("\n Artefacts sauvegardés dans models/")