import pandas as pd
import os

os.makedirs("data", exist_ok=True)
cols = ["timestamp", "accuracy", "precision", "recall", "f1_score", "num_predictions", "num_labels_received"]
pd.DataFrame(columns=cols).to_csv("data/metrics_history.csv", index=False)
print("✅ CSV initialisé.")