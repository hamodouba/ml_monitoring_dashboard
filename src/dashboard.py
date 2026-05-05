import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Monitoring MLOps - Maintenance Prédictive", layout="wide")
st.title("🏭 Tableau de Bord MLOps - Surveillance de Modèle Industriel")
st.markdown("Suivi continu des performances & détection de dérive capteurs (Data Drift)")

@st.cache_data(ttl=60)
def load_metrics():
    try:
        df = pd.read_csv("data/metrics_history.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except FileNotFoundError:
        st.error("❌ `data/metrics_history.csv` introuvable. Exécutez `src/periodic_metrics.py`")
        return pd.DataFrame()

df = load_metrics()
if df.empty: st.stop()

# KPIs
st.subheader("📊 Dernières Métriques")
last = df.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{last['accuracy']:.3f}")
c2.metric("Précision", f"{last['precision']:.3f}")
c3.metric("Rappel (Recall)", f"{last['recall']:.3f}")
c4.metric("F1-Score", f"{last['f1_score']:.3f}")

# Graphique
st.subheader("📈 Évolution Temporelle")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["accuracy"], mode="lines+markers", name="Accuracy"))
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["precision"], mode="lines+markers", name="Precision"))
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["recall"], mode="lines+markers", name="Recall"))
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["f1_score"], mode="lines+markers", name="F1"))
fig.update_layout(xaxis_title="Temps", yaxis_title="Score", yaxis_range=[0, 1.05], hovermode="x unified", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# Alerte
st.subheader("⚠️ Système d'Alerte - Data Drift")
window = 5
if len(df) >= window:
    drop = df["accuracy"].iloc[:window].mean() - df["accuracy"].iloc[-window:].mean()
    if drop > 0.10:
        st.error(f"🔴 **CRITIQUE** : Chute d'accuracy de **{drop:.2f}**. Réentraînement ou calibration requis.")
    elif drop > 0.05:
        st.warning(f"🟠 **ATTENTION** : Baisse de **{drop:.2f}**. Surveillance accrue.")
    else:
        st.success(" **NORMAL** : Performances stables.")
else:
    st.info(f"ℹ️ Minimum {window} batches requis pour l'analyse.")

with st.expander("📂 Historique brut"):
    st.dataframe(df.sort_values("timestamp", ascending=False).reset_index(drop=True))

st.markdown("---")
st.caption("🛠️ Cycle Ingénieur MLOps | Maintenance Prédictive Industrielle | Avril 2026")