import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLOps Monitor — Maintenance Prédictive",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏭",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  /* Root theme */
  :root {
    --bg-primary: #0d1117;
    --bg-card: #161b22;
    --bg-card-hover: #1c2230;
    --border: #30363d;
    --text-primary: #e6edf3;
    --text-muted: #8b949e;
    --accent-green: #3fb950;
    --accent-orange: #d29922;
    --accent-red: #f85149;
    --accent-blue: #58a6ff;
    --accent-purple: #bc8cff;
  }

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
  }

  /* Header */
  .main-header {
    background: linear-gradient(135deg, #161b22 0%, #1c2230 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .main-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
    letter-spacing: -0.5px;
  }
  .main-header .subtitle {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 4px;
    font-weight: 300;
    letter-spacing: 0.5px;
  }
  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.5px;
  }
  .status-ok   { background: rgba(63,185,80,.15);  border: 1px solid rgba(63,185,80,.4);  color: #3fb950; }
  .status-warn { background: rgba(210,153,34,.15); border: 1px solid rgba(210,153,34,.4); color: #d29922; }
  .status-crit { background: rgba(248,81,73,.15);  border: 1px solid rgba(248,81,73,.4);  color: #f85149; }

  /* KPI Cards */
  .kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
  }
  .kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
  }
  .kpi-card.green::before  { background: var(--accent-green); }
  .kpi-card.orange::before { background: var(--accent-orange); }
  .kpi-card.red::before    { background: var(--accent-red); }
  .kpi-card.blue::before   { background: var(--accent-blue); }
  .kpi-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    font-family: 'Space Mono', monospace;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 8px;
  }
  .kpi-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
  }
  .kpi-delta {
    font-size: 0.75rem;
    margin-top: 6px;
    font-family: 'Space Mono', monospace;
  }
  .kpi-delta.pos { color: var(--accent-green); }
  .kpi-delta.neg { color: var(--accent-red); }
  .kpi-delta.neu { color: var(--text-muted); }

  /* Alert banner */
  .alert-banner {
    border-radius: 10px;
    padding: 16px 20px;
    margin: 12px 0;
    font-weight: 500;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    font-size: 0.9rem;
  }
  .alert-crit { background: rgba(248,81,73,.1); border: 1px solid rgba(248,81,73,.5); color: #ffa198; }
  .alert-warn { background: rgba(210,153,34,.1); border: 1px solid rgba(210,153,34,.5); color: #e3b341; }
  .alert-ok   { background: rgba(63,185,80,.1);  border: 1px solid rgba(63,185,80,.5);  color: #56d364; }
  .alert-info { background: rgba(88,166,255,.1); border: 1px solid rgba(88,166,255,.5); color: #79c0ff; }

  /* Section titles */
  .section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-muted);
    border-left: 3px solid var(--accent-blue);
    padding-left: 10px;
    margin: 28px 0 16px 0;
  }

  /* Health score ring label */
  .health-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--text-muted);
    text-align: center;
    margin-top: -8px;
  }

  /* Dataframe */
  .stDataFrame { border: 1px solid var(--border) !important; border-radius: 8px !important; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] p {
    font-size: 0.82rem !important;
    color: var(--text-muted) !important;
  }

  /* Divider */
  hr { border-color: var(--border) !important; margin: 24px 0 !important; }

  /* Footer */
  .footer {
    text-align: center;
    font-size: 0.72rem;
    color: var(--text-muted);
    font-family: 'Space Mono', monospace;
    margin-top: 32px;
    padding: 16px;
    border-top: 1px solid var(--border);
  }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#8b949e", family="DM Sans"),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#30363d"),
    margin=dict(l=0, r=0, t=32, b=0),
)

METRIC_COLORS = {
    "accuracy":  "#58a6ff",
    "precision": "#bc8cff",
    "recall":    "#3fb950",
    "f1_score":  "#f78166",
}

def color_for_score(v, warn=0.80, crit=0.70):
    if v >= warn: return "#3fb950"
    if v >= crit: return "#d29922"
    return "#f85149"

def health_score(row):
    return round((row["accuracy"] + row["f1_score"] * 1.5 + row["precision"] + row["recall"]) / 4.5, 4)

# ─── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_metrics(path="data/metrics_history.csv"):
    try:
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["health"] = df.apply(health_score, axis=1)
        return df
    except FileNotFoundError:
        return None

# ─── Demo data fallback ───────────────────────────────────────────────────────
@st.cache_data
def generate_demo():
    np.random.seed(42)
    n = 60
    t = [datetime(2026, 3, 1) + timedelta(hours=6 * i) for i in range(n)]
    acc  = np.clip(0.94 - np.linspace(0, 0.12, n) + np.random.normal(0, 0.008, n), 0.60, 1.0)
    prec = np.clip(acc  + np.random.normal(0, 0.012, n), 0.60, 1.0)
    rec  = np.clip(acc  - 0.02 + np.random.normal(0, 0.012, n), 0.60, 1.0)
    f1   = np.clip(2 * prec * rec / (prec + rec), 0.60, 1.0)
    df = pd.DataFrame({"timestamp": t, "accuracy": acc, "precision": prec,
                       "recall": rec, "f1_score": f1})
    df["health"] = df.apply(health_score, axis=1)
    return df

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    data_path = st.text_input("📁 Fichier CSV", value="data/metrics_history.csv",
                              help="Chemin vers metrics_history.csv")

    st.markdown("**Seuils d'alerte (Accuracy)**")
    thresh_crit = st.slider("🔴 Critique (chute)", 0.05, 0.30, 0.10, 0.01,
                            format="%.2f")
    thresh_warn = st.slider("🟠 Attention (chute)", 0.02, 0.15, 0.05, 0.01,
                            format="%.2f")

    st.markdown("**Fenêtre glissante**")
    window = st.slider("Taille de fenêtre (batches)", 3, 20, 5)

    st.markdown("**Affichage**")
    show_bands = st.checkbox("Bandes de confiance", value=True)
    show_annotations = st.checkbox("Annotations d'alerte", value=True)
    selected_metrics = st.multiselect(
        "Métriques visibles",
        ["accuracy", "precision", "recall", "f1_score", "health"],
        default=["accuracy", "precision", "recall", "f1_score"],
    )

    st.markdown("---")
    demo_mode = st.checkbox("🔧 Mode démo (données simulées)", value=False)
    if st.button("🔄 Actualiser les données"):
        st.cache_data.clear()
        st.rerun()

# ─── Load data ────────────────────────────────────────────────────────────────
df = None if demo_mode else load_metrics(data_path)
using_demo = df is None

if using_demo:
    df = generate_demo()
    if not demo_mode:
        st.markdown("""<div class="alert-banner alert-info">
            ℹ️  <strong>data/metrics_history.csv</strong> introuvable — affichage en mode démo.
            Exécutez <code>src/periodic_metrics.py</code> puis désactivez le mode démo.
        </div>""", unsafe_allow_html=True)

# ─── Drift Analysis ───────────────────────────────────────────────────────────
def compute_drift(series, window):
    if len(series) < window * 2:
        return None, None
    baseline = series.iloc[:window].mean()
    recent   = series.iloc[-window:].mean()
    drop     = baseline - recent
    return drop, recent

drop, recent_acc = compute_drift(df["accuracy"], window)

if drop is None:
    status_label = "INSUFFISANT"
    status_class = "status-ok"
    alert_class  = "alert-info"
    alert_icon   = "ℹ️"
    alert_msg    = f"Minimum {window * 2} batches requis pour l'analyse de dérive."
elif drop > thresh_crit:
    status_label = "CRITIQUE"
    status_class = "status-crit"
    alert_class  = "alert-crit"
    alert_icon   = "🔴"
    alert_msg    = (f"Chute d'accuracy de <strong>{drop:.3f}</strong> détectée sur les "
                    f"{window} derniers batches. Réentraînement ou recalibration requis immédiatement.")
elif drop > thresh_warn:
    status_label = "ATTENTION"
    status_class = "status-warn"
    alert_class  = "alert-warn"
    alert_icon   = "🟠"
    alert_msg    = (f"Baisse d'accuracy de <strong>{drop:.3f}</strong> — surveillance accrue recommandée. "
                    f"Envisager une inspection des capteurs sources.")
else:
    status_label = "NOMINAL"
    status_class = "status-ok"
    alert_class  = "alert-ok"
    alert_icon   = "✅"
    alert_msg    = f"Performances stables. Variation détectée : <strong>{drop:.3f}</strong> (sous les seuils)."

# ─── Header ───────────────────────────────────────────────────────────────────
last = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else last

st.markdown(f"""
<div class="main-header">
  <div style="font-size:2.4rem;">🏭</div>
  <div>
    <h1>MLOps Monitor — Maintenance Prédictive</h1>
    <div class="subtitle">
      Surveillance temps réel · Data Drift · {len(df)} batches chargés ·
      Dernier batch : {last['timestamp'].strftime('%d %b %Y %H:%M')}
      {'&nbsp;&nbsp;<em>[DÉMO]</em>' if using_demo else ''}
    </div>
  </div>
  <div style="margin-left:auto;">
    <span class="status-pill {status_class}">● {status_label}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── KPI Row ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Métriques Courantes</div>', unsafe_allow_html=True)

def kpi_html(label, value, prev_val, color_class="blue"):
    delta = value - prev_val
    sign = "+" if delta >= 0 else ""
    dc   = "pos" if delta > 0.001 else ("neg" if delta < -0.001 else "neu")
    vc   = color_for_score(value)
    return f"""
    <div class="kpi-card {color_class}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value" style="color:{vc}">{value:.3f}</div>
      <div class="kpi-delta {dc}">{sign}{delta:.4f}</div>
    </div>"""

cols = st.columns(5)
kpi_defs = [
    ("ACCURACY",  "accuracy",  "blue"),
    ("PRÉCISION", "precision", "green"),
    ("RECALL",    "recall",    "blue"),
    ("F1-SCORE",  "f1_score",  "blue"),
    ("SANTÉ",     "health",    "blue"),
]
for col, (label, key, cc) in zip(cols, kpi_defs):
    with col:
        st.markdown(kpi_html(label, last[key], prev[key], cc), unsafe_allow_html=True)

# ─── Alert Banner ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="alert-banner {alert_class}">
  <span style="font-size:1.1rem">{alert_icon}</span>
  <div><strong>Data Drift :</strong> {alert_msg}</div>
</div>
""", unsafe_allow_html=True)

# ─── Charts ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Évolution Temporelle</div>', unsafe_allow_html=True)

col_main, col_gauge = st.columns([3, 1])

with col_main:
    fig = go.Figure()

    color_map = {
        "accuracy":  "#58a6ff",
        "precision": "#bc8cff",
        "recall":    "#3fb950",
        "f1_score":  "#f78166",
        "health":    "#ffa657",
    }
    label_map = {
        "accuracy": "Accuracy", "precision": "Précision",
        "recall": "Recall", "f1_score": "F1-Score", "health": "Santé Globale",
    }

    for metric in selected_metrics:
        color = color_map.get(metric, "#8b949e")
        y = df[metric]

        if show_bands:
            roll_std = y.rolling(window, min_periods=1).std().fillna(0)
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            fig.add_trace(go.Scatter(
                x=pd.concat([df["timestamp"], df["timestamp"].iloc[::-1]]),
                y=pd.concat([y + roll_std, (y - roll_std).iloc[::-1]]),
                fill="toself",
                fillcolor=f"rgba({r},{g},{b},0.10)",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))

        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=y,
            mode="lines+markers",
            name=label_map.get(metric, metric),
            line=dict(color=color, width=2),
            marker=dict(size=4, color=color),
            hovertemplate=f"<b>{label_map.get(metric, metric)}</b>: %{{y:.4f}}<extra></extra>",
        ))

    # Alert threshold line
    if drop is not None and show_annotations:
        baseline_val = df["accuracy"].iloc[:window].mean()
        crit_line = baseline_val - thresh_crit
        warn_line = baseline_val - thresh_warn
        fig.add_hline(y=warn_line, line_dash="dot", line_color="#d29922", line_width=1,
                      annotation_text="⚠ Seuil attention", annotation_position="bottom right",
                      annotation_font_color="#d29922", annotation_font_size=10)
        fig.add_hline(y=crit_line, line_dash="dot", line_color="#f85149", line_width=1,
                      annotation_text="🔴 Seuil critique", annotation_position="bottom right",
                      annotation_font_color="#f85149", annotation_font_size=10)

    fig.update_layout(
        **PLOTLY_THEME,
        yaxis_range=[0.55, 1.02],
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        height=360,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_gauge:
    # Health gauge
    health_val = float(last["health"])
    hc = color_for_score(health_val, warn=0.82, crit=0.72)
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(health_val * 100, 1),
        number=dict(suffix="%", font=dict(size=28, color=hc, family="Space Mono")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=0, tickcolor="rgba(0,0,0,0)",
                      tickfont=dict(color="#8b949e", size=9)),
            bar=dict(color=hc, thickness=0.28),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, 70],  color="rgba(248,81,73,0.12)"),
                dict(range=[70, 82], color="rgba(210,153,34,0.12)"),
                dict(range=[82, 100], color="rgba(63,185,80,0.12)"),
            ],
            threshold=dict(line=dict(color=hc, width=2), value=round(health_val * 100, 1)),
        ),
    ))
    gauge_layout = {**PLOTLY_THEME, "margin": dict(l=16, r=16, t=48, b=0)}
    gauge.update_layout(
        **gauge_layout,
        height=240,
        title=dict(text="Score Santé", font=dict(size=11, color="#8b949e",
                   family="Space Mono"), x=0.5),
    )
    st.plotly_chart(gauge, use_container_width=True)

    # Rolling accuracy
    roll_acc = df["accuracy"].rolling(window).mean().dropna()
    if len(roll_acc) >= 2:
        trend = roll_acc.iloc[-1] - roll_acc.iloc[-2]
        trend_icon = "▲" if trend > 0.001 else ("▼" if trend < -0.001 else "→")
        trend_color = "#3fb950" if trend > 0.001 else ("#f85149" if trend < -0.001 else "#8b949e")
        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
                    padding:14px;text-align:center;margin-top:-8px">
          <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
                      color:#8b949e;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px">
            Tendance Acc. (×{window})
          </div>
          <div style="font-family:'Space Mono',monospace;font-size:1.5rem;
                      font-weight:700;color:{trend_color}">
            {trend_icon} {trend:+.4f}
          </div>
        </div>""", unsafe_allow_html=True)

# ─── Distribution & Drift Heatmap ────────────────────────────────────────────
st.markdown('<div class="section-title">Analyse de Distribution & Dérive</div>',
            unsafe_allow_html=True)

col_dist, col_corr = st.columns(2)

with col_dist:
    # Violin / box comparison: first half vs second half
    half = len(df) // 2
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    fig2 = go.Figure()
    for m in metrics:
        c = color_map[m]
        fig2.add_trace(go.Box(
            y=df[m].iloc[:half], name=label_map[m],
            marker_color=c, line_color=c,
            fillcolor=f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.2)",
            boxmean=True, legendgroup=m, showlegend=True,
            hovertemplate=f"<b>{label_map[m]} (baseline)</b><br>%{{y:.4f}}<extra></extra>",
        ))
    for m in metrics:
        c = color_map[m]
        fig2.add_trace(go.Box(
            y=df[m].iloc[half:], name=label_map[m],
            marker_color=c, line_color=c,
            fillcolor=f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.05)",
            boxmean=True, legendgroup=m, showlegend=False,
            hovertemplate=f"<b>{label_map[m]} (récent)</b><br>%{{y:.4f}}<extra></extra>",
        ))
    fig2.update_layout(
        **PLOTLY_THEME,
        title=dict(text="Distribution : Baseline (plein) vs Récent (pointillé)",
                   font=dict(size=11, color="#8b949e")),
        yaxis_range=[0.55, 1.05],
        boxmode="group",
        height=300,
        legend=dict(orientation="h", y=1.08, font=dict(size=10)),
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_corr:
    # Rolling mean heatmap over time
    step = max(1, len(df) // 20)
    roll_df = df[["timestamp"] + metrics].copy()
    for m in metrics:
        roll_df[m] = roll_df[m].rolling(window, min_periods=1).mean()
    roll_df = roll_df.iloc[::step].reset_index(drop=True)

    z = roll_df[metrics].values.T
    x_labels = [t.strftime("%d/%m %H:%M") for t in roll_df["timestamp"]]

    fig3 = go.Figure(go.Heatmap(
        z=z,
        x=x_labels,
        y=[label_map[m] for m in metrics],
        colorscale=[
            [0.0, "#f85149"], [0.3, "#d29922"],
            [0.6, "#3fb950"], [1.0, "#58a6ff"],
        ],
        zmin=0.65, zmax=1.0,
        hovertemplate="<b>%{y}</b><br>%{x}<br>Score : %{z:.4f}<extra></extra>",
        colorbar=dict(
            tickfont=dict(color="#8b949e", size=9),
            outlinecolor="#30363d", outlinewidth=1,
            thickness=12,
        ),
    ))
    fig3_layout = {
        **PLOTLY_THEME,
        "xaxis": {**PLOTLY_THEME["xaxis"], "tickangle": 45, "tickfont": dict(size=8, color="#8b949e")},
        "yaxis": {**PLOTLY_THEME["yaxis"], "tickfont": dict(size=10)},
    }
    fig3.update_layout(
        **fig3_layout,
        title=dict(text="Carte Thermique — Moyenne Glissante",
                   font=dict(size=11, color="#8b949e")),
        height=300,
    )
    st.plotly_chart(fig3, use_container_width=True)

# ─── Drift per metric bar chart ───────────────────────────────────────────────
if len(df) >= window * 2:
    st.markdown('<div class="section-title">Delta de Dérive par Métrique</div>',
                unsafe_allow_html=True)
    drifts, drift_colors = [], []
    for m in metrics:
        d = df[m].iloc[:window].mean() - df[m].iloc[-window:].mean()
        drifts.append(d)
        drift_colors.append(
            "#f85149" if d > thresh_crit else
            "#d29922" if d > thresh_warn else "#3fb950"
        )
    fig4 = go.Figure(go.Bar(
        x=[label_map[m] for m in metrics],
        y=drifts,
        marker_color=drift_colors,
        text=[f"{d:+.4f}" for d in drifts],
        textposition="outside",
        textfont=dict(family="Space Mono", size=11),
        hovertemplate="<b>%{x}</b><br>Dérive : %{y:+.4f}<extra></extra>",
    ))
    fig4.add_hline(y=thresh_warn, line_dash="dot", line_color="#d29922", line_width=1)
    fig4.add_hline(y=thresh_crit, line_dash="dot", line_color="#f85149", line_width=1)
    fig4_layout = {
        **PLOTLY_THEME,
        "yaxis": {**PLOTLY_THEME["yaxis"], "title": "Δ (Baseline − Récent)", "tickformat": "+.3f"},
    }
    fig4.update_layout(
        **fig4_layout,
        height=260,
        showlegend=False,
    )
    st.plotly_chart(fig4, use_container_width=True)

# ─── Raw Data ─────────────────────────────────────────────────────────────────
with st.expander("📂 Historique brut des batches"):
    display_df = df.copy()
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    for m in ["accuracy", "precision", "recall", "f1_score", "health"]:
        display_df[m] = display_df[m].map("{:.4f}".format)
    st.dataframe(
        display_df.sort_values("timestamp", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=280,
    )

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
  🏭 Cycle Ingénieur MLOps · Maintenance Prédictive Industrielle ·
  Dernier rafraîchissement : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
  {'· MODE DÉMO' if using_demo else ''}
</div>
""", unsafe_allow_html=True)