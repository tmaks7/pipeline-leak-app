import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec
from collections import deque
from datetime import datetime

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Pipeline SCADA — Leak Detector",
    page_icon="🛢️",
    layout="wide"
)

plt.rcParams.update({
    "figure.facecolor":  "#0e1117",
    "axes.facecolor":    "#1a1f2e",
    "axes.edgecolor":    "#2a3a4a",
    "axes.labelcolor":   "#c8e4f8",
    "xtick.color":       "#8a9ab0",
    "ytick.color":       "#8a9ab0",
    "grid.color":        "#2a3a4a",
    "text.color":        "#c8e4f8",
})

# ── CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
  .scada-header {
    background: linear-gradient(90deg, #0e1117, #1a2a3a);
    border-left: 4px solid #00d4ff;
    padding: 14px 20px; margin-bottom: 18px;
  }
  .scada-header h2 { color:#00d4ff; font-family:monospace; margin:0; font-size:1.3rem; letter-spacing:.12em; }
  .scada-header p  { color:#8a9ab0; font-family:monospace; font-size:.7rem; margin:2px 0 0; letter-spacing:.15em; }
  .kpi-card {
    background:#1a1f2e; border:1px solid #2a3a4a;
    border-top: 3px solid; padding:14px 18px; border-radius:4px;
  }
  .kpi-card.blue  { border-top-color:#00d4ff; }
  .kpi-card.green { border-top-color:#00ff88; }
  .kpi-card.red   { border-top-color:#ff4466; }
  .kpi-card.amber { border-top-color:#ffaa00; }
  .kpi-label { font-family:monospace; font-size:.6rem; letter-spacing:.18em; color:#8a9ab0; text-transform:uppercase; }
  .kpi-value { font-family:monospace; font-size:1.7rem; font-weight:700; margin:4px 0 2px; }
  .kpi-sub   { font-family:monospace; font-size:.6rem; color:#8a9ab0; }
  .alarm-row { background:#2a1a1a; border-left:3px solid #ff4466; padding:6px 12px; margin:3px 0;
               font-family:monospace; font-size:.72rem; color:#ff8888; border-radius:2px; }
  .normal-row{ background:#1a2a1a; border-left:3px solid #00ff88; padding:6px 12px; margin:3px 0;
               font-family:monospace; font-size:.72rem; color:#88ffaa; border-radius:2px; }
  .tag       { display:inline-block; padding:1px 7px; border-radius:2px; font-size:.6rem;
               font-family:monospace; letter-spacing:.1em; font-weight:700; }
  .tag-leak  { background:rgba(255,68,102,.2); color:#ff4466; border:1px solid #ff4466; }
  .tag-ok    { background:rgba(0,255,136,.15); color:#00ff88; border:1px solid #00ff88; }
  .section-title { font-family:monospace; font-size:.65rem; letter-spacing:.2em; color:#00d4ff;
                   text-transform:uppercase; border-bottom:1px solid #2a3a4a; padding-bottom:6px; margin-bottom:12px; }
</style>
""", unsafe_allow_html=True)

# ── Session state: rolling history ────────────────────────────
HISTORY = 60
if "history" not in st.session_state:
    st.session_state.history = {
        "pressure_diff":      deque([5.0]*HISTORY, maxlen=HISTORY),
        "flow_diff":          deque([3.0]*HISTORY, maxlen=HISTORY),
        "pressure_roll_mean": deque([5.0]*HISTORY, maxlen=HISTORY),
        "flow_roll_std":      deque([1.0]*HISTORY, maxlen=HISTORY),
        "prob":               deque([0.0]*HISTORY, maxlen=HISTORY),
        "leak":               deque([0]*HISTORY,   maxlen=HISTORY),
        "timestamps":         deque(["--"]*HISTORY, maxlen=HISTORY),
    }
if "event_log" not in st.session_state:
    st.session_state.event_log = []

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")
model = load_model()
features = ["pressure_diff", "flow_diff", "pressure_roll_mean", "flow_roll_std"]

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="scada-header">
  <h2>🛢 PIPELINE SCADA — LEAK DETECTION SYSTEM</h2>
  <p>SUPERVISORY CONTROL AND DATA ACQUISITION · ML-POWERED ANOMALY DETECTION · REV 2.4</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-title">⚙ Sensor Input Panel</div>', unsafe_allow_html=True)

    sim_mode = st.toggle("🔄 Auto-simulate sensor feed", value=False)

    if sim_mode:
        inject_leak = st.toggle("💥 Inject leak event", value=False)
        st.caption("Simulates live sensor readings each run")
        np.random.seed(int(datetime.now().microsecond))
        if inject_leak:
            pressure_diff      = np.random.normal(5, 0.5) - np.random.uniform(5, 10)
            flow_diff          = np.random.normal(3, 0.5) + np.random.uniform(10, 30)
            pressure_roll_mean = np.random.normal(5, 0.5) - np.random.uniform(3, 7)
            flow_roll_std      = np.random.normal(1, 0.2) + np.random.uniform(4, 8)
        else:
            pressure_diff      = np.random.normal(5.0, 0.4)
            flow_diff          = np.random.normal(3.0, 0.3)
            pressure_roll_mean = np.random.normal(5.0, 0.3)
            flow_roll_std      = np.random.normal(1.0, 0.15)
    else:
        pressure_diff      = st.number_input("Pressure Difference (kPa)",  value=5.0, step=0.1)
        flow_diff          = st.number_input("Flow Difference (L/s)",       value=3.0, step=0.1)
        pressure_roll_mean = st.number_input("Pressure Rolling Mean (kPa)", value=5.0, step=0.1)
        flow_roll_std      = st.number_input("Flow Rolling Std (L/s)",      value=1.0, step=0.1)

    st.divider()
    threshold   = st.slider("Decision Threshold", 0.05, 0.95, 0.50, step=0.01)
    auto_refresh = st.slider("Auto-refresh (sec)", 1, 10, 3) if sim_mode else None
    predict_btn = st.button("▶ Submit Reading", use_container_width=True, type="primary")

    if sim_mode and auto_refresh:
        st.caption(f"⏱ Refreshing every {auto_refresh}s")

# ── Predict ───────────────────────────────────────────────────
input_arr = np.array([[pressure_diff, flow_diff, pressure_roll_mean, flow_roll_std]])
prob = model.predict_proba(input_arr)[0][1]
pred = int(prob >= threshold)
ts   = datetime.now().strftime("%H:%M:%S")

# Push to history on button press or sim mode
if predict_btn or sim_mode:
    h = st.session_state.history
    h["pressure_diff"].append(pressure_diff)
    h["flow_diff"].append(flow_diff)
    h["pressure_roll_mean"].append(pressure_roll_mean)
    h["flow_roll_std"].append(flow_roll_std)
    h["prob"].append(prob)
    h["leak"].append(pred)
    h["timestamps"].append(ts)
    if pred == 1:
        st.session_state.event_log.insert(0, {
            "time": ts, "prob": prob,
            "pressure_diff": pressure_diff, "flow_diff": flow_diff
        })
        st.session_state.event_log = st.session_state.event_log[:50]

h = st.session_state.history

# ── KPI strip ─────────────────────────────────────────────────
alarm_count = sum(h["leak"])
avg_prob    = np.mean(h["prob"])
status_col  = "red" if pred == 1 else "green"
status_txt  = "⚠ LEAK DETECTED" if pred == 1 else "✔ NORMAL"

k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    (k1, "blue",  "LEAK PROBABILITY",     f"{prob:.3f}",       "current reading"),
    (k2, status_col, "SYSTEM STATUS",     status_txt,          f"threshold @ {threshold:.2f}"),
    (k3, "amber", "PRESSURE DIFF",        f"{pressure_diff:.2f} kPa", "current"),
    (k4, "green", "FLOW DIFF",            f"{flow_diff:.2f} L/s",     "current"),
    (k5, "red",   "ALARMS (LAST 60)",     str(alarm_count),    f"avg p={avg_prob:.3f}"),
]
for col, cls, label, val, sub in kpis:
    col.markdown(f"""
    <div class="kpi-card {cls}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{val}</div>
      <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── SCADA trend panel ─────────────────────────────────────────
st.markdown('<div class="section-title">◈ Live Trend Panel — Last 60 Readings</div>', unsafe_allow_html=True)

x = list(range(HISTORY))
leak_arr = np.array(h["leak"])
prob_arr = np.array(h["prob"])

fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
fig.patch.set_facecolor("#0e1117")
fig.subplots_adjust(hspace=0.08)

configs = [
    ("pressure_diff",      "Pressure Diff (kPa)",  "#00d4ff", (0, 20)),
    ("flow_diff",          "Flow Diff (L/s)",       "#00ff88", (0, 40)),
    ("pressure_roll_mean", "P Roll Mean (kPa)",     "#ffaa00", (-5, 15)),
    ("flow_roll_std",      "Flow Roll Std (L/s)",   "#cc88ff", (0, 12)),
]

for ax, (key, ylabel, color, ylim) in zip(axes, configs):
    vals = np.array(h[key])
    ax.fill_between(x, vals, alpha=0.12, color=color)
    ax.plot(x, vals, color=color, lw=1.5, zorder=3)

    # Shade leak regions
    for i, lk in enumerate(leak_arr):
        if lk == 1:
            ax.axvspan(i - 0.5, i + 0.5, color="#ff4466", alpha=0.18, zorder=2)

    ax.set_ylabel(ylabel, fontsize=8, labelpad=4)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.2, lw=0.5)
    ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3a4a")

axes[-1].set_xlabel("Reading index (newest → right)", fontsize=8)

# Shared legend
leak_patch   = mpatches.Patch(color="#ff4466", alpha=0.5, label="Leak event")
normal_patch = mpatches.Patch(color="#1a1f2e", label="Normal")
fig.legend(handles=[leak_patch, normal_patch], loc="upper right",
           fontsize=8, framealpha=0.3, labelcolor="white")

st.pyplot(fig)
plt.close()

# ── Gauge row + probability trend ─────────────────────────────
st.markdown('<div class="section-title">◈ Instrument Panel</div>', unsafe_allow_html=True)

g1, g2, g3 = st.columns([1, 1, 2])

def draw_gauge(ax, value, vmin, vmax, label, color, unit=""):
    """Half-circle gauge."""
    ax.set_facecolor("#0e1117")
    theta = np.linspace(np.pi, 0, 200)
    # Track
    ax.plot(np.cos(theta), np.sin(theta), color="#2a3a4a", lw=14, solid_capstyle="round")
    # Fill
    frac = np.clip((value - vmin) / (vmax - vmin), 0, 1)
    theta_fill = np.linspace(np.pi, np.pi - frac * np.pi, 200)
    ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=color, lw=14, solid_capstyle="round")
    # Needle
    angle = np.pi - frac * np.pi
    ax.annotate("", xy=(0.62 * np.cos(angle), 0.62 * np.sin(angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.8))
    ax.text(0, -0.22, f"{value:.2f}{unit}", ha="center", va="center",
            fontsize=13, fontweight="bold", color=color, fontfamily="monospace")
    ax.text(0, -0.48, label, ha="center", va="center",
            fontsize=7, color="#8a9ab0", fontfamily="monospace", letter_spacing=2)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.65, 1.15)
    ax.axis("off")

with g1:
    fig_g, ax_g = plt.subplots(figsize=(3.5, 2.2))
    fig_g.patch.set_facecolor("#0e1117")
    c = "#ff4466" if pressure_diff < 0 else "#00d4ff"
    draw_gauge(ax_g, pressure_diff, -5, 20, "PRESSURE DIFF", c, " kPa")
    st.pyplot(fig_g); plt.close()

with g2:
    fig_g, ax_g = plt.subplots(figsize=(3.5, 2.2))
    fig_g.patch.set_facecolor("#0e1117")
    c = "#ff4466" if flow_diff > 15 else "#00ff88"
    draw_gauge(ax_g, flow_diff, 0, 40, "FLOW DIFF", c, " L/s")
    st.pyplot(fig_g); plt.close()

with g3:
    fig_p, ax_p = plt.subplots(figsize=(7, 2.2))
    fig_p.patch.set_facecolor("#0e1117")
    ax_p.set_facecolor("#1a1f2e")
    ax_p.fill_between(x, prob_arr, alpha=0.2, color="#c8420a")
    ax_p.plot(x, prob_arr, color="#c8420a", lw=1.5)
    ax_p.axhline(threshold, color="white", lw=1, linestyle="--", alpha=0.5, label=f"Threshold {threshold:.2f}")
    ax_p.fill_between(x, prob_arr, threshold,
                       where=(prob_arr >= threshold),
                       color="#ff4466", alpha=0.35, label="Alarm zone")
    ax_p.set_ylim(0, 1); ax_p.set_ylabel("P(leak)", fontsize=8)
    ax_p.set_xlabel("Reading index", fontsize=8)
    ax_p.legend(fontsize=7, framealpha=0.2, labelcolor="white")
    ax_p.grid(alpha=0.2, lw=0.5)
    ax_p.tick_params(labelsize=7)
    for sp in ax_p.spines.values(): sp.set_edgecolor("#2a3a4a")
    ax_p.set_title("LEAK PROBABILITY TREND", fontsize=8, color="#00d4ff",
                   fontfamily="monospace", loc="left", pad=8)
    st.pyplot(fig_p); plt.close()

# ── Pipeline diagram ──────────────────────────────────────────
st.markdown('<div class="section-title">◈ Pipeline Schematic</div>', unsafe_allow_html=True)

fig_s, ax_s = plt.subplots(figsize=(14, 2.8))
fig_s.patch.set_facecolor("#0e1117")
ax_s.set_facecolor("#0e1117")
ax_s.set_xlim(0, 10); ax_s.set_ylim(0, 3); ax_s.axis("off")

# Pipe body
pipe_color = "#ff4466" if pred == 1 else "#00d4ff"
pipe = plt.Rectangle((0.5, 1.1), 9, 0.8, color=pipe_color, alpha=0.15, zorder=1)
ax_s.add_patch(pipe)
ax_s.plot([0.5, 9.5], [1.1, 1.1], color=pipe_color, lw=2, alpha=0.6)
ax_s.plot([0.5, 9.5], [1.9, 1.9], color=pipe_color, lw=2, alpha=0.6)

# Flow arrow
ax_s.annotate("", xy=(8.8, 1.5), xytext=(1.2, 1.5),
              arrowprops=dict(arrowstyle="-|>", color=pipe_color, lw=2, alpha=0.5))
ax_s.text(5, 1.5, "FLOW →", ha="center", va="center",
          color=pipe_color, fontfamily="monospace", fontsize=9, alpha=0.7)

# Sensor nodes
sensors = [
    (1.5, "P_IN",  f"{pressure_diff+5:.1f}kPa", "#00d4ff"),
    (3.5, "F_IN",  f"{flow_diff+3:.1f}L/s",     "#00ff88"),
    (6.5, "F_OUT", f"{flow_diff:.1f}L/s",        "#00ff88"),
    (8.5, "P_OUT", f"5.0kPa",                    "#00d4ff"),
]
for sx, slabel, sval, scol in sensors:
    ax_s.plot(sx, 1.9, "o", color=scol, ms=8, zorder=5)
    ax_s.plot([sx, sx], [1.9, 2.35], color=scol, lw=1, alpha=0.6)
    ax_s.text(sx, 2.5, slabel, ha="center", fontsize=7,
              color=scol, fontfamily="monospace", fontweight="bold")
    ax_s.text(sx, 2.2, sval, ha="center", fontsize=6.5, color="#c8e4f8", fontfamily="monospace")

# Leak marker
if pred == 1:
    lx = 5.0
    for i in range(4):
        angle = 80 + i * 15
        rad   = np.radians(angle)
        ax_s.annotate("", xy=(lx + 0.22*np.cos(rad), 1.1 - 0.22*np.sin(rad)),
                      xytext=(lx, 1.1),
                      arrowprops=dict(arrowstyle="-|>", color="#ff4466", lw=1.2))
    ax_s.plot(lx, 1.1, "v", color="#ff4466", ms=12, zorder=6)
    ax_s.text(lx, 0.65, "⚠ LEAK", ha="center", color="#ff4466",
              fontfamily="monospace", fontsize=9, fontweight="bold")

# Status bar
status_bg = "#2a1020" if pred == 1 else "#0a2a18"
status_fc = "#ff4466"  if pred == 1 else "#00ff88"
status_msg = f"⚠  ALARM — LEAK DETECTED  |  P(leak)={prob:.3f}  |  {ts}" \
             if pred == 1 else \
             f"✔  SYSTEM NORMAL  |  P(leak)={prob:.3f}  |  {ts}"
ax_s.add_patch(plt.Rectangle((0, 0), 10, 0.45, color=status_bg, zorder=4))
ax_s.text(5, 0.22, status_msg, ha="center", va="center",
          color=status_fc, fontfamily="monospace", fontsize=8, fontweight="bold", zorder=5)

st.pyplot(fig_s); plt.close()

# ── Event log ─────────────────────────────────────────────────
st.markdown('<div class="section-title">◈ Alarm & Event Log</div>', unsafe_allow_html=True)

if not st.session_state.event_log:
    st.markdown('<div class="normal-row">No alarm events recorded in this session.</div>',
                unsafe_allow_html=True)
else:
    for ev in st.session_state.event_log[:10]:
        st.markdown(f"""
        <div class="alarm-row">
          <span class="tag tag-leak">LEAK</span>
          &nbsp; {ev['time']}
          &nbsp;|&nbsp; P(leak)=<b>{ev['prob']:.4f}</b>
          &nbsp;|&nbsp; ΔP={ev['pressure_diff']:.2f} kPa
          &nbsp;|&nbsp; ΔF={ev['flow_diff']:.2f} L/s
        </div>""", unsafe_allow_html=True)

if st.button("🗑 Clear Event Log"):
    st.session_state.event_log = []
    st.rerun()

# ── Auto refresh ───────────────────────────────────────────────
if sim_mode and auto_refresh:
    import time
    time.sleep(auto_refresh)
    st.rerun()