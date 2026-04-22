import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay
)

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Pipeline Leak Detector",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 Pipeline Leak Detection — RF Classifier")
st.caption("Feature engineering → model training → evaluation, end-to-end")

# ── Sidebar controls ───────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    n_samples     = st.slider("Dataset size (n)", 500, 5000, 2000, step=500)
    n_leaks       = st.slider("Number of leak events", 50, 300, 150, step=10)
    test_size     = st.slider("Test split", 0.10, 0.40, 0.20, step=0.05)
    threshold     = st.slider("Decision threshold", 0.05, 0.95, 0.50, step=0.01)
    random_seed   = st.number_input("Random seed", value=42)
    st.divider()
    run = st.button("▶ Run / Retrain", use_container_width=True, type="primary")

# ── Generate data ──────────────────────────────────────────────
@st.cache_data
def generate_data(n, n_leaks, seed):
    np.random.seed(seed)
    time         = pd.date_range(start="2024-01-01", periods=n, freq="h")
    pressure_in  = np.random.normal(100, 2, n)
    pressure_out = pressure_in - np.random.normal(5, 1, n)
    flow_in      = np.random.normal(500, 20, n)
    flow_out     = flow_in - np.random.normal(3, 2, n)
    temperature  = np.random.normal(30, 1, n)
    leak         = np.zeros(n)

    leak_indices = np.random.choice(range(200, n - 200), size=n_leaks, replace=False)
    for i in leak_indices:
        leak[i]          = 1
        pressure_out[i] -= np.random.uniform(5, 10)
        flow_out[i]     -= np.random.uniform(10, 30)

    return pd.DataFrame({
        "time": time, "pressure_in": pressure_in,
        "pressure_out": pressure_out, "flow_in": flow_in,
        "flow_out": flow_out, "temperature": temperature, "leak": leak
    })

# ── Feature engineering ────────────────────────────────────────
@st.cache_data
def engineer_features(df):
    df = df.copy()
    df["pressure_diff"]      = df["pressure_in"] - df["pressure_out"]
    df["flow_diff"]          = df["flow_in"] - df["flow_out"]
    df["pressure_roll_mean"] = df["pressure_diff"].rolling(5).mean()
    df["flow_roll_std"]      = df["flow_diff"].rolling(5).std()
    return df.dropna().reset_index(drop=True)

# ── Train model ────────────────────────────────────────────────
@st.cache_resource
def train_model(df, test_size, seed):
    features = ["pressure_diff", "flow_diff", "pressure_roll_mean", "flow_roll_std"]
    X = df[features]
    y = df["leak"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return model, X_test, y_test, y_pred, y_prob, features

# ── Run pipeline ───────────────────────────────────────────────
df_raw  = generate_data(n_samples, n_leaks, int(random_seed))
df      = engineer_features(df_raw)
model, X_test, y_test, y_pred_default, y_prob, features = train_model(
    df, test_size, int(random_seed)
)

# Apply chosen threshold
y_pred = (y_prob >= threshold).astype(int)

# ── KPI strip ──────────────────────────────────────────────────
auc_score = roc_auc_score(y_test, y_prob)
report    = classification_report(y_test, y_pred, output_dict=True)
acc       = report["accuracy"]
f1_leak   = report.get("1.0", report.get("1", {})).get("f1-score", 0)
recall_leak = report.get("1.0", report.get("1", {})).get("recall", 0)

c1, c2, c3, c4 = st.columns(4)
c1.metric("ROC–AUC",      f"{auc_score:.3f}")
c2.metric("Accuracy",     f"{acc*100:.1f}%")
c3.metric("F1 — Leak",    f"{f1_leak:.3f}")
c4.metric("Recall — Leak",f"{recall_leak:.3f}")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Evaluation",
    "📈 Feature Analysis",
    "🔬 Raw Predictions",
    "📋 Data Explorer"
])

# ────────────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns(2)

    # Confusion matrix
    with col_left:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4, 3.5))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Leak"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Threshold = {threshold:.2f}", fontsize=10)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Classification report
    with col_right:
        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).T.drop(
            columns=["support"], errors="ignore"
        ).round(3)
        st.dataframe(report_df.style.background_gradient(cmap="Blues", axis=None), use_container_width=True)

    # ROC + PR side by side
    col_roc, col_pr = st.columns(2)

    with col_roc:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color="#c8420a", lw=2, label=f"AUC = {auc_score:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="#aaa", lw=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col_pr:
        st.subheader("Precision–Recall Curve")
        prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(rec_c, prec_c, color="#2b5fa3", lw=2)
        baseline = y_test.mean()
        ax.axhline(baseline, linestyle="--", color="#aaa", lw=1, label=f"Baseline ({baseline:.2f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        fig.tight_layout(); st.pyplot(fig); plt.close()

# ────────────────────────────────────────────────────────────────
with tab2:
    col_fi, col_dist = st.columns(2)

    with col_fi:
        st.subheader("Feature Importance")
        fi = pd.Series(model.feature_importances_, index=features).sort_values()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fi.plot(kind="barh", ax=ax, color=["#2b5fa3","#1a8a4a","#c8420a","#8a4fc8"])
        ax.set_xlabel("Importance"); ax.grid(axis="x", alpha=0.3)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col_dist:
        st.subheader("Predicted Probability Distribution")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(y_prob[y_test == 0], bins=25, alpha=0.6, label="Normal",  color="#2b5fa3")
        ax.hist(y_prob[y_test == 1], bins=25, alpha=0.7, label="Leak",    color="#c8420a")
        ax.axvline(threshold, color="black", linestyle="--", lw=1.5, label=f"Threshold={threshold:.2f}")
        ax.set_xlabel("P(leak)"); ax.set_ylabel("Count")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(6, 4))
    corr = df[features + ["leak"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
    fig.tight_layout(); st.pyplot(fig); plt.close()

# ────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Prediction Results on Test Set")
    results = X_test.copy()
    results["true_label"] = y_test.values
    results["predicted"]  = y_pred
    results["probability"] = y_prob.round(4)
    results["correct"]     = (results["true_label"] == results["predicted"])

    st.dataframe(
        results.style
            .map(lambda v: "background-color:#fde8e8" if v == False else "", subset=["correct"])
            .format({"probability": "{:.4f}"}),
        use_container_width=True,
        height=420
    )

    wrong = results[~results["correct"]]
    st.caption(f"Misclassified: **{len(wrong)}** / {len(results)} ({len(wrong)/len(results)*100:.1f}%)")

# ────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Engineered Feature Dataset")
    st.dataframe(df.head(500), use_container_width=True)
    st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols after dropna | Leak rate: {df['leak'].mean()*100:.1f}%")

    csv = df.to_csv(index=False).encode()
    st.download_button(
        "⬇ Download processed CSV",
        data=csv,
        file_name="pipeline_features.csv",
        mime="text/csv"
    )