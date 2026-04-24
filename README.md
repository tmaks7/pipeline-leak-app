# 🛢️ Pipeline Leak Detection System

> A machine learning-powered SCADA dashboard for real-time pipeline anomaly detection, built with Random Forest classification and deployed on Streamlit Community Cloud.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview

This project simulates a real-world oil & gas pipeline monitoring system that detects leak events from sensor telemetry data using a trained Random Forest classifier. The application combines traditional SCADA (Supervisory Control and Data Acquisition) visual design with modern machine learning inference, deployed as an interactive web application.

The system ingests four engineered sensor features, runs them through a trained model, and outputs a real-time leak probability with visual alarms — mimicking the kind of monitoring dashboard used in industrial pipeline operations.

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pipeline-leak-app-dklxdmarvknwjkyanytx7a.streamlit.app/)


---

## 📸 Screenshots

| SCADA Live Panel | Model Evaluation |
|---|---|
| ![SCADA](https://github.com/tmaks7/pipeline-leak-app/blob/master/Pipeline_SCADA_Live_Leak%20Detector_page-0001.jpg) | ![Eval](https://github.com/tmaks7/pipeline-leak-app/blob/master/Pipeline_SCADA_Leak%20Detector_modelevaluation_page-0001.jpg) |

---

## 🧠 How It Works

### 1. Data Generation
Synthetic pipeline sensor data is generated at hourly intervals across 2,000 readings, capturing four sensor channels: inlet pressure, outlet pressure, inlet flow, and outlet flow. Leak events are injected at random intervals with realistic anomaly magnitudes — pressure drops of 5–10 kPa and flow losses of 10–30 L/s.

### 2. Feature Engineering
Four features are derived from the raw sensor readings:

| Feature | Description | Importance |
|---|---|---|
| `pressure_diff` | Inlet minus outlet pressure | ~18% |
| `flow_diff` | Inlet minus outlet flow rate | ~32% |
| `pressure_roll_mean` | 5-reading rolling mean of pressure diff | ~9% |
| `flow_roll_std` | 5-reading rolling std of flow diff | ~41% |

The rolling standard deviation of flow differential is the strongest signal — a leak creates sudden erratic variance in flow loss that normal operation does not produce.

### 3. Model Training
A Random Forest Classifier is trained with `class_weight='balanced'` to handle the class imbalance inherent in leak detection datasets (~7.5% positive rate). The model is evaluated with stratified cross-validation and a tuned decision threshold of 0.30 to maximise recall on the minority (leak) class.

### 4. SCADA Dashboard
The deployed app provides five views:

- **🖥 SCADA Live** — real-time gauge instruments, scrolling trend panel, pipeline schematic with live leak marker, and an alarm event log
- **📊 Model Evaluation** — confusion matrix, ROC curve, precision-recall curve, and classification report
- **📈 Feature Analysis** — feature importances, probability distribution, and correlation heatmap
- **🔬 Raw Predictions** — full test set with per-row correct/incorrect highlighting
- **📋 Data Explorer** — browse and download the processed feature dataset

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| ROC–AUC | ~0.99 |
| Accuracy | ~97% |
| F1 — Leak class | ~0.95 |
| Recall — Leak class | ~0.97 |
| CV ROC-AUC (5-fold) | ~0.99 |

> Metrics are evaluated on a held-out 20% stratified test set at decision threshold 0.30.

---

## 🗂️ Project Structure

```
pipeline_leak/
│
├── .streamlit/
│   └── config.toml          # Dark theme configuration
│
├── app.py                   # Main Streamlit SCADA application
├── resave_model.py          # Model training and serialisation script
├── model.pkl                # Trained Random Forest model
├── requirements.txt         # Pinned Python dependencies
├── .gitignore               # Excludes venv, raw data, cache
└── README.md                # This file
```

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.11+
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# 2. Create and activate virtual environment
python -m venv .venv

# Windows CMD
.venv\Scripts\activate

# Git Bash / Mac / Linux
source .venv/Scripts/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Retrain the model (optional)

If you want to retrain from scratch with your own data:

```bash
python resave_model.py
```

### Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📦 Dependencies

```
streamlit>=1.35.0
pandas
numpy
scikit-learn==1.5.2
matplotlib
seaborn
joblib
```

> `scikit-learn` is pinned to ensure `model.pkl` loads without dtype compatibility errors across environments.

---

## 🔧 Configuration

The decision threshold defaults to **0.30** and is adjustable via the sidebar slider at runtime. Lowering the threshold increases recall (fewer missed leaks) at the cost of more false alarms — the right trade-off for safety-critical pipeline monitoring.

The dark theme is defined in `.streamlit/config.toml` and is automatically applied on both local and cloud deployments.

---

## ☁️ Deployment

The app is deployed on **Streamlit Community Cloud**. Any push to the `main` branch triggers an automatic redeploy — no manual steps required.

To deploy your own instance:

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo, set branch to `main` and main file to `app.py`
5. Click **Deploy**

---

## 🛣️ Roadmap

- [ ] Connect to live OPC-UA / MQTT sensor feed
- [ ] Add LSTM-based temporal anomaly detection
- [ ] Multi-pipeline segment support
- [ ] Email / SMS alerting on alarm trigger
- [ ] Model retraining pipeline with MLflow tracking

---

## 👤 Author

**Your Name**
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- LinkedIn: [your-linkedin](https://linkedin.com/in/your-linkedin)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Sensor anomaly detection patterns inspired by real-world oil & gas SCADA systems
- Built as part of an oil & gas data science training programme
- Deployed infrastructure provided by [Streamlit Community Cloud](https://streamlit.io/cloud)
