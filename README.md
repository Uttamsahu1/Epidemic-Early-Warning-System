# 🦠 AI Epidemic Early Warning System

> Real-time outbreak detection and 30-day case forecasting across 195+ countries

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=flat-square)
![Track C](https://img.shields.io/badge/CodeCure-Track%20C-green?style=flat-square)

---

## 📌 Problem Statement

Predicting the spread of infectious diseases is essential for public health preparedness. This project (Track C — Epidemic Spread Prediction) builds an AI-powered early warning system that monitors global outbreak risk in real time.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🌍 **Overview Dashboard** | Global risk distribution, top 10 countries by cases & growth rate |
| 🔬 **Country Analysis** | Deep-dive trend charts for any country — cases, daily new cases, deaths |
| 🔮 **30-Day Forecast** | Ridge Regression ML model with lag features & confidence bands |
| 🚨 **Risk Alerts** | 🔴 High / 🟡 Medium / 🟢 Low alert system based on 7-day growth rate |

---

## 🗂️ Project Structure

```
epidemic-early-warning/
│
├── app.py              ← Streamlit dashboard (main entry)
├── data.py             ← Data loading, cleaning & feature engineering
├── model.py            ← ML forecasting model + alert logic
├── requirements.txt    ← Python dependencies
└── README.md
```

---

## 📊 Dataset

**Primary:** [Johns Hopkins CSSE COVID-19 Time Series](https://github.com/CSSEGISandData/COVID-19)
- Daily confirmed cases & deaths across 195+ countries
- Loaded directly via URL — no manual download needed

**Features engineered:**
- Daily new cases (diff of cumulative)
- 7-day rolling average (smoothed trend)
- Growth rate (% change over 7 days)
- Doubling time (days at current growth rate)
- Risk level: 🔴 HIGH (≥20%) · 🟡 MEDIUM (5–20%) · 🟢 LOW (<5%)

---

## 🤖 ML Model

**Algorithm:** Ridge Regression via scikit-learn Pipeline

**Features used:**
- `Lag1`, `Lag7`, `Lag14` — previous day case counts
- `Roll7`, `Roll14` — rolling averages
- `Day` — days since pandemic start
- `DayOfWeek`, `Month` — temporal features

---

## ⚙️ Setup & Run

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/epidemic-early-warning.git
cd epidemic-early-warning

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app.py
```

No dataset download needed — data loads automatically from GitHub.

---

## 🏆 Built for CodeCure AI Hackathon — Track C
