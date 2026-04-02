# 🦠 EpiWatch — AI Epidemic Early Warning System
🚀 Live Demo: https://your-app-name.streamlit.app
> An AI-powered web application that detects global outbreak risk and forecasts disease spread using Machine Learning, built with Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview

**EpiWatch** is a machine learning web application that monitors real-time epidemic trends across 195+ countries and predicts disease spread before outbreaks escalate. It classifies countries into three alert levels — 🟢 Low Risk, 🟡 Medium Risk, and 🔴 High Risk — enabling public health authorities to respond early.

Built for the **CodeCure AI Hackathon 2026 — Track C (Epidemic Spread Prediction)**.

---

## 🖥️ App Preview

| Overview | Country Analysis | Forecast | Risk Alerts |
|---|---|---|---|
| Global KPI dashboard with risk distribution charts | Per-country case trend, daily cases & deaths | 30-day ML forecast with confidence bands | 🔴 High / 🟡 Medium alert tables with country search |

---

## ✨ Features

- **Overview Dashboard** — Total confirmed cases, country-level risk distribution, top 10 countries by case count and growth rate
- **Country Analysis** — Deep-dive trend charts for any country — cumulative cases, daily new cases, and death trajectory
- **30-Day Forecast** — Ridge Regression ML model with lag features, rolling averages, and confidence bands
- **Risk Alert System** — Real-time 🔴 HIGH / 🟡 MEDIUM / 🟢 LOW classification based on 7-day growth rate
- **Professional Dark UI** — Custom CSS injection for a polished, production-grade Streamlit interface

---

## 🗂️ Project Structure

```
epidemic-early-warning/
│
├── app.py              ← Main Streamlit web application
├── data.py             ← Data loading, cleaning & feature engineering
├── model.py            ← Ridge Regression forecaster + alert logic
├── requirements.txt    ← Python dependencies
└── README.md
```

---

## 📊 Dataset

**Primary Dataset:** [Johns Hopkins CSSE COVID-19 Time Series](https://github.com/CSSEGISandData/COVID-19)

Loaded directly via URL — no manual download required.

| Property | Value |
|---|---|
| Countries Tracked | 195+ |
| Date Range | Jan 2020 → Mar 2023 |
| Data Format | Daily time-series CSV |
| Source | Johns Hopkins CSSE |

**Features engineered:**

| Feature | Description |
|---|---|
| `Daily_Cases` | New cases per day (diff of cumulative) |
| `Daily_Deaths` | New deaths per day |
| `Rolling_7day` | 7-day rolling average (smoothed trend) |
| `Growth_Rate` | % change in rolling average over 7 days |
| `Doubling_Time` | Days until cases double at current rate |
| `Risk_Level` | 🔴 HIGH / 🟡 MEDIUM / 🟢 LOW classification |

---

## 🤖 ML Model

| Property | Value |
|---|---|
| Algorithm | Ridge Regression |
| Framework | scikit-learn Pipeline + StandardScaler |
| Forecast Horizon | Up to 60 days |
| Confidence Band | ±30% around prediction |

**Lag features used for forecasting:**

```python
Features = ["Day", "Lag1", "Lag7", "Lag14", "Roll7", "Roll14", "DayOfWeek", "Month"]
```

---

## 🚨 Risk Level Classification

| Level | Label | Condition | Description |
|---|---|---|---|
| 🟢 | **Low Risk** | Growth Rate < 5% | Stable — no significant outbreak signal |
| 🟡 | **Medium Risk** | Growth Rate 5–20% | Elevated transmission — enhanced surveillance needed |
| 🔴 | **High Risk** | Growth Rate ≥ 20% | Critical outbreak — immediate intervention required |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/epidemic-early-warning.git
cd epidemic-early-warning
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

> ✅ No dataset download needed — data loads automatically from GitHub.

---

## 📦 Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
requests>=2.31.0
```

---

## 🏗️ Built With

- **[Streamlit](https://streamlit.io)** — Web app framework
- **[scikit-learn](https://scikit-learn.org)** — Ridge Regression ML model
- **[Pandas](https://pandas.pydata.org)** — Data processing & feature engineering
- **[NumPy](https://numpy.org)** — Numerical computing
- **[Matplotlib](https://matplotlib.org)** — Charts and visualizations
- **[Johns Hopkins CSSE](https://github.com/CSSEGISandData/COVID-19)** — COVID-19 dataset

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

Built with ❤️ for **CodeCure AI Hackathon 2026 — Track C**

> *"An outbreak detected a week early can save thousands of lives. EpiWatch monitors every country, every day — so no signal goes unnoticed."*
