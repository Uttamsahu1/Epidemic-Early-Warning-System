import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_features(country_df):
    """Build time-series features for a single country."""
    df = country_df.copy().reset_index(drop=True)
    df["Day"]        = (df["Date"] - df["Date"].min()).dt.days
    df["Lag1"]       = df["Daily_Cases"].shift(1).fillna(0)
    df["Lag7"]       = df["Daily_Cases"].shift(7).fillna(0)
    df["Lag14"]      = df["Daily_Cases"].shift(14).fillna(0)
    df["Roll7"]      = df["Daily_Cases"].rolling(7,  min_periods=1).mean()
    df["Roll14"]     = df["Daily_Cases"].rolling(14, min_periods=1).mean()
    df["DayOfWeek"]  = df["Date"].dt.dayofweek
    df["Month"]      = df["Date"].dt.month
    return df

FEATURES = ["Day","Lag1","Lag7","Lag14","Roll7","Roll14","DayOfWeek","Month"]

def train_model(country_df):
    df = build_features(country_df)
    df = df.dropna(subset=FEATURES + ["Daily_Cases"])
    if len(df) < 30:
        return None, None
    X = df[FEATURES]
    y = df["Daily_Cases"]
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=10.0))
    ])
    model.fit(X, y)
    return model, df

def forecast(country_df, days=30):
    """Forecast next `days` daily cases for a country."""
    model, df = train_model(country_df)
    if model is None:
        return None

    last_date  = df["Date"].max()
    last_day   = df["Day"].max()
    history    = df["Daily_Cases"].values.tolist()
    roll_buf   = history[-14:] if len(history) >= 14 else history

    preds, dates = [], []
    for i in range(1, days + 1):
        future_date = last_date + pd.Timedelta(days=i)
        lag1  = history[-1]  if len(history) >= 1  else 0
        lag7  = history[-7]  if len(history) >= 7  else 0
        lag14 = history[-14] if len(history) >= 14 else 0
        roll7  = np.mean(history[-7:])
        roll14 = np.mean(history[-14:])

        row = pd.DataFrame([{
            "Day":       last_day + i,
            "Lag1":      lag1,
            "Lag7":      lag7,
            "Lag14":     lag14,
            "Roll7":     roll7,
            "Roll14":    roll14,
            "DayOfWeek": future_date.dayofweek,
            "Month":     future_date.month,
        }])
        pred = max(0, model.predict(row)[0])
        preds.append(pred)
        dates.append(future_date)
        history.append(pred)

    return pd.DataFrame({"Date": dates, "Forecast": preds})

def get_alert_level(growth_rate, doubling_time):
    """Return alert metadata based on growth rate."""
    if growth_rate >= 20:
        return {
            "level": "HIGH",
            "color": "#ef4444",
            "bg":    "#1c0a0a",
            "border":"#991b1b",
            "emoji": "🔴",
            "msg":   "Critical outbreak risk. Immediate intervention needed.",
        }
    elif growth_rate >= 5:
        return {
            "level": "MEDIUM",
            "color": "#f59e0b",
            "bg":    "#1c1008",
            "border":"#92400e",
            "emoji": "🟡",
            "msg":   "Elevated transmission detected. Enhanced surveillance required.",
        }
    else:
        return {
            "level": "LOW",
            "color": "#10b981",
            "bg":    "#052e16",
            "border":"#166534",
            "emoji": "🟢",
            "msg":   "Stable — no significant outbreak signal detected.",
        }
