import pandas as pd
import numpy as np

CONFIRMED_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
DEATHS_URL    = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"

def _melt(df, value_name):
    df = df.drop(columns=["Province/State", "Lat", "Long"], errors="ignore")
    df = df.groupby("Country/Region").sum().reset_index()
    df = df.melt(id_vars="Country/Region", var_name="Date", value_name=value_name)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["Country/Region", "Date"]).reset_index(drop=True)

def load_data():
    confirmed = _melt(pd.read_csv(CONFIRMED_URL), "Confirmed")
    deaths    = _melt(pd.read_csv(DEATHS_URL),    "Deaths")
    df = confirmed.merge(deaths, on=["Country/Region", "Date"])

    df["Daily_Cases"] = (
        df.groupby("Country/Region")["Confirmed"]
        .diff().fillna(0).clip(lower=0)
    )
    df["Daily_Deaths"] = (
        df.groupby("Country/Region")["Deaths"]
        .diff().fillna(0).clip(lower=0)
    )
    df["Rolling_7day"] = (
        df.groupby("Country/Region")["Daily_Cases"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )
    df["Growth_Rate"] = (
        df.groupby("Country/Region")["Rolling_7day"]
        .transform(lambda x: x.pct_change(periods=7) * 100)
    ).fillna(0).replace([np.inf, -np.inf], 0).round(2)

    df["Doubling_Time"] = (
        7 * np.log(2) / np.log1p(df["Growth_Rate"] / 100)
    ).replace([np.inf, -np.inf], np.nan).round(1)

    def assign_risk(gr):
        if gr >= 20:   return "HIGH"
        elif gr >= 5:  return "MEDIUM"
        else:          return "LOW"

    df["Risk_Level"] = df["Growth_Rate"].apply(assign_risk)
    return df

def get_country_list(df):
    return sorted(df["Country/Region"].unique().tolist())

def get_latest_snapshot(df):
    return df.sort_values("Date").groupby("Country/Region").tail(1).reset_index(drop=True)

def get_country_data(df, country):
    return df[df["Country/Region"] == country].reset_index(drop=True)

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"Shape: {df.shape}")
    print(f"Countries: {df['Country/Region'].nunique()}")
    snap = get_latest_snapshot(df)
    print(snap[["Country/Region","Confirmed","Daily_Cases","Growth_Rate","Risk_Level"]].head(10).to_string(index=False))
