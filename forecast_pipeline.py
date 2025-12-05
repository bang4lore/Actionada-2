import pandas as pd
import numpy as np
import joblib
import logging
from datetime import timedelta

REQUIRED_COLUMNS = [...]

class CashForecastPipeline:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger()

    def validate_input(self, df):
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            self.logger.error(f"Missing columns: {missing}")
            raise ValueError(f"Missing columns: {missing}")
        if df.isna().any().any():
            self.logger.warning("Input dataframe contains NaNs")

    def preprocess(self, df):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["atm_id", "date"])
        return df

    def make_features(self, df):
        df = df.copy()
        df["net_flow"] = df["deposits"] - df["withdrawals"]
        for lag in [1, 2, 7]:
            df[f"net_flow_lag_{lag}"] = df.groupby("atm_id")["net_flow"].shift(lag)
        df["net_flow_roll_7"] = df.groupby("atm_id")["net_flow"].rolling(window=7, min_periods=3).mean().reset_index(level=0, drop=True)
        df["holiday_tomorrow"] = df.groupby("atm_id")["is_holiday"].shift(-1)
        df = df.dropna()
        return df

    def forecast_14_days(self, df_raw):
        self.validate_input(df_raw)
        df = self.preprocess(df_raw)
        forecasts = []
        for atm_id, group in df.groupby("atm_id"):
            group = group.sort_values("date")
            for day in range(14):
                features = self.make_features(group)
                X = features.drop(columns=["date", "atm_id"])
                pred = self.model.predict(X.iloc[[-1]])[0]
                forecast_date = group["date"].max() + timedelta(days=day + 1)
                forecasts.append({"atm_id": atm_id, "date": forecast_date, "predicted_balance": pred})
        return pd.DataFrame(forecasts)

