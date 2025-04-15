import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta, date
import holidays
import pickle
import sys
import os
from sklearn.metrics import mean_absolute_error

from src.models.prophet.train_model import mape, smape

# Load data
total_sales_df = pd.read_csv("../../data/processed/sales_for_fc.csv")
total_sales_df["order_date"] = pd.to_datetime(total_sales_df["order_date"])
total_sales_df.set_index("order_date", inplace=True)
total_sales_df = total_sales_df.drop(columns=["Unnamed: 0"])

# Hardcode the path to src/models/prophet
prophet_dir = (
    "c:/Users/sebas/Documents/Data_Science/14-Dave_Ebbelaar/Time_Series_sktime/models"
)
sys.path.insert(0, prophet_dir)

# Define the file path using the hardcoded directory
file_path = os.path.join(prophet_dir, "prophet_params.pkl")

# Load the dictionary from the Pickle file
try:
    with open(file_path, "rb") as f:
        dicts_loaded = pickle.load(f)
    print("Loaded dicts:", dicts_loaded)
except FileNotFoundError:
    print(
        f"Error: {file_path} not found. Ensure train_prophet_model.py has been run to create the file."
    )
    dicts_loaded = {}
except Exception as e:
    print(f"Error loading Pickle file: {e}")
    dicts_loaded = {}

# PROPHET MODEL

prediction_days = 30
forecast_start_date = max(total_sales_df.index) - timedelta(prediction_days)

us_holidays = holidays.US(years=[2015, 2019])
holiday_df = pd.DataFrame(
    {
        "holiday": "US_Holidays",
        "ds": pd.to_datetime(list(us_holidays.keys())),
        "lower_window": -1,  # Day before
        "upper_window": 1,  # Day after
    }
)

forecasted_dfs = []

for feature in total_sales_df.columns:
    # Formatting
    df_copy = total_sales_df[feature].copy().reset_index()
    df_copy.columns = ["ds", "y"]
    df_copy[["y"]] = df_copy[["y"]].apply(pd.to_numeric)
    df_copy["ds"] = pd.to_datetime(df_copy["ds"])

    df_copy_ = df_copy[df_copy["ds"] < forecast_start_date]

    # Access the nested parameters using dicts_loaded[feature][0]
    params_dict = dicts_loaded[feature][0]  # Access the nested dictionary

    # Model
    model = Prophet(
        changepoint_prior_scale=params_dict["changepoint_prior_scale"],
        seasonality_prior_scale=params_dict["seasonality_prior_scale"],
        seasonality_mode=params_dict["seasonality_mode"],
        changepoint_range=params_dict.get("changepoint_range", 0.8),
        holidays=holiday_df,
    )

    model.fit(df_copy_)

    future = model.make_future_dataframe(periods=prediction_days)
    fcst_prophet_train = model.predict(future)

    filter = fcst_prophet_train["ds"] >= forecast_start_date
    predicted_df = fcst_prophet_train[filter][["ds", "yhat"]]
    predicted_df = predicted_df.merge(df_copy)

    print(
        feature,
        "MAPE:",
        mape(predicted_df["y"], predicted_df["yhat"]),
        "SMAPE:",
        smape(predicted_df["y"], predicted_df["yhat"]),
        "MAE:",
        mean_absolute_error(predicted_df["y"], predicted_df["yhat"]),
    )
