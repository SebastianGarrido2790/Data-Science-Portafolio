import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta, date
import holidays
import pickle
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import seaborn as sns

# Import mape and smape from train_prophet_model (though not used here, kept for consistency)
from src.models.prophet.train_model import mape, smape

# Load data
total_sales_df = pd.read_csv("../../data/processed/sales_for_fc.csv")
total_sales_df["order_date"] = pd.to_datetime(total_sales_df["order_date"])
total_sales_df.set_index("order_date", inplace=True)
total_sales_df = total_sales_df.drop(columns=["Unnamed: 0"])

# Hardcode the path to src/models/prophet
prophet_dir = (
    r"C:\Users\sebas\Documents\Data_Science\14-Dave_Ebbelaar\Time_Series_sktime\models"
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
forecast_start_date = max(total_sales_df.index)
forecasted_dfs = []

us_holidays = holidays.US(years=[2015, 2019])
holiday_df = pd.DataFrame(
    {
        "holiday": "US_Holidays",
        "ds": pd.to_datetime(list(us_holidays.keys())),
        "lower_window": -1,
        "upper_window": 1,
    }
)

# Define the root directory and figures path
figures_dir = r"c:/Users/sebas/Documents/Data_Science/14-Dave_Ebbelaar/Time_Series_sktime/reports/figures"
# figures_dir = os.path.join("reports", "figures")
os.makedirs(figures_dir, exist_ok=True)  # Create directory if it doesn't exist

for feature in total_sales_df.columns[:5]:
    # Formatting
    df_copy = total_sales_df[feature].copy().reset_index()
    df_copy.columns = ["ds", "y"]
    df_copy[["y"]] = df_copy[["y"]].apply(pd.to_numeric)
    df_copy["ds"] = pd.to_datetime(df_copy["ds"])

    df_copy_ = df_copy[df_copy["ds"] < forecast_start_date]

    # Access the parameters, handling both flat and nested structures
    if feature in dicts_loaded and isinstance(dicts_loaded[feature], dict):
        if "0" in dicts_loaded[feature]:  # Nested structure
            params_dict = dicts_loaded[feature]["0"]
        else:  # Flat structure
            params_dict = dicts_loaded[feature]
    else:
        print(
            f"Warning: No valid parameters found for feature {feature}. Using defaults."
        )
        params_dict = {}  # Default to empty if not found

    # Debug print to verify parameters
    print("Params for", feature, ":", params_dict)

    # Model (use .get() to safely handle missing keys with defaults)
    model = Prophet(
        changepoint_prior_scale=params_dict.get("changepoint_prior_scale", 0.5),
        seasonality_prior_scale=params_dict.get("seasonality_prior_scale", 10),
        seasonality_mode=params_dict.get("seasonality_mode", "multiplicative"),
        changepoint_range=params_dict.get("changepoint_range", 0.8),
        interval_width=0.95,  # Widen confidence intervals
        holidays=holiday_df,
    )

    model.fit(df_copy_)

    future = model.make_future_dataframe(periods=prediction_days)
    fcst_prophet_train = model.predict(future)

    # Plot forecast and components
    fig1 = model.plot(fcst_prophet_train)
    plt.title(f"Forecast for {feature}")
    # Save forecast plot with 100 dpi
    forecast_plot_path = os.path.join(figures_dir, f"{feature}_forecast.png")
    plt.savefig(forecast_plot_path, dpi=100, bbox_inches="tight")
    plt.show()
    plt.close()  # Close the figure to free memory

    fig2 = model.plot_components(fcst_prophet_train)
    plt.title(f"Components for {feature}")
    # Save components plot with 100 dpi
    components_plot_path = os.path.join(figures_dir, f"{feature}_components.png")
    plt.savefig(components_plot_path, dpi=100, bbox_inches="tight")
    plt.show()
    plt.close()  # Close the figure to free memory

    forecasted_df = fcst_prophet_train[fcst_prophet_train["ds"] >= forecast_start_date]
    forecasted_dfs.append(forecasted_df)
