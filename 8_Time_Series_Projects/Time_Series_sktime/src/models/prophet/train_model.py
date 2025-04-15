import numpy as np
import pandas as pd
import pickle
import os

import math
import itertools
from scipy import stats
import time

from dateutil import parser
from datetime import datetime, timedelta, date

from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
from prophet.diagnostics import cross_validation

import warnings

warnings.simplefilter("ignore")


def mape(actual, pred):
    """
    Mean Absolute Percentage Error (MAPE) Function

    input: list/series for actual values and predicted values
    output: mape value
    """
    actual, pred = np.array(actual), np.array(pred)
    mask = actual != 0  # Exclude zeros
    return np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100


def smape(actual, pred):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE) Function

    input: list/series for actual values and predicted values
    output: smape value
    """
    actual, pred = np.array(actual), np.array(pred)
    return 100 * np.mean(2 * np.abs(pred - actual) / (np.abs(actual) + np.abs(pred)))


# Only run the tuning and saving when the script is run directly
if __name__ == "__main__":
    # Load data
    total_sales_df = pd.read_csv("../../data/processed/sales_for_fc.csv")
    total_sales_df["order_date"] = pd.to_datetime(total_sales_df["order_date"])
    total_sales_df.set_index("order_date", inplace=True)
    total_sales_df = total_sales_df.drop(columns=["Unnamed: 0"])

    # Create time series parameters
    changepoint_prior_scale_range = np.linspace(0.1, 0.5, num=5).tolist()
    seasonality_prior_scale_range = np.linspace(1.0, 20, num=5).tolist()
    holidays_prior_scale_range = np.linspace(0.01, 10, num=5).tolist()
    seasonality_mode_options = ["additive", "multiplicative"]
    changepoint_range = list(np.linspace(0.5, 0.95, num=5))

    start_time = time.time()
    dicts = {}

    # Hyperparameters
    for feature in total_sales_df.columns:
        category_df = total_sales_df[feature].copy().reset_index()
        category_df.columns = ["ds", "y"]
        category_df[["y"]] = category_df[["y"]].apply(pd.to_numeric)
        category_df["ds"] = pd.to_datetime(category_df["ds"])

        param_grid = {
            "changepoint_prior_scale": changepoint_prior_scale_range,
            "seasonality_prior_scale": seasonality_prior_scale_range,
            "holidays_prior_scale": holidays_prior_scale_range,
            "seasonality_mode": seasonality_mode_options,
            "changepoint_range": changepoint_range,
        }

        all_params = [
            dict(zip(param_grid.keys(), v))
            for v in itertools.product(*param_grid.values())
        ]
        all_params = np.random.choice(all_params, size=10, replace=False)
        smapes = []

        for params in all_params:
            model = Prophet(**params).fit(category_df)
            df_cv = cross_validation(
                model, initial="730 days", period="30 days", horizon="30 days"
            )
            df_p = performance_metrics(df_cv, rolling_window=1)
            print(df_p)
            smapes.append(df_p["smape"].values[0])

        tuning_results = pd.DataFrame(all_params)
        tuning_results["smape"] = smapes
        print(feature)
        print(tuning_results.head())

        params_dict = dict(
            tuning_results.sort_values("smape").reset_index(drop=True).iloc[0]
        )
        params_dict["column"] = feature
        dicts[feature] = params_dict

    print("--- %s seconds ---" % (time.time() - start_time))

    # Hardcode the path to src/models/prophet using raw string
    file_path = r"C:\Users\sebas\Documents\Data_Science\14-Dave_Ebbelaar\Time_Series_sktime\models\prophet_params.pkl"

    # Save dicts to a Pickle file
    with open(file_path, "wb") as f:
        pickle.dump(dicts, f)
