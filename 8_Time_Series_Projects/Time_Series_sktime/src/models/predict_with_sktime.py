import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sktime.forecasting.base import ForecastingHorizon, BaseForecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.theta import ThetaForecaster
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.utils.plotting import plot_series
from sklearn.metrics import mean_absolute_error, r2_score
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.fbprophet import Prophet

sys.path.append("..")
import utility.plot_settings  # Custom styling module

import warnings

warnings.simplefilter("ignore")

total_sales_df = pd.read_csv("../../data/processed/sales_for_fc.csv")
total_sales_df["order_date"] = pd.to_datetime(total_sales_df["order_date"])
total_sales_df.set_index("order_date", inplace=True)
# Drop unnecessary column
total_sales_df = total_sales_df.drop(columns=["Unnamed: 0"])

# Handle outliers
Q1 = total_sales_df["sales"].quantile(0.25)
Q3 = total_sales_df["sales"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
total_sales_df["sales"] = total_sales_df["sales"].clip(lower_bound, upper_bound)

total_sales_df.index

#! Predict on the available data first to get an idea of the overall performance of the model
# Convert frequency to day
forecast_df = total_sales_df.resample("D").sum()

# Convert index to datetime
forecast_df.index = pd.to_datetime(forecast_df.index)

# Split dataset: 80 % train, 20 % test
y_train, y_test = temporal_train_test_split(forecast_df["sales"], test_size=0.2)

# Initialize AutoARIMA model with weekly seasonality
model = AutoARIMA(sp=7)  # 'sp=7' assumes weekly seasonality
model.fit(y_train)

# In-sample predictions
y_pred_train = pd.Series(
    model._forecaster.model_.predict_in_sample(), index=y_train.index
)

# Out-of-sample predictions
fh_test = ForecastingHorizon(y_test.index, is_relative=False)
y_pred_test = model.predict(fh_test)

# Combine predictions and actuals
y_pred_all = pd.concat([y_pred_train, y_pred_test])
y_all = pd.concat([y_train, y_test])

# Compute overall MAE
mae_all = mean_absolute_error(y_all, y_pred_all)
print(f"Overall MAE: {mae_all:.2f}")

# Compute overall MAPE, excluding zeros
non_zero_idx = y_all != 0
if non_zero_idx.sum() > 0:
    mape_all = mean_absolute_percentage_error(
        y_all[non_zero_idx], y_pred_all[non_zero_idx]
    )
    print(f"Overall MAPE (excluding zeros): {mape_all:.2%}")
else:
    print("All values are zero, MAPE undefined.")

# In-sample and out-of-sample metrics
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
print(f"In-sample MAE: {mae_train:.2f}")
print(f"Out-of-sample MAE: {mae_test:.2f}")

non_zero_idx_train = y_train != 0
if non_zero_idx_train.sum() > 0:
    mape_train = mean_absolute_percentage_error(
        y_train[non_zero_idx_train], y_pred_train[non_zero_idx_train]
    )
    print(f"In-sample MAPE (excluding zeros): {mape_train:.2%}")

non_zero_idx_test = y_test != 0
if non_zero_idx_test.sum() > 0:
    mape_test = mean_absolute_percentage_error(
        y_test[non_zero_idx_test], y_pred_test[non_zero_idx_test]
    )
    print(f"Out-of-sample MAPE (excluding zeros): {mape_test:.2%}")


#! Function to streamlines forecasting process
def sktime_forecast(
    dataset: pd.DataFrame,
    horizon: int,
    forecaster: BaseForecaster,
    validation: bool = False,
    confidence: float = 0.9,
    frequency: str = "D",
    plot: bool = True,
    n_hist: int = None,
    metrics: list = ["mae"],
    interpolation_method: str = "time",
):
    """
    Loop over a time series DataFrame, train an sktime forecaster, and optionally visualize results.

    Args:
        dataset (pd.DataFrame): Time series data with a datetime index.
        horizon (int): Number of periods to forecast.
        forecaster (BaseForecaster): Configured sktime forecaster instance.
        validation (bool, optional): If True, split data for validation; else forecast beyond data. Defaults to False.
        confidence (float, optional): Confidence level for prediction intervals (0-1). Defaults to 0.9.
        frequency (str, optional): Resampling frequency (e.g., 'D' for daily). Defaults to "D".
        plot (bool, optional): If True, plot the results. Defaults to True.
        n_hist (int, optional): Number of historical periods to plot. Defaults to horizon*3.
        metrics (list, optional): Metrics to compute during validation (e.g., ["mae", "rmse"]). Defaults to ["mae"].
        interpolation_method (str, optional): Method for interpolating missing values. Defaults to "time".

    Returns:
        dict: Results for each column with keys 'y_pred', 'ci', and metrics (if validation=True).
    """
    # Input Validation
    if not isinstance(dataset.index, pd.DatetimeIndex):
        raise ValueError("Dataset must have a datetime index.")

    # Adjust frequency
    forecast_df = dataset.resample(rule=frequency).sum()

    # Handle Missing Values
    forecast_df = forecast_df.interpolate(method=interpolation_method)
    forecast_df = forecast_df.fillna(method="bfill").fillna(method="ffill")

    if forecast_df.isna().any().any():
        warnings.warn("NaNs remain after interpolation; consider reviewing data.")

    results = {}

    for col in dataset.columns:
        df = forecast_df[col].dropna()

        if validation:
            # Simple train-test split for validation
            y_train = df[:-horizon]
            y_test = df.tail(horizon)

            try:
                forecaster.fit(y_train)
                fh = ForecastingHorizon(y_test.index, is_relative=False)
                y_pred = forecaster.predict(fh)

                # Confidence Interval Handling
                ci = (
                    forecaster.predict_interval(fh, coverage=confidence)
                    if hasattr(forecaster, "predict_interval")
                    else None
                )

                # Compute Multiple Metrics
                metric_results = {}
                if "mae" in metrics:
                    metric_results["mae"] = mean_absolute_error(y_test, y_pred)
                if "rmse" in metrics:
                    metric_results["rmse"] = np.sqrt(np.mean((y_test - y_pred) ** 2))
                if "mape" in metrics:
                    non_zero_idx = y_test != 0
                    if non_zero_idx.sum() > 0:
                        metric_results["mape"] = (
                            np.mean(
                                np.abs(
                                    (y_test[non_zero_idx] - y_pred[non_zero_idx])
                                    / y_test[non_zero_idx]
                                )
                            )
                            * 100
                        )
                    else:
                        metric_results["mape"] = np.nan

                results[col] = {
                    "y_pred": y_pred,
                    "ci": ci,
                    "y_test": y_test,
                    **metric_results,
                }

            except Exception as e:
                raise RuntimeError(f"Validation failed for column {col}: {str(e)}")

        else:
            # Forecast beyond the dataset
            try:
                forecaster.fit(df)
                last_date = df.index.max()
                fh = ForecastingHorizon(
                    pd.date_range(last_date, periods=horizon + 1, freq=frequency)[1:],
                    is_relative=False,
                )
                y_pred = forecaster.predict(fh)
                ci = (
                    forecaster.predict_interval(fh, coverage=confidence)
                    if hasattr(forecaster, "predict_interval")
                    else None
                )
                results[col] = {"y_pred": y_pred, "ci": ci}

            except Exception as e:
                raise RuntimeError(f"Forecasting failed for column {col}: {str(e)}")

        # Plotting with Forecaster Name
        if plot:
            forecaster_name = forecaster.__class__.__name__
            n_hist = n_hist or horizon * 3
            if validation:
                plot_series(
                    y_train[-n_hist:], y_test, y_pred, labels=["Train", "Test", "Pred"]
                )
            else:
                plot_series(df[-n_hist:], y_pred, labels=["Historical", "Forecast"])
            if ci is not None:
                plt.gca().fill_between(
                    y_pred.index, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.1
                )
            plt.grid(False)
            if validation and "mae" in metrics:
                plt.title(
                    f"{forecaster_name} - {col} - {horizon}-step forecast (MAE: {results[col]['mae']:.2f})"
                )
            else:
                plt.title(f"{forecaster_name} - {col} - {horizon}-step forecast")
            plt.show()

    return results


if __name__ == "__main__":
    # Configure forecaster
    forecaster = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    # forecaster = AutoARIMA(sp=7, suppress_warnings=True)

    # Validation mode: forecast last 30 days and evaluate
    # Splits the last 30 days for testing, computes MAE, RMSE, and MAPE, and plots the results
    results_val = sktime_forecast(
        total_sales_df,
        horizon=30,
        forecaster=forecaster,
        validation=True,
        confidence=0.95,
        frequency="D",
        plot=True,
        n_hist=90,
        metrics=["mae", "rmse", "mape"],
    )

    # Forecasting mode: predict next 30 days beyond the dataset
    # Predicts 30 days beyond the dataset and plots the forecast with confidence intervals
    results_fc = sktime_forecast(
        total_sales_df,
        horizon=30,
        forecaster=forecaster,
        validation=False,
        confidence=0.95,
        frequency="D",
        plot=True,
    )

    # Access results
    print(
        "Validation MAE:",
        results_val["sales"]["mae"],
        "\n" "Validation RMSE:",
        results_val["sales"]["rmse"],
        "\n" "Validation MAPE:",
        results_val["sales"]["mape"],
        "\n",
    )
    print("Forecasted values:\n", results_fc["sales"]["y_pred"])
