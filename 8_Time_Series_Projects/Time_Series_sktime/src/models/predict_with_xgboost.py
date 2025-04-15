import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import holidays

# Load data
total_sales_df = pd.read_csv("../../data/processed/sales_for_fc.csv")
total_sales_df["order_date"] = pd.to_datetime(total_sales_df["order_date"])
total_sales_df.set_index("order_date", inplace=True)
total_sales_df = total_sales_df.drop(columns=["Unnamed: 0"])

# Handle outliers
Q1 = total_sales_df["sales"].quantile(0.25)
Q3 = total_sales_df["sales"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
total_sales_df["sales"] = total_sales_df["sales"].clip(lower_bound, upper_bound)


# Feature engineering
def prepare_features(df):
    df = df.copy()
    df["lag_1"] = df["sales"].shift(1)
    df["lag_7"] = df["sales"].shift(7)
    df["lag_30"] = df["sales"].shift(30)
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(int)
    df["rolling_mean_7"] = df["sales"].rolling(window=7).mean()
    df["rolling_std_7"] = df["sales"].rolling(window=7).std()
    us_holidays = holidays.US()
    df["is_holiday"] = df.index.map(lambda x: 1 if x in us_holidays else 0)
    df["week_of_year"] = df.index.isocalendar().week
    return df.dropna()


total_sales_df = prepare_features(total_sales_df)

# Train-Test Split
# Last 30 days for testing (matching the Prophet validation horizon)
train_df = total_sales_df.iloc[:-30]
test_df = total_sales_df.iloc[-30:]

# Features (X) and target (y)
features = [
    "lag_1",
    "lag_7",
    "lag_30",
    "day_of_week",
    "month",
    "is_weekend",
    "rolling_mean_7",
    "rolling_std_7",
    "is_holiday",
    "week_of_year",
]
X_train = train_df[features]
y_train = train_df["sales"]
X_test = test_df[features]
y_test = test_df["sales"]

# Model training
# Define XGBoost model and hyperparameter distributions
model = xgb.XGBRegressor(objective="reg:squarederror")
param_dist = {
    "max_depth": [3, 5, 7, 10],
    "n_estimators": [50, 100, 200, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.7, 0.8, 1.0],  # Fraction of samples used per tree
    "colsample_bytree": [0.6, 0.8, 1.0],  # Fraction of features used per tree
    "min_child_weight": [1, 5, 10],  # Minimum sum of instance weight in a child
    "gamma": [
        0,
        0.1,
        0.5,
    ],  # Minimum loss reduction required to make a further partition
    "lambda": [0.1, 1, 10],  # L2 regularization
    "alpha": [0.1, 1, 10],  # L1 regularization
}

# Use TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Perform randomized search
random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=100,  # Number of parameter settings to sample
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    random_state=42,  # For reproducibility
)
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

# Predictions and evaluation
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
smape = 100 * np.mean(2 * np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred)))

print(f"XGBoost MAE: {mae}")
print(f"XGBoost RMSE: {rmse}")
print(f"XGBoost SMAPE: {smape}%")

# Forecasting
future_dates = pd.date_range(
    start=total_sales_df.index[-1] + pd.Timedelta(days=1), periods=30, freq="D"
)
future_df = pd.DataFrame(index=future_dates)

# Initialize with last 30 known values
last_data = total_sales_df.tail(30).copy()

predictions = []
for i in range(30):
    # Prepare features for the next day
    next_day = pd.DataFrame(index=[future_dates[i]])
    next_day["lag_1"] = last_data["sales"].iloc[-1]
    next_day["lag_7"] = last_data["sales"].iloc[-7] if len(last_data) >= 7 else np.nan
    next_day["lag_30"] = (
        last_data["sales"].iloc[-30] if len(last_data) >= 30 else np.nan
    )
    next_day["day_of_week"] = future_dates[i].dayofweek
    next_day["month"] = future_dates[i].month
    next_day["is_weekend"] = int(future_dates[i].dayofweek in [5, 6])
    next_day["rolling_mean_7"] = last_data["sales"].tail(7).mean()
    next_day["rolling_std_7"] = last_data["sales"].tail(7).std()
    next_day["is_holiday"] = int(future_dates[i] in holidays.US())
    next_day["week_of_year"] = future_dates[i].isocalendar().week

    # Predict
    pred = best_model.predict(next_day[features])[0]
    predictions.append(pred)

    # Update last_data with the prediction
    new_row = next_day.copy()
    new_row["sales"] = pred
    last_data = pd.concat([last_data, new_row])

# Results
forecast_df = pd.DataFrame({"y_pred": predictions}, index=future_dates)
print("Forecasted values:\n", forecast_df)

# Plot Predicted vs Actual Values
y_pred_forecast = pd.Series(predictions, index=future_dates)

plt.figure(figsize=(12, 6))

# Plot historical data (last 90 days for context)
plt.plot(
    total_sales_df["sales"][-90:], label="Historical Sales", color="black", alpha=0.5
)

# Plot actual test values
plt.plot(y_test.index, y_test, label="Actual (Test)", color="blue", linewidth=2)

# Plot predicted test values
plt.plot(
    y_test.index,
    y_pred,
    label="Predicted (Test)",
    color="red",
    linestyle="--",
    linewidth=2,
)

# Plot forecasted values beyond dataset
plt.plot(
    y_pred_forecast.index,
    y_pred_forecast,
    label="Forecasted",
    color="green",
    linestyle="--",
    linewidth=2,
)

# Customize plot
plt.title(
    f"XGBoost: Predicted vs Actual Sales (MAE: {mae:.2f}, RMSE: {rmse:.2f}, SMAPE: {smape:.2f}%)"
)
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
