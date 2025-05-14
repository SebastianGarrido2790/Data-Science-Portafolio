# Insights on the Forecast Plot for "Sales"

The forecast plot titled **"Forecast for sales"** visualizes both historical sales data and the model's predictions over time.

## Historical Trend
- **Time Period:** April 2015 to October 2019.
- **Growth:** Sales increase from approximately 0–5,000 units in 2015 to peaks exceeding 20,000 units by 2019, indicating consistent long-term growth.
- **Data Patterns:** Black dots representing historical data show periodic spikes and dips, suggesting seasonal patterns or significant sales events.

## Forecasted Trend
- **Trend Continuation:** The blue forecast line extends beyond mid-2019 with a smoother trajectory, reflecting the long-term growth trend.
- **Model Influence:** This smoothness is influenced by the tuned parameter `changepoint_prior_scale=0.3`, which allows moderate flexibility in trend changes.

## Confidence Intervals
- **Visual Representation:** The shaded blue area around the forecast line represents the 95% confidence interval (`interval_width=0.95`).
- **Uncertainty:** The interval widens further into the future, ranging from approximately -5,000 to 25,000. The inclusion of negative values is unexpected for sales data, which should be non-negative.

## Potential Issues
### Negative Values
- **Observation:** The y-axis extending to -5,000 suggests that the model predicts negative sales.
- **Possible Causes:**
  - Data preprocessing errors (e.g., negative adjustments for returns).
  - The use of an additive seasonality mode (`seasonality_mode="additive"`) allowing negative predictions when seasonal effects significantly subtract from the trend.
  - Lack of constraints to enforce non-negative outputs.

### Outliers
- **Observation:** Isolated points above 20,000 (e.g., around 2019) may represent significant sales events or anomalies.
- **Interpretation:** Although the widened confidence intervals account for variability, they may not fully explain these outliers.

## Seasonality
- **General Insight:** Fluctuations in historical data hint at seasonal patterns (e.g., weekly, yearly, or holiday-related).
- **Model Components:** Prophet captures these patterns through seasonality and holiday components, using parameters like `seasonality_prior_scale=15.25` and `holidays_prior_scale=7.5`.

# Insights on the Components Plots for "Sales"

The components plots break down the forecast into the following aspects: trend, holidays, weekly seasonality, and yearly seasonality.

## 1. Trend Component
- **Observation:** 
  - The trend starts at around 1,400 units in mid-2015, dips to 1,200 units by mid-2016, and then rises steadily to nearly 2,000 units by April 2019.
  - The forecast continues this upward movement.
- **Model Impact:** The smooth trend is a result of `changepoint_prior_scale=0.3`, balancing flexibility with stability.
- **Insight:** Prophet effectively captures the long-term growth trend, with the dip in 2016 potentially indicating a structural change. The `changepoint_range=0.6125` ensures that changepoints are considered across most of the historical data.

## 2. Holidays Component
- **Observation:**
  - The plot shows the effect of U.S. holidays on sales, with impacts ranging from -40% to +60%.
  - Significant effects are visible around late 2015, mid-2016, and late 2018, likely corresponding to holidays such as Thanksgiving, Christmas, or New Year’s Day.
- **Insight:**
  - Holidays have a notable, variable impact on sales.
  - The effects are modeled with `holidays_prior_scale=7.5` and a window of -1 to +1 days around each holiday.
  - Refining holiday definitions (e.g., adding specific events or adjusting weights) could improve forecast accuracy.

## 3. Weekly Seasonality Component
- **Observation:**
  - Weekly effects fluctuate between -60% and +40%.
  - Tuesdays show the highest positive effect (+20%), Thursdays the largest negative effect (-60%), and Saturdays a strong positive effect (+40%), with Sundays near neutral.
- **Insight:**
  - Weekly seasonality is a significant driver of sales.
  - The pronounced mid-week peaks and weekend boosts are effectively captured by the `seasonality_prior_scale=15.25` parameter.

## 4. Yearly Seasonality Component
- **Observation:**
  - The yearly pattern oscillates between -100% and +100%.
  - Peaks occur in March (+75%), July (+50%), and November (+75%), while troughs are seen in May (-50%) and September (-25%).
- **Insight:**
  - Strong annual cycles in sales behavior are evident.
  - This component likely reflects seasonal events or consumer trends, and the high `seasonality_prior_scale=15.25` ensures these fluctuations are captured.

# General Insights for Time Series Forecasting with Prophet

## Model Fit
- **Parameters:**
  - `changepoint_prior_scale=0.3`
  - `seasonality_prior_scale=15.25`
  - `holidays_prior_scale=7.5`
  - `seasonality_mode="additive"`
  - `changepoint_range=0.6125`
- **Performance:** The SMAPE of 0.913% indicates excellent accuracy, though the unusually low value might suggest a reporting error (typical SMAPE values are around 5–20%).

## Key Drivers
- The components plots highlight that:
  - Weekly and yearly seasonality.
  - Holiday effects.
  - Overall trend.
- These factors collectively explain short- and medium-term variations in sales.

## Strengths
- **Growth and Seasonality:** Prophet excels at capturing long-term growth and seasonal patterns.
- **Holiday Effects:** Including U.S. holidays helps model real-world events that impact sales.

## Challenges
### Negative Predictions
- **Issue:** Negative sales predictions are critical, as they misalign with realistic expectations.
  
### Uncertainty
- **Observation:** Widening confidence intervals suggest increasing uncertainty for longer forecasts.

### Holiday Variability
- **Insight:** Inconsistent holiday effects indicate potential for model refinement.

# Recommendations for Improvement

## Address Negative Values
- **Investigate:** Examine the source of negative values (e.g., returns or data preprocessing errors).
- **Adjust:** Consider switching to a multiplicative seasonality mode (`seasonality_mode="multiplicative"`) if sales scales vary significantly.
- **Constrain:** Apply a log transformation or constraints (e.g., `yhat = max(0, yhat)`) to ensure non-negative predictions.

## Refine Holiday Modeling
- **Tuning:** Adjust `holidays_prior_scale` or add custom holidays specific to your sales context.
- **Validation:** Compare holiday effects with actual sales spikes to ensure accuracy.

## Enhance Trend Flexibility
- **Experiment:** Try a higher `changepoint_prior_scale` (e.g., 0.5) if the model misses sudden shifts.
- **Focus:** Adjust `changepoint_range` to concentrate on recent data in cases of structural breaks.

## Reduce Uncertainty
- **External Factors:** Incorporate external regressors (e.g., marketing campaigns, economic indicators) to improve precision.
- **Forecast Horizon:** Consider narrowing the forecast horizon or increasing `interval_width` beyond 0.95 for more conservative bounds.

## Validate Accuracy
- **Recheck Metrics:** Confirm the SMAPE value (0.913% may be unusually low) against other metrics such as MAPE and MAE.
- **Cross-Validation:** Perform cross-validation with a longer forecast horizon if necessary.
