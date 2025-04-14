import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Ensure floating-point numbers are displayed with two decimal places and a width of 20 characters for alignment
pd.options.display.float_format = lambda x: f"{x:20.2f}"
# Show all columns on output
pd.set_option("display.max_columns", 999)

df = pd.read_csv(
    "../../data/interim/online_retail_2009_2010_clean.csv", parse_dates=["InvoiceDate"]
)

df["TotalSales"] = df["Price"] * df["Quantity"]

# Define a reference date (e.g., one day after the last transaction)
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

# Group by Customer ID to compute RFM metrics
rfm = df.groupby(by="CustomerID", as_index=False).agg(
    {
        "InvoiceDate": "max",  # Get Last Purchase Date
        "Invoice": "nunique",  # Frequency: Number of unique purchases
        "TotalSales": "sum",  # Monetary: Total spending
    }
)

# Compute Recency in days
rfm["Recency"] = (reference_date - rfm["InvoiceDate"]).dt.days

# Rename columns
rfm.rename(
    columns={
        "InvoiceDate": "LastInvoiceDate",  # Keep last purchase date
        "Invoice": "Frequency",
        "TotalSales": "Monetary",
    },
    inplace=True,
)
rfm.head()

# Save RFM metrics to CSV
rfm.to_csv("../../data/processed/rfm_metrics.csv", index=False)

# ----------------------------------------------
# Scatter Plot: Recency vs Monetary
# ----------------------------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(data=rfm, x="Recency", y="Monetary", s=50, color="navy")
plt.title("Recency vs Monetary Value")
plt.xlabel("Recency (Days since last purchase)")
plt.ylabel("Monetary (Total Spend)")
plt.show()

# ----------------------------------------------
# Scatter Plot: Frequency vs Monetary
# ----------------------------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(data=rfm, x="Frequency", y="Monetary", s=50, color="darkgreen")
plt.title("Frequency vs Monetary Value")
plt.xlabel("Frequency (Number of Transactions)")
plt.ylabel("Monetary (Total Spend)")
plt.show()

# ----------------------------------------------
# Bar Plot: Top 10 Customers by Monetary Value
# ----------------------------------------------
top10 = rfm.sort_values("Monetary", ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(
    x=top10.index.astype(str),
    y=top10["Monetary"],
    palette="Blues_d",
    hue=top10["Monetary"],
)
plt.title("Top 10 Customers by Monetary Value")
plt.xlabel("CustomerID")
plt.ylabel("Monetary (Total Spend)")
plt.xticks(rotation=45)
plt.show()


# ----------------------------------------------
# Box Plot: Distribution of Monetary Value
# ----------------------------------------------
def plot_rfm_boxplots(rfm_df):
    """
    Plots boxplots for Recency, Frequency, and Monetary metrics in an RFM dataset.

    Parameters:
    rfm_df (pd.DataFrame): DataFrame containing RFM metrics (Recency, Frequency, and Monetary).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Define RFM metrics and corresponding colors
    metrics = ["Recency", "Frequency", "Monetary"]
    colors = ["skyblue", "lightgreen", "salmon"]
    titles = ["Recency Distribution", "Frequency Distribution", "Monetary Distribution"]
    y_labels = ["Recency (Days)", "Frequency (Transactions)", "Monetary (Total Spend)"]

    # Create boxplots for each metric
    for i, metric in enumerate(metrics):
        sns.boxplot(data=rfm_df, y=metric, ax=axes[i], color=colors[i])
        axes[i].set_title(titles[i])
        axes[i].set_ylabel(y_labels[i])

    plt.tight_layout()
    plt.suptitle("Boxplots of RFM Metrics", fontsize=16, y=1.05)
    plt.show()


plot_rfm_boxplots(rfm)


# Function to remove outliers using IQR
def remove_outliers(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[
            (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        ]
    return df_clean


# Remove outliers from RFM data
rfm_clean = remove_outliers(rfm, ["Frequency", "Monetary"])
rfm_clean.describe()

# Save data without outliers
rfm_clean.to_csv(
    "../../data/processed/online_retail_2009_2010_without_outliers.csv", index=False
)

# Plot without outliers
plot_rfm_boxplots(rfm_clean)

# Plot data to verify the scale
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d")
scatter = ax.scatter(
    rfm_clean["Monetary"], rfm_clean["Frequency"], rfm_clean["Recency"]
)
ax.set_xlabel("Monetary Value")
ax.set_ylabel("Frequency")
ax.set_zlabel("Recency")
ax.set_title("3D Scatter Plot of Customer Data")
plt.show()

# Scale data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(
    rfm_clean[
        [
            "Frequency",
            "Monetary",
            "Recency",
        ]
    ]
)

# Convert scaled data back to DataFrame
rfm_scaled_df = pd.DataFrame(
    rfm_scaled, columns=["Frequency", "Monetary", "Recency"], index=rfm_clean.index
)

# Save scaled data
rfm_scaled_df.to_csv(
    "../../data/processed/online_retail_2009_2010_scaled.csv", index=True
)
