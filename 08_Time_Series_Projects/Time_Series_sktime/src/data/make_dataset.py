import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv("../../data/raw/train.csv")

df.info()


def missing_data(input_data):
    """
    This function returns dataframe with information about the percentage of nulls in each column and the column data type.

    input: pandas df
    output: pandas df

    """

    total = input_data.isnull().sum()
    percent = input_data.isnull().sum() / input_data.isnull().count() * 100
    table = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
    types = []
    for col in input_data.columns:
        dtype = str(input_data[col].dtype)
        types.append(dtype)
    table["Types"] = types
    return pd.DataFrame(table)


missing_data(df)

df.columns = df.columns.str.replace(" ", "_").str.lower()

# Drop irrelevant column with missing values
df = df.drop(columns=["postal_code"])

# Convert dates to datetime format
df["order_date"] = pd.to_datetime(df["order_date"], format="%d/%m/%Y")

# Check timestamp data
print(min(df["order_date"]), max(df["order_date"]))

# Category grouping
agg_categories = (
    df.groupby(["order_date", "sub-category"])
    .agg({"sales": "sum"})
    .reset_index()
    .sort_values(["sub-category", "order_date"])
)

total_sales_df = agg_categories.pivot(
    index="order_date", columns="sub-category", values="sales"
)

for column in total_sales_df.columns:
    plt.plot(total_sales_df[column])
    plt.title(column)
    plt.show()

# Need to break out each into it's own dataframe for prediction since each will have different rows affected
prediction_df_list = []

# Cleaning up dataframe using z-score to remove outliers which heavily bias the model
for column in total_sales_df.columns:
    df_clean = total_sales_df[[column]].reset_index()

    z = np.abs(stats.zscore(df_clean[column]))
    outlier_index = np.where(z > 2)[
        0
    ]  # 95.44 % of the data points lie between 3 standard deviations (Gaussian Distribution)
    print(
        "Dropping "
        + str(len(outlier_index))
        + " rows for following category: "
        + column
    )
    df_clean.drop(index=outlier_index, inplace=True)
    df_clean.set_index("order_date", inplace=True)
    prediction_df_list.append(df_clean)

# Save to study dataset by thirds
total_sales_df.to_csv("../../data/interim/total_sales.csv", index=True)

# Aggregate sales by Order Date
df_ts = df.groupby("order_date")["sales"].sum().reset_index()

df_ts = df_ts.sort_values(by="order_date").reset_index(drop=True)

df_ts.info()

df_ts.to_csv("../../data/interim/sales_ts.csv", index=False)
