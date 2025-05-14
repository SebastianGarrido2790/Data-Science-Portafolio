import pandas as pd

# Ensure floating-point numbers are displayed with two decimal places and a width of 20 characters for alignment
pd.options.display.float_format = lambda x: f"{x:20.2f}"
# Show all columns on output
pd.set_option("display.max_columns", 999)

df = pd.read_csv("../../data/interim/online_retail_2009_2010.csv")

df_clean = df.copy()

df_clean.info()

df_clean["Invoice"] = df_clean["Invoice"].astype(str)
filter_invoice = df_clean["Invoice"].str.match("^\\d{6}$") == True
df_clean = df_clean[filter_invoice]

df_clean["StockCode"] = df_clean["StockCode"].astype(str)
filter_stockcode = (
    (df_clean["StockCode"].str.match("^\\d{5}$") == True)
    | (df_clean["StockCode"].str.match("^\\d{5}[a-zA-Z]+$") == True)
    | (df_clean["StockCode"].str.match("^PADS$") == True)
)

df_clean.dropna(subset=["Customer ID"], inplace=True)

# Check if negative quantity still exists
df_clean.describe()

# Check 0 prices
len(df_clean[df_clean["Price"] == 0])

df_clean = df_clean[df_clean["Price"] > 0.0]

df_clean["Price"].min()

# How much data we dropped?
(len(df) - len(df_clean)) / len(df) * 100

# Convert InvoiceDate from object to datetime
df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])

# Rename columns
df_clean.rename(columns={"Customer ID": "CustomerID"}, inplace=True)

# Save cleaned data
df_clean.to_csv("../../data/interim/online_retail_2009_2010_clean.csv", index=False)
