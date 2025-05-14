import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")

# Load raw data
df = pd.read_csv("../../data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Step 1: Data Cleaning
# Drop redundant columns (identified in Step 2)
df = df.drop(columns=["Over18", "EmployeeCount", "StandardHours"])

# Step 2: Feature Engineering
# Tenure Ratio
df["TenureRatio"] = df["YearsAtCompany"] / df["TotalWorkingYears"].replace(
    0, 1
)  # Avoid division by 0

# Satisfaction Score
df["SatisfactionScore"] = df[
    ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction"]
].mean(axis=1)

# Age Bins
df["AgeGroup"] = pd.cut(
    df["Age"], bins=[0, 30, 40, 100], labels=["<30", "30-40", ">40"]
)

# Income-to-Level Ratio
df["IncomeToLevelRatio"] = df["MonthlyIncome"] / df["JobLevel"]

# Long Commute Flag
df["LongCommute"] = (df["DistanceFromHome"] > 10).astype(int)

# Step 3: Define features and target
X = df.drop(columns=["Attrition"])
y = df["Attrition"].map({"Yes": 1, "No": 0})  # Encode target

# Step 4: Define preprocessing pipeline
# Numerical and categorical columns
numerical_cols = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EmployeeNumber",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "TenureRatio",
    "SatisfactionScore",
    "IncomeToLevelRatio",
    "LongCommute",
]
categorical_cols = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
    "AgeGroup",
]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols),
    ]
)

# Full pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

# Step 5: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 6: Apply pipeline
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)

# Convert processed data back to DataFrame for storage
feature_names = numerical_cols + list(
    pipeline.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .get_feature_names_out(categorical_cols)
)
X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

# Step 7: Save processed data
X_train_df.to_csv("../../data/processed/X_train.csv", index=False)
X_test_df.to_csv("../../data/processed/X_test.csv", index=False)
y_train.to_csv("../../data/processed/y_train.csv", index=False)
y_test.to_csv("../../data/processed/y_test.csv", index=False)

print("Processed data saved to data/processed/")
print("X_train shape:", X_train_df.shape)
print("X_test shape:", X_test_df.shape)
