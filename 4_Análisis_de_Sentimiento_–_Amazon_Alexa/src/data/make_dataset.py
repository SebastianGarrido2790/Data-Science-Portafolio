import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

df = pd.read_csv("../../data/raw/amazon_alexa.tsv", delimiter="\t", quoting=3)

df.isnull().sum()

# Print the missing values samples
df[df["verified_reviews"].isna() == True]

# We'll drop that one null record
df.dropna(inplace=True)

df.info()

# Explore rating column
df["rating"].value_counts()
df["rating"].value_counts().plot(kind="bar")

# Check percentanges
round(df["feedback"].value_counts() / df.shape[0] * 100, 2)

# Extracting the 'verified_reviews' value for one record with feedback = 0
df[df["feedback"] == 0].iloc[0]["verified_reviews"]

# Extracting the 'verified_reviews' value for one record with feedback = 1
df[df["feedback"] == 1].iloc[0]["verified_reviews"]

df[df["feedback"] == 1]["rating"].value_counts()
df[df["feedback"] == 0]["rating"].value_counts()

# --------------------
# Data Preprocessing
# --------------------


# Define feedback based on your rules
def map_feedback(rating):
    if rating in [1, 2]:
        return 0  # Negative
    elif rating == 3:
        return None  # Neutral - exclude
    elif rating in [4, 5]:
        return 1  # Positive


df["feedback"] = df["rating"].apply(map_feedback)
df = df.dropna(subset=["feedback"])  # Remove neutral reviews


# Preprocessing function
def preprocess_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    # Handle non-string inputs (e.g., NaN, None) by converting to empty string
    text = str(text) if pd.notna(text) else ""
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    words = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return " ".join(words) if words else ""  # Return empty string if no words remain


df["processed_reviews"] = df["verified_reviews"].apply(preprocess_text)

# Check for any remaining NaN values
print(df["processed_reviews"].isna().sum())

# Save the preprocessed data
df.to_csv("../../data/processed/preprocessed_data.csv", index=False)
