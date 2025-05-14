import joblib
import pandas as pd
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Load the scaler and label encoder (if needed for plan_type)
scaler = joblib.load(os.path.join(PATHS["models"], "scaler.pkl"))

# Step 1: Define the new data (including raw ticket notes)
new_ticket_notes = "Customer reported a billing issue that was resolved quickly."

# Step 2: Summarize the ticket notes using the same summarization model as training
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
try:
    summary = summarizer(
        new_ticket_notes,
        max_length=5,  # Adjust based on training settings
        min_length=2,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )[0]["summary_text"]
except Exception as e:
    logger.error(f"Summarization failed: {e}")
    raise

# Step 3: Generate the embedding for the summary
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Same model as training
summary_embedding = embedding_model.encode([summary])[0].tolist()  # Shape: (768,)

# Step 4: Create the new_data DataFrame with the actual embedding
new_data = pd.DataFrame(
    {
        "age": [50],
        "tenure": [10],
        "spend_rate": [100],
        "plan_type_encoded": [1],
    }
)

# Step 5: Scale the numeric features
numeric_features = ["age", "tenure", "spend_rate"]
new_data_scaled = new_data.copy()
new_data_scaled[numeric_features] = scaler.transform(new_data[numeric_features])

# Step 6: Add the summary_embedding as separate columns (to match training data structure)
embedding_dim = 768  # Same as training
embedding_columns = [f"embedding_{i}" for i in range(embedding_dim)]
embedding_df = pd.DataFrame([summary_embedding], columns=embedding_columns)

# Concatenate the scaled numeric features, plan_type_encoded, and embedding
new_data_final = pd.concat([new_data_scaled, embedding_df], axis=1)

# Step 7: Ensure all columns match the training data
# (Assuming X_train from training had these columns)
expected_columns = list(X_train.columns)
new_data_final = new_data_final[expected_columns]

# Step 8: Make the prediction
prediction = model.predict(new_data_final)
logger.info(f"Prediction for new data: {prediction[0]}")
