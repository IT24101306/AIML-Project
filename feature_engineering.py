import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# ============================================================
# STEP 1 - Load dataset
# ============================================================
df = pd.read_csv("hotel_reviews_preprocessed.csv")

print("=" * 60)
print("FEATURE ENGINEERING (FIXED VERSION)")
print("=" * 60)
print("Shape:", df.shape)
print("Columns:", list(df.columns))

# ============================================================
# STEP 2 - Create output folder
# ============================================================
os.makedirs("feature_files", exist_ok=True)

# ============================================================
# STEP 3 - TF-IDF (Review Text)
# ============================================================
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9,
    sublinear_tf=True
)

df["Review_Text"] = df["Review_Text"].fillna("")
tfidf_matrix = tfidf_vectorizer.fit_transform(df["Review_Text"])

joblib.dump(tfidf_vectorizer, "feature_files/tfidf_vectorizer.pkl")

print("TF-IDF review shape:", tfidf_matrix.shape)

# ============================================================
# STEP 4 - TF-IDF (Tags)
# ============================================================
df["Tags"] = df["Tags"].fillna("")

tags_vectorizer = TfidfVectorizer(
    max_features=200,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95
)

tags_matrix = tags_vectorizer.fit_transform(df["Tags"])

joblib.dump(tags_vectorizer, "feature_files/tags_vectorizer.pkl")

print("TF-IDF tags shape:", tags_matrix.shape)

# ============================================================
# STEP 5 - SENTIMENT LABEL HANDLING (FIXED PART)
# ============================================================
# Prefer fused / existing labels from preprocessed CSV; use stars only when sentiment is missing.
if "Sentiment_Label" in df.columns and df["Sentiment_Label"].notna().any():
    print("\nUsing existing Sentiment_Label column (e.g. from batch / logistic fusion)...")
elif "Reviewer_Score" in df.columns:
    print("\nUsing Reviewer_Score to generate sentiment (>=7 => Positive)...")
    df["Sentiment_Label"] = df["Reviewer_Score"].apply(
        lambda x: "Positive" if float(x) >= 7 else "Negative"
    )
else:
    raise Exception(
        "No sentiment source found! Need either Sentiment_Label or Reviewer_Score"
    )

# Encode labels
label_encoder = LabelEncoder()
df["Sentiment_Encoded"] = label_encoder.fit_transform(df["Sentiment_Label"])

joblib.dump(label_encoder, "feature_files/label_encoder.pkl")

print("\nLabel distribution:")
print(df["Sentiment_Label"].value_counts())

# ============================================================
# STEP 6 - FIX NaN ISSUES (IMPORTANT FIX #2)
# ============================================================
df["Review_Length"] = df["Review_Text"].apply(lambda x: len(str(x).split()))

min_len = df["Review_Length"].min()
max_len = df["Review_Length"].max()

df["Review_Length_Normalized"] = (
    (df["Review_Length"] - min_len) /
    (max_len - min_len + 1e-8)   # prevent divide by zero
)

joblib.dump({"min": min_len, "max": max_len}, "feature_files/length_stats.pkl")

# ============================================================
# STEP 6b - Reviewer score (stars) for extended features / retraining
# ============================================================
if "Reviewer_Score" not in df.columns:
    df["Reviewer_Score"] = 5.5
df["Reviewer_Score"] = pd.to_numeric(df["Reviewer_Score"], errors="coerce").fillna(5.5)
use_five_scale = df["Reviewer_Score"].max() <= 5.5
if use_five_scale:
    df["Reviewer_Score_Normalized"] = (df["Reviewer_Score"].clip(1, 5) - 1.0) / 4.0
else:
    df["Reviewer_Score_Normalized"] = (df["Reviewer_Score"].clip(1, 10) - 1.0) / 9.0
joblib.dump({"use_five_scale": bool(use_five_scale)}, "feature_files/rating_scale_meta.pkl")

# ============================================================
# STEP 7 - Keyword features
# ============================================================
def count_keywords(text, keywords):
    text = str(text).lower()
    return sum(1 for w in keywords if w in text)

negative_keywords = ["dirty","rude","bad","worst","terrible","horrible"]
positive_keywords = ["clean","good","great","excellent","amazing","friendly"]

df["Negative_Keyword_Count"] = df["Review_Text"].apply(
    lambda x: count_keywords(x, negative_keywords)
)

df["Positive_Keyword_Count"] = df["Review_Text"].apply(
    lambda x: count_keywords(x, positive_keywords)
)

df["Keyword_Sentiment_Score"] = (
    df["Positive_Keyword_Count"] - df["Negative_Keyword_Count"]
)

# ============================================================
# STEP 8 - Category encoding (FIX #3 safety)
# ============================================================
if "Category" not in df.columns:
    df["Category"] = "Other"

category_encoder = LabelEncoder()
df["Category_Encoded"] = category_encoder.fit_transform(df["Category"])

joblib.dump(category_encoder, "feature_files/category_encoder.pkl")

# ============================================================
# STEP 9 - Final feature matrix
# ============================================================
numeric_features = sp.csr_matrix(df[[
    "Review_Length_Normalized",
    "Negative_Keyword_Count",
    "Positive_Keyword_Count",
    "Keyword_Sentiment_Score",
    "Reviewer_Score_Normalized",
]].values)

final_matrix = sp.hstack([
    tfidf_matrix,
    tags_matrix,
    numeric_features
])

print("\nFinal feature shape:", final_matrix.shape)

sp.save_npz("feature_files/final_feature_matrix.npz", final_matrix)

# ============================================================
# STEP 10 - Save labels
# ============================================================
np.save("feature_files/sentiment_labels.npy", df["Sentiment_Encoded"].values)
np.save("feature_files/category_labels.npy", df["Category_Encoded"].values)

# ============================================================
# FINAL CHECK
# ============================================================
print("\nFEATURE ENGINEERING COMPLETE SUCCESSFULLY")
print("=" * 60)
print("No KeyError issues")
print("No NaN crashes")
print("Pipeline stable for training")

# ============================================================
# STEP - SAVE FINAL ENGINEERED DATASET (CSV)
# ============================================================
output_csv_path = "hotel_reviews_feature_engineered.csv"

df.to_csv(output_csv_path, index=False)

print("\n" + "=" * 60)
print("DATASET SAVED SUCCESSFULLY")
print("=" * 60)
print(f"Saved file: {output_csv_path}")
print(f"Shape      : {df.shape}")
print("This file contains all cleaned + engineered features")