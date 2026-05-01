import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ============================================================
# STEP 1 - Load dataset
# ============================================================
df = pd.read_csv("hotel_reviews_clean.csv")

print("Initial shape:", df.shape)

# ============================================================
# STEP 2 - Drop null / empty rows
# ============================================================
df.dropna(subset=["Review_Text", "Reviewer_Score", "Tags"], inplace=True)

df["Review_Text"] = df["Review_Text"].astype(str)
df["Tags"] = df["Tags"].astype(str)

df = df[df["Review_Text"].str.strip() != ""]
df = df[df["Tags"].str.strip() != ""]

print("After removing null/empty:", df.shape)

# ============================================================
# STEP 3 - Remove placeholder-only rows
# ============================================================
both_placeholder = (
    df["Review_Text"].str.contains("No Positive", case=False, na=False) &
    df["Review_Text"].str.contains("No Negative", case=False, na=False)
)
df = df[~both_placeholder]

# ============================================================
# STEP 4 - Remove placeholder phrases
# ============================================================
df["Review_Text"] = df["Review_Text"].str.replace(
    r"No Positive[\.\s]*", "", case=False, regex=True
)
df["Review_Text"] = df["Review_Text"].str.replace(
    r"No Negative[\.\s]*", "", case=False, regex=True
)

df["Review_Text"] = df["Review_Text"].str.strip()

# ============================================================
# STEP 5 - Remove duplicates (better version)
# ============================================================
df.drop_duplicates(subset=["Review_Text", "Tags"], inplace=True)

print("After duplicate removal:", df.shape)

# ============================================================
# STEP 6 - Sentiment Label (Improved)
# ============================================================
def get_sentiment(score):
    if score >= 7:
        return "Positive"
    elif score <= 4:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment_Label"] = df["Reviewer_Score"].apply(get_sentiment)

# Optional: remove neutral
df = df[df["Sentiment_Label"] != "Neutral"]

df.drop(columns=["Reviewer_Score"], inplace=True)

print("Sentiment distribution:")
print(df["Sentiment_Label"].value_counts())

# ============================================================
# STEP 7 - Clean Tags
# ============================================================
def clean_tags(tag_str):
    tag_str = str(tag_str)
    tag_str = re.sub(r"[\[\]'\"]", "", tag_str)
    tag_str = tag_str.lower()
    tag_str = re.sub(r"\s+", " ", tag_str)
    return tag_str.strip()

df["Tags"] = df["Tags"].apply(clean_tags)

# ============================================================
# STEP 8 - Clean Review_Text
# ============================================================
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Keep important sentiment words
keep_words = {
    "no","not","never","good","bad","great","terrible",
    "excellent","horrible","amazing","awful","best","worst",
    "clean","dirty","friendly","rude","comfortable","uncomfortable"
}

stop_words = stop_words - keep_words

def clean_review_text(text):
    text = str(text).lower()
    
    # Remove URLs & emails
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)

    # Keep numbers
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()

    # Remove stopwords
    tokens = [w for w in tokens if w not in stop_words]

    # Lemmatize
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    # Keep words length >= 3
    tokens = [w for w in tokens if len(w) > 2]

    return " ".join(tokens)

print("Cleaning text...")
df["Review_Text"] = df["Review_Text"].apply(clean_review_text)

# ============================================================
# STEP 9 - Remove very short reviews
# ============================================================
df = df[df["Review_Text"].str.split().str.len() >= 3]

print("After removing short reviews:", df.shape)

# ============================================================
# STEP 10 - Review Length Feature
# ============================================================
df["Review_Length"] = df["Review_Text"].apply(lambda x: len(x.split()))

# ============================================================
# STEP 11 - Category Assignment (Improved)
# ============================================================
food_keywords = set(["food","breakfast","meal","restaurant","dinner","lunch","coffee","buffet"])
staff_keywords = set(["staff","service","rude","friendly","manager","reception","helpful"])
rooms_keywords = set(["room","bed","bathroom","clean","dirty","noise","noisy","ac","wifi","tv"])

def assign_category(row):
    words = (row["Review_Text"] + " " + row["Tags"]).lower().split()

    food_score = sum(1 for w in words if w in food_keywords)
    staff_score = sum(1 for w in words if w in staff_keywords)
    rooms_score = sum(1 for w in words if w in rooms_keywords)

    scores = {"Food": food_score, "Staff": staff_score, "Rooms": rooms_score}
    best = max(scores, key=scores.get)

    return best if scores[best] > 0 else "Other"

df["Category"] = df.apply(assign_category, axis=1)

print("Category distribution:")
print(df["Category"].value_counts())

# ============================================================
# STEP 12 - Final cleanup
# ============================================================
df.reset_index(drop=True, inplace=True)

print("Final shape:", df.shape)

# ============================================================
# STEP 13 - Save dataset
# ============================================================
df.to_csv("hotel_reviews_preprocessed.csv", index=False)

print("Saved successfully: hotel_reviews_preprocessed.csv")