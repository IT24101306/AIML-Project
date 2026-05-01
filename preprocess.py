import pandas as pd

# Load original dataset
df = pd.read_csv("Hotel_Reviews.csv")

print("Original shape:", df.shape)
print("Original columns:", list(df.columns))

# Step 1 — Combine Positive + Negative into one new column
df["Review_Text"] = df["Positive_Review"] + ". " + df["Negative_Review"]

# Step 2 — Drop all 15 unwanted columns
columns_to_drop = [
    "Hotel_Address",
    "Hotel_Name",
    "Average_Score",
    "Additional_Number_of_Scoring",
    "Total_Number_of_Reviews",
    "Total_Number_of_Reviews_Reviewer_Has_Given",
    "Review_Total_Negative_Word_Counts",
    "Review_Total_Positive_Word_Counts",
    "Reviewer_Nationality",
    "Review_Date",
    "days_since_review",
    "lat",
    "lng",
    "Positive_Review",
    "Negative_Review"
]

df_clean = df.drop(columns=columns_to_drop)

# Step 3 — Save new clean dataset
df_clean.to_csv("hotel_reviews_clean.csv", index=False)

print("\nCleaned shape:", df_clean.shape)
print("Remaining columns:", list(df_clean.columns))
print(df_clean.head())