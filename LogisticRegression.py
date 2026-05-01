import numpy as np
import scipy.sparse as sp
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

import pandas as pd

# ============================================================
# STEP 1 - Output folders
# ============================================================
os.makedirs("models", exist_ok=True)
os.makedirs("models/plots", exist_ok=True)

print("=" * 60)
print("LOGISTIC REGRESSION - SENTIMENT CLASSIFIER (IMPROVED)")
print("=" * 60)

# ============================================================
# STEP 2 - Load dataset
# ============================================================
X = sp.load_npz("feature_files/final_feature_matrix.npz")
y = np.load("feature_files/sentiment_labels.npy")

label_encoder = joblib.load("feature_files/label_encoder.pkl")

print("\nDataset Overview")
print("Shape:", X.shape)
print("Labels:", y.shape)
print("Classes:", label_encoder.classes_)

# ============================================================
# STEP 3 - Train test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split")
print("Train:", X_train.shape)
print("Test :", X_test.shape)

# ============================================================
# STEP 4 - MODEL SUITABILITY EXPLANATION (for report)
# ============================================================
print("\nMODEL SUITABILITY:")
print("- Logistic Regression is suitable for high-dimensional sparse TF-IDF text data")
print("- Works well for binary classification (Positive / Negative)")
print("- Produces probabilistic outputs for confidence scoring")
print("- Efficient for large datasets (>400K reviews)")

# ============================================================
# STEP 5 - BASE MODEL (Logistic Regression)
# ============================================================
base_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    solver="lbfgs",
    random_state=42,
    n_jobs=-1
)

base_model.fit(X_train, y_train)

y_pred_base = base_model.predict(X_test)
y_prob_base = base_model.predict_proba(X_test)[:, 1]

# ============================================================
# STEP 6 - HYPERPARAMETER TUNING (GridSearchCV)
# ============================================================
print("\nRunning GridSearchCV (Hyperparameter Tuning)...")

param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "max_iter": [500, 1000]
}

cv = StratifiedKFold(n_splits=5)

grid = GridSearchCV(
    LogisticRegression(class_weight="balanced", solver="lbfgs", n_jobs=-1),
    param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest Parameters Found:")
print(grid.best_params_)

# ============================================================
# STEP 7 - Evaluate Base Model vs Tuned Model
# ============================================================

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "auc": roc_auc_score(y_test, y_prob),
        "pred": y_pred,
        "prob": y_prob
    }

base_metrics = evaluate(base_model, X_test, y_test)
tuned_metrics = evaluate(best_model, X_test, y_test)

print("\nMODEL COMPARISON")
print("=" * 40)

print("\nBase Model:")
print(f"Accuracy : {base_metrics['accuracy']:.4f}")
print(f"F1 Score : {base_metrics['f1']:.4f}")
print(f"AUC      : {base_metrics['auc']:.4f}")

print("\nTuned Model:")
print(f"Accuracy : {tuned_metrics['accuracy']:.4f}")
print(f"F1 Score : {tuned_metrics['f1']:.4f}")
print(f"AUC      : {tuned_metrics['auc']:.4f}")

# Choose best model
final_model = best_model if tuned_metrics["f1"] > base_metrics["f1"] else base_model

print("\nFinal Model Selected:", "Tuned Model" if final_model == best_model else "Base Model")

# ============================================================
# STEP 8 - CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_test, final_model.predict(X_test))

plt.figure()
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.savefig("models/plots/confusion_matrix.png")
plt.close()

# ============================================================
# STEP 9 - ROC CURVE
# ============================================================
y_prob = final_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve")
plt.legend()
plt.savefig("models/plots/roc_curve.png")
plt.close()

# ============================================================
# STEP 10 - Save model
# ============================================================
joblib.dump(final_model, "models/classifier.pkl")

# ============================================================
# STEP 11 - Feature importance (interpretability)
# ============================================================
feature_names = joblib.load("feature_files/tfidf_vectorizer.pkl").get_feature_names_out()
coef = final_model.coef_[0]

top_negative = np.argsort(coef)[:20]
top_positive = np.argsort(coef)[-20:][::-1]

print("\nTop Negative Words:")
for i in top_negative:
    print(feature_names[i], coef[i])

print("\nTop Positive Words:")
for i in top_positive:
    print(feature_names[i], coef[i])

# ============================================================
# STEP 12 - FINAL REPORT SUMMARY (for assignment)
# ============================================================
print("\nFINAL SUMMARY")
print("=" * 60)
print("\nTraining Completed Successfully")