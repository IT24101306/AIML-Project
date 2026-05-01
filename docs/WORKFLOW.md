# Hotel Review Intelligence — End-to-End Workflow

This document describes the **completed workflow** implemented in this repository: how data moves from raw sources through **offline ML training**, **batch upload processing**, and the **Flask + React dashboard**. It maps each stage to the **code files** that implement it.

---

## 1. Architecture overview

```
┌─────────────────┐     HTTP (JSON)      ┌──────────────────┐
│  React UI       │ ◄──────────────────► │  Flask API       │
│  HotelDashboard │                      │  api/app.py      │
└─────────────────┘                      └────────┬─────────┘
                                                │
                    ┌───────────────────────────┼───────────────────────────┐
                    │                           │                           │
                    ▼                           ▼                           ▼
            hotel_reviews_              models/*.pkl                 api/results/
            preprocessed.csv            feature_files/*.pkl          monthly_*.csv
            negative_reviews_          (trained offline)            (latest batch)
            clustered.csv
```

- There is **no SQL database** for review rows. The operational “source of truth” for the dashboard is **CSV files** at the project root and under `api/results/`.
- **Trained ML** lives in `models/` and `feature_files/` as **joblib / numpy** artifacts produced by offline scripts.

---

## 2. Offline ML training pipeline (not run on each upload)

These steps **build or refresh** the classifiers and cluster models. Run them on your machine when you need new models (e.g. after changing training data or features).

| Order | Script | Purpose | Main outputs |
|------:|--------|---------|----------------|
| 1 | `preprocess.py` | Shape raw `Hotel_Reviews.csv` into a cleaner table. | `hotel_reviews_clean.csv` |
| 2 | `data_cleaning.py` | Text cleaning, rules, sentiment labels from reviewer score, etc. | `hotel_reviews_preprocessed.csv` |
| 3 | `feature_engineering.py` | TF‑IDF (text + tags), numeric features, label encoder, sparse matrix. | `feature_files/*` (e.g. vectorizers, `final_feature_matrix.npz`, `sentiment_labels.npy`) |
| 4 | `clustering.py` | KMeans + SVD (or related) on features; cluster → category mapping. | `models/kmeans_model.pkl`, `svd_reducer.pkl`, `cluster_label_mapping.pkl`, … |
| 5 | `LogisticRegression.py` | Trains **sentiment** classifier on engineered features. | `models/classifier.pkl` |

**Label semantics (training):** Sentiment labels in the training CSV are derived from **reviewer score rules** in the cleaning / feature scripts (e.g. high scores → Positive). That is independent of the **upload CSV** used in the app.

**Important:** **Uploading a CSV in the web app does not retrain** these models. Upload only **applies** existing pickles.

---

## 3. Application runtime workflow (dashboard + API)

### 3.1 Frontend

| Component | Path | Role |
|-----------|------|------|
| Dashboard | `frontend/src/pages/HotelDashboard.jsx` | Metrics, tables, upload, export PDF, category actions, timeframe filter. |
| API base URL | `REACT_APP_API_URL` (optional) | Defaults to `http://127.0.0.1:5000/api`. |

The UI calls endpoints such as `/summary`, `/reviews`, `/category_actions`, `/upload_monthly`, `/clear_reviews`, etc.

### 3.2 Flask API (HTTP layer)

| Path | File | Role |
|------|------|------|
| Core app | `api/app.py` | Routes, CORS, loads CSVs, lazy-loads models, applies timeframe slices, Claude for per-review solutions and category-level actions. |
| Claude client | `api/claude_client.py` | Anthropic API wrapper (`ANTHROPIC_API_KEY`, optional `ANTHROPIC_MODEL`). |

**Key CSV paths (from `app.py`):**

- `hotel_reviews_preprocessed.csv` — all scored reviews for summary / tables (project root).
- `negative_reviews_clustered.csv` — negative rows with cluster metadata when aligned with preprocessing.
- `api/results/monthly_reviews.csv` — last uploaded file.
- `api/results/monthly_results.csv` — last batch output table.

### 3.3 Upload → batch processing

When the user uploads a CSV:

1. **`POST /api/upload_monthly`** (`app.py`) saves the file to `api/results/monthly_reviews.csv`.
2. **`reset_preprocessed_and_cluster_csvs()`** resets the **root** preprocessed + clustered CSVs to headers (and removes stale alternate clustered file under `models/` if present) so a new batch does not **stack** on old rows.
3. **`batch_processor.py`** is run as a subprocess (same Python as the server).

**Batch processor** (`api/batch_processor.py`):

- Reads the uploaded CSV (`utf-8-sig`, Python engine).
- Normalizes columns (`review_text`, `tags`, `reviewer_score`).
- Drops **empty / invalid / duplicate** rows (avoids junk lines from spreadsheets).
- Optionally caps rows if `MONTHLY_BATCH_MAX_ROWS` is set to a **positive** integer (default in code is **no cap** — process full cleaned file unless env overrides).
- For each remaining row, calls **`process_review()`** in `solution_generator.py` (loads models once at import: classifier, KMeans, SVD, vectorizers, etc.).
- Writes:
  - `api/results/monthly_results.csv` (batch table),
  - **Overwrites** `hotel_reviews_preprocessed.csv` and `negative_reviews_clustered.csv` with **only this batch’s** rows (no append drift).

**Inference in batch** (`api/solution_generator.py`):

- Cleans text, builds the same style of **sparse feature vector** as training (TF‑IDF + tags + length / lexicon-style numerics).
- **Sentiment:** `classifier.predict` + `label_encoder` → `Positive` / `Negative`.
- **If negative:** severity heuristics, **cluster** prediction → category bucket (Food / Staff / Rooms / Other via mapping), then **Claude** can generate structured solution JSON for that row.

**Note:** `api/ml_utils.py` is a small alternate helper surface; the main batch path uses `solution_generator.py`.

### 3.4 Clear data

- **`POST /api/clear_reviews`** resets preprocessed + clustered CSVs and removes `api/results/monthly_reviews.csv` and `monthly_results.csv` (see `clear_all_review_data()` in `app.py`).

---

## 4. What the dashboard “total reviews” means

- **`GET /api/summary`** counts rows in **`hotel_reviews_preprocessed.csv`** (after optional timeframe slice).
- After an upload, that file should match the **number of valid, deduplicated rows** processed in the last batch (not spreadsheet blank lines).

---

## 5. Environment variables (reference)

| Variable | Where used | Purpose |
|----------|------------|---------|
| `ANTHROPIC_API_KEY` | `api/claude_client.py` | Claude API for solutions / category actions. |
| `ANTHROPIC_MODEL` | `api/claude_client.py` | Model id override. |
| `MONTHLY_BATCH_MAX_ROWS` | `api/batch_processor.py` | If set to **> 0**, only first *N* valid rows per upload (faster; totals partial). |

---

## 6. Quick file index

| Area | Files |
|------|--------|
| UI | `frontend/src/pages/HotelDashboard.jsx`, `frontend/src/App.js` |
| API | `api/app.py`, `api/claude_client.py` |
| Batch | `api/batch_processor.py`, `api/solution_generator.py` |
| Train | `preprocess.py`, `data_cleaning.py`, `feature_engineering.py`, `clustering.py`, `LogisticRegression.py` |
| Artifacts | `models/*.pkl`, `feature_files/*`, `hotel_reviews_preprocessed.csv`, `negative_reviews_clustered.csv` |

---

## 7. Is this the “completed” workflow?

**Yes** for this repo: it covers **(A)** offline training → artifacts, **(B)** upload → batch inference + CSV updates, **(C)** dashboard reads + optional Claude summaries/actions.  

It does **not** include items you would add separately in production (auth, job queues for long batches, multi-tenant DB, automated retraining CI, etc.).

---

*Last aligned with the codebase layout and behavior as of the documentation date.*
