# api/app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import re
import nltk
import scipy.sparse as sp
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
from claude_client import claude_complete
import subprocess
import sys
import json
import math

# Always load api/.env regardless of cwd (e.g. `python api/app.py` from repo root).
_API_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_API_DIR, ".env"))
from mongo_sync import fetch_mongo_reviews_dataframe

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',  quiet=True)

app    = Flask(__name__)
CORS(app)

# ── paths ──────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE, "models")
FEAT_DIR    = os.path.join(BASE, "feature_files")
PREPROC_CSV = os.path.join(BASE, "hotel_reviews_preprocessed.csv")
CLUSTER_CSV = os.path.join(BASE, "negative_reviews_clustered.csv")
CLUSTER_CSV_ALT = os.path.join(MODELS_DIR, "negative_reviews_clustered.csv")

# Cap rows for /api/reviews vectorization (newest negatives only); keeps UI fast on huge CSVs
_REVIEWS_NEG_POOL = 8000

# Sidebar category filter must match these labels
_REVIEW_TABLE_CATEGORIES = frozenset({"Food", "Staff", "Rooms", "Other"})

# ── load models (lazy loading - on demand) ──────────────────
_models = {}
_models_loaded = False

def load_models():
    global _models, _models_loaded
    if _models_loaded:
        return
    
    print("Loading models...")
    required_files = [
        "classifier.pkl", "kmeans_model.pkl", "svd_reducer.pkl",
        "cluster_label_mapping.pkl", "tfidf_vectorizer.pkl",
        "tags_vectorizer.pkl", "label_encoder.pkl", "length_stats.pkl"
    ]
    
    missing = [f for f in required_files if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        print(f"Warning: Missing model files: {missing}")
        print("Please run training scripts first: preprocess.py, clustering.py, LogisticRegression.py")
        _models_loaded = True
        return
    
    _models["classifier"] = joblib.load(os.path.join(MODELS_DIR, "classifier.pkl"))
    _models["kmeans_model"] = joblib.load(os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    _models["svd_reducer"] = joblib.load(os.path.join(MODELS_DIR, "svd_reducer.pkl"))
    _models["cluster_label_mapping"] = joblib.load(os.path.join(MODELS_DIR, "cluster_label_mapping.pkl"))
    _models["tfidf_vectorizer"] = joblib.load(os.path.join(FEAT_DIR, "tfidf_vectorizer.pkl"))
    _models["tags_vectorizer"] = joblib.load(os.path.join(FEAT_DIR, "tags_vectorizer.pkl"))
    _models["label_encoder"] = joblib.load(os.path.join(FEAT_DIR, "label_encoder.pkl"))
    _models["length_stats"] = joblib.load(os.path.join(FEAT_DIR, "length_stats.pkl"))
    print("Models loaded.")
    _models_loaded = True

def get_model(name):
    load_models()
    return _models.get(name)

# ── NLP helpers ────────────────────────────────────────────
_stop_words = set(stopwords.words("english")) - {
    "no","not","never","nothing","good","bad","great","terrible",
    "excellent","horrible","amazing","awful","best","worst",
    "clean","dirty","friendly","rude","comfortable",
    "uncomfortable","quiet","noisy"
}
_lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text   = str(text).lower()
    text   = re.sub(r"[^a-zA-Z\s]", "", text)
    text   = re.sub(r"\s+", " ", text).strip()
    tokens = [_lemmatizer.lemmatize(w)
              for w in text.split()
              if w not in _stop_words and len(w) > 2]
    return " ".join(tokens)

def build_features(cleaned_review, cleaned_tags):
    tfidf_vectorizer = get_model("tfidf_vectorizer")
    tags_vectorizer = get_model("tags_vectorizer")
    length_stats = get_model("length_stats")
    
    if not all([tfidf_vectorizer, tags_vectorizer, length_stats]):
        raise Exception("Models not loaded. Please train the models first.")
    
    r_tfidf  = tfidf_vectorizer.transform([cleaned_review])
    t_tfidf  = tags_vectorizer.transform([cleaned_tags])
    length   = len(cleaned_review.split())
    mn, mx   = length_stats["min"], length_stats["max"]
    norm_len = (length - mn) / (mx - mn) if mx != mn else 0

    neg_kw = ["dirty","rude","bad","worst","terrible","horrible"]
    pos_kw = ["clean","good","great","excellent","amazing","friendly"]

    neg_c  = sum(1 for w in neg_kw  if w in cleaned_review)
    pos_c  = sum(1 for w in pos_kw  if w in cleaned_review)

    numeric = sp.csr_matrix([[
        norm_len, neg_c, pos_c, pos_c - neg_c
    ]])
    combined = sp.hstack([r_tfidf, t_tfidf, numeric])
    return combined, neg_c, len(cleaned_review.split())

def calculate_severity(neg_keyword_count, review_length):
    score = 0
    score += 3 if neg_keyword_count >= 5 else 2 if neg_keyword_count >= 3 else 1 if neg_keyword_count >= 1 else 0
    score += 2 if review_length >= 50 else 1 if review_length >= 20 else 0
    return "High" if score >= 5 else "Medium" if score >= 3 else "Low"

# Keywords for severity / table (substring match, same semantics as old per-row apply)
_SEVERITY_NEG_KWS = (
    "dirty", "rude", "broken", "terrible", "horrible", "awful",
    "worst", "bad", "poor", "disappointing", "disgusting",
)

def _neg_keyword_counts_series(review_text: pd.Series) -> pd.Series:
    low = review_text.fillna("").str.lower()
    cnt = pd.Series(0, index=review_text.index, dtype=np.int32)
    for w in _SEVERITY_NEG_KWS:
        cnt += low.str.contains(re.escape(w), regex=True, na=False).astype(np.int32)
    return cnt

def _severity_series(neg_kw: pd.Series, review_len: pd.Series) -> pd.Series:
    nk = neg_kw.to_numpy(dtype=np.int32, copy=False)
    rl = review_len.to_numpy(dtype=np.float64, copy=False)
    score = np.zeros(len(nk), dtype=np.int32)
    score += np.where(nk >= 5, 3, np.where(nk >= 3, 2, np.where(nk >= 1, 1, 0)))
    score += np.where(rl >= 50, 2, np.where(rl >= 20, 1, 0))
    labels = np.where(score >= 5, "High", np.where(score >= 3, "Medium", "Low"))
    return pd.Series(labels, index=neg_kw.index, dtype=object)


def _sentiment_bucket_series(df: pd.DataFrame) -> pd.Series:
    """Normalize Sentiment_Label to 'positive' | 'negative' | 'other' (case-insensitive)."""
    if "Sentiment_Label" not in df.columns:
        return pd.Series("other", index=df.index, dtype=object)
    if df.empty:
        return pd.Series(dtype=object, index=df.index)
    raw = df["Sentiment_Label"].astype(str).str.strip().str.lower()
    out = pd.Series("other", index=df.index, dtype=object)
    out = out.mask(raw == "positive", "positive")
    out = out.mask(raw == "negative", "negative")
    return out


def _is_negative_sentiment_df(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: same rows as summary 'negative' bucket (case-insensitive)."""
    return _sentiment_bucket_series(df) == "negative"


def predict_category(feature_vector):
    from sklearn.preprocessing import normalize
    svd_reducer = get_model("svd_reducer")
    kmeans_model = get_model("kmeans_model")
    cluster_label_mapping = get_model("cluster_label_mapping")
    
    if not all([svd_reducer, kmeans_model, cluster_label_mapping]):
        raise Exception("Models not loaded. Please train the models first.")
    
    dense   = feature_vector.toarray() if sp.issparse(feature_vector) else feature_vector
    reduced = normalize(svd_reducer.transform(dense))
    cid     = kmeans_model.predict(reduced)[0]
    return cluster_label_mapping.get(int(cid), "Other"), int(cid)

def get_solution_from_claude(review_text, category, severity, tags):
    cat_ctx = {
        "Food" : "Complaint about food quality, breakfast, meals, or dining.",
        "Staff": "Complaint about staff behavior, receptionist, or service quality.",
        "Rooms": "Complaint about room condition, cleanliness, noise, or amenities.",
        "Other": "General complaint about WiFi, parking, facilities, or pricing."
    }
    sev_ctx = {
        "High"  : "CRITICAL — urgent action required, serious reputation risk.",
        "Medium": "MODERATE — needs attention to prevent recurrence.",
        "Low"   : "MINOR — note and address as quality improvement."
    }
    prompt = f"""You are an expert hotel management consultant.
A guest left a negative review.

Category : {category}
Review   : {review_text}
Severity : {severity}
Tags     : {tags}

Context  : {cat_ctx.get(category,'')}
Severity context: {sev_ctx.get(severity,'')}

Respond ONLY with valid JSON — no markdown, no extra text:
{{
  "immediate_action": "...",
  "short_term_fix": "...",
  "long_term_improvement": "...",
  "guest_response": "...",
  "department_responsible": "...",
  "estimated_resolution_time": "...",
  "prevention_tip": "..."
}}"""
    try:
        text = claude_complete(prompt, max_tokens=1200)
        text = re.sub(r"```json|```", "", text).strip()
        return {"success": True, "data": json.loads(text)}
    except Exception:
        # Fallback to high-quality rule-based suggestions if API fails (e.g. quota exceeded)
        fallbacks = {
            "Food": {
                "immediate_action": "Offer a complimentary meal replacement or discount.",
                "short_term_fix": "Review ingredient freshness and discuss with head chef.",
                "long_term_improvement": "Update the menu based on consistent guest feedback and train kitchen staff.",
                "guest_response": "We sincerely apologize for your dining experience and would love to invite you back for a complimentary meal.",
                "department_responsible": "Food & Beverage",
                "estimated_resolution_time": "24 hours",
                "prevention_tip": "Daily tasting audits before service begins."
            },
            "Staff": {
                "immediate_action": "Manager to personally apologize to the guest.",
                "short_term_fix": "Identify the staff member and provide immediate corrective feedback.",
                "long_term_improvement": "Implement monthly hospitality and conflict-resolution training.",
                "guest_response": "We deeply regret the service you received. This does not reflect our standards and we are addressing it with our team.",
                "department_responsible": "Human Resources / Front Desk",
                "estimated_resolution_time": "Immediate",
                "prevention_tip": "Reward programs for staff who receive positive guest mentions."
            },
            "Rooms": {
                "immediate_action": "Relocate guest to an upgraded room or dispatch housekeeping immediately.",
                "short_term_fix": "Block the room from future bookings until fully inspected and repaired.",
                "long_term_improvement": "Revamp the housekeeping checklist and schedule deep cleaning rotations.",
                "guest_response": "Please accept our apologies for the room conditions. We have escalated this to our Head Housekeeper.",
                "department_responsible": "Housekeeping / Maintenance",
                "estimated_resolution_time": "1-2 hours",
                "prevention_tip": "Supervisors must randomly audit 10% of cleaned rooms daily."
            },
            "Other": {
                "immediate_action": "Acknowledge the specific issue and provide a tailored concession (e.g. free WiFi, waived parking).",
                "short_term_fix": "Log the complaint in the property management system for tracking.",
                "long_term_improvement": "Review amenity contracts and negotiate better terms with vendors (e.g. ISP).",
                "guest_response": "Thank you for bringing this to our attention. We are actively working to resolve this facility issue.",
                "department_responsible": "Operations",
                "estimated_resolution_time": "Variable",
                "prevention_tip": "Regular preventive maintenance schedules for all guest amenities."
            }
        }
        return {"success": True, "data": fallbacks.get(category, fallbacks["Other"])}

# ── helper to load CSVs safely ─────────────────────────────
_preprocessed_cache = None
_preprocessed_cache_mtime = None
_clustered_cache = None
_clustered_cache_source = None
_clustered_cache_mtime = None


def _file_mtime(path: str):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None

def load_preprocessed():
    global _preprocessed_cache, _preprocessed_cache_mtime
    if not os.path.exists(PREPROC_CSV):
        raise FileNotFoundError(f"Preprocessed data not found: {PREPROC_CSV}")

    curr_mtime = _file_mtime(PREPROC_CSV)
    if _preprocessed_cache is None or _preprocessed_cache_mtime != curr_mtime:
        _preprocessed_cache = pd.read_csv(PREPROC_CSV)
        _preprocessed_cache_mtime = curr_mtime
    return _preprocessed_cache


def reset_preprocessed_and_cluster_csvs():
    """
    Empty dashboard row stores (headers only). Does not touch api/results/*.
    Call before a new upload so batch_processor append = dataset for this file only.
    """
    preproc_header = "Tags,Review_Text,Sentiment_Label,Review_Length,Category,Reviewer_Score\n"
    cluster_header = (
        "Tags,Review_Text,Sentiment_Label,Review_Length,Category,"
        "predicted_cluster,display_category\n"
    )
    with open(PREPROC_CSV, "w", encoding="utf-8", newline="") as f:
        f.write(preproc_header)
    with open(CLUSTER_CSV, "w", encoding="utf-8", newline="") as f:
        f.write(cluster_header)
    if os.path.isfile(CLUSTER_CSV_ALT):
        try:
            os.remove(CLUSTER_CSV_ALT)
        except OSError:
            pass


def clear_all_review_data():
    """
    Reset dashboard CSV data: preprocessed reviews, clustered negatives, monthly batch files.
    Does not delete trained model pickles under models/ or feature_files/.
    """
    api_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(api_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    reset_preprocessed_and_cluster_csvs()

    for name in ("monthly_reviews.csv", "monthly_results.csv"):
        path = os.path.join(results_dir, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass


def load_clustered():
    global _clustered_cache, _clustered_cache_source, _clustered_cache_mtime
    source = None
    for path in (CLUSTER_CSV, CLUSTER_CSV_ALT):
        if os.path.exists(path):
            source = path
            break

    if source is None:
        _clustered_cache = pd.DataFrame()
        _clustered_cache_source = None
        _clustered_cache_mtime = None
        return _clustered_cache

    curr_mtime = _file_mtime(source)
    if (
        _clustered_cache is None
        or _clustered_cache_source != source
        or _clustered_cache_mtime != curr_mtime
    ):
        _clustered_cache = pd.read_csv(source)
        _clustered_cache_source = source
        _clustered_cache_mtime = curr_mtime
    return _clustered_cache

def apply_timeframe(df, cdf, timeframe):
    """Simulates time filtering by taking a proportional slice of the dataset."""
    if timeframe == "all" or timeframe == "1y" or not timeframe:
        return df, cdf
    
    total = len(df)
    if timeframe == "1m":
        ratio = 0.083
    elif timeframe == "3m":
        ratio = 0.25
    elif timeframe == "6m":
        ratio = 0.50
    else:
        ratio = 1.0

    cutoff_df = int(total * ratio)
    cutoff_cdf = int(len(cdf) * ratio)
    
    return df.tail(cutoff_df), cdf.tail(cutoff_cdf)


def _assign_display_category(neg_df: pd.DataFrame, cdf: pd.DataFrame) -> pd.DataFrame:
    """Add display_category to a negative-only dataframe. cdf rows align 1:1 with neg_df when lengths match."""
    out = neg_df.copy().reset_index(drop=True)
    cdf_ok = len(cdf) > 0 and len(cdf) == len(out)
    if cdf_ok:
        cr = cdf.reset_index(drop=True)
        if "predicted_cluster" in cr.columns:
            out["predicted_cluster"] = cr["predicted_cluster"].values
        if "original_category" in cr.columns:
            out["original_category"] = cr["original_category"].values

    cluster_label_mapping = get_model("cluster_label_mapping") or {}
    if "predicted_cluster" in out.columns and cluster_label_mapping:
        pc = pd.to_numeric(out["predicted_cluster"], errors="coerce")
        lut = {}
        for k, v in cluster_label_mapping.items():
            ik = int(k)
            lut[ik] = v
            lut[float(ik)] = v
        out["display_category"] = pc.map(lut).fillna("Other").astype(str)
    elif "original_category" in out.columns:
        out["display_category"] = out["original_category"].astype(str)
    elif "Category" in out.columns:
        out["display_category"] = out["Category"].astype(str)
    else:
        out["display_category"] = "Other"
    out["display_category"] = out["display_category"].astype(str).str.strip()
    return out


def _normalize_category_counts(raw: dict) -> dict:
    """Merge counts into canonical Food / Staff / Rooms / Other (unknown labels → Other)."""
    base = {"Rooms": 0, "Staff": 0, "Food": 0, "Other": 0}
    for k, v in raw.items():
        key = str(k).strip()
        if key in base:
            base[key] += int(v)
        else:
            base["Other"] += int(v)
    return base


_CATEGORIES_ORDER = ("Rooms", "Staff", "Food", "Other")
_MAX_EXCERPTS_PER_CATEGORY = 28
_MAX_EXCERPT_CHARS = 140
_MAX_INTERNAL_BRIEF_CHARS = 3200


def _build_internal_category_brief(neg_in_category: pd.DataFrame) -> str:
    """Server-side brief from review excerpts (not shown to end users)."""
    texts = neg_in_category["Review_Text"].dropna().astype(str).str.strip()
    texts = texts[texts != ""].tail(_MAX_EXCERPTS_PER_CATEGORY)
    lines = []
    for t in texts:
        one = " ".join(t.split())
        if len(one) > _MAX_EXCERPT_CHARS:
            one = one[: _MAX_EXCERPT_CHARS - 1].rsplit(" ", 1)[0] + "…"
        lines.append(f"- {one}")
    body = "\n".join(lines)
    if len(body) > _MAX_INTERNAL_BRIEF_CHARS:
        body = body[: _MAX_INTERNAL_BRIEF_CHARS - 1].rsplit("\n", 1)[0] + "\n…"
    return body


def _fallback_actions_for_category(category: str) -> list:
    fb = {
        "Food": [
            "Taste breakfast and hot items daily before service.",
            "Brief kitchen on portion size and holding temperatures.",
            "Track repeat food complaints in a simple log.",
        ],
        "Staff": [
            "Managers acknowledge issues with guests within 24 hours.",
            "Short refresher on check-in tone and problem escalation.",
            "Spot-check front desk interactions weekly.",
        ],
        "Rooms": [
            "Inspect reported rooms same day and fix or reassign.",
            "Deep-clean high-complaint room types this month.",
            "Verify housekeeping checklist against guest comments.",
        ],
        "Other": [
            "Log amenity issues (WiFi, parking, AC) in one place.",
            "Assign an owner and a target fix date per recurring issue.",
            "Tell guests what you fixed when they check out or by email.",
        ],
    }
    return list(fb.get(category, fb["Other"]))


def get_simplified_actions_from_claude(category_briefs: dict) -> dict:
    """
    One Claude call: category_briefs maps category name -> internal excerpt brief.
    Returns dict category -> list of short action strings.
    """
    if not category_briefs:
        return {c: [] for c in _CATEGORIES_ORDER}

    blocks = []
    for cat in _CATEGORIES_ORDER:
        brief = category_briefs.get(cat, "").strip()
        if brief:
            blocks.append(f"### {cat}\n{brief}")
    combined = "\n\n".join(blocks)

    prompt = f"""You are a hotel operations advisor. Below is an INTERNAL brief of guest feedback excerpts by category (staff only; not for guests).

{combined}

Some lines may be neutral, mixed, or mostly positive with a small nitpick—recommend proportionate, calm actions only where there is a real operational issue. Do not assume every line is a severe complaint or write alarmist remediation.

Respond ONLY with valid JSON — no markdown, no extra text. Use exactly these keys: "Rooms", "Staff", "Food", "Other".
For each key that had a section above, set a value to an array of 3 to 5 strings. Each string is ONE very short, simple sentence: a practical action for hotel management.
For keys with no section above, use an empty array [].
Example shape: {{"Rooms":["..."],"Staff":[],"Food":["..."],"Other":["..."]}}"""

    empty = {c: [] for c in _CATEGORIES_ORDER}
    try:
        text = claude_complete(prompt, max_tokens=900)
        text = re.sub(r"```json|```", "", text).strip()
        data = json.loads(text)
        out = {c: [] for c in _CATEGORIES_ORDER}
        for c in _CATEGORIES_ORDER:
            raw = data.get(c)
            if isinstance(raw, list):
                out[c] = [str(x).strip() for x in raw if str(x).strip()]
            elif isinstance(raw, str) and raw.strip():
                out[c] = [raw.strip()]
        # Ensure categories we sent to the model have something if model returned empty
        for c in category_briefs:
            if c in out and not out[c]:
                out[c] = _fallback_actions_for_category(c)
        return out
    except Exception:
        return {c: _fallback_actions_for_category(c) if c in category_briefs else [] for c in _CATEGORIES_ORDER}


# ──────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────

@app.route("/api/summary", methods=["GET"])
def get_summary():
    """Main dashboard metrics + category + severity breakdown."""
    try:
        timeframe = request.args.get("timeframe", "all")
        df  = load_preprocessed()
        cdf = load_clustered()
        
        df, cdf = apply_timeframe(df, cdf, timeframe)

        total = len(df)
        sent = _sentiment_bucket_series(df)
        pos = int((sent == "positive").sum())
        neg = int((sent == "negative").sum())
        sentiment_other = max(0, total - pos - neg)
        neg_df = df[sent == "negative"].copy().reset_index(drop=True)
        avg_len = (
            round(float(df["Review_Length"].mean()), 1) if total else 0.0
        )

        # Category counts must use the *same* negative rows as `neg` above. A clustered CSV
        # from a full historical run (10k+ rows) must not be counted when preproc only has 47 negatives.
        cdf_for_neg = cdf if len(cdf) == len(neg_df) else pd.DataFrame()
        neg_labeled = _assign_display_category(neg_df, cdf_for_neg)
        cat_counts = _normalize_category_counts(
            neg_labeled["display_category"].value_counts().to_dict()
        )

        # severity from preprocessed negative reviews (vectorized — large CSV safe)
        neg_kw = _neg_keyword_counts_series(neg_df["Review_Text"])
        sev = _severity_series(neg_kw, neg_df["Review_Length"])
        sev_counts = sev.value_counts().to_dict()

        return jsonify({
            "total"            : int(total),
            "positive"         : int(pos),
            "negative"         : int(neg),
            "sentiment_other"  : int(sentiment_other),
            "avg_length"       : avg_len,
            "pos_pct"          : round(pos / total * 100, 1) if total else 0.0,
            "neg_pct"          : round(neg / total * 100, 1) if total else 0.0,
            "categories"       : cat_counts,
            "severities"       : {
                "High"  : int(sev_counts.get("High",   0)),
                "Medium": int(sev_counts.get("Medium", 0)),
                "Low"   : int(sev_counts.get("Low",    0)),
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/category_actions", methods=["GET"])
def get_category_actions():
    """
    Build an internal brief from negative reviews per category, send one batched
    request to Claude for simplified actions. Summary is not returned to clients.
    """
    try:
        timeframe = request.args.get("timeframe", "all")
        df = load_preprocessed()
        cdf = load_clustered()
        df, cdf = apply_timeframe(df, cdf, timeframe)

        neg_df = df[_is_negative_sentiment_df(df)].copy().reset_index(drop=True)
        cdf_for_neg = cdf if len(cdf) == len(neg_df) else pd.DataFrame()
        neg_labeled = _assign_display_category(neg_df, cdf_for_neg)

        briefs = {}
        counts = {c: 0 for c in _CATEGORIES_ORDER}
        for cat in _CATEGORIES_ORDER:
            sub = neg_labeled[neg_labeled["display_category"] == cat]
            counts[cat] = int(len(sub))
            if len(sub) > 0:
                briefs[cat] = _build_internal_category_brief(sub)

        actions_by_cat = (
            get_simplified_actions_from_claude(briefs)
            if briefs
            else {c: [] for c in _CATEGORIES_ORDER}
        )

        payload = {}
        for cat in _CATEGORIES_ORDER:
            payload[cat] = {
                "actions": actions_by_cat.get(cat, []),
                "count": counts[cat],
            }
        return jsonify({"categories": payload})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reviews", methods=["GET"])
def get_reviews():
    """All reviews (positive + negative + other) for the table. Category filter keeps negatives in that
    bucket and still lists positive/other rows so the menu is not empty when most reviews are positive."""
    try:
        timeframe = request.args.get("timeframe", "all")
        category = request.args.get("category", "").strip()
        df  = load_preprocessed()
        cdf = load_clustered()

        df, cdf = apply_timeframe(df, cdf, timeframe)

        df_reset = df.reset_index(drop=True)
        if df_reset.empty:
            return jsonify({"reviews": [], "total": 0, "total_matching": 0})

        sent = _sentiment_bucket_series(df_reset)
        is_neg = sent == "negative"

        neg_only = df_reset[is_neg].reset_index(drop=True)
        neg_n = len(neg_only)
        cdf_ok = len(cdf) > 0 and len(cdf) == neg_n
        labeled_neg = _assign_display_category(neg_only, cdf if cdf_ok else pd.DataFrame())

        display_categories: list[str] = []
        severities: list[str] = []
        sentiments: list[str] = []
        neg_j = 0
        for i in range(len(df_reset)):
            buck = sent.iloc[i]
            if buck == "positive":
                sentiments.append("Positive")
                display_categories.append("Positive")
                severities.append("—")
            elif buck == "negative":
                sentiments.append("Negative")
                ln = labeled_neg.iloc[neg_j]
                display_categories.append(str(ln["display_category"]))
                rt = df_reset.iloc[i]["Review_Text"]
                rl = df_reset.iloc[i]["Review_Length"]
                nk = _neg_keyword_counts_series(pd.Series([rt])).iloc[0]
                severities.append(
                    _severity_series(pd.Series([nk]), pd.Series([rl])).iloc[0]
                )
                neg_j += 1
            else:
                sentiments.append("Other")
                display_categories.append("Other")
                severities.append("—")

        work_base = df_reset.copy()
        work_base["display_category"] = display_categories
        work_base["severity"] = severities
        work_base["sentiment"] = sentiments

        dc_ser = pd.Series(display_categories, index=work_base.index)
        if category in _REVIEW_TABLE_CATEGORIES:
            mask = (
                ((sent == "negative") & (dc_ser == category))
                | (sent == "positive")
                | (sent == "other")
            )
        else:
            mask = pd.Series(True, index=work_base.index)

        work = work_base.loc[mask].copy()
        total_matching = int(len(work))

        if total_matching > _REVIEWS_NEG_POOL:
            work = work.iloc[-_REVIEWS_NEG_POOL:].copy()
        work = work.iloc[::-1].head(100).reset_index(drop=True)

        reviews = []
        for i, row in work.iterrows():
            rs_cell = row.get("Reviewer_Score")
            rs_out = None
            if rs_cell is not None and not pd.isna(rs_cell):
                try:
                    rs_out = float(rs_cell)
                    if not math.isfinite(rs_out):
                        rs_out = None
                except (TypeError, ValueError):
                    rs_out = None
            reviews.append({
                "id"             : int(i),
                "review_text"    : str(row["Review_Text"])[:300],
                "tags"           : str(row.get("Tags", "leisure trip")),
                "category"       : str(row["display_category"]),
                "severity"       : str(row["severity"]),
                "sentiment"      : str(row["sentiment"]),
                "review_length"  : int(row["Review_Length"]),
                "reviewer_score" : rs_out,
                "status"         : "pending",
            })

        return jsonify({
            "reviews": reviews,
            "total": len(reviews),
            "total_matching": int(total_matching),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


_POSITIVE_REVIEW_SOLUTION = {
    "immediate_action": "No corrective action is required — sentiment for this review is positive.",
    "short_term_fix": "Thank the guest (reply publicly or privately) and share the praise with staff who helped.",
    "long_term_improvement": "Use strong reviews in onboarding and quality standards as examples of the target experience.",
    "guest_response": "Thank you so much for your kind words — we are delighted you enjoyed your stay and hope to welcome you back soon.",
    "department_responsible": "Guest relations",
    "estimated_resolution_time": "N/A",
    "prevention_tip": "Keep monitoring feedback themes; no issue remediation needed for this review.",
}


@app.route("/api/review/solution", methods=["POST"])
def get_solution():
    """Generate Claude API solution for a single review (complaint flow). Skips remediation if text+rating read as positive."""
    try:
        body        = request.get_json()
        review_text = body.get("review_text", "")
        category    = body.get("category", "Other")
        severity    = body.get("severity", "Medium")
        tags        = body.get("tags", "leisure trip")

        if not review_text:
            return jsonify({"error": "review_text required"}), 400

        rs = body.get("reviewer_score", body.get("rating"))
        if rs is not None and rs != "":
            try:
                rs = float(rs)
                if not math.isfinite(rs):
                    rs = None
            except (TypeError, ValueError):
                rs = None
        else:
            rs = None

        from solution_generator import build_feature_vector, sentiment_from_rating_and_lr

        X, _, _ = build_feature_vector(str(review_text).strip(), str(tags))
        sent_lab, _, _ = sentiment_from_rating_and_lr(X, rs)
        if sent_lab == "Positive":
            return jsonify({"solution": _POSITIVE_REVIEW_SOLUTION, "sentiment": "Positive"})

        result = get_solution_from_claude(review_text, category, severity, tags)
        if result["success"]:
            return jsonify({"solution": result["data"], "sentiment": "Negative"})
        else:
            return jsonify({"error": result["error"]}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sparkline", methods=["GET"])
def get_sparkline():
    """Daily review volume — simulated from row index buckets."""
    try:
        timeframe = request.args.get("timeframe", "all")
        df    = load_preprocessed()
        df, _ = apply_timeframe(df, pd.DataFrame(), timeframe)
        
        total = len(df)
        # split into 20 equal buckets to simulate daily volume
        bucket_size = max(total // 20, 1)
        buckets     = []
        for i in range(20):
            start   = i * bucket_size
            end     = start + bucket_size
            chunk   = df.iloc[start:end]
            neg_cnt = int(_is_negative_sentiment_df(chunk).sum())
            buckets.append({
                "day"   : i + 1,
                "total" : len(chunk),
                "negative": neg_cnt
            })
        return jsonify({"sparkline": buckets})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze_new_review():
    """Analyze a brand new review submitted from the website."""
    try:
        body        = request.get_json()
        review_text = body.get("review_text", "").strip()
        tags        = body.get("tags", "leisure trip")

        if not review_text:
            return jsonify({"error": "review_text required"}), 400

        classifier = get_model("classifier")
        if not classifier:
            return jsonify({"error": "Models not loaded. Please train the models first."}), 500

        cleaned  = clean_text(review_text)
        features, neg_c, rev_len = build_features(cleaned, tags.lower())

        label_encoder = get_model("label_encoder")
        if not label_encoder:
            return jsonify({"error": "Models not loaded. Please train the models first."}), 500

        from solution_generator import sentiment_from_rating_and_lr

        rs = body.get("reviewer_score", body.get("rating"))
        if rs is not None and rs != "":
            try:
                rs = float(rs)
            except (TypeError, ValueError):
                rs = None
        else:
            rs = None
        sentiment, confidence, sent_detail = sentiment_from_rating_and_lr(
            features, rs, clf=classifier, enc=label_encoder
        )
        lr_probs = sent_detail.get("lr_probs") or {}

        if sentiment == "Positive":
            return jsonify({
                "sentiment"  : "Positive",
                "confidence" : confidence,
                "lr_probs"   : lr_probs,
                "positivity_score": sent_detail.get("positivity_score_0_100"),
                "rating_rule": sent_detail.get("rating_rule"),
                "message"    : "Positive review — no action required"
            })

        severity = calculate_severity(neg_c, rev_len)
        category, cluster_id = predict_category(features)
        solution = get_solution_from_claude(review_text, category, severity, tags)

        return jsonify({
            "sentiment"   : "Negative",
            "confidence"  : confidence,
            "lr_probs"    : lr_probs,
            "positivity_score": sent_detail.get("positivity_score_0_100"),
            "rating_rule": sent_detail.get("rating_rule"),
            "category"    : category,
            "cluster_id"  : cluster_id,
            "severity"    : severity,
            "solution"    : solution.get("data") if solution["success"] else None,
            "error"       : solution.get("error")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/monthly_report", methods=["GET"])
def get_monthly_report():
    """Returns the processed monthly results."""
    try:
        api_dir = os.path.dirname(os.path.abspath(__file__))
        results_csv = os.path.join(api_dir, "results", "monthly_results.csv")
        if not os.path.exists(results_csv):
            return jsonify({"data": []})
        df = pd.read_csv(results_csv)
        return jsonify({"data": df.to_dict(orient="records")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/clear_reviews", methods=["POST"])
def clear_reviews():
    """Wipe review CSVs used by the dashboard and monthly batch outputs."""
    try:
        clear_all_review_data()
        global _preprocessed_cache, _preprocessed_cache_mtime
        global _clustered_cache, _clustered_cache_source, _clustered_cache_mtime
        _preprocessed_cache = None
        _preprocessed_cache_mtime = None
        _clustered_cache = None
        _clustered_cache_source = None
        _clustered_cache_mtime = None
        return jsonify({"success": True, "message": "All review data cleared."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload_monthly", methods=["POST"])
def upload_monthly():
    """Upload a CSV, save as monthly_reviews.csv, and run batch_processor.py"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        api_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(api_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        target_path = os.path.join(results_dir, "monthly_reviews.csv")
        file.save(target_path)

        # batch_processor appends to preproc/cluster; reset so this upload is not stacked on old data
        reset_preprocessed_and_cluster_csvs()

        subprocess.run(
            [sys.executable, "batch_processor.py"],
            cwd=api_dir,
            check=True,
            capture_output=True,
            text=True,
        )

        global _preprocessed_cache, _preprocessed_cache_mtime
        global _clustered_cache, _clustered_cache_source, _clustered_cache_mtime
        _preprocessed_cache = None
        _preprocessed_cache_mtime = None
        _clustered_cache = None
        _clustered_cache_source = None
        _clustered_cache_mtime = None

        rows_processed = 0
        results_csv = os.path.join(results_dir, "monthly_results.csv")
        if os.path.isfile(results_csv):
            try:
                rows_processed = int(len(pd.read_csv(results_csv)))
            except Exception:
                rows_processed = 0

        return jsonify(
            {
                "success": True,
                "message": "Batch processing complete",
                "rows_processed": rows_processed,
            }
        )
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or e.stdout or "").strip() or str(e)
        return jsonify({"error": f"Batch processing failed: {detail}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sync_from_mongo", methods=["POST"])
def sync_from_mongo():
    """
    Pull reviews from MongoDB (reviews collection), write a batch CSV, and run
    the same pipeline as monthly CSV upload (batch_processor → preprocessed + clustered CSVs).
    """
    try:
        df, mongo_stats = fetch_mongo_reviews_dataframe()
        api_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(api_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        target_path = os.path.join(results_dir, "mongo_sync_reviews.csv")
        df.to_csv(target_path, index=False, encoding="utf-8")

        reset_preprocessed_and_cluster_csvs()

        from batch_processor import run_monthly_batch

        run_monthly_batch(target_path, use_llm_solution=False)

        global _preprocessed_cache, _preprocessed_cache_mtime
        global _clustered_cache, _clustered_cache_source, _clustered_cache_mtime
        _preprocessed_cache = None
        _preprocessed_cache_mtime = None
        _clustered_cache = None
        _clustered_cache_source = None
        _clustered_cache_mtime = None

        rows_synced = int(len(df))
        rows_processed = 0
        results_csv = os.path.join(results_dir, "monthly_results.csv")
        if os.path.isfile(results_csv):
            try:
                rows_processed = int(len(pd.read_csv(results_csv)))
            except Exception:
                rows_processed = rows_synced

        return jsonify(
            {
                "success": True,
                "message": "MongoDB sync complete",
                "rows_synced": rows_synced,
                "rows_processed": rows_processed,
                **mongo_stats,
            }
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models": "loaded"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)