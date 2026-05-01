# =========================================================
# solution_generator.py
# =========================================================

import os
import re
import json
import math
import joblib
import numpy as np
import scipy.sparse as sp
from dotenv import load_dotenv
from claude_client import claude_complete

load_dotenv()

from sklearn.preprocessing import normalize

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))

ROOT = os.path.dirname(BASE)

MODELS_DIR = os.path.join(ROOT,"models")
FEAT_DIR   = os.path.join(ROOT,"feature_files")


# ---------------------------------------------------------
# LOAD SAVED FILES
# ---------------------------------------------------------

print("Loading models...")

classifier = joblib.load(
    os.path.join(MODELS_DIR,"classifier.pkl")
)

kmeans_model = joblib.load(
    os.path.join(MODELS_DIR,"kmeans_model.pkl")
)

svd_reducer = joblib.load(
    os.path.join(MODELS_DIR,"svd_reducer.pkl")
)

cluster_map = joblib.load(
    os.path.join(MODELS_DIR,"cluster_label_mapping.pkl")
)

tfidf_vectorizer = joblib.load(
    os.path.join(FEAT_DIR,"tfidf_vectorizer.pkl")
)

tags_vectorizer = joblib.load(
    os.path.join(FEAT_DIR,"tags_vectorizer.pkl")
)

label_encoder = joblib.load(
    os.path.join(FEAT_DIR,"label_encoder.pkl")
)

length_stats = joblib.load(
    os.path.join(FEAT_DIR,"length_stats.pkl")
)

print("Models loaded successfully")


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

def clean_text(text):

    text=str(text).lower()

    text=re.sub(r"http\S+","",text)
    text=re.sub(r"[^a-zA-Z ]"," ",text)

    text=re.sub(r"\s+"," ",text).strip()

    return text


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

def build_feature_vector(review,tags):

    review=clean_text(review)
    tags=clean_text(tags)

    review_tfidf = tfidf_vectorizer.transform(
        [review]
    )

    tags_tfidf = tags_vectorizer.transform(
        [tags]
    )

    words=review.split()

    review_len=len(words)

    min_len=length_stats["min"]
    max_len=length_stats["max"]

    if max_len==min_len:
        norm_len=0
    else:
        norm_len=(review_len-min_len)/(max_len-min_len)

    negative_words=["dirty","rude","bad","worst","terrible","horrible"]
    positive_words=["clean","good","great","excellent","amazing","friendly"]

    neg_count=sum(1 for w in negative_words if w in review)
    pos_count=sum(1 for w in positive_words if w in review)
    sentiment_score = pos_count - neg_count

    numeric=sp.csr_matrix([[
        norm_len,
        neg_count,
        pos_count,
        sentiment_score
    ]])

    X=sp.hstack([
        review_tfidf,
        tags_tfidf,
        numeric
    ])

    return X,neg_count,review_len


# ---------------------------------------------------------
# SENTIMENT
# ---------------------------------------------------------

def predict_sentiment(X, clf=None, enc=None):
    """
    Sentiment from the trained LogisticRegression only (predict_proba).
    Does not blend reviewer_score or any separate 'sentiment score' heuristic.
    Returns: label, confidence_pct (max class prob * 100), probs_by_label (e.g. {"Negative": 0.82, "Positive": 0.18}).

    If predict_proba fails (e.g. sklearn version mismatch on pickled model), falls back to predict() +
    a sharp one-hot-ish distribution so rating+LR rules in sentiment_from_rating_and_lr still run.
    """
    clf = clf or classifier
    enc = enc or label_encoder
    classes = enc.classes_
    try:
        probs = clf.predict_proba(X)[0]
    except Exception:
        pred = int(clf.predict(X)[0])
        label = enc.inverse_transform([pred])[0]
        n = len(classes)
        probs = np.zeros(n, dtype=float)
        probs[pred] = 0.82
        for j in range(n):
            if j != pred:
                probs[j] = (1.0 - 0.82) / max(1, n - 1)
        probs_by_label = {str(classes[i]): float(probs[i]) for i in range(n)}
        return label, round(float(np.max(probs)) * 100.0, 2), probs_by_label

    pred = int(np.argmax(probs))
    label = enc.inverse_transform([pred])[0]
    confidence = float(np.max(probs)) * 100.0
    probs_by_label = {
        str(classes[i]): float(probs[i])
        for i in range(len(probs))
    }
    return label, round(confidence, 2), probs_by_label


def sentiment_from_rating_and_lr(X, reviewer_score, clf=None, enc=None):
    """
    Sentiment tag using the database rating plus LogisticRegression.

    - If rating <= RATING_NEGATIVE_MAX (default 2) -> label Negative (rating rule).
    - If rating >= RATING_POSITIVE_MIN (default 3) -> label Positive (rating rule).
    - If rating is strictly between those bounds, or missing/invalid -> label from LR only.

    Confidence for rule cases is LR probability for the chosen class (how much the model agrees).
    positivity_score_0_100 is always P(Positive) from LR * 100 for dashboards / Mongo score field.

    For 1–10 guest scores, set e.g. RATING_NEGATIVE_MAX=3 and RATING_POSITIVE_MIN=7 in api/.env.
    """
    label_lr, conf_lr, lr_probs = predict_sentiment(X, clf=clf, enc=enc)
    p_pos = float(lr_probs.get("Positive", 0.0))
    p_neg = float(lr_probs.get("Negative", max(0.0, 1.0 - p_pos)))
    positivity_lr = round(p_pos * 100.0, 2)

    r = None
    try:
        if reviewer_score is not None and not (
            isinstance(reviewer_score, float) and math.isnan(reviewer_score)
        ):
            r = float(reviewer_score)
            if not math.isfinite(r):
                r = None
    except (TypeError, ValueError):
        r = None

    neg_max = float(os.environ.get("RATING_NEGATIVE_MAX", "2"))
    pos_min = float(os.environ.get("RATING_POSITIVE_MIN", "3"))

    rating_rule = "lr_only"
    if r is not None:
        if r <= neg_max:
            label = "Negative"
            rating_rule = "rating_lte_neg_max"
            conf = round(p_neg * 100.0, 2)
        elif r >= pos_min:
            label = "Positive"
            rating_rule = "rating_gte_pos_min"
            conf = round(p_pos * 100.0, 2)
        else:
            label = label_lr
            conf = conf_lr
            rating_rule = "lr_between_thresholds"
    else:
        label = label_lr
        conf = conf_lr

    return (
        label,
        conf,
        {
            "lr_probs": lr_probs,
            "positivity_score_0_100": positivity_lr,
            "p_negative_fused": None,
            "rating_rule": rating_rule,
            "reviewer_rating_used": r,
        },
    )


def _rating_negative_prior(reviewer_score) -> float:
    """
    Star-driven P(negative) prior only (no text). Low stars -> higher value.
    Scale: auto (<=5 treated as 1-5 stars, else 1-10) or REVIEW_RATING_MAX=5|10 in env.
    """
    try:
        rs = float(reviewer_score)
    except (TypeError, ValueError):
        return 0.5
    mx = str(os.environ.get("REVIEW_RATING_MAX", "auto")).strip().lower()
    if mx == "5":
        rs = max(1.0, min(5.0, rs))
        p = (5.0 - rs) / 4.0
    elif mx == "10":
        rs = max(1.0, min(10.0, rs))
        p = (10.0 - rs) / 9.0
    else:
        if rs <= 5.5:
            rs = max(1.0, min(5.0, rs))
            p = (5.0 - rs) / 4.0
        else:
            rs = max(1.0, min(10.0, rs))
            p = (10.0 - rs) / 9.0
    return float(np.clip(p, 0.02, 0.98))


def fused_sentiment_from_models(X, reviewer_score, clf=None, enc=None):
    """
    Combine LogisticRegression(text) with star rating (same signal family as training:
    labels were score-based: >=7 Positive, <=4 Negative on a 10pt scale).

    Returns (label, confidence_pct, details) where details has lr_probs, positivity_score_0_100,
    and p_negative_fused. Pure text-only LR is often too harsh on upbeat but casual wording;
    fusion fixes many false negatives when the guest gave a high rating.

    Env: SENTIMENT_FUSION_TEXT_WEIGHT / SENTIMENT_FUSION_STAR_WEIGHT (defaults favor stars slightly).
    SENTIMENT_NEGATIVE_THRESHOLD: call Negative only if fused P(neg) >= this (default 0.5; try 0.55
    if you still see too many false negatives).
    """
    clf = clf or classifier
    enc = enc or label_encoder
    probs = clf.predict_proba(X)[0]
    classes = list(enc.classes_)
    lr_probs = {str(classes[i]): float(probs[i]) for i in range(len(classes))}

    if "Negative" not in classes or "Positive" not in classes:
        pred = int(np.argmax(probs))
        label = enc.inverse_transform([pred])[0]
        confidence = float(np.max(probs)) * 100.0
        return (
            label,
            round(confidence, 2),
            {
                "lr_probs": lr_probs,
                "positivity_score_0_100": None,
                "p_negative_fused": None,
            },
        )

    neg_i = classes.index("Negative")
    p_neg_text = float(np.clip(probs[neg_i], 1e-4, 1.0 - 1e-4))
    p_star = _rating_negative_prior(reviewer_score)
    p_star = float(np.clip(p_star, 1e-4, 1.0 - 1e-4))
    # Slightly star-heavy default: matches score-derived training labels and reduces false negatives.
    w_text = float(os.environ.get("SENTIMENT_FUSION_TEXT_WEIGHT", "0.45"))
    w_star = float(os.environ.get("SENTIMENT_FUSION_STAR_WEIGHT", "0.55"))
    s = w_text + w_star
    w_text, w_star = w_text / s, w_star / s
    z = w_text * np.log(p_neg_text / (1.0 - p_neg_text)) + w_star * np.log(
        p_star / (1.0 - p_star)
    )
    p_fused = 1.0 / (1.0 + np.exp(-z))
    thr = float(os.environ.get("SENTIMENT_NEGATIVE_THRESHOLD", "0.5"))
    thr = float(np.clip(thr, 0.01, 0.99))
    label = "Negative" if p_fused >= thr else "Positive"
    conf = max(p_fused, 1.0 - p_fused) * 100.0
    positivity = round((1.0 - p_fused) * 100.0, 2)
    return label, round(conf, 2), {
        "lr_probs": lr_probs,
        "positivity_score_0_100": positivity,
        "p_negative_fused": round(float(p_fused), 4),
    }


# ---------------------------------------------------------
# SEVERITY
# ---------------------------------------------------------

def calculate_severity(neg_count,length):

    score=0

    if neg_count>=4:
        score+=3

    elif neg_count>=2:
        score+=2

    elif neg_count>=1:
        score+=1

    if length>=50:
        score+=2

    elif length>=20:
        score+=1


    if score>=5:
        return "High"

    elif score>=3:
        return "Medium"

    else:
        return "Low"


# ---------------------------------------------------------
# CLUSTER PREDICTION (local k-means + mapping — no external API)
# ---------------------------------------------------------

def canonical_complaint_category(raw) -> str:
    """
    Map k-means cluster labels / noisy strings to exactly Food | Staff | Rooms | Other.
    """
    if raw is None:
        return "Other"
    s = str(raw).strip()
    if not s:
        return "Other"
    key = s.lower().replace("_", " ").strip()
    exact = {
        "food": "Food",
        "staff": "Staff",
        "room": "Rooms",
        "rooms": "Rooms",
        "housekeeping": "Rooms",
        "accommodation": "Rooms",
        "other": "Other",
        "general": "Other",
        "facilities": "Other",
        "wifi": "Other",
        "parking": "Other",
    }
    if key in exact:
        return exact[key]
    if any(w in key for w in ("food", "breakfast", "meal", "dining", "restaurant", "chef", "buffet")):
        return "Food"
    if any(w in key for w in ("staff", "service", "reception", "rude", "manager", "front desk")):
        return "Staff"
    if any(w in key for w in ("room", "bed", "bathroom", "noisy", "housekeep", "pillow", "mattress")):
        return "Rooms"
    return "Other"


def predict_category(X):

    dense=X.toarray()

    reduced=svd_reducer.transform(
        dense
    )

    reduced=normalize(reduced)

    cluster_id=int(
        kmeans_model.predict(reduced)[0]
    )

    category=cluster_map.get(
        cluster_id,
        "Other"
    )

    distances=kmeans_model.transform(
        reduced
    )[0]

    confidence=(
        1-distances[cluster_id]/
        distances.sum()
    )*100

    return (
        category,
        cluster_id,
        round(confidence,2)
    )


# Rule-based action templates (Mongo/batch can use these only — no LLM API)
_RULE_BASED_SOLUTIONS = {
    "Food": {
        "immediate_action": "Offer a complimentary meal replacement or discount.",
        "short_term_fix": "Review ingredient freshness and discuss with head chef.",
        "long_term_improvement": "Update the menu based on consistent guest feedback and train kitchen staff.",
        "guest_response": "We sincerely apologize for your dining experience and would love to invite you back for a complimentary meal.",
        "department_responsible": "Food & Beverage",
        "estimated_resolution_time": "24 hours",
        "prevention_tip": "Daily tasting audits before service begins.",
    },
    "Staff": {
        "immediate_action": "Manager to personally apologize to the guest.",
        "short_term_fix": "Identify the staff member and provide immediate corrective feedback.",
        "long_term_improvement": "Implement monthly hospitality and conflict-resolution training.",
        "guest_response": "We deeply regret the service you received. This does not reflect our standards and we are addressing it with our team.",
        "department_responsible": "Human Resources / Front Desk",
        "estimated_resolution_time": "Immediate",
        "prevention_tip": "Reward programs for staff who receive positive guest mentions.",
    },
    "Rooms": {
        "immediate_action": "Relocate guest to an upgraded room or dispatch housekeeping immediately.",
        "short_term_fix": "Block the room from future bookings until fully inspected and repaired.",
        "long_term_improvement": "Revamp the housekeeping checklist and schedule deep cleaning rotations.",
        "guest_response": "Please accept our apologies for the room conditions. We have escalated this to our Head Housekeeper.",
        "department_responsible": "Housekeeping / Maintenance",
        "estimated_resolution_time": "1-2 hours",
        "prevention_tip": "Supervisors must randomly audit 10% of cleaned rooms daily.",
    },
    "Other": {
        "immediate_action": "Acknowledge the specific issue and provide a tailored concession (e.g. free WiFi, waived parking).",
        "short_term_fix": "Log the complaint in the property management system for tracking.",
        "long_term_improvement": "Review amenity contracts and negotiate better terms with vendors (e.g. ISP).",
        "guest_response": "Thank you for bringing this to our attention. We are actively working to resolve this facility issue.",
        "department_responsible": "Operations",
        "estimated_resolution_time": "Variable",
        "prevention_tip": "Regular preventive maintenance schedules for all guest amenities.",
    },
}


def get_rule_based_solution(category: str) -> dict:
    c = canonical_complaint_category(category)
    return dict(_RULE_BASED_SOLUTIONS[c])


# ---------------------------------------------------------
# CLAUDE API SOLUTION GENERATOR (optional)
# ---------------------------------------------------------

def generate_solution(review_text, tags, category, severity, use_llm: bool = True):
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
    cat = canonical_complaint_category(category)
    if not use_llm:
        return get_rule_based_solution(cat)

    prompt = f"""You are an expert hotel management consultant.
A guest left a negative review.

Category : {cat}
Review   : {review_text}
Severity : {severity}
Tags     : {tags}

Context  : {cat_ctx.get(cat,'')}
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
        if not text or not str(text).strip():
            raise ValueError("empty model response")
        text = re.sub(r"```json|```", "", str(text)).strip()
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("JSON root must be an object")
        return data
    except Exception:
        return get_rule_based_solution(cat)


# ---------------------------------------------------------
# FULL PIPELINE
# ---------------------------------------------------------

def process_review(
    review_text,
    tags="leisure trip",
    reviewer_score=None,
    use_llm_solution: bool = True,
):

    result={

        "review_text":review_text,
        "tags":tags,
        "reviewer_score":reviewer_score,

        "sentiment":None,
        "sentiment_conf":None,
        "sentiment_lr_probs":None,
        "sentiment_positivity_score":None,

        "category":None,
        "category_conf":None,
        "cluster_id":None,

        "severity":None,

        "solution":None,
        "error":None
    }


    try:

        X,neg_count,length=build_feature_vector(
            review_text,
            tags
        )

        sentiment, conf, sent_detail = sentiment_from_rating_and_lr(X, reviewer_score)

        result["sentiment"] = sentiment
        result["sentiment_conf"] = conf
        result["sentiment_lr_probs"] = sent_detail.get("lr_probs")
        result["sentiment_positivity_score"] = sent_detail.get("positivity_score_0_100")


        if sentiment=="Positive":

            result["solution"]={
                "message":
                "Positive review. No action required."
            }

            return result


        severity=calculate_severity(
            neg_count,
            length
        )

        result["severity"]=severity


        category_raw,cluster_id,cconf=\
            predict_category(X)

        category = canonical_complaint_category(category_raw)
        result["category"]=category
        result["cluster_id"]=cluster_id
        result["category_conf"]=cconf


        result["solution"] = generate_solution(
            review_text,
            tags,
            category,
            severity,
            use_llm=use_llm_solution,
        )
        if not isinstance(result["solution"], dict):
            result["solution"] = _empty_solution_dict()


    except Exception as e:

        result["error"] = str(e)
        if not isinstance(result.get("solution"), dict):
            result["solution"] = _empty_solution_dict()


    return result


def _empty_solution_dict():
    return {
        "immediate_action": "",
        "short_term_fix": "",
        "long_term_improvement": "",
        "guest_response": "",
        "department_responsible": "",
        "estimated_resolution_time": "",
        "prevention_tip": "",
    }



# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__=="__main__":

    review="""
    Room dirty with mold in bathroom.
    Bed sheets terrible and cockroaches.
    """

    result=process_review(review)

    print(result)