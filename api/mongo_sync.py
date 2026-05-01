"""
Load review documents from MongoDB and prepare the same CSV shape that batch_processor.run_monthly_batch expects.

During sync, sentiment matches the dashboard: DB rating thresholds (default rating <= 2 => Negative,
rating >= 3 => Positive) with LogisticRegression when the rating is between those values or missing.
Mongo score field is P(Positive) from LR × 100. Tune RATING_NEGATIVE_MAX / RATING_POSITIVE_MIN in api/.env.

After sync, complaint category is assigned by the local pipeline (TF-IDF + k-means + cluster map), not by any
category field stored in MongoDB. Sync runs the batch with use_llm_solution=False (rule-based actions only).

Configure via environment (api/.env):
  MONGODB_URI                 — required for sync (e.g. mongodb+srv://user:pass@cluster/...)
  MONGODB_DB                  — database name (default: hotel)
  MONGODB_REVIEWS_COLLECTION  — collection name (default: reviews)
  MONGODB_SYNC_ONLY_ACTIVE    — if true/1, only documents with status == \"active\"
  MONGODB_SYNC_FILTER         — optional JSON object merged into the find query (e.g. {\"hotelId\": \"...\"})
  MONGODB_REVIEW_TEXT_FIELD   — optional: single field name to use for review body (overrides auto-detect)
  MONGODB_RATING_FIELD        — optional: exact Mongo field for numeric rating (overrides auto-detect list)
  MONGODB_WRITE_LR_SENTIMENT  — if true/1 (default), update each doc with LR label + score
  MONGODB_LR_SENTIMENT_LABEL_FIELD   — Mongo field for label (default: sentimentLabel)
  MONGODB_LR_SENTIMENT_SCORE_FIELD   — Mongo field for score (default: sentimentalScore)
  MONGODB_LR_SENTIMENT_CONF_FIELD    — optional Mongo field for model confidence %% on predicted class
  RATING_NEGATIVE_MAX               — rating <= this => Negative (default 2)
  RATING_POSITIVE_MIN               — rating >= this => Positive (default 3)
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

# Tried in order until a non-empty string is found (your app may use `text`, `cleanedPreview`, etc.)
_REVIEW_TEXT_KEYS = (
    "text",
    "review_text",
    "reviewText",
    "content",
    "body",
    "full_review",
    "fullReview",
    "Full_Review",
    "cleanedPreview",
    "cleaned_preview",
    "review",
)

# Tried in order for numeric guest rating (sentiment rules + CSV reviewer_score).
_RATING_KEYS = (
    "rating",
    "stars",
    "starRating",
    "star_rating",
    "reviewerScore",
    "reviewer_score",
    "Reviewer_Score",
    "score",
    "guestRating",
    "overallRating",
    "overall_rating",
    "hotelRating",
)


def _rating_from_doc(doc: dict[str, Any]) -> float | None:
    """Parse rating from common field names. None if missing (do not guess — avoids false Positive)."""
    override = _strip_env(os.environ.get("MONGODB_RATING_FIELD"))
    keys = (override,) if override else ()
    keys = keys + _RATING_KEYS
    seen: set[str] = set()
    for key in keys:
        if not key or key in seen:
            continue
        seen.add(key)
        if key not in doc:
            continue
        raw = doc.get(key)
        if raw is None:
            continue
        s = str(raw).strip()
        if not s or s.lower() in {"nan", "none", "null", ""}:
            continue
        try:
            return float(s)
        except ValueError:
            continue
    return None


def _truthy(val: str | None) -> bool:
    if not val:
        return False
    return val.strip().lower() in ("1", "true", "yes", "on")


def _strip_env(val: str | None) -> str:
    if not val:
        return ""
    s = str(val).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s


def _review_body_from_doc(doc: dict[str, Any]) -> str:
    override = _strip_env(os.environ.get("MONGODB_REVIEW_TEXT_FIELD"))
    if override and override in doc:
        return str(doc.get(override) or "").strip()
    for key in _REVIEW_TEXT_KEYS:
        if key not in doc:
            continue
        raw = doc.get(key)
        if raw is None:
            continue
        s = str(raw).strip()
        if s:
            return s
    return ""


def _doc_to_row(doc: dict[str, Any]) -> dict[str, Any] | None:
    text = _review_body_from_doc(doc)
    if not text or len(text) < 2:
        return None
    bad = {"", "nan", "none", "null", "<na>"}
    if text.lower() in bad:
        return None

    tokens = doc.get("mlTokens")
    if isinstance(tokens, list) and tokens:
        tags = " ".join(str(t).strip() for t in tokens if str(t).strip()) or "leisure trip"
    else:
        tags = "leisure trip"

    reviewer_score = _rating_from_doc(doc)

    return {
        "review_text": text,
        "tags": tags,
        "reviewer_score": reviewer_score,
    }


def _mongo_find_filter() -> dict[str, Any]:
    q: dict[str, Any] = {}
    if _truthy(os.environ.get("MONGODB_SYNC_ONLY_ACTIVE")):
        q["status"] = "active"
    extra = os.environ.get("MONGODB_SYNC_FILTER", "").strip()
    if extra:
        try:
            parsed = json.loads(extra)
            if isinstance(parsed, dict):
                q.update(parsed)
            else:
                raise ValueError("MONGODB_SYNC_FILTER must be a JSON object")
        except json.JSONDecodeError as e:
            raise ValueError(f"MONGODB_SYNC_FILTER is not valid JSON: {e}") from e
    return q


def fetch_mongo_reviews_dataframe() -> tuple[pd.DataFrame, dict[str, Any]]:
    uri = _strip_env(os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URI"))
    if not uri:
        raise ValueError(
            "MONGODB_URI is not set. Add it to api/.env (next to app.py), e.g. "
            "MONGODB_URI=mongodb://localhost:27017/ or mongodb+srv://user:pass@cluster.mongodb.net/ "
            "Then restart the Flask process."
        )

    db_name = _strip_env(os.environ.get("MONGODB_DB")) or "hotel"
    coll_name = _strip_env(os.environ.get("MONGODB_REVIEWS_COLLECTION")) or "reviews"

    try:
        from pymongo import MongoClient
        from pymongo.errors import PyMongoError
        from pymongo.operations import UpdateOne
    except ImportError as e:
        raise RuntimeError(
            "pymongo is not installed. Run: pip install pymongo"
        ) from e

    client = MongoClient(uri, serverSelectionTimeoutMS=20000)
    try:
        client.admin.command("ping")
    except PyMongoError as e:
        client.close()
        raise RuntimeError(f"MongoDB connection failed: {e}") from e

    diag = ""
    filt: dict[str, Any] = {}
    docs: list[dict[str, Any]] = []
    coll = client[db_name][coll_name]
    try:
        filt = _mongo_find_filter()
        docs = list(coll.find(filt))
        if not docs:
            try:
                db = client[db_name]
                cols = db.list_collection_names()
                raw_count = db[coll_name].count_documents({})
                skip_dbs = frozenset({"admin", "local", "config"})
                other_dbs = [d for d in sorted(client.list_database_names()) if d not in skip_dbs]
                diag = (
                    f" Diagnostics: on database `{db_name}`, collection `{coll_name}` has "
                    f"count_documents({{}})={raw_count}. Collections in this DB: {cols}. "
                    f"Databases on this server (names are case-sensitive): {other_dbs[:25]}"
                    + ("…" if len(other_dbs) > 25 else "")
                    + ". If your data is under `simple_app` but .env says `Simple_app`, fix MONGODB_DB to match Compass exactly."
                )
            except Exception as ex:
                diag = f" (Could not inspect server: {ex})"
    except Exception:
        client.close()
        raise

    from solution_generator import build_feature_vector, sentiment_from_rating_and_lr

    write_lr = _truthy(os.environ.get("MONGODB_WRITE_LR_SENTIMENT", "true"))
    label_field = _strip_env(os.environ.get("MONGODB_LR_SENTIMENT_LABEL_FIELD")) or "sentimentLabel"
    score_field = _strip_env(os.environ.get("MONGODB_LR_SENTIMENT_SCORE_FIELD")) or "sentimentalScore"
    conf_field = _strip_env(os.environ.get("MONGODB_LR_SENTIMENT_CONF_FIELD"))

    rows: list[dict[str, Any]] = []
    bulk_ops: list[UpdateOne] = []

    for d in docs:
        row = _doc_to_row(d)
        if not row:
            continue
        oid = d.get("_id")
        lr_label = None
        lr_score = None
        lr_conf = None
        try:
            X, _, _ = build_feature_vector(row["review_text"], row["tags"])
            lr_label, lr_conf, detail = sentiment_from_rating_and_lr(
                X, row["reviewer_score"]
            )
            lr_score = detail.get("positivity_score_0_100")
            if lr_score is None and detail.get("lr_probs"):
                lr_score = round(
                    float(detail["lr_probs"].get("Positive", 0.0)) * 100.0, 2
                )
        except Exception:
            pass

        enriched = {
            **row,
            "lr_sentiment_label": lr_label,
            "lr_sentiment_score": lr_score,
            "lr_sentiment_confidence": lr_conf,
        }
        rows.append(enriched)

        if write_lr and oid is not None and lr_label is not None:
            to_set: dict[str, Any] = {label_field: lr_label}
            if lr_score is not None:
                to_set[score_field] = lr_score
            if conf_field and lr_conf is not None:
                to_set[conf_field] = lr_conf
            bulk_ops.append(UpdateOne({"_id": oid}, {"$set": to_set}))

    if write_lr and bulk_ops:
        try:
            coll.bulk_write(bulk_ops, ordered=False)
        except Exception as e:
            client.close()
            raise RuntimeError(f"MongoDB bulk update (LR sentiment) failed: {e}") from e

    client.close()

    if not rows:
        if not docs:
            raise ValueError(
                "MongoDB returned 0 documents. Check MONGODB_DB, MONGODB_REVIEWS_COLLECTION, "
                f"and find filter {filt!r}. If you use MONGODB_SYNC_ONLY_ACTIVE=true, documents must have status \"active\". "
                "Try MONGODB_SYNC_ONLY_ACTIVE=false in api/.env."
                + (f" {diag}" if diag else "")
            )
        sample = docs[0]
        keys = sorted(str(k) for k in sample.keys() if not str(k).startswith("_"))
        keys_hint = ", ".join(keys[:25]) + ("…" if len(keys) > 25 else "")
        raise ValueError(
            f"Found {len(docs)} document(s) but none had usable review text. "
            "The sync looks for text in fields such as: "
            + ", ".join(_REVIEW_TEXT_KEYS[:8])
            + ". "
            f"First document top-level keys: {keys_hint}. "
            "Set MONGODB_REVIEW_TEXT_FIELD=myFieldName in api/.env if your body field has another name, then restart Flask."
        )

    stats: dict[str, Any] = {
        "mongo_lr_updates": len(bulk_ops),
        "mongo_rows_scored": sum(1 for r in rows if r.get("lr_sentiment_label") is not None),
    }
    return pd.DataFrame(rows), stats
