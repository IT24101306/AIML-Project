import os
import time
import math
import pandas as pd
import numpy as np
from datetime import datetime

_PREPROC_COLS = [
    "Tags",
    "Review_Text",
    "Sentiment_Label",
    "Review_Length",
    "Category",
    "Reviewer_Score",
]
_CLUSTER_COLS = [
    "Tags",
    "Review_Text",
    "Sentiment_Label",
    "Review_Length",
    "Category",
    "predicted_cluster",
    "display_category",
]

from solution_generator import process_review


BASE=os.path.dirname(
    os.path.abspath(__file__)
)

# Default: process every row in the uploaded CSV so dashboard totals match the file (no silent cap).
# Set MONTHLY_BATCH_MAX_ROWS to a positive number to limit rows per upload (faster, but totals are partial).
_DEFAULT_MONTHLY_CAP = 0


def _normalize_sentiment_csv(val) -> str:
    """Canonical labels for dashboard CSV (must match _sentiment_bucket_series in app.py)."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    t = str(val).strip().lower()
    if t == "positive":
        return "Positive"
    if t == "negative":
        return "Negative"
    return ""


def _normalize_monthly_columns(df):
    """Map exports like Full_Review / Reviewer_Score to review_text / reviewer_score."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "review_text" not in df.columns:
        for alt in ("full_review", "review", "negative_review", "positive_review", "combined_review"):
            if alt in df.columns:
                df["review_text"] = df[alt]
                break
    if "review_text" not in df.columns:
        raise ValueError(
            "CSV needs a review text column (e.g. review_text, Full_Review, review)."
        )
    if "tags" not in df.columns:
        df["tags"] = "leisure trip"
    else:
        df["tags"] = df["tags"].fillna("leisure trip").astype(str)
    if "reviewer_score" not in df.columns:
        for alt in ("rating", "stars", "star_rating", "reviewer_rating"):
            if alt in df.columns:
                df["reviewer_score"] = df[alt]
                break
    if "reviewer_score" not in df.columns:
        df["reviewer_score"] = np.nan
    df["reviewer_score"] = pd.to_numeric(df["reviewer_score"], errors="coerce")
    default_rs = os.environ.get("BATCH_DEFAULT_REVIEWER_SCORE", "").strip()
    if default_rs:
        try:
            df["reviewer_score"] = df["reviewer_score"].fillna(float(default_rs))
        except ValueError:
            pass
    return df[["review_text", "tags", "reviewer_score"]]


def _drop_invalid_and_dedupe_reviews(df):
    """
    Strip empty / NaN review rows (common when Excel exports trailing blank lines).
    Remove duplicate reviews (same text + tags). Keeps first occurrence only.
    """
    if df.empty:
        return df
    out = df.copy()
    rt = out["review_text"].astype(str).str.strip()
    bad = {"", "nan", "none", "null", "<na>"}
    mask = rt.notna() & ~rt.str.lower().isin(bad) & (rt.str.len() >= 2)
    out = out.loc[mask].copy()
    out["review_text"] = rt.loc[out.index]

    out["_norm_text"] = (
        out["review_text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    out["_norm_tags"] = out["tags"].astype(str).str.strip().str.lower()
    before = len(out)
    out = out.drop_duplicates(subset=["_norm_text", "_norm_tags"], keep="first")
    out = out.drop(columns=["_norm_text", "_norm_tags"])
    dropped = before - len(out)
    if dropped:
        print(f"Deduped: removed {dropped} duplicate review row(s).")
    return out.reset_index(drop=True)

RESULTS=os.path.join(
    BASE,
    "results"
)

os.makedirs(
    RESULTS,
    exist_ok=True
)


def run_monthly_batch(input_csv, use_llm_solution=None):
    """
    use_llm_solution: if None, read BATCH_USE_LLM_SOLUTION env (default true).
    Mongo sync passes False so sentiment/category stay local-only (no Claude).
    """
    if use_llm_solution is None:
        v = str(os.environ.get("BATCH_USE_LLM_SOLUTION", "true")).lower()
        use_llm_solution = v in ("1", "true", "yes")

    # utf-8-sig strips Excel BOM; python engine is more forgiving of odd CSVs
    df = pd.read_csv(input_csv, encoding="utf-8-sig", engine="python")
    df = _normalize_monthly_columns(df)
    n_raw = len(df)
    df = _drop_invalid_and_dedupe_reviews(df)
    n_skip = n_raw - len(df)
    if n_skip:
        print(
            f"Skipped {n_skip} empty/invalid or duplicate row(s) in upload "
            f"(raw {n_raw} -> {len(df)} to process).",
            flush=True,
        )

    cap = int(os.environ.get("MONTHLY_BATCH_MAX_ROWS", str(_DEFAULT_MONTHLY_CAP)))
    if cap > 0 and len(df) > cap:
        print(
            f"NOTE: {len(df)} rows in file; processing first {cap} only "
            f"(MONTHLY_BATCH_MAX_ROWS={cap}). Remove or set 0 for full file."
        )
        df = df.iloc[:cap].copy().reset_index(drop=True)

    n_rows = len(df)
    throttle = 0.2 if n_rows <= 80 else (0.02 if n_rows <= 500 else 0.0)

    results=[]
    preproc_records=[]
    cluster_records=[]

    pos=0
    neg=0

    category_counts={
        "Food":0,
        "Staff":0,
        "Rooms":0,
        "Other":0
    }

    severity_counts={
        "High":0,
        "Medium":0,
        "Low":0
    }


    for i,row in df.iterrows():

        print(
          f"Processing {i+1}/{len(df)}"
        )

        rs_raw = row["reviewer_score"]
        if pd.isna(rs_raw):
            rs_pass = None
        else:
            try:
                rs_pass = float(rs_raw)
                if not math.isfinite(rs_pass):
                    rs_pass = None
            except (TypeError, ValueError):
                rs_pass = None

        r=process_review(
            row["review_text"],
            row["tags"],
            rs_pass,
            use_llm_solution=use_llm_solution,
        )

        sent = r.get("sentiment")
        if sent == "Positive":
            pos += 1
        elif sent == "Negative":
            neg += 1
            cat = r.get("category")
            if cat in category_counts:
                category_counts[cat] += 1
            sev = r.get("severity")
            if sev:
                severity_counts[sev] += 1
        else:
            # Pipeline error — do not label as Negative in exports (avoids false negatives in CSV).
            neg += 1
            sent = ""

        sent = _normalize_sentiment_csv(sent)

        sol = r.get("solution")
        if not isinstance(sol, dict):
            sol = {}

        review_len = len(str(r.get("review_text") or "").split())
        try:
            rs = float(row["reviewer_score"])
            if not math.isfinite(rs):
                rs = float("nan")
        except (TypeError, ValueError, KeyError):
            rs = float("nan")
        preproc_records.append({
            "Tags": r.get("tags") or "leisure trip",
            "Review_Text": r.get("review_text") or "",
            "Sentiment_Label": sent,
            "Review_Length": review_len,
            "Category": (r.get("category") or "Other"),
            "Reviewer_Score": rs,
        })

        if sent == "Negative":
            cluster_records.append({
                "Tags": r.get("tags") or "leisure trip",
                "Review_Text": r.get("review_text") or "",
                "Sentiment_Label": sent,
                "Review_Length": review_len,
                "Category": (r.get("category") or "Other"),
                "predicted_cluster": r.get("cluster_id", -1),
                "display_category": (r.get("category") or "Other"),
            })

        results.append({

            "review_text":
            r.get("review_text") or "",

            "sentiment":
            sent,

            "category":
            r.get("category"),

            "severity":
            r.get("severity"),

            "immediate_action":
            sol.get(
             "immediate_action"
            ),

            "short_term_fix":
            sol.get(
             "short_term_fix"
            ),

            "department":
            sol.get(
             "department_responsible"
            )
        })

        if throttle:
            time.sleep(throttle)

    output=os.path.join(
        RESULTS,
        "monthly_results.csv"
    )

    pd.DataFrame(
        results
    ).to_csv(
        output,
        index=False
    )

    # Overwrite dashboard CSVs from this batch only (exactly len(df) preproc rows; no append drift)
    ROOT = os.path.dirname(BASE)
    preproc_csv = os.path.join(ROOT, "hotel_reviews_preprocessed.csv")
    cluster_csv = os.path.join(ROOT, "negative_reviews_clustered.csv")

    if preproc_records:
        pd.DataFrame(preproc_records).to_csv(
            preproc_csv, index=False, encoding="utf-8"
        )
    else:
        pd.DataFrame(columns=_PREPROC_COLS).to_csv(
            preproc_csv, index=False, encoding="utf-8"
        )

    if cluster_records:
        pd.DataFrame(cluster_records).to_csv(
            cluster_csv, index=False, encoding="utf-8"
        )
    else:
        pd.DataFrame(columns=_CLUSTER_COLS).to_csv(
            cluster_csv, index=False, encoding="utf-8"
        )

    print("\nSaved:",output)

    print("\nCategory Counts")
    print(category_counts)

    print("\nSeverity Counts")
    print(severity_counts)



if __name__=="__main__":

    sample=os.path.join(
        RESULTS,
        "monthly_reviews.csv"
    )

    if not os.path.exists(sample):

        demo=pd.DataFrame([
        {
        "review_text":
        "Dirty room and mold in bathroom",
        "tags":"leisure trip",
        "reviewer_score":2
        },
        {
        "review_text":
        "Staff rude at reception",
        "tags":"business trip",
        "reviewer_score":3
        }
        ])

        demo.to_csv(
            sample,
            index=False
        )


    run_monthly_batch(sample)