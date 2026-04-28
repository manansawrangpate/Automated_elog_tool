"""
pipeline.py — keyword pre-filter + Complement Naive Bayes classifier.

Two stages:
  1. Keyword check — instant flag for obvious issue terms
  2. ML classifier  — TF-IDF + Complement Naive Bayes on everything else

To tune sensitivity: adjust THRESHOLD below.
  Lower  -> more flags, fewer misses  (more conservative)
  Higher -> fewer flags, more misses  (less conservative)

"""

import re
import pickle
import pandas as pd
import numpy as np
from io import BytesIO
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

#change these 2 things 
THRESHOLD = 0.15   # probability above which a log is flagged

#keyword filter
KEYWORDS = [
    "oos", "out of service", "fault", "faulted", "alarm", "tripped",
    "failed", "failure", "not working", "overtorque", "overheating",
    "blown", "seized", "broken", "damaged", "misalignment",
    "work order", "replacement", "troubleshoot", "repair",
    "remove for service", "for service", "on route", "corroded",
    "replace", "install", "commission", "out of alignment",
    "overflow", "spill", "bypass", "leak", "clog", "clogged",
    "communications failure", "power fault", "generator fault", "fault"
]

MODEL_PATH = Path(__file__).parent / "model.pkl"


#helpers

def _normalize(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace("_x000d_", " ").replace("\\r\\n", " ").replace("\r\n", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def _keyword_hit(text: str) -> tuple[bool, str]:
    norm = _normalize(text)
    for kw in KEYWORDS:
        if kw in norm:
            return True, kw
    return False, ""


# classifier

def train(texts: list, labels: list) -> Pipeline:
    """Train and return a TF-IDF + ComplementNB pipeline."""
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=5000,
            sublinear_tf=True,
        )),
        ("clf", ComplementNB(alpha=0.05)),
    ])
    model.fit([_normalize(t) for t in texts], labels)
    return model


def save_model(model: Pipeline) -> None:
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)


def load_model():
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None


#2 stage pipeline

def run(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """
    Run triage on a DataFrame.

    Stage 1: keyword check  -> FLAG immediately
    Stage 2: ML classifier  -> FLAG if probability >= THRESHOLD
                            -> CLEAR otherwise

    If no model is provided, falls back to keyword-only (flags uncertain logs).
    """
    df = df.copy()
    decisions, keywords, probas, stages = [], [], [], []

    for text in df["LDTEXT"].fillna(""):
        hit, kw = _keyword_hit(text)
        if hit:
            decisions.append("FLAG")
            keywords.append(kw)
            probas.append(1.0)
            stages.append("keyword")
        else:
            decisions.append(None)
            keywords.append("")
            probas.append(None)
            stages.append(None)

    if model is not None:
        # run ML on logs not already flagged by keywords
        needs_ml = [i for i, d in enumerate(decisions) if d is None]
        if needs_ml:
            ml_texts = [_normalize(df["LDTEXT"].fillna("").iloc[i]) for i in needs_ml]
            ml_proba = model.predict_proba(ml_texts)[:, 1]
            for j, i in enumerate(needs_ml):
                p = float(ml_proba[j])
                probas[i] = round(p, 4)
                if p >= THRESHOLD:
                    decisions[i] = "FLAG"
                    stages[i]    = "ml"
                else:
                    decisions[i] = "CLEAR"
                    stages[i]    = "ml"
    else:
        # no model — flag conservatively rather than silently clear
        for i, d in enumerate(decisions):
            if d is None:
                decisions[i] = "FLAG"
                stages[i]    = "keyword_fallback"
                probas[i]    = 0.0

    df["triage_decision"] = decisions
    df["matched_keyword"] = keywords
    df["ml_probability"]  = probas
    df["triage_stage"]    = stages
    return df


# output

def to_excel(results: pd.DataFrame, prefix: str) -> bytes:
    """Single sheet with all flagged issues, sorted by site then date."""
    flagged  = results[results["triage_decision"] == "FLAG"].copy()
    out_cols = [c for c in [
        "EVENTDATE", "DESCRIPTION", "PERSONGROUP",
        "LDTEXT", "matched_keyword", "ml_probability", "triage_stage",
    ] if c in flagged.columns]
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        (flagged[out_cols]
            .sort_values(["DESCRIPTION", "EVENTDATE"], na_position="last")
            .to_excel(writer, sheet_name="Predicted Issues", index=False))
    return buf.getvalue()