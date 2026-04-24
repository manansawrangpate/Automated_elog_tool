"""
trains a recall-based classifier. 
- logistic regression on tf-idf w/ engineered features
- timeseries cross validation
- set thresold is 0.97
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from scipy.sparse import hstack, csr_matrix
import joblib

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

#target min recall where decision threshold is set
TARGET_RECALL = 0.97

#engineered features
ENGINEERED_FEATURE_COLS = [
    "feat_text_len",
    "feat_word_count",
    "feat_line_count",
    "feat_has_numbers",
    "feat_has_worder",
    "feat_has_timestamp",
    "feat_kw_critical",
    "feat_kw_equipment_failure",
    "feat_kw_maintenance_action",
    "feat_kw_regulatory",
    "feat_kw_control_system",
    "feat_kw_infrastructure",
    "feat_kw_safety",
    "feat_kw_total_categories",
    "feat_any_issue_keyword",
    "feat_negated",
    "feat_contingency",
    "feat_is_remote",
    "feat_regulatory",
    "feat_escalation",
    "feat_oos",
    "feat_alarm",
    "feat_repeated_fault",
]


def build_feature_matrix(df: pd.DataFrame, tfidf: Optional[TfidfVectorizer] = None, fit: bool = False):
    """
    Build a combined feature matrix from TF-IDF text features +
    engineered numerical features.

    Parameters
    ----------
    df     : DataFrame with ldtext_norm and all feat_* columns
    tfidf  : existing TfidfVectorizer (None = create new)
    fit    : if True, fit the vectorizer on this data

    Returns: (X_sparse, tfidf_vectorizer)
    """
    # TF-IDF on normalized log text
    texts = df["ldtext_norm"].fillna("").tolist()

    if tfidf is None:
        tfidf = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=2,
            max_features=8000,
            sublinear_tf=True,
            analyzer="word",
            token_pattern=r"(?u)\b[a-z][a-z0-9]{1,}\b",
        )

    if fit:
        X_tfidf = tfidf.fit_transform(texts)
    else:
        X_tfidf = tfidf.transform(texts)

    # Engineered features (fill missing cols with 0)
    eng_cols = [c for c in ENGINEERED_FEATURE_COLS if c in df.columns]
    X_eng = df[eng_cols].fillna(0).values.astype(np.float32)
    X_eng_sparse = csr_matrix(X_eng)

    # Combine
    X = hstack([X_tfidf, X_eng_sparse])
    return X, tfidf



def find_recall_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_recall: float = TARGET_RECALL,
) -> float:
    """
    threshold finder (tries to find highest threshold that NEVER misses a single issue)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # Find all thresholds where recall >= target
    valid = [
        (t, p, r)
        for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds)
        if r >= target_recall
    ]

    if not valid:
        logger.warning(
            "Cannot achieve recall=%.2f — using lowest available threshold 0.10",
            target_recall,
        )
        return 0.10

    # Among valid thresholds, pick the one with highest precision
    best = max(valid, key=lambda x: x[1])
    logger.info(
        "Threshold %.3f achieves recall=%.3f, precision=%.3f",
        best[0], best[2], best[1],
    )
    return best[0]



def train_classifier(
    df: pd.DataFrame,
    save_dir: Optional[str] = None,
    n_cv_splits: int = 3,
) -> dict:
    """
    training a logistic regression classifier 

    df: dataframe
    save_dir: save model here
    n_cv_splits : number of TimeSeriesSplit folds

    Returns dict with keys: model, tfidf, threshold, metrics, feature_importances
    """
    logger.info("Starting classifier training on %d samples", len(df))

    # Sort by date for temporal integrity
    if "date_str" in df.columns:
        df = df.sort_values("date_str").reset_index(drop=True)

    y = df["is_issue"].values
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    logger.info("Class distribution: %d issues, %d routine (ratio 1:%.0f)",
                pos_count, neg_count, neg_count / max(pos_count, 1))

    # Build feature matrix (fit vectorizer on all data — ok since we're not
    # leaking labels, only vocabulary)
    X, tfidf = build_feature_matrix(df, fit=True)

    # Class weights — heavily favor the positive (issue) class
    # Ratio-based: if 1% issues, weight = 99; cap at 50 for stability
    raw_weight = min(neg_count / max(pos_count, 1), 50)
    class_weight = {0: 1, 1: raw_weight}
    logger.info("Using class_weight: %s", class_weight)

    # Base classifier
    base_clf = LogisticRegression(
        C=1.0,
        class_weight=class_weight,
        max_iter=1000,
        solver="saga", n_jobs=1,
        random_state=42,
    )

    # Calibrated wrapper for reliable probability scores
    clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=3)

    # --------------- Time-series cross-validation ---------------
    tscv = TimeSeriesSplit(n_splits=n_cv_splits)
    cv_recalls = []
    cv_precisions = []
    cv_aucs = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if y_val.sum() == 0:
            logger.warning("Fold %d: no positive samples in validation set, skipping", fold)
            continue

        clf.fit(X_train, y_train)
        y_proba_val = clf.predict_proba(X_val)[:, 1]
        threshold = find_recall_threshold(y_val, y_proba_val)
        y_pred_val = (y_proba_val >= threshold).astype(int)

        report = classification_report(y_val, y_pred_val, output_dict=True, zero_division=0)
        recall = report.get("1", {}).get("recall", 0)
        precision = report.get("1", {}).get("precision", 0)
        auc = roc_auc_score(y_val, y_proba_val)

        cv_recalls.append(recall)
        cv_precisions.append(precision)
        cv_aucs.append(auc)

        logger.info(
            "Fold %d | recall=%.3f precision=%.3f AUC=%.3f threshold=%.3f | val_size=%d pos=%d",
            fold, recall, precision, auc, threshold, len(y_val), y_val.sum(),
        )

    # --------------- Final model on all data ---------------
    clf.fit(X, y)
    y_proba_all = clf.predict_proba(X)[:, 1]
    final_threshold = find_recall_threshold(y, y_proba_all)

    y_pred_final = (y_proba_all >= final_threshold).astype(int)
    final_report = classification_report(y, y_pred_final, output_dict=True, zero_division=0)

    # Feature importances from the underlying logistic regression
    # (grab from the last calibrated estimator)
    try:
        base = clf.calibrated_classifiers_[-1].estimator
        coefs = base.coef_[0]
        feature_names = list(tfidf.get_feature_names_out()) + [
            c for c in ENGINEERED_FEATURE_COLS if c in df.columns
        ]
        importance_df = pd.DataFrame({
            "feature": feature_names[:len(coefs)],
            "coefficient": coefs,
        }).sort_values("coefficient", ascending=False)
    except Exception:
        importance_df = pd.DataFrame()

    metrics = {
        "cv_recall_mean": np.mean(cv_recalls) if cv_recalls else None,
        "cv_recall_std": np.std(cv_recalls) if cv_recalls else None,
        "cv_precision_mean": np.mean(cv_precisions) if cv_precisions else None,
        "cv_auc_mean": np.mean(cv_aucs) if cv_aucs else None,
        "final_recall": final_report.get("1", {}).get("recall"),
        "final_precision": final_report.get("1", {}).get("precision"),
        "final_f1": final_report.get("1", {}).get("f1-score"),
        "threshold": final_threshold,
        "n_train": len(df),
        "n_issues": int(pos_count),
    }

    logger.info("Final model metrics: %s", metrics)

    result = {
        "model": clf,
        "tfidf": tfidf,
        "threshold": final_threshold,
        "metrics": metrics,
        "feature_importances": importance_df,
    }

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, save_path / "classifier.joblib")
        joblib.dump(tfidf, save_path / "tfidf.joblib")
        joblib.dump(final_threshold, save_path / "threshold.joblib")
        if not importance_df.empty:
            importance_df.to_csv(save_path / "feature_importances.csv", index=False)
        logger.info("Model artifacts saved to %s", save_path)

    return result

def predict(
    df: pd.DataFrame,
    model: CalibratedClassifierCV,
    tfidf: TfidfVectorizer,
    threshold: float,
) -> pd.DataFrame:
    """
    run classifier on the dataframe

    return dataframe with new columns
    - clf_proba    : float  (probability of being an issue)
    - clf_label    : int    (1=issue, 0=routine at given threshold) 
    - clf_confident_clear : bool (True only if model is very confident it's routine)
    """
    df = df.copy()
    X, _ = build_feature_matrix(df, tfidf=tfidf, fit=False)
    proba = model.predict_proba(X)[:, 1]

    df["clf_proba"] = proba
    df["clf_label"] = (proba >= threshold).astype(int)

    # Only clear a log if the model is very confident it's routine.
    # Threshold is intentionally low (0.05) — we prefer false positives over false negatives.
    # Also require: no issue keywords fired, no contingency flag, not very short text
    # (short logs are harder to classify and should be reviewed)
    df["clf_confident_clear"] = (
        (proba < 0.01)
        & (df.get("feat_any_issue_keyword", pd.Series(0, index=df.index)) == 0)
        & (df.get("stage1_label", pd.Series(0, index=df.index)) != 1)
        & (df.get("feat_word_count", pd.Series(999, index=df.index)) >= 8)
        & (df.get("feat_contingency", pd.Series(0, index=df.index)) == 0)
    )

    return df


def load_model(model_dir: str) -> dict:
    """Load saved model artifacts from disk."""
    model_dir = Path(model_dir)
    return {
        "model": joblib.load(model_dir / "classifier.joblib"),
        "tfidf": joblib.load(model_dir / "tfidf.joblib"),
        "threshold": joblib.load(model_dir / "threshold.joblib"),
    }


def get_top_issue_terms(tfidf: TfidfVectorizer, model: CalibratedClassifierCV, n: int = 50) -> list[str]:
    """Return the top N terms most associated with ISSUE class."""
    try:
        base = model.calibrated_classifiers_[-1].estimator
        coefs = base.coef_[0]
        names = tfidf.get_feature_names_out()
        n_tfidf = len(names)
        tfidf_coefs = coefs[:n_tfidf]
        top_idx = np.argsort(tfidf_coefs)[-n:][::-1]
        return [names[i] for i in top_idx]
    except Exception:
        return []
