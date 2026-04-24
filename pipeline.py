"""
pipeline.py

stage 1 - keyword identifier 
stage 2 - ML classifier 
stage 3 - LLM 

Notes: 
-
  - A log can only be flagged or moved to next stage
  - a log is only dismissed if ALL stages agree
  - Stage 1 flags are never overridden by later stages
  - False negatives are treated as critical failures

Output columns added to each input row:
    triage_decision   : 'FLAG' | 'CLEAR' | 'REVIEW'
    triage_stage      : which stage made the final decision
    triage_score      : float confidence (0-1)
    triage_reason     : human-readable explanation
    stage1_label      : int (-1=clear boilerplate, 0=pass, 1=flag)
    stage1_reason     : str
    clf_proba         : float (Stage 2 model probability)
    clf_label         : int (Stage 2 decision at threshold)
    llm_label         : int or None (Stage 3 LLM decision)
    llm_reason        : str or None (Stage 3 LLM explanation)
    matching_keywords : list[str] (keywords that triggered Stage 1)
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from features import normalize_text, apply_keyword_stage, ISSUE_KEYWORDS, ALL_ISSUE_KEYWORDS, engineer_features
from classifier import predict, build_feature_matrix, load_model

logger = logging.getLogger(__name__)


# Stage 2: below this → confident CLEAR; above → FLAG or REVIEW
CLEAR_THRESHOLD = 0.15
# Stage 2: above this → confident FLAG (skip LLM)
FLAG_THRESHOLD = 0.60

# LLM settings
LLM_MODEL = "mistral"          # Change to "llama3.1" or any Ollama model
LLM_TIMEOUT = 30               # seconds
LLM_MAX_LOG_CHARS = 600        # truncate very long logs for LLM input


def extract_matching_keywords(text: str) -> list[str]:
    """Return all issue keywords found in the text (for explainability)."""
    text_lower = text.lower()
    found = []
    for category, keywords in ISSUE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                found.append(f"{kw} [{category}]")
    return found



LLM_SYSTEM_PROMPT = """You are a wastewater and industrial facility risk analyst reviewing operator log entries.

Your task: Classify each log entry as ISSUE or ROUTINE.

ISSUE = any log describing:
- Equipment failures, faults, trips, or alarms that are active or unresolved
- Operational abnormalities (unusual readings, unexpected behavior)
- Infrastructure problems (leaks, clogs, blockages, bypasses)
- Maintenance actions triggered by a failure (work order created, parts ordered)
- Regulatory notifications or spill events
- Safety incidents or hazards
- Any situation requiring follow-up action

ROUTINE = any log describing:
- Normal daily operations, shift checks, rounds
- Completed scheduled maintenance with no findings
- Standard lab work, data entry, sample collection
- Equipment rotating normally as part of scheduled duty rotation

IMPORTANT: When in doubt, say ISSUE. Missing a real problem is far worse than a false alarm.

Respond with EXACTLY this format (nothing else):
CLASSIFICATION: ISSUE or ROUTINE
REASON: One sentence explaining your decision."""

FEW_SHOT_EXAMPLES = """
EXAMPLE 1:
Log: "blower 3 faulted on high temp, reset twice - still faulting. Created work order W12345."
CLASSIFICATION: ISSUE
REASON: Repeated fault with failed resets indicates a persistent equipment problem requiring maintenance intervention.

EXAMPLE 2:
Log: "Performed station check. All pumps running normally. Entered readings into SharePoint. No issues noted."
CLASSIFICATION: ROUTINE
REASON: Standard station check completed without any abnormal findings.

EXAMPLE 3:
Log: "Pump 4 OOS - still awaiting parts from supplier."
CLASSIFICATION: ISSUE
REASON: Equipment is out of service and the repair is pending, indicating an ongoing operational issue.

EXAMPLE 4:
Log: "Decanted sludge thickening tanks 1 and 2. Completed lab and entered data."
CLASSIFICATION: ROUTINE
REASON: Standard sludge handling and lab work with no reported problems.

EXAMPLE 5:
Log: "OCU and travelling screen remain OOS. Unable to decant STT - hose has multiple holes producing no flow."
CLASSIFICATION: ISSUE
REASON: Multiple pieces of equipment out of service and a hose failure preventing normal operations.

EXAMPLE 6:
Log: "0645 - Informed SAC of ongoing spill. Spoke with environmental officer assigned event #1QG1QRG."
CLASSIFICATION: ISSUE
REASON: Active spill with regulatory notification and assigned event number - serious environmental incident.
"""


def _call_ollama(text: str, model: str = LLM_MODEL) -> tuple[Optional[int], Optional[str]]:
    """
    Call local Ollama LLM for log classification.

    Returns: (label, reason) where label is 1=ISSUE, 0=ROUTINE, None=error
    """
    try:
        import requests
    except ImportError:
        logger.error("requests not available — install with: pip install requests")
        return None, "LLM unavailable (requests not installed)"

    truncated = text[:LLM_MAX_LOG_CHARS]
    if len(text) > LLM_MAX_LOG_CHARS:
        truncated += "... [truncated]"

    prompt = f"{FEW_SHOT_EXAMPLES}\n\nNow classify this log:\nLog: \"{truncated}\""

    payload = {
        "model": model,
        "system": LLM_SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 80},
    }

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=LLM_TIMEOUT,
        )
        response.raise_for_status()
        output = response.json().get("response", "").strip()

        # Parse the structured response
        label = None
        reason = None

        class_match = re.search(r"CLASSIFICATION:\s*(ISSUE|ROUTINE)", output, re.IGNORECASE)
        reason_match = re.search(r"REASON:\s*(.+?)(?:\n|$)", output, re.IGNORECASE)

        if class_match:
            label = 1 if class_match.group(1).upper() == "ISSUE" else 0
        if reason_match:
            reason = reason_match.group(1).strip()

        if label is None:
            # Fallback: if response contains ISSUE anywhere, flag it (conservative)
            label = 1 if "issue" in output.lower() else 0
            reason = f"Parsed from: {output[:100]}"

        return label, reason

    except requests.exceptions.ConnectionError:
        logger.warning("Ollama not running — Stage 3 unavailable. Install: https://ollama.ai")
        return None, "LLM unavailable (Ollama not running)"
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return None, f"LLM error: {str(e)[:80]}"


def _llm_available() -> bool:
    """Check if Ollama is available."""
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False
    

class TriagePipeline:
    """
    3-stage log pipeline 

    Usage:
        pipeline = TriagePipeline.from_model_dir("models/")
        results = pipeline.run(df_logs)
        flagged = results[results['triage_decision'] == 'FLAG']
    """

    def __init__(
        self,
        model: object,
        tfidf: object,
        threshold: float,
        use_llm: bool = True,
        llm_model: str = LLM_MODEL,
    ):
        self.model = model
        self.tfidf = tfidf
        self.threshold = threshold
        self.use_llm = use_llm and _llm_available()
        self.llm_model = llm_model

        if use_llm and not self.use_llm:
            logger.warning(
                "LLM requested but Ollama is not available. "
                "Stage 3 will pass uncertain logs directly to REVIEW. "
                "To enable: install Ollama (https://ollama.ai) and run: ollama pull %s",
                llm_model,
            )

    @classmethod
    def from_model_dir(cls, model_dir: str, **kwargs) -> "TriagePipeline":
        """Load a trained pipeline from saved model directory."""
        artifacts = load_model(model_dir)
        return cls(
            model=artifacts["model"],
            tfidf=artifacts["tfidf"],
            threshold=artifacts["threshold"],
            **kwargs,
        )

    def run(self, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        Run all pipeline stages on a DataFrame of logs.

        Input: DataFrame with at least LDTEXT column (and ideally all raw columns)
        Output: Same DataFrame with triage decision columns added.
        """
        df = df.copy()
        n = len(df)
        logger.info("Running triage pipeline on %d logs", n)

        # ---- Pre-processing ----
        if "ldtext_norm" not in df.columns:
            df["ldtext_norm"] = df["LDTEXT"].fillna("").apply(normalize_text)
        df = engineer_features(df)

        # Initialize output columns
        df["triage_decision"] = "REVIEW"
        df["triage_stage"] = ""
        df["triage_score"] = 0.5
        df["triage_reason"] = ""
        df["matching_keywords"] = df["ldtext_norm"].apply(
            lambda t: "; ".join(extract_matching_keywords(t))
        )
        df["llm_label"] = None
        df["llm_reason"] = None

        #
        df = apply_keyword_stage(df)

        # Stage 1 FLAG: critical keyword match
        s1_flag = df["stage1_label"] == 1
        df.loc[s1_flag, "triage_decision"] = "FLAG"
        df.loc[s1_flag, "triage_stage"] = "stage1_keyword"
        df.loc[s1_flag, "triage_score"] = 0.95
        df.loc[s1_flag, "triage_reason"] = (
            "Critical keyword match: " + df.loc[s1_flag, "matching_keywords"]
        )

        # Also always flag LOGENTRY_CONTINGENCY=1
        if "LOGENTRY_CONTINGENCY" in df.columns:
            contingency_flag = (df["LOGENTRY_CONTINGENCY"] == 1) & (df["triage_decision"] != "FLAG")
            df.loc[contingency_flag, "triage_decision"] = "FLAG"
            df.loc[contingency_flag, "triage_stage"] = "stage1_contingency"
            df.loc[contingency_flag, "triage_score"] = 1.0
            df.loc[contingency_flag, "triage_reason"] = "LOGENTRY_CONTINGENCY=1 (operator-marked event)"

        # Track remaining logs for Stage 2
        needs_stage2 = df["triage_decision"] == "REVIEW"
        n_s1_flagged = s1_flag.sum()
        logger.info(
            "Stage 1: %d flagged, %d pass to Stage 2",
            n_s1_flagged, needs_stage2.sum(),
        )


        #ML classifier
        if needs_stage2.sum() > 0:
            df_stage2 = df[needs_stage2].copy()
            df_stage2 = predict(df_stage2, self.model, self.tfidf, self.threshold)

            # Write back Stage 2 scores
            df.loc[needs_stage2, "clf_proba"] = df_stage2["clf_proba"].values
            df.loc[needs_stage2, "clf_label"] = df_stage2["clf_label"].values

            # Stage 2 FLAG: model says issue with reasonable confidence
            s2_flag_mask = needs_stage2 & (df["clf_proba"] >= FLAG_THRESHOLD)
            df.loc[s2_flag_mask, "triage_decision"] = "FLAG"
            df.loc[s2_flag_mask, "triage_stage"] = "stage2_classifier"
            df.loc[s2_flag_mask, "triage_score"] = df.loc[s2_flag_mask, "clf_proba"]
            df.loc[s2_flag_mask, "triage_reason"] = (
                "ML classifier: high issue probability ("
                + df.loc[s2_flag_mask, "clf_proba"].round(3).astype(str) + ")"
            )

            # Stage 2 CLEAR: model very confident it's routine AND no keywords
            s2_clear_mask = (
                needs_stage2
                & (df.get("clf_confident_clear", pd.Series(False, index=df.index)))
            )
            df.loc[s2_clear_mask, "triage_decision"] = "CLEAR"
            df.loc[s2_clear_mask, "triage_stage"] = "stage2_classifier"
            df.loc[s2_clear_mask, "triage_score"] = 1.0 - df.loc[s2_clear_mask, "clf_proba"]
            df.loc[s2_clear_mask, "triage_reason"] = (
                "ML classifier: low issue probability ("
                + df.loc[s2_clear_mask, "clf_proba"].round(3).astype(str) + "), no keywords"
            )

            n_s2_flagged = s2_flag_mask.sum()
            n_s2_cleared = s2_clear_mask.sum()
            logger.info(
                "Stage 2: %d flagged, %d cleared, %d uncertain → Stage 3",
                n_s2_flagged, n_s2_cleared,
                (df["triage_decision"] == "REVIEW").sum(),
            )

        # LLM classifier 

        needs_stage3 = df["triage_decision"] == "REVIEW"

        if needs_stage3.sum() > 0:
            if self.use_llm:
                logger.info("Stage 3: running LLM on %d uncertain logs", needs_stage3.sum())
                llm_labels = []
                llm_reasons = []

                for idx, row in df[needs_stage3].iterrows():
                    label, reason = _call_ollama(row["ldtext_norm"], model=self.llm_model)
                    llm_labels.append(label)
                    llm_reasons.append(reason)

                    if verbose:
                        decision = "FLAG" if label == 1 else ("CLEAR" if label == 0 else "REVIEW")
                        logger.debug(
                            "  LLM [%s]: %s | %s",
                            row.get("EVENTDATE", "?"),
                            decision,
                            str(reason)[:80],
                        )

                df.loc[needs_stage3, "llm_label"] = llm_labels
                df.loc[needs_stage3, "llm_reason"] = llm_reasons

                # LLM says ISSUE → FLAG
                llm_issue_mask = needs_stage3 & (df["llm_label"] == 1)
                df.loc[llm_issue_mask, "triage_decision"] = "FLAG"
                df.loc[llm_issue_mask, "triage_stage"] = "stage3_llm"
                df.loc[llm_issue_mask, "triage_score"] = 0.80
                df.loc[llm_issue_mask, "triage_reason"] = (
                    "LLM: " + df.loc[llm_issue_mask, "llm_reason"].fillna("")
                )

                # LLM says ROUTINE AND ML score was low → CLEAR
                llm_clear_mask = (
                    needs_stage3
                    & (df["llm_label"] == 0)
                    & (df.get("clf_proba", pd.Series(0.5, index=df.index)) < 0.35)
                )
                df.loc[llm_clear_mask, "triage_decision"] = "CLEAR"
                df.loc[llm_clear_mask, "triage_stage"] = "stage3_llm"
                df.loc[llm_clear_mask, "triage_score"] = 0.75
                df.loc[llm_clear_mask, "triage_reason"] = (
                    "LLM: " + df.loc[llm_clear_mask, "llm_reason"].fillna("")
                )

                # Remaining REVIEW: LLM error or still uncertain → always FLAG
                # (conservative: never leave uncertain as CLEAR)
                still_review = df["triage_decision"] == "REVIEW"
                df.loc[still_review, "triage_decision"] = "FLAG"
                df.loc[still_review, "triage_stage"] = "stage3_uncertain"
                df.loc[still_review, "triage_score"] = 0.50
                df.loc[still_review, "triage_reason"] = "Uncertain — flagged conservatively for human review"

            else:
                # No LLM: uncertain logs go to REVIEW queue (human review)
                # But also check the ML score and flag anything above lower threshold
                uncertain_flag = needs_stage3 & (
                    df.get("clf_proba", pd.Series(0.5, index=df.index)) >= self.threshold
                )
                df.loc[uncertain_flag, "triage_decision"] = "FLAG"
                df.loc[uncertain_flag, "triage_stage"] = "stage2_low_confidence"
                df.loc[uncertain_flag, "triage_score"] = df.loc[uncertain_flag, "clf_proba"]
                df.loc[uncertain_flag, "triage_reason"] = (
                    "ML: above threshold (no LLM available) — "
                    + df.loc[uncertain_flag, "clf_proba"].round(3).astype(str)
                )

                # True uncertain (below threshold, no LLM) → conservative FLAG
                still_review = df["triage_decision"] == "REVIEW"
                df.loc[still_review, "triage_decision"] = "FLAG"
                df.loc[still_review, "triage_stage"] = "stage2_uncertain"
                df.loc[still_review, "triage_score"] = 0.45
                df.loc[still_review, "triage_reason"] = "ML uncertain — flagged conservatively (no LLM)"

        decision_counts = df["triage_decision"].value_counts().to_dict()
        total = len(df)
        logger.info(
            "Pipeline complete: %d total | FLAG=%d (%.1f%%) | CLEAR=%d (%.1f%%) | REVIEW=%d",
            total,
            decision_counts.get("FLAG", 0),
            100 * decision_counts.get("FLAG", 0) / total,
            decision_counts.get("CLEAR", 0),
            100 * decision_counts.get("CLEAR", 0) / total,
            decision_counts.get("REVIEW", 0),
        )

        return df

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Evaluate pipeline performance on a labeled dataset.
        Computes recall, precision, false negative rate.
        """
        results = self.run(df)
        y_true = df["is_issue"].values
        y_pred = (results["triage_decision"] == "FLAG").astype(int).values

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())

        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        fnr = fn / max(tp + fn, 1)

        metrics = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "false_negative_rate": round(fnr, 4),
            "flags_sent_to_human": int(y_pred.sum()),
            "logs_cleared": int((y_pred == 0).sum()),
            "reduction_pct": round(100 * (y_pred == 0).sum() / max(len(y_pred), 1), 1),
        }

        if fn > 0:
            fn_logs = results[(y_true == 1) & (y_pred == 0)][
                ["LDTEXT", "ldtext_norm", "triage_stage", "clf_proba", "matching_keywords"]
            ]
            logger.warning("FALSE NEGATIVES (%d):", fn)
            for _, row in fn_logs.iterrows():
                logger.warning("  [%s] prob=%.3f kw=[%s]",
                               str(row.get("LDTEXT", ""))[:80],
                               row.get("clf_proba", 0),
                               row.get("matching_keywords", ""))

        return metrics
