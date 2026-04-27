"""
pipeline.py — keyword detecter.

keyword triage, single file.

To add a keyword: add a string to KEYWORDS below, then rerun.
To remove a noisy keyword: delete it from KEYWORDS.
"""

import re
import pandas as pd
from io import BytesIO

#change this to tune this filter
KEYWORDS = [
    #equipment 
    "oos", "out of service", "fault", "faulted", "alarm", "tripped",
    "failed", "failure", "not working", "overtorque", "overheating",
    "blown", "seized", "broken", "damaged", "misalignment",
    #maintenance
    "work order", "replacement", "troubleshoot", "repair",
    "remove for service", "for service", "on route", "corroded",
    "replace", "install", "commission", "out of alignment",
    #problem events
    "overflow", "spill", "bypass", "leak", "clog", "clogged",
    "communications failure", "power fault", "generator fault",
    "electrical storm", "diverted", "isolated",
]


def _normalize(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace("_x000d_", " ").replace("\\r\\n", " ").replace("\r\n", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def _flag(text: str) -> tuple[bool, str]:
    """Returns (is_flagged, first_matched_keyword)."""
    norm = _normalize(text)
    for kw in KEYWORDS:
        if kw in norm:
            return True, kw
    return False, ""


def run(df: pd.DataFrame) -> pd.DataFrame:
    """Run keyword triage. Returns df with triage columns added."""
    df = df.copy()
    decisions, keywords = [], []
    for text in df["LDTEXT"].fillna(""):
        is_flagged, matched = _flag(text)
        decisions.append("FLAG" if is_flagged else "CLEAR")
        keywords.append(matched)
    df["triage_decision"] = decisions
    df["matched_keyword"] = keywords
    return df


def to_excel(results: pd.DataFrame, prefix: str) -> bytes:
    """Build output Excel: single sheet with all flagged issues only."""
    flagged  = results[results["triage_decision"] == "FLAG"].copy()
    out_cols = [c for c in ["EVENTDATE", "DESCRIPTION", "PERSONGROUP", "LDTEXT", "matched_keyword"]
                if c in flagged.columns]
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        (flagged[out_cols]
            .sort_values(["DESCRIPTION", "EVENTDATE"], na_position="last")
            .to_excel(writer, sheet_name="Predicted Issues", index=False))
    return buf.getvalue()