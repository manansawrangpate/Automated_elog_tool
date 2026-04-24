"""
features.py

normalize text, abbreviation dictionary (built manually), feature engineering
"""

import re
from typing import Optional

import pandas as pd

# Abbreviation dictionary
#add to this as needed

ABBREV_MAP: dict[str, str] = {
    # Operator shorthand
    "wo": "work order",
    "wos": "work orders",
    "oos": "out of service",
    "o/s": "out of service",
    "nb": "note",
    "chkd": "checked",
    "chk": "check",
    "lvl": "level",
    "lvls": "levels",
    "req'd": "required",
    "req": "required",
    "maint": "maintenance",
    "mx": "maintenance",
    "insp": "inspection",
    "inspt": "inspected",
    "ops": "operations",
    "op": "operator",
    "stn": "station",
    "svc": "service",
    "avail": "available",
    "unavail": "unavailable",
    "temp": "temperature",
    "temps": "temperatures",
    "equip": "equipment",
    "elec": "electrical",
    "mech": "mechanical",
    "inst": "instrumentation",
    "cntrl": "control",
    "ctrl": "control",
    "pres": "pressure",
    "diff": "differential",
    "iso": "isolated",
    "isol": "isolated",
    "rplc": "replace",
    "repl": "replace",
    "rep": "replace",
    "ord": "ordered",
    "rcvd": "received",
    "deliv": "delivered",
    "sched": "scheduled",
    "compl": "completed",
    "comp": "completed",
    "ok": "normal",
    "n/a": "not applicable",
    "tbd": "to be determined",
    "asap": "as soon as possible",
    "approx": "approximately",
    "w/": "with",
    "w/o": "without",
    # Equipment shorthand
    "blwr": "blower",
    "blwrs": "blowers",
    "trp": "trip",
    "trpd": "tripped",
    "flt": "fault",
    "flts": "faults",
    "fltd": "faulted",
    "alm": "alarm",
    "alms": "alarms",
    "scada": "scada",
    "ocu": "ocu",
    "uvd": "uv disinfection",
    "uv": "uv",
    "roc": "remote operations center",
    "pcs": "process control system",
    "mcc": "motor control center",
    "vfd": "variable frequency drive",
    "ao": "automatic operation",
    "ato": "automatic operation",
    "man": "manual",
    "manl": "manual",
    "auto": "automatic",
    "hmi": "human machine interface",
    "plc": "programmable logic controller",
    "io": "input output",
    "stt": "sludge thickening tank",
    "sht": "sludge holding tank",
    "was": "waste activated sludge",
    "ras": "return activated sludge",
    "mlss": "mixed liquor suspended solids",
    "do": "dissolved oxygen",
    "bod": "biochemical oxygen demand",
    "tss": "total suspended solids",
    "eff": "effluent",
    "infl": "influent",
    "sps": "sewage pumping station",
    "wrrf": "water resource recovery facility",
    "wwtp": "wastewater treatment plant",
    "ehs": "environmental health and safety",
    "lscrca": "lake simcoe conservation authority",
    "sac": "spill action center",
    "ceww": "central east wastewater",
    "wwn": "wastewater north",
    "wws": "wastewater south",
    "wwc": "wastewater central",
    "pm": "preventive maintenance",
    "cse": "confined space entry",
    "jhsc": "joint health safety committee",
    "loto": "lockout tagout",
    "ekey": "electronic key",
    "oic": "operator in charge",
    "oit": "operator in training",
    "tl": "team lead",
    "supr": "supervisor",
    "tech": "technician",
    "elec": "electrician",
    "n/s": "not in service",
    "n/r": "not responding",
    "n/f": "not functioning",
    "c/o": "changeover",
    "f/u": "follow up",
    "f/w": "follow with",
    "r/s": "reset",
    "rst": "reset",
    "rstd": "reset",
    "clrd": "cleared",
    "clr": "cleared",
    "notif": "notified",
    "notfd": "notified",
    "ntfd": "notified",
    "wkng": "working",
    "rng": "running",
    "run": "running",
    "dn": "down",
    "dwn": "down",
    "hr": "hour",
    "hrs": "hours",
    "min": "minutes",
    "mins": "minutes",
    "yr": "year",
    "mo": "month",
}

#false positive patterns like "no leak", "no issyes" 
NEGATION_PATTERNS = re.compile(
    r"\b(no|not|none|without|cleared|resolved|okay|ok|normal|"
    r"completed without|no issues|all clear|confirmed ok|"
    r"checked .{0,20} ok|inspected .{0,20} ok)\b",
    re.IGNORECASE,
)

# Issue-related keywords for feature engineering
ISSUE_KEYWORDS: dict[str, list[str]] = {
    "critical": [
        "overflow", "spill", "bypass", "sewage release", "sanitary overflow",
        "sso", "emergency", "lightning strike", "flood", "explosion",
        "gas leak", "fire", "collapse", "structural failure",
    ],
    "equipment_failure": [
        "fault", "faulted", "faulting", "failed", "failure", "tripped", "trip",
        "out of service", "oos", "not responding", "not running", "not working",
        "seized", "seized up", "burned out", "burnt out", "overheated",
        "overtemp", "over temperature", "short circuit", "scr shorted",
        "motor failure", "bearing failure", "shaft failure", "impeller",
        "cavitation", "vibration", "noise", "unusual sound", "strange sound",
        "smoke", "sparks", "burning smell",
    ],
    "maintenance_action": [
        "work order", "work order created", "wo created", "wo raised",
        "wo submitted", "replacement ordered", "parts ordered", "repair",
        "maintenance required", "follow up required", "follow-up",
        "notified maintenance", "called maintenance", "contacted maintenance",
        "submitted ticket", "archibus",
    ],
    "regulatory": [
        "notified ehs", "notified moe", "notified tssa", "notified ministry",
        "notified sac", "spill action", "environmental officer",
        "regulatory", "exceedance", "permit limit", "compliance",
        "health unit", "conservation authority", "lscrca", "event number",
        "sample collected", "spill sample", "chain of custody",
    ],
    "control_system": [
        "scada", "alarm", "alarms", "alert", "communication failure",
        "loss of communication", "no communication", "comms failure",
        "communication error", "plc fault", "panel fault", "remote fault",
        "hmi error", "no signal", "signal lost", "trending abnormal",
        "abnormal reading", "abnormal level", "unexpected reading",
        "flow gap", "flow discrepancy",
    ],
    "infrastructure": [
        "leak", "leaking", "leakage", "drip", "seepage",
        "clog", "clogged", "plugged", "plugging", "blockage", "blocked",
        "overflow", "surcharge", "backup", "pipe break", "pipe burst",
        "valve stuck", "valve failed", "actuator fault",
        "pump failed", "pump failure", "pump not starting", "pump not running",
        "pump tripped", "pump fault",
    ],
    "safety": [
        "injury", "incident", "near miss", "unsafe", "hazard",
        "spill cleanup", "absorbent pads", "containment",
        "gas detected", "h2s", "hydrogen sulfide", "gas alarm",
    ],
}

ALL_ISSUE_KEYWORDS: set[str] = {
    kw for kws in ISSUE_KEYWORDS.values() for kw in kws
}


def normalize_text(text: str) -> str:
    """
    normalize elog for processing 
    - lowercase, line breaks, know nabbreviations, shorten extended whitespace
    """
    if not text or not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Normalize line breaks
    text = text.replace("\\r\\n", " ").replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    text = text.replace("_x000D_", " ")

    # Remove bullet points and list markers
    text = re.sub(r"^\s*[-•\*]\s*", " ", text, flags=re.MULTILINE)

    # Expand abbreviations (whole-word match only)
    words = text.split()
    expanded = []
    for word in words:
        # Strip trailing punctuation for lookup but preserve it
        clean = re.sub(r"[.,;:!?\"'()\[\]]", "", word)
        expanded.append(ABBREV_MAP.get(clean, word))
    text = " ".join(expanded)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text



def has_negated_issue_term(text: str) -> bool:
    """
    returns true if text has issue but has negatation like "no leaks" 
    note: only if ALL issues are negated not just 1 

    """
    text_lower = text.lower()
    issue_hits = [kw for kw in ALL_ISSUE_KEYWORDS if kw in text_lower]
    if not issue_hits:
        return False

    # Check if each hit is preceded by a negation within ~5 words
    for kw in issue_hits:
        pattern = rf"\b(?:no|not|none|cleared|resolved|ok|okay|normal)\b.{{0,40}}{re.escape(kw)}"
        if not re.search(pattern, text_lower):
            return False  # At least one un-negated issue term found
    return True


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    
    add featuyres to data frame 

    input: DataFrame with at least ['LDTEXT', 'ldtext_norm', 'LOGENTRY_CONTINGENCY',
                                       'ISREMOTE', 'DESCRIPTION', 'PERSONGROUP']
    returns: same DataFrame with additional feature columns.
    """
    df = df.copy()

    # Ensure normalized text column exists
    if "ldtext_norm" not in df.columns:
        df["ldtext_norm"] = df["LDTEXT"].fillna("").apply(normalize_text)

    text = df["ldtext_norm"]

    # ---- Basic text statistics ----
    df["feat_text_len"] = text.str.len()
    df["feat_word_count"] = text.str.split().str.len()
    df["feat_line_count"] = df["LDTEXT"].fillna("").str.count(r"\n|\r\n|\\r\\n")
    df["feat_has_numbers"] = text.str.contains(r"\d+", na=False).astype(int)
    df["feat_has_worder"] = text.str.contains(
        r"\b(work order|wo\s*#?\d+|w\d{7,})\b", case=False, na=False
    ).astype(int)
    df["feat_has_timestamp"] = text.str.contains(
        r"\b\d{4}h?r?s?\b|\b\d{1,2}:\d{2}\b", case=False, na=False
    ).astype(int)

    # ---- Issue keyword signals ----
    for category, keywords in ISSUE_KEYWORDS.items():
        pattern = "|".join(re.escape(kw) for kw in keywords)
        df[f"feat_kw_{category}"] = text.str.contains(
            pattern, case=False, na=False
        ).astype(int)

    # Total keyword categories hit
    kw_cols = [c for c in df.columns if c.startswith("feat_kw_")]
    df["feat_kw_total_categories"] = df[kw_cols].sum(axis=1)
    df["feat_any_issue_keyword"] = (df["feat_kw_total_categories"] > 0).astype(int)

    # ---- Negation flag ----
    df["feat_negated"] = text.apply(has_negated_issue_term).astype(int)

    # ---- Strong signals ----
    df["feat_contingency"] = df["LOGENTRY_CONTINGENCY"].fillna(0).astype(int)
    df["feat_is_remote"] = df["ISREMOTE"].fillna(0).astype(int)

    # Regulatory notification terms
    df["feat_regulatory"] = text.str.contains(
        r"notif|sac|ehs|moe|ministry|health unit|conservation authority|"
        r"event.{0,5}#|spill.{0,10}sample|chain of custody",
        case=False, na=False,
    ).astype(int)

    # Maintenance escalation signals
    df["feat_escalation"] = text.str.contains(
        r"work order|replacement ordered|parts ordered|archibus|"
        r"called maintenance|called electrician|on route|on.site|dispatched",
        case=False, na=False,
    ).astype(int)

    # OOS (out of service) - high signal but common in routine too
    df["feat_oos"] = text.str.contains(
        r"\boos\b|out of service|o/s\b|n/s\b", case=False, na=False
    ).astype(int)

    # Alarm-related
    df["feat_alarm"] = text.str.contains(
        r"\balarm\b|\bfault\b|\btrip\b|\balert\b", case=False, na=False
    ).astype(int)

    # Repeated fault / reset cycles (strong issue signal)
    df["feat_repeated_fault"] = text.str.contains(
        r"reset.{0,20}(again|still|x\s*[23456789]|\d+ times?)|"
        r"still fault|still not|continues to fault|recurring",
        case=False, na=False,
    ).astype(int)

    # ---- Site / group one-hot (top groups) ----
    top_groups = ["WW-NORTH", "WW-CEAST", "WW-WEST", "CENTRAL", "EAST", "WEST", "W-MAINT", "ROC"]
    for grp in top_groups:
        safe_name = grp.replace("-", "_").replace(" ", "_")
        df[f"feat_group_{safe_name}"] = (
            df["PERSONGROUP"].fillna("") == grp
        ).astype(int)

    return df


CRITICAL_PATTERNS = re.compile(
    r"\b("
    r"overflow|spill|bypass|sewage.{0,10}release|emergency|flood"
    r"|lightning strike|explosion|fire|gas leak|collapse"
    r"|pump.{0,15}fail|pump.{0,10}fault|pump.{0,10}trip"
    r"|not respond|no communication|comm.{0,10}fail"
    r"|work order.{0,10}creat|wo.{0,5}(raised|created|submitted)"
    r"|replacement ordered|parts ordered"
    r"|notif.{0,10}(ehs|moe|sac|ministry|health unit|conservation)"
    r"|spill.{0,10}sample|chain of custody|event.{0,5}(#|number)"
    r"|out of service|oos\b"
    r"|alarm.{0,20}(still|active|uncleared|ongoing)"
    r"|still fault|still not.{0,20}(work|run|respond)"
    r"|leak.{0,10}(found|detect|identified|ongoing|active)"
    r"|clog|plugged|blockage|backup\b|surcharge"
    r"|oil leak|blower.{0,10}(oil|leak|fault|oos|out of service)"
    r"|damaged.{0,20}(equipment|device|pump|sensor|meter|transmitter)"
    r"|e.?stop|emergency stop"
    r"|chlorine.{0,10}(leak|spill|line)"
    r"|level transmitter|flowmeter.{0,10}(replace|fail|damaged)"
    r"|maximo.{0,10}(down|offline|unavailable|server)"
    r"|elog.{0,10}(down|offline|unavailable)"
    r"|contingency log|paper log"
    r"|commission|verify.{0,10}(meter|transmitter|sensor)"
    r"|devices.{0,10}(damaged|failed|replaced)"
    r"|electrical.{0,10}(strike|damage|fault|problem)"
    r")\b",
    re.IGNORECASE,
)

ROUTINE_PATTERNS = re.compile(
    r"^("
    r"shift start|shift end|shift handover|end of shift"
    r"|operator sign.{0,5}(in|out)|daily rounds|daily check"
    r"|routine check|station check.{0,10}complete"
    r")\s*[.,;]?\s*$",
    re.IGNORECASE,
)


def keyword_stage_label(text: str) -> tuple[int, str]:
    """

    keyword rules
    returns (label, reason)
    label = 1 -> flagged as potential issye
    label = 0 -> likely routine (we are not sure)
    label = -1 -> 100% routine (this elog is safely ignored)
    """
    if not text:
        return 0, "empty"

    norm = normalize_text(text)

    # Clear routine boilerplate
    if ROUTINE_PATTERNS.match(norm.strip()):
        return -1, "clear_routine_boilerplate"

    # Critical keyword hit
    if CRITICAL_PATTERNS.search(norm):
        # Check if the match is negated
        if has_negated_issue_term(norm):
            return 0, "keyword_hit_but_negated"
        return 1, "critical_keyword_match"

    return 0, "no_keyword_match"


def apply_keyword_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Stage 1 keyword labels to a DataFrame."""
    df = df.copy()
    results = df["ldtext_norm"].apply(keyword_stage_label)
    df["stage1_label"] = [r[0] for r in results]
    df["stage1_reason"] = [r[1] for r in results]
    return df
