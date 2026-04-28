# Automated Elogbook tool 

Every month we have 4000-5000 operator logs that consist of notes that operators leave regarding routine visits to the various facities. This tool scans every log and identifies which logs indicate an operational issue which must be reported during the monthy operational status reports.

---

## Architecture

**Stage 1 — Keyword Filter**
Every operational log is firstly scanned for a list of keywords that likely classify the log as an operational issue. If any keyword is found in a log it is flagged and otherwise marked as clear. 

**Stage 2 — ML Classifier**

All operational logs that didn't trigger a keyword are scored by a ML model (TF IDF + Complement Naive Bayes). If the model's score exceeds the threshold, the log is flagged and otherwise cleared. 

This model must be trained once before used (Use the **Train model** tab in the website). 


## Performance

Tested on Feb 2026 and Mar 2026 data:

| Month | Total logs | Issues | Recall | Missed |
|-------|-----------|--------|--------|--------|
| Feb 2026 | 4,354 | 52 | 100% | 0 |
| Mar 2026 | 4,958 | 45 | 100% | 0 |

**Recall** = % of real issues the tool caught.   
**False positives** are expected, the tool intentionally over-flags. Reviewers see ~2,000 logs instead of 4,500+, and real issues are not missed.


## File structure

```
elogbook_triage/
├── app.py            ← Streamlit web app (connect repository that store these files)
├── pipeline.py       ← keyword filter, ML classifier, threshold (edit this to tune model / keyword filter)
├── model.pkl         ← trained model (created after first training run)
└── requirements.txt
```


## Setup

In VS code terminal

```bash
# 1. Install dependencies 
pip install streamlit pandas openpyxl rapidfuzz scikit-learn

# 2. Run the app
python -m streamlit run app.py
```

Opens at http://localhost:8501

---

## File naming convention

Files must follow the format below

| File | Format | Example |
|------|--------|---------|
| RAW export | `YYYY-MM RAW.xlsx` | `2026-04 RAW.xlsx` |
| SORTED answer key | `YYYY-MM SORTED.xlsx` | `2026-04 SORTED.xlsx` |

---

## Using this tool

### Tab 1: Running the tool

Use this every month to process a new RAW export.

1. Upload a `YYYY-MM RAW.xlsx` file
2. Click **Run triage**
3. Click **Download sorted results**

Output file: `YYYY-MM SORTED predicted.xlsx`  
Contains one sheet "**Predicted Issues**" with every flagged log, sorted by site then date.

Output columns:

| Column | Description |
|--------|-------------|
| `EVENTDATE` | Date of the log entry |
| `DESCRIPTION` | Site name |
| `PERSONGROUP` | Operator group |
| `LDTEXT` | Original log text |
| `matched_keyword` | Keyword that triggered Stage 1 (blank if flagged by ML) |
| `ml_probability` | ML model confidence score (0–1) |
| `triage_stage` | Whether flagged by `keyword` or `ml` |

---

### Tab 2: Train model

Use this once at the start, and again whenever you want to improve the model with new months of data.

1. Upload all your historical `YYYY-MM RAW.xlsx` files
2. Upload all the corresponding `YYYY-MM SORTED.xlsx` answer keys
3. Click **Train model**

The model is saved as `model.pkl` next to `app.py` and used automatically by Tab 1.

**When to retrain:** when you have accumulated new monthly data!

---

### Tab 3: Evaluate

Use this to check and tune model on 1 month of data

1. Upload a `YYYY-MM RAW.xlsx` and its `YYYY-MM SORTED.xlsx`
2. Click **Evaluate**

Shows:
- **Recall** — % of real issues caught 
- **Precision** — % of flags that were real issues
- **Missed (FN)** — real issues the tool failed to flag
- **False alarms (FP)** — routine logs incorrectly flagged

Three tabs break down: correctly caught issues, missed issues, and false alarms.  
Download the full evaluation report as Excel.
