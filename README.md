# Automated elog tool

A tool for flagging operational issues for manual review to differentiate between 
real and routine elogs. 

# overview

```
Every month ~4000-5000 raw elogs 

======= Stage 1: Keyword detecter =============

Regex (Regular Expressions)

rule-based pattern matching system for text
(helps define patterns instead of exact words)

Example: 
"pump failed"
"pump failure"
"pump failing"

fail(ed|ure|ing)?

fuzzy match (helps catch spelling issues / abbresviations)

Any keyword detected -> flagged as operational issue for manula review.

======= Stage 2: ML classifier =============

TF-IDF: a method for converting text into numbers so ML models can process it. 
(turning sentences into embeddings)

TF = term frequency "how frequent does a word appears in a log"

IDF = Inverse Document Frequency "Gives more importance to rare/specialized words"

1-3 grams: model is looking at single vs two vs third word phrases

We built 8000 potential features. 

Calibrated Logistic Regression
- outputs probablities 
- recall threshold >= 0.97 (catch 97% of actual issues for training)
- clear only if prob < 0.05 and no keywords and no flagging 

======= Stage 3: ML classifier =============

if any uncertainty in outputs -> human review
- sorted by confidence score
- keyword matches
```

---

Training Metrics

Recall: 1.0 
False Negative Rate: 0.0
Precision: 0.01 - 0.03 
AUC: 0.937
Class imbalance: 1:83

---

#structure

```
log_triage/
├── data_loader.py    #loads training csv sets and fuzzy matches issues labels 
├── features.py       #text normalization, abbreviation expansion, feature engineering
├── classifier.py     #tf-idf logistic regression 
├── pipeline.py       #3 stage training & evaluator
├── run_triage.py     #main run file
├── requirements.txt
├── models/           #saved models 
│   ├── classifier.joblib
│   ├── tfidf.joblib
│   ├── threshold.joblib
│   └── feature_importances.csv
└── README.md
```

---

#installation for training

```bash
#install dependancies 
pip install -r requirements.txt

#organize data fles
# data/
#   2026-01_RAW.csv
#   2026-01_SORTED.xlsx
#   2026-02_RAW.csv
#   2026-02_SORTED.xlsx
#   etc.
```

---

## Usage

### Quick demo (uses the uploaded files)
```bash
cd log_triage
python run_triage.py demo
```

### Train on all available months
```bash
python run_triage.py train --data-dir /path/to/data --model-dir models/
```

### Run triage on a new month (no answer key needed)
```bash
python run_triage.py run \
  --raw /path/to/2026-04_RAW.csv \
  --model-dir models/ \
  --output 2026-04_triage.xlsx \
  --use-llm   # add this flag if Ollama is running
```

### Evaluate on a labeled month pair
```bash
python run_triage.py evaluate \
  --raw /path/to/2026-03_RAW.csv \
  --sorted /path/to/2026-03_SORTED.xlsx \
  --model-dir models/
```

---

## Output Format

The triage Excel output contains four sheets:

- **All Results** — every log with triage columns
- **Flagged Issues** — only flagged logs (sorted by confidence)
- **Cleared Routine** — logs the system cleared with high confidence
- **Summary** — aggregate statistics

Key output columns:

| Column | Description |
|--------|-------------|
| `triage_decision` | `FLAG` / `CLEAR` / `REVIEW` |
| `triage_stage` | Which stage made the decision |
| `triage_score` | Confidence (0–1) |
| `triage_reason` | Human-readable explanation |
| `matching_keywords` | Which keywords triggered Stage 1 |
| `clf_proba` | ML model probability of being an issue |
| `llm_reason` | LLM explanation (if Stage 3 ran) |

---

## How to Add More Training Data

Add new month pairs to your data directory and retrain:

```bash
# Add new files:
# data/2025-04_RAW.csv
# data/2025-04_SORTED.xlsx

python run_triage.py train --data-dir data/ --model-dir models/
```

The data loader auto-discovers all `*_RAW.csv` / `*_SORTED.xlsx` pairs.
Retraining takes ~30–60 seconds for 11 months of data.

---

## Active Learning / Human-in-the-Loop

When reviewers mark a flagged log as "false positive," feed that back:

```python
# In your review tool, after a reviewer marks a log:
# Add it to a feedback CSV with columns: LDTEXT, is_issue (0 or 1), source_month
# Then retrain:

import pandas as pd
feedback = pd.read_csv("human_feedback.csv")
# Add to training data and retrain monthly
```

Each false positive that gets corrected makes the model more precise
without reducing recall — the active learning loop tightens the system
over time.

---

## Extending the Keyword List

The keyword dictionary in `features.py` (`CRITICAL_PATTERNS`, `ISSUE_KEYWORDS`)
should be updated as you discover new issue patterns.

To find new discriminative terms automatically:

```python
from data_loader import load_all_pairs
from features import engineer_features
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

df = load_all_pairs("data/")
issue_text = df[df['is_issue']==1]['ldtext_norm']
routine_text = df[df['is_issue']==0]['ldtext_norm']

vec = TfidfVectorizer(ngram_range=(1,2), min_df=2)
vec.fit(df['ldtext_norm'])

issue_scores = vec.transform(issue_text).mean(axis=0)
routine_scores = vec.transform(routine_text).mean(axis=0)

lift = pd.Series(
    (issue_scores.A1 + 1e-6) / (routine_scores.A1 + 1e-6),
    index=vec.get_feature_names_out()
).sort_values(ascending=False)

print(lift.head(30))  # Top discriminative terms — add to CRITICAL_PATTERNS
```

---

## Threshold Tuning

The decision threshold (currently ~0.17) is calibrated at `TARGET_RECALL = 0.97`
in `classifier.py`. To adjust:

```python
# More conservative (higher recall, more false alarms):
TARGET_RECALL = 0.99

# Slightly less conservative (reduces false alarms, may miss ~1-2% of issues):
TARGET_RECALL = 0.95
```

**Recommendation:** Never go below 0.95 for this use case.

---

## Known Limitations & Roadmap

**Current limitations:**
1. Clearing rate is ~0% with 3 months of data — more months needed
2. No embedding-based semantic similarity (Stage 2 is TF-IDF only)
3. LLM stage requires Ollama running locally
4. Some borderline "issues" in answer keys are marginal (Maximo downtime, test entries)

**Roadmap (add when you have 8+ months of data):**
- [ ] Add `sentence-transformers` (SetFit) as a second Stage 2 classifier
- [ ] Build active learning feedback loop from reviewer decisions  
- [ ] Add drift monitoring (track embedding centroid shift over time)
- [ ] Add SHAP explainability for ML decisions
- [ ] Build a simple Streamlit review UI with "Confirm / False Positive" buttons
- [ ] Add cross-month pattern detection (same site having repeated issues)

---

## Data Notes (from exploration)

- **RAW CSV columns:** `LDTEXT` (free text), `EVENTDATE`, `LOCATION`, `LOGENTRY_CONTINGENCY`,
  `ISREMOTE`, `PERSONGROUP`, `DESCRIPTION` (site name), plus several operator reference fields
- **LOGENTRY_CONTINGENCY = 1:** Operator-marked contingency events — always true issues (13–14/month)
- **Issue rate:** ~1.1–1.2% across Jan/Feb/Mar (53–54 issues per ~4,300–4,900 logs/month)
- **Sites:** 25 sites across 9 person groups (WW-NORTH, WW-CEAST, WW-WEST, CENTRAL, EAST, WEST,
  W-MAINT, LINEAR, ROC)
- **SORTED XLSX format changed between Mar and earlier months** — data_loader handles both
- **Matching strategy:** fuzzy text match (rapidfuzz) by date + first 60 chars of LDTEXT, 
  achieving ~87–100% match rate across months


# Install dependencies (one time)
pip install streamlit pandas openpyxl scikit-learn rapidfuzz joblib numpy

# Run the app
cd elogbook_triage
streamlit run app.py