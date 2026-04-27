import streamlit as st
import pandas as pd
import tempfile
import sys
import warnings
import logging
from pathlib import Path
from io import BytesIO

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Elogbook Sorting Tool")
st.title("Elogbook Sorting Tool")

tab1, tab2 = st.tabs(["Run triage", "Evaluate"])

# ── TAB 1 ──────────────────────────────────────────────────────────────────
with tab1:
    st.write("Upload the RAW elogbook export and download the predicted sorted issues.")
    st.info("File must be named **YYYY-MM RAW.xlsx** — e.g. `2026-04 RAW.xlsx`")

    raw_file = st.file_uploader("Choose a RAW elogbook file (.xlsx)", type=["xlsx"], key="run_raw")

    if raw_file:
        st.success("File uploaded successfully!")
        if st.button("Run triage"):
            try:
                import pipeline
                prefix  = raw_file.name.replace(" RAW.xlsx", "").replace(".xlsx", "")
                df      = pd.read_excel(raw_file)
                results = pipeline.run(df)

                n_flag  = (results["triage_decision"] == "FLAG").sum()
                n_clear = (results["triage_decision"] == "CLEAR").sum()
                st.write(f"**{n_flag}** predicted issues · **{n_clear}** routine · **{len(results)}** total")

                st.download_button(
                    label="Download sorted results",
                    data=pipeline.to_excel(results, prefix),
                    file_name=f"{prefix} SORTED predicted.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as e:
                st.exception(e)

# ── TAB 2 ──────────────────────────────────────────────────────────────────
with tab2:
    st.write("Upload a RAW + SORTED pair to check accuracy.")
    st.info(
        "RAW file must be named **YYYY-MM RAW.xlsx** — e.g. `2026-04 RAW.xlsx`\n\n"
        "SORTED file must be named **YYYY-MM SORTED.xlsx** — e.g. `2026-04 SORTED.xlsx`"
    )

    col1, col2 = st.columns(2)
    with col1:
        eval_raw    = st.file_uploader("RAW file (.xlsx)",          type=["xlsx"], key="eval_raw")
    with col2:
        eval_sorted = st.file_uploader("SORTED answer key (.xlsx)", type=["xlsx"], key="eval_sorted")

    if eval_raw and eval_sorted:
        st.success("Both files uploaded!")
        if st.button("Evaluate"):
            try:
                import pipeline
                from rapidfuzz import fuzz
                from openpyxl import load_workbook
                import re

                df = pd.read_excel(eval_raw)
                df["is_issue"] = 0
                df["date_str"] = pd.to_datetime(df["EVENTDATE"], errors="coerce").dt.strftime("%Y-%m-%d")

                with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                    tmp.write(eval_sorted.read())
                    tmp_path = tmp.name

                wb = load_workbook(tmp_path, read_only=True)
                ws = wb["Sheet1"]
                for row in ws.iter_rows(values_only=True):
                    if not row[3]: continue
                    t = str(row[3]).replace("_x000D_", " ").replace("\r\n", " ").replace("\n", " ")
                    parts = re.split(r"[\u2014]{5,}", t)
                    body = (parts[1].strip() if len(parts) >= 2 else t)[:80]
                    date_str = row[2].strftime("%Y-%m-%d") if hasattr(row[2], "strftime") else str(row[2])[:10]
                    cands = df[df["date_str"] == date_str]
                    best_idx, best_score = None, 0
                    for idx, cand in cands.iterrows():
                        score = fuzz.partial_ratio(body[:60], str(cand["LDTEXT"] or "").lower()[:80])
                        if score > best_score:
                            best_score, best_idx = score, idx
                    if best_idx is not None and best_score >= 75:
                        df.loc[best_idx, "is_issue"] = 1

                results = pipeline.run(df)
                y_true  = df["is_issue"].values
                y_pred  = (results["triage_decision"] == "FLAG").astype(int).values

                tp = int(((y_true==1) & (y_pred==1)).sum())
                fp = int(((y_true==0) & (y_pred==1)).sum())
                fn = int(((y_true==1) & (y_pred==0)).sum())
                tn = int(((y_true==0) & (y_pred==0)).sum())
                recall    = round(tp / max(tp+fn, 1), 4)
                precision = round(tp / max(tp+fp, 1), 4)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Recall",           f"{recall:.4f}")
                c2.metric("Precision",         f"{precision:.4f}")
                c3.metric("Missed (FN)",        fn,
                          delta="CRITICAL" if fn > 0 else "None",
                          delta_color="inverse")
                c4.metric("False alarms (FP)", fp)

                if fn > 0:
                    st.error(f" {fn} real issue(s) missed — add their keywords to KEYWORDS in pipeline.py")
                else:
                    st.success(" No issues missed.")

                COLS = [c for c in ["EVENTDATE", "DESCRIPTION", "PERSONGROUP", "LDTEXT",
                                    "matched_keyword", "is_issue", "triage_decision"]
                        if c in results.columns]

                t1, t2, t3 = st.tabs([f"Caught ({tp})", f" Missed ({fn})", f" False alarms ({fp})"])
                with t1:
                    st.dataframe(results[(y_true==1) & (y_pred==1)][COLS].reset_index(drop=True),
                                 use_container_width=True)
                with t2:
                    if fn > 0:
                        st.dataframe(results[(y_true==1) & (y_pred==0)][COLS].reset_index(drop=True),
                                     use_container_width=True)
                    else:
                        st.info("No missed issues.")
                with t3:
                    st.dataframe(results[(y_true==0) & (y_pred==1)][COLS].head(100).reset_index(drop=True),
                                 use_container_width=True)

            except Exception as e:
                st.exception(e)