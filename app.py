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

tab1, tab2 = st.tabs(["Run triage", "Train model"])

with tab1:
    st.write("Upload the RAW elogbook export and download the predicted sorted issues.")
     
    raw_file = st.file_uploader(
    "Choose a RAW elogbook file (.csv)",
    type=["csv"],
    key="triage_raw",
    )

    if raw_file is not None:
        st.success("File uploaded successfully!")

        if st.button("Run triage"):
            try:
                from features  import engineer_features, normalize_text
                from pipeline  import TriagePipeline

                MODEL_DIR = Path(__file__).parent / "models_6mo"

                if not MODEL_DIR.exists():
                    st.error(
                        "No trained model found. Make sure the models_6mo/ folder "
                        "is in the same directory as app.py."
                    )
                    st.stop()

                prefix = raw_file.name.replace("_RAW.csv", "").replace(".csv", "")

                with st.spinner("Running triage…"):
                    with tempfile.TemporaryDirectory() as tmp:
                        tmp = Path(tmp)
                        raw_path = tmp / raw_file.name
                        raw_path.write_bytes(raw_file.read())

                        df = pd.read_csv(str(raw_path), dtype={"LOCATION": str})
                        df["is_issue"] = -1
                        df["source_month"] = prefix
                        df = engineer_features(df)

                        pipe    = TriagePipeline.from_model_dir(str(MODEL_DIR))
                        results = pipe.run(df)

                flagged = results[results["triage_decision"] == "FLAG"].copy()
                n_total  = len(results)
                n_flagged = len(flagged)
                n_cleared = n_total - n_flagged

                st.write(
                    f"**{n_flagged}** predicted issues · "
                    f"**{n_cleared}** routine · "
                    f"**{n_total}** total"
                )

                # Build output Excel matching SORTED format:
                # one sheet per site (DESCRIPTION), containing flagged logs only,
                # sorted by date then confidence score
                buf = BytesIO()

                with pd.ExcelWriter(buf, engine="openpyxl") as writer:

                    # Sheet 1: All predicted issues (most useful single view)
                    summary_cols = [c for c in [
                        "EVENTDATE", "DESCRIPTION", "PERSONGROUP",
                        "LDTEXT", "triage_score", "triage_reason",
                        "matching_keywords", "clf_proba", "triage_stage",
                    ] if c in flagged.columns]

                    (flagged[summary_cols]
                        .sort_values(["DESCRIPTION", "EVENTDATE"], na_position="last")
                        .to_excel(writer, sheet_name="All Predicted Issues", index=False))

                    # One sheet per site — mirrors the real SORTED structure
                    if "DESCRIPTION" in flagged.columns:
                        sites = sorted(flagged["DESCRIPTION"].dropna().unique())
                        for site in sites:
                            site_df = flagged[flagged["DESCRIPTION"] == site][summary_cols]
                            site_df = site_df.sort_values("EVENTDATE", na_position="last")

                            # Truncate sheet name to Excel 31-char limit
                            sheet_name = str(site)[:31]
                            site_df.to_excel(writer, sheet_name=sheet_name, index=False)

                st.download_button(
                    label="Download sorted results",
                    data=buf.getvalue(),
                    file_name=f"{prefix}_SORTED_predicted.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            except Exception as e:
                st.exception(e)

with tab2:
    st.write(
        "Upload a RAW + SORTED pair to evaluate the current model and see "
        "which logs were missed or incorrectly flagged."
    )

    col1, col2 = st.columns(2)
    with col1:
        train_raw = st.file_uploader(
            "RAW elogbook export (.csv)",
            type=["csv"],
            key="train_raw",
        )
    with col2:
        train_sorted = st.file_uploader(
            "SORTED answer key (.xlsx)",
            type=["xlsx"],
            key="train_sorted",
        )

    if train_raw and train_sorted:
        st.success("Both files uploaded!")

        if st.button("Evaluate model"):
            try:
                from data_loader import load_month_pair
                from features    import engineer_features
                from pipeline    import TriagePipeline

                MODEL_DIR = Path(__file__).parent / "models_6mo"

                if not MODEL_DIR.exists():
                    st.error("No trained model found in models_6mo/. Run training via command line first.")
                    st.stop()

                prefix = train_raw.name.replace("_RAW.csv", "").replace(".csv", "")

                with st.spinner("Loading and evaluating…"):
                    with tempfile.TemporaryDirectory() as tmp:
                        tmp = Path(tmp)
                        raw_path    = tmp / train_raw.name
                        sorted_path = tmp / train_sorted.name
                        raw_path.write_bytes(train_raw.read())
                        sorted_path.write_bytes(train_sorted.read())

                        df = load_month_pair(str(raw_path), str(sorted_path))
                        df["source_month"] = prefix
                        df = engineer_features(df)

                        pipe    = TriagePipeline.from_model_dir(str(MODEL_DIR))
                        results = pipe.run(df)

                y_true = df["is_issue"].values
                y_pred = (results["triage_decision"] == "FLAG").astype(int).values

                tp = int(((y_true==1) & (y_pred==1)).sum())
                fp = int(((y_true==0) & (y_pred==1)).sum())
                fn = int(((y_true==1) & (y_pred==0)).sum())
                tn = int(((y_true==0) & (y_pred==0)).sum())
                recall    = round(tp / max(tp+fn, 1), 4)
                precision = round(tp / max(tp+fp, 1), 4)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Recall",    f"{recall:.4f}", help="% of real issues caught")
                c2.metric("Precision", f"{precision:.4f}", help="% of flags that are real issues")
                c3.metric("Missed (FN)", fn,
                          delta=f"{'⚠️ CRITICAL' if fn > 0 else '✅ None'}",
                          delta_color="inverse")
                c4.metric("False alarms (FP)", fp)

                if fn > 0:
                    st.error(f"⚠️ {fn} real issue(s) were MISSED by the pipeline.")
                else:
                    st.success("✅ No issues missed — perfect recall on this month.")

                OUTPUT_COLS = [c for c in [
                    "EVENTDATE", "DESCRIPTION", "PERSONGROUP", "LDTEXT",
                    "triage_decision", "triage_stage", "triage_score",
                    "triage_reason", "matching_keywords", "clf_proba", "is_issue",
                ] if c in results.columns]

                if fn > 0:
                    st.subheader("Missed issues (false negatives)")
                    st.caption("These are real issues the model failed to flag. Add their key phrases to features.py → CRITICAL_PATTERNS.")
                    missed = results[(y_true==1) & (y_pred==0)][OUTPUT_COLS]
                    st.dataframe(missed.reset_index(drop=True), use_container_width=True)

                t1, t2, t3 = st.tabs([
                    f"✅ Correctly flagged issues ({tp})",
                    f"❌ Missed issues ({fn})",
                    f"⚠️ False alarms ({fp})",
                ])

                with t1:
                    correct = results[(y_true==1) & (y_pred==1)][OUTPUT_COLS]
                    st.dataframe(correct.reset_index(drop=True), use_container_width=True)

                with t2:
                    if fn > 0:
                        missed = results[(y_true==1) & (y_pred==0)][OUTPUT_COLS]
                        st.dataframe(missed.reset_index(drop=True), use_container_width=True)
                    else:
                        st.info("No missed issues.")

                with t3:
                    false_alarms = results[(y_true==0) & (y_pred==1)][OUTPUT_COLS].head(200)
                    st.dataframe(false_alarms.reset_index(drop=True), use_container_width=True)
                    if fp > 200:
                        st.caption(f"Showing first 200 of {fp} false alarms.")

                buf = BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    summary = pd.DataFrame([
                        {"Metric": "Total logs",              "Value": len(df)},
                        {"Metric": "True issues in month",    "Value": int(y_true.sum())},
                        {"Metric": "Recall",                  "Value": recall},
                        {"Metric": "Precision",               "Value": precision},
                        {"Metric": "Correctly flagged (TP)",  "Value": tp},
                        {"Metric": "Missed issues (FN)",      "Value": fn},
                        {"Metric": "False alarms (FP)",       "Value": fp},
                        {"Metric": "Correctly cleared (TN)",  "Value": tn},
                    ])
                    summary.to_excel(writer, sheet_name="Summary", index=False)
                    results[(y_true==1) & (y_pred==0)][OUTPUT_COLS].to_excel(
                        writer, sheet_name="MISSED ISSUES", index=False)
                    results[(y_true==1) & (y_pred==1)][OUTPUT_COLS].to_excel(
                        writer, sheet_name="Correctly flagged", index=False)
                    results[(y_true==0) & (y_pred==1)][OUTPUT_COLS].to_excel(
                        writer, sheet_name="False alarms", index=False)
                    results[OUTPUT_COLS].to_excel(
                        writer, sheet_name="All logs", index=False)

                st.download_button(
                    label="Download evaluation report",
                    data=buf.getvalue(),
                    file_name=f"{prefix}_evaluation.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            except Exception as e:
                st.exception(e)