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
st.write("Upload the RAW elogbook export and download the predicted sorted issues.")

raw_file = st.file_uploader(
    "Choose a RAW elogbook file (.csv)",
    type=["csv"],
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