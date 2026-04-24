"""
run_triage.py

main entry point for operational log

examples: 

  #training 
  python run_triage.py train --data-dir /path/to/data

  #run on raw csv
  python run_triage.py run --raw /path/to/2026-04_RAW.csv --model-dir models/

  #evaluate pipeline on labeled month pair

  python run_triage.py evaluate --raw 2026-03_RAW.csv --sorted 2026-03_SORTED.xlsx

#demo 
#   python run_triage.py demo
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)




UPLOAD_DIR = "/mnt/user-data/uploads"
MODEL_DIR = "/home/claude/log_triage/models"
OUTPUT_DIR = "/mnt/user-data/outputs"


def demo_train_and_evaluate():
    """
    Train on Jan+Feb, evaluate on March.
    Prints a detailed report of pipeline performance.
    """
    from data_loader import load_month_pair, load_all_pairs
    from features import engineer_features
    from classifier import train_classifier
    from pipeline import TriagePipeline

    logger.info("=" * 60)
    logger.info("DEMO: Loading Jan + Feb for training, March for evaluation")
    logger.info("=" * 60)

    # Load Jan + Feb as training data
    frames = []
    for month in ["2026-01", "2026-02"]:
        raw = f"{UPLOAD_DIR}/{month}_RAW.csv"
        sorted_ = f"{UPLOAD_DIR}/{month}_SORTED.xlsx"
        df = load_month_pair(raw, sorted_)
        df["source_month"] = month
        frames.append(df)
        logger.info("  %s: %d rows, %d issues", month, len(df), df["is_issue"].sum())

    df_train = pd.concat(frames, ignore_index=True)
    df_train = engineer_features(df_train)

    logger.info("\n--- Training classifier ---")
    artifacts = train_classifier(df_train, save_dir=MODEL_DIR, n_cv_splits=3)

    logger.info("\n--- Cross-validation metrics ---")
    m = artifacts["metrics"]
    logger.info("  CV Recall:    %.3f ± %.3f", m["cv_recall_mean"] or 0, m["cv_recall_std"] or 0)
    logger.info("  CV Precision: %.3f", m["cv_precision_mean"] or 0)
    logger.info("  CV AUC:       %.3f", m["cv_auc_mean"] or 0)
    logger.info("  Threshold:    %.3f", m["threshold"])

    # Show top issue-predictive terms
    from classifier import get_top_issue_terms
    top_terms = get_top_issue_terms(artifacts["tfidf"], artifacts["model"], n=20)
    logger.info("\n--- Top 20 issue-predictive terms ---")
    logger.info("  %s", ", ".join(top_terms))

    # ---- Evaluate on March ----
    logger.info("\n--- Loading March for evaluation ---")
    df_march = load_month_pair(
        f"{UPLOAD_DIR}/2026-03_RAW.csv",
        f"{UPLOAD_DIR}/2026-03_SORTED.xlsx",
    )
    df_march = engineer_features(df_march)

    logger.info("\n--- Running pipeline on March (no LLM) ---")
    pipeline = TriagePipeline(
        model=artifacts["model"],
        tfidf=artifacts["tfidf"],
        threshold=artifacts["threshold"],
        use_llm=False,  # Set True if Ollama is running
    )

    eval_metrics = pipeline.evaluate(df_march)

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS — March 2026")
    logger.info("=" * 60)
    logger.info("  Total logs:        %d", len(df_march))
    logger.info("  True issues:       %d", df_march["is_issue"].sum())
    logger.info("  TP (caught):       %d", eval_metrics["tp"])
    logger.info("  FP (false alarms): %d", eval_metrics["fp"])
    logger.info("  FN (MISSED!):      %d  ← THIS MUST BE NEAR ZERO", eval_metrics["fn"])
    logger.info("  Recall:            %.4f", eval_metrics["recall"])
    logger.info("  Precision:         %.4f", eval_metrics["precision"])
    logger.info("  False Negative Rate: %.4f", eval_metrics["false_negative_rate"])
    logger.info("  Logs cleared (reduced workload): %d (%.1f%%)",
                eval_metrics["logs_cleared"], eval_metrics["reduction_pct"])
    logger.info("  Flags sent to human review: %d", eval_metrics["flags_sent_to_human"])
    logger.info("=" * 60)

    # ---- Save detailed output ----
    results = pipeline.run(df_march)
    save_results(results, "march_triage_results.xlsx")

    return results, eval_metrics


def save_results(df: pd.DataFrame, filename: str):
    """Save triage results to Excel with color-coded sheets."""
    import openpyxl
    from openpyxl.styles import PatternFill, Font

    path = Path(OUTPUT_DIR) / filename
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    output_cols = [
        "EVENTDATE", "DESCRIPTION", "PERSONGROUP",
        "LDTEXT", "triage_decision", "triage_stage",
        "triage_score", "triage_reason", "matching_keywords",
        "clf_proba", "stage1_label", "llm_reason",
    ]
    available_cols = [c for c in output_cols if c in df.columns]

    with pd.ExcelWriter(str(path), engine="openpyxl") as writer:
        # All results
        df[available_cols].to_excel(writer, sheet_name="All Results", index=False)

        # Flagged only
        flagged = df[df["triage_decision"] == "FLAG"][available_cols]
        flagged.to_excel(writer, sheet_name="Flagged Issues", index=False)

        # Cleared only
        cleared = df[df["triage_decision"] == "CLEAR"][available_cols]
        cleared.to_excel(writer, sheet_name="Cleared Routine", index=False)

        # Summary stats
        summary = pd.DataFrame([{
            "Metric": "Total logs",     "Value": len(df)},
            {"Metric": "Flagged",        "Value": (df["triage_decision"]=="FLAG").sum()},
            {"Metric": "Cleared",        "Value": (df["triage_decision"]=="CLEAR").sum()},
            {"Metric": "Review",         "Value": (df["triage_decision"]=="REVIEW").sum()},
            {"Metric": "Flag rate (%)",  "Value": round(100*(df["triage_decision"]=="FLAG").sum()/len(df),1)},
            {"Metric": "Reduction (%)",  "Value": round(100*(df["triage_decision"]=="CLEAR").sum()/len(df),1)},
        ])
        summary.to_excel(writer, sheet_name="Summary", index=False)

    logger.info("Results saved to %s", path)
    return str(path)



def main():
    parser = argparse.ArgumentParser(description="Operational Log Triage System")
    subparsers = parser.add_subparsers(dest="command")

    # Demo command
    subparsers.add_parser("demo", help="Run full demo on uploaded files")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train on a directory of RAW/SORTED pairs")
    train_parser.add_argument("--data-dir", required=True)
    train_parser.add_argument("--model-dir", default=MODEL_DIR)

    # Run command
    run_parser = subparsers.add_parser("run", help="Run triage on a new raw CSV")
    run_parser.add_argument("--raw", required=True)
    run_parser.add_argument("--model-dir", default=MODEL_DIR)
    run_parser.add_argument("--output", default=None)
    run_parser.add_argument("--use-llm", action="store_true")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate on a labeled pair")
    eval_parser.add_argument("--raw", required=True)
    eval_parser.add_argument("--sorted", required=True)
    eval_parser.add_argument("--model-dir", default=MODEL_DIR)

    args = parser.parse_args()

    if args.command == "demo":
        results, metrics = demo_train_and_evaluate()
        sys.exit(0 if metrics["false_negative_rate"] < 0.05 else 1)

    elif args.command == "train":
        from data_loader import load_all_pairs
        from features import engineer_features
        from classifier import train_classifier
        df = load_all_pairs(args.data_dir)
        df = engineer_features(df)
        artifacts = train_classifier(df, save_dir=args.model_dir)
        logger.info("Training complete. Metrics: %s", artifacts["metrics"])

    elif args.command == "run":
        from data_loader import load_month_pair
        from features import normalize_text, engineer_features
        from pipeline import TriagePipeline

        df = pd.read_csv(args.raw, dtype={"LOCATION": str})
        df["ldtext_norm"] = df["LDTEXT"].fillna("").apply(normalize_text)
        df = engineer_features(df)
        pipeline = TriagePipeline.from_model_dir(args.model_dir, use_llm=args.use_llm)
        results = pipeline.run(df)
        out_file = args.output or f"triage_output.xlsx"
        save_results(results, out_file)

    elif args.command == "evaluate":
        from data_loader import load_month_pair
        from features import engineer_features
        from pipeline import TriagePipeline
        df = load_month_pair(args.raw, args.sorted)
        df = engineer_features(df)
        pipeline = TriagePipeline.from_model_dir(args.model_dir)
        metrics = pipeline.evaluate(df)
        logger.info("Evaluation: %s", metrics)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
