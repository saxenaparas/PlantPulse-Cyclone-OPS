import os
import argparse
import logging
import pandas as pd
from src.config import CFG
from src.utils_io import setup_logging, read_any, enforce_5min_grid, ensure_outdir

def load_and_resample(data_path: str, resample_rule: str):
    df = read_any(data_path)
    logging.info("Loaded data: %s rows, %s to %s", len(df), df.index.min(), df.index.max())
    df = enforce_5min_grid(df, resample_rule)
    logging.info("Resampled to %s grid: %s rows", resample_rule, len(df))
    return df

def parse_args(default_out="outputs"):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="Path to .csv or .xlsx")
    ap.add_argument("--outdir", default=default_out, help="Output directory")
    ap.add_argument("--logdir", default="logs", help="Log directory")
    ap.add_argument("--resample", default=CFG.RESAMPLE_RULE, help="Resample rule (e.g., 5min)")
    ap.add_argument("--target_col", default=CFG.TARGET_COL, help="Target column name")
    return ap.parse_args()

import pandas as pd
from src.detect_anomalies import run_anomaly_detection

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--logdir", default="logs")
    ap.add_argument("--resample", default=CFG.RESAMPLE_RULE)
    ap.add_argument("--target_col", default=CFG.TARGET_COL)
    ap.add_argument("--clusters_csv", default=None, help="Path to outputs/state_labels.csv (from clustering step)")
    args = ap.parse_args()

    setup_logging(args.logdir, "run_detect_anomalies")
    ensure_outdir(args.outdir)
    CFG.TARGET_COL = args.target_col

    df = load_and_resample(args.data_path, args.resample)

    # Load labels
    if args.clusters_csv and os.path.exists(args.clusters_csv):
        try:
            lab = pd.read_csv(args.clusters_csv, parse_dates=["timestamp"])
            lab = lab.set_index("timestamp")["cluster"]
            lab = lab.reindex(df.index).astype(float)
        except Exception:
            # If user accidentally points to clusters_summary.csv, fallback to state_labels.csv
            alt = os.path.join(args.outdir, "state_labels.csv")
            lab = pd.read_csv(alt, parse_dates=["timestamp"]).set_index("timestamp")["cluster"].reindex(df.index).astype(float)
    else:
        # default try outputs/state_labels.csv
        lab_path = os.path.join(args.outdir, "state_labels.csv")
        lab = pd.read_csv(lab_path, parse_dates=["timestamp"]).set_index("timestamp")["cluster"].reindex(df.index).astype(float)

    anomalies = run_anomaly_detection(df, args.outdir, lab)

if __name__ == "__main__":
    main()
