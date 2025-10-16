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
from src.detect_shutdowns import detect_shutdown_mask, run_shutdown_detection
from src.cluster_states import run_clustering

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--logdir", default="logs")
    ap.add_argument("--resample", default=CFG.RESAMPLE_RULE)
    ap.add_argument("--target_col", default=CFG.TARGET_COL)
    ap.add_argument("--shutdowns_csv", default=None, help="Optional precomputed shutdowns csv to exclude")
    args = ap.parse_args()

    setup_logging(args.logdir, "run_cluster_states")
    ensure_outdir(args.outdir)
    CFG.TARGET_COL = args.target_col

    df = load_and_resample(args.data_path, args.resample)

    # Build shutdown mask
    if args.shutdowns_csv and os.path.exists(args.shutdowns_csv):
        sd = pd.read_csv(args.shutdowns_csv, parse_dates=["start","end"])
        mask = pd.Series(False, index=df.index)
        for _, r in sd.iterrows():
            mask.loc[r["start"]:r["end"]] = True
    else:
        mask = detect_shutdown_mask(df)

    states_df, full_labels = run_clustering(df, args.outdir, exclude_mask=mask)
    # Also save aligned label series for downstream steps
    aligned = pd.DataFrame({"timestamp": full_labels.index, "cluster": full_labels.values}).dropna()
    aligned.to_csv(os.path.join(args.outdir, "state_labels.csv"), index=False)
    logging.info("Saved state labels: %s", os.path.join(args.outdir, "state_labels.csv"))

if __name__ == "__main__":
    main()
