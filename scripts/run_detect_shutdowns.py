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

from src.detect_shutdowns import run_shutdown_detection

def main():
    args = parse_args()
    setup_logging(args.logdir, "run_detect_shutdowns")
    ensure_outdir(args.outdir)
    CFG.TARGET_COL = args.target_col

    df = load_and_resample(args.data_path, args.resample)
    run_shutdown_detection(df, args.outdir)

if __name__ == "__main__":
    main()
