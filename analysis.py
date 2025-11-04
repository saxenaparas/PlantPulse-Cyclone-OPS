#!/usr/bin/env python3
"""
analysis.py - single runnable orchestration entrypoint for PlantPulse-Cyclone-OPS

Features:
- Runs the pipeline steps in order by invoking the scripts as modules:
    python -m scripts.run_eda ...
- Checks for expected output CSVs and skips steps if files already exist (unless --force).
- Streams subprocess stdout/stderr to console and to logs/<step>.log.
- Creates logs/analysis.log with detailed timestamps and a final summary.
- Per-step timeout support so long-running steps can be bounded (--timeout).
- Robust reading of CFG from src.config whether CFG is a dict or class instance.
"""

from __future__ import annotations
import argparse
import subprocess
import sys
import os
from pathlib import Path
import logging
import time
import datetime
import shlex
import threading

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
OUT_DIR_DEFAULT = ROOT / "outputs"

# Pipeline step definitions: name -> (module, list_of_args_as_strings, expected_output_paths)
# We will call: python -m <module> <args...>
STEPS = [
    {
        "name": "eda",
        "module": "scripts.run_eda",
        "args": ["--data_path", "{data_path}", "--outdir", "{outdir}"],
        "expected": ["{outdir}/eda_summary.csv", "{outdir}/correlations.png"]
    },
    {
        "name": "shutdowns",
        "module": "scripts.run_detect_shutdowns",
        "args": ["--data_path", "{data_path}", "--outdir", "{outdir}"],
        "expected": ["{outdir}/shutdown_periods.csv", "{outdir}/shutdown_plot.png"]
    },
    {
        "name": "clusters",
        "module": "scripts.run_cluster_states",
        "args": ["--data_path", "{data_path}", "--outdir", "{outdir}", "--shutdowns_csv", "{outdir}/shutdown_periods.csv"],
        "expected": ["{outdir}/clusters_summary.csv", "{outdir}/state_labels.csv", "{outdir}/cluster_scatter.png"]
    },
    {
        "name": "anomalies",
        "module": "scripts.run_detect_anomalies",
        "args": ["--data_path", "{data_path}", "--outdir", "{outdir}", "--clusters_csv", "{outdir}/state_labels.csv"],
        "expected": ["{outdir}/anomalous_periods.csv", "{outdir}/anomalies_plot.png"]
    },
    {
        "name": "forecast",
        "module": "scripts.run_forecast_1h",
        "args": ["--data_path", "{data_path}", "--outdir", "{outdir}"],
        "expected": ["{outdir}/forecasts.csv", "{outdir}/backtest_metrics.csv", "{outdir}/forecast_plot.png"]
    },
]


def setup_logging(logpath: Path):
    logpath.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("analysis")
    logger.setLevel(logging.INFO)
    # clear any handlers
    if logger.handlers:
        for h in logger.handlers[:]:
            logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(str(logpath), mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def safe_get_cfg_attr(cfg, name, default=None):
    """Return attribute either from object with getattr or mapping .get"""
    try:
        return getattr(cfg, name)
    except Exception:
        try:
            return cfg.get(name, default)  # type: ignore
        except Exception:
            return default


def check_expected_files(outdir: Path, expected_patterns: list[str]) -> tuple[bool, list[Path]]:
    """Return (all_exist, list_missing)"""
    missing = []
    for pat in expected_patterns:
        p = Path(pat.format(outdir=str(outdir)))
        if not p.exists():
            missing.append(p)
    return (len(missing) == 0, missing)


def stream_subprocess(cmd: list[str], logfile: Path, timeout: int | None, logger: logging.Logger):
    """
    Run subprocess, stream stdout/stderr to console and write to logfile.
    Returns exitcode. Raises subprocess.TimeoutExpired on timeout.
    """
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("w", encoding="utf-8", buffering=1) as fh:
        fh.write(f"# Command: {' '.join(shlex.quote(s) for s in cmd)}\n")
        fh.write(f"# Started: {datetime.datetime.now().isoformat()}\n\n")
        fh.flush()
        # Start process
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True)
        start = time.time()

        def reader_thread(pipe, fh, logger):
            for line in iter(pipe.readline, ""):
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                out_line = f"{ts} | {line.rstrip()}"
                # write to file
                fh.write(out_line + "\n")
                fh.flush()
                # also print to console
                print(out_line)
            pipe.close()

        thread = threading.Thread(target=reader_thread, args=(proc.stdout, fh, logger), daemon=True)
        thread.start()

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            thread.join(timeout=2)
            fh.write(f"\n# TIMEOUT after {timeout} seconds. Process killed.\n")
            fh.flush()
            logger.error("Process timed out and was killed: %s", cmd)
            raise
        thread.join(timeout=2)
        end = time.time()
        fh.write(f"\n# Finished: {datetime.datetime.now().isoformat()} (duration {end - start:.1f}s)\n")
        fh.flush()
        return proc.returncode


def run_step(step_def: dict, data_path: str, outdir: Path, logger: logging.Logger, force: bool, timeout: int | None):
    name = step_def["name"]
    module = step_def["module"]
    # prepare expected patterns and args
    expected = [p.format(outdir=str(outdir)) for p in step_def.get("expected", [])]
    expected_paths = [Path(p) for p in expected]
    all_exist = all(p.exists() for p in expected_paths)
    if all_exist and not force:
        logger.info("SKIP %s: expected outputs exist (%s). Use --force to re-run.", name, ", ".join(str(p.name) for p in expected_paths))
        return {"name": name, "skipped": True, "code": 0, "missing": []}
    # Build command
    args = []
    for a in step_def.get("args", []):
        args.append(a.format(data_path=data_path, outdir=str(outdir)))
    cmd = [sys.executable, "-m", module] + args
    logger.info("START step=%s module=%s cmd=%s", name, module, " ".join(shlex.quote(x) for x in cmd))
    step_log = LOG_DIR / f"{name}.log"
    try:
        code = stream_subprocess(cmd, step_log, timeout, logger)
        if code != 0:
            logger.error("Step %s finished with non-zero exit code: %s (see %s)", name, code, step_log)
        else:
            logger.info("OK   step=%s finished successfully (see %s)", name, step_log)
        # after run check missing outputs
        missing = [str(p) for p in expected_paths if not p.exists()]
        if missing:
            logger.warning("Step %s completed but these expected files are missing: %s", name, missing)
        return {"name": name, "skipped": False, "code": code, "missing": missing}
    except subprocess.TimeoutExpired:
        logger.exception("Step %s timed out after %s seconds. See %s for partial logs.", name, timeout, step_log)
        return {"name": name, "skipped": False, "code": -1, "missing": [str(p) for p in expected_paths if not p.exists()]}


def main():
    parser = argparse.ArgumentParser(description="Run pipeline orchestration (analysis.py)")
    parser.add_argument("--data_path", required=True, help="path to data.xlsx or csv")
    parser.add_argument("--outdir", default=str(OUT_DIR_DEFAULT), help="outputs directory")
    parser.add_argument("--plots_dir", default=str(ROOT / "plots"), help="where to copy/write plots (optional)")
    parser.add_argument("--quick", action="store_true", help="quick mode: still runs missing steps but uses conservative timeouts")
    parser.add_argument("--force", action="store_true", help="force re-run steps even if outputs exist")
    parser.add_argument("--timeout", type=int, default=None, help="global per-step timeout in seconds (overrides quick defaults)")
    args = parser.parse_args()

    data_path = str(Path(args.data_path).resolve())
    outdir = Path(args.outdir)
    plots_dir = Path(args.plots_dir)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    analysis_log = LOG_DIR / "analysis.log"
    logger = setup_logging(analysis_log)

    logger.info("analysis.py started")
    logger.info("data_path=%s outdir=%s plots_dir=%s quick=%s force=%s", data_path, outdir, plots_dir, args.quick, args.force)

    # smart defaults for quick mode
    if args.timeout is not None:
        global_timeout = args.timeout
    else:
        if args.quick:
            # small timeouts to avoid getting stuck (sensible defaults)
            global_timeout = 600  # 10 minutes per step
        else:
            global_timeout = None  # no timeout

    # Validate data file
    if not Path(data_path).exists():
        logger.error("Data file not found: %s", data_path)
        sys.exit(2)

    # ensure outdir and plots_dir exist
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Try to load CFG if present (non-fatal)
    try:
        from src import config as _cfg_mod  # may raise
        CFG = getattr(_cfg_mod, "CFG", _cfg_mod)
    except Exception as e:
        CFG = {}
        logger.warning("Could not import src.config (proceeding with defaults). Error: %s", e)

    # ensure deterministic seed is honored if present
    DEFAULT_SEED = 42
    seed = safe_get_cfg_attr(CFG, "RANDOM_SEED", DEFAULT_SEED)
    try:
        seed = int(seed)
    except Exception:
        seed = DEFAULT_SEED
    logger.info("Using RANDOM_SEED=%s", seed)

    # Run steps in order
    results = []
    for s in STEPS:
        info = run_step(s, data_path, outdir, logger, force=args.force, timeout=global_timeout)
        results.append(info)
        # short pause between steps so console output is readable
        time.sleep(0.5)

    # final summary
    logger.info("Pipeline Summary:")
    for r in results:
        logger.info(" - %s: skipped=%s code=%s missing=%s", r["name"], bool(r.get("skipped")), r.get("code"), r.get("missing"))

    # copy core PNGs into plots_dir (if present)
    pngs = list((outdir).glob("*.png"))
    for p in pngs:
        tgt = plots_dir / p.name
        try:
            # atomic copy
            from shutil import copy2
            copy2(p, tgt)
        except Exception as e:
            logger.warning("Failed to copy %s -> %s : %s", p, tgt, e)

    logger.info("Copied %d png(s) to %s", len(pngs), plots_dir)
    logger.info("analysis.py finished")
    print("\nSUMMARY (also written to logs/analysis.log):")
    for r in results:
        print(f" - {r['name']}: skipped={r.get('skipped')} code={r.get('code')} missing={r.get('missing')}")
    # exit code rule: 0 if all codes 0 or skipped, else 1
    bad = [r for r in results if r.get("code", 0) not in (0, None)]
    if bad:
        logger.error("One or more steps failed or timed out.")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
