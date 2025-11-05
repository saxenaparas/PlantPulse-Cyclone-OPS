# 🤖 PlantPulse-Cyclone-OPS
> - Time-series analysis pipeline for cyclone/plant telemetry — shutdown detection, behavioral clustering, contextual anomaly detection, and 1‑hour forecasting.
> - Cyclone Machine Time-Series — EDA ▪ Shutdowns ▪ Operating States ▪ Contextual Anomalies ▪ 1-Hour Forecast  
> - Clean, deterministic, **Pure-Python CLI Script (Windows-friendly, deterministic & no notebooks)** that run the full pipeline on `data.xlsx` (5‑minute telemetry).

**Author:** Paras Saxena  

It's a scriptable pipeline for **industrial 5-minute telemetry** (cyclone machine) & helps to discover **shutdown/idle periods**, **characterize operating states**, flag **contextual anomalies** safely, and **forecast 1 hour ahead** to support **reliability & maintenance planning**.  

- **Steps:** EDA → Shutdown detection → Clustering (states) → Contextual anomalies → 1‑hour forecasting (backtest + model comparison).
- **Entrypoint:** `analysis.py` — single orchestrator that will re-run missing steps or run selected steps.  
- **Inputs:** CSV or Excel (`.xlsx`) with a timestamp column and sensor variables.
- **Granularity:** Resampled to a **strict 5-minute grid** (deterministic).  
- **Outputs:** All CSVs/PNGs in `outputs/`, logs in `logs/`. *(Committing outputs + index + corpus so results can be verifed instantly.)*

---

## ✨ Highlights

* **End-to-end CLI** — each step is a command with logs + reproducible artifacts
* **Strict 5-min index** — consistent grid for EDA, clustering & forecasting
* **Rule-based shutdowns** — minimum duration + configurable thresholds
* **Interpretable states** — rolling-window features + KMeans + silhouette selection
* **Contextual anomalies** — per-state IsolationForest + rolling-MAD fallback
* **Forecasting (1 h / 12×5-min)** — naïve vs seasonal-naïve vs ARIMA vs GBM lags
* **Deterministic** — global `RANDOM_SEED`, pinned deps, same data ⇒ same results

---

## 🌟 Important Features

- `analysis.py` orchestrates the full pipeline (full or quick modes). It checks for required CSV artifacts and will re-run missing steps unless `--quick` is used.
- Per‑step timeouts: `--quick` uses a conservative per‑step timeout (default **600s**). Use `--timeout <seconds>` to override for all steps. Use `--timeout 0` to disable timeouts (or provide a very large value like `--timeout 86400`) if you want the pipeline to run until completion.
- Forecasting can compares ≥2 methods and writes the comparison to `outputs/backtest_metrics.csv` and `outputs/forecasts.csv` (model, y_true, y_pred, timestamp). See **Forecast notes** below.
- Plots: `analysis.py` copies key PNGs to `plots/` (single-year view, anomaly annotations, forecast plot) so the visual story is easy to browse on GitHub.

---

## 🗂️ Repository Structure

```
PlantPulse-Cyclone-OPS/
├─ README.md
├─ analysis.py                 # Single CLI orchestration entrypoint
├─ data.xlsx                   # analysis data (committed for review not committed for public use unless allowed)
├─ logs/                       # run logs
<<<<<<< HEAD
│   ├─ analysis.log
│   ├─ run_eda.log
│   ├─ run_detect_shutdowns.log
│   ├─ run_cluster_states.log
│   ├─ run_detect_anomalies.log
│   └─ run_forecast_1h.log
├─ outputs/                    # CSVs & PNGs (created by scripts)
│   ├─ eda_summary.csv
│   ├─ correlations.png
│   ├─ week_view.png
│   ├─ year_view.png
│   ├─ shutdown_periods.csv
│   ├─ shutdown_plot.png
│   ├─ clusters_summary.csv
│   ├─ state_labels.csv
│   ├─ cluster_scatter.png
│   ├─ elbow.png
│   ├─ silhouette.png
│   ├─ anomalous_periods.csv
│   ├─ anomalies_plot.png
│   ├─ forecasts.csv            # created after full forecast run
│   ├─ backtest_metrics.csv     # created after full forecast run (MAE/RMSE)
│   └─ forecast_plot.png
├─ plots/                       # README-friendly copy of important PNGs (created by analysis.py)
├─ requirements.txt
├─ scripts/
│   ├─ __init__.py              # enables `python -m scripts.run_*`
│   ├─ run_eda.py
│   ├─ run_detect_shutdowns.py
│   ├─ run_cluster_states.py
│   ├─ run_detect_anomalies.py
│   └─ run_forecast_1h.py
└─ src/
    ├─ __init__.py
    ├─ config.py                 # all constants with comments
    ├─ utils_io.py               # robust IO, timestamp parse, 5-min resample
    ├─ eda.py
    ├─ detect_shutdowns.py
    ├─ cluster_states.py
    ├─ detect_anomalies.py
    ├─ features.py               # rolling features (15/30/60 min)
    └─ forecast_1h.py
=======
│  ├─ analysis.log
│  ├─ run_eda.log
│  ├─ run_detect_shutdowns.log
│  ├─ run_cluster_states.log
│  ├─ run_detect_anomalies.log
│  └─ run_forecast_1h.log
├─ outputs/                    # CSVs & PNGs (created by scripts)
│  ├─ eda_summary.csv
│  ├─ correlations.png
│  ├─ week_view.png
│  ├─ year_view.png
│  ├─ shutdown_periods.csv
│  ├─ shutdown_plot.png
│  ├─ clusters_summary.csv
│  ├─ state_labels.csv
│  ├─ cluster_scatter.png
│  ├─ elbow.png
│  ├─ silhouette.png
│  ├─ anomalous_periods.csv
│  ├─ anomalies_plot.png
│  ├─ forecasts.csv             # created after full forecast run
│  ├─ backtest_metrics.csv      # created after full forecast run (MAE/RMSE)
│  └─ forecast_plot.png
├─ plots/                       # README-friendly copy of important PNGs (created by analysis.py)
├─ requirements.txt
├─ scripts/
│  ├─ __init__.py               # enables `python -m scripts.run_*`
│  ├─ run_eda.py
│  ├─ run_detect_shutdowns.py
│  ├─ run_cluster_states.py
│  ├─ run_detect_anomalies.py
│  └─ run_forecast_1h.py
└─ src/
├─ **init**.py
├─ config.py                    # all constants with comments
├─ utils_io.py                  # robust IO, timestamp parse, 5-min resample
├─ eda.py
├─ detect_shutdowns.py
├─ cluster_states.py
├─ detect_anomalies.py
├─ features.py                  # rolling features (15/30/60 min)
└─ forecast_1h.py
>>>>>>> 249dd1df46b1cac7eaa6ed069e555d8f1cb39c60
```

> Tip: keeping `__init__.py` in `scripts/` lets you run commands in **module mode** (`python -m scripts.run_*`), which avoids PYTHONPATH issues.

---

## ✅ Dev Preflight Checklist [Setup-Summary] *(copy-paste)*

> Run from project root `PlantPulse-Cyclone-OPS/`

**Windows PowerShell (Python 3.11 recommended):**

```powershell
py -0p
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
if (!(Test-Path scripts\__init__.py)) { New-Item -ItemType File -Path scripts\__init__.py | Out-Null }
```

**Linux/macOS (bash):**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
: > scripts/__init__.py
```

---

### 🚀 Quick Start (Full Pipeline)

>**Quick preflight (fast) — runs only checks and quick versions of heavy steps**

```powershell
python analysis.py --data_path "data.xlsx" --outdir "outputs" --plots_dir "plots" --quick
```


### 💼 Full run (no time limit): disable timeouts or increase timeout

**Option A:** disable timeouts (if supported)
```powershell
python analysis.py --data_path "data.xlsx" --outdir "outputs" --plots_dir "plots" --force --timeout 0
```
**Option B:** allow long but finite time (e.g., 1 hour per step)
```
python analysis.py --data_path "data.xlsx" --outdir "outputs" --plots_dir "plots" --force --timeout 3600
```

#### 🚩 Final One‑Liner (Run Everything, Full Fidelity)
```powershell
python analysis.py --data_path "data.xlsx" --outdir "outputs" --plots_dir "plots" --force --timeout 0 
```

> 📜 **Notes:**
>
> * `--quick` = conservative, short timeouts (600s per step) and sampling where applicable so reviewers can get a result fast.
> * `--force` forces re-run of steps even if outputs already exist.
> * If you encounter a heavy step (ARIMA fitting / clustering on full records), use `--timeout` to increase or `--force` + `--quick` to run a sampled/faster variant.

> 📌 **Practical Guidance for Use:**
> - If you have a laptop/desktop with 8+ cores and 16+ GB RAM: try `--timeout 3600` (1 hour) for full fidelity. For exact reproducibility on the original dataset, supply `--timeout 0` to let the run finish.
> - If you need a fast check: use the `--quick` flag.


### 🆚 Quick vs Full mode (detailed)

- **Quick mode** (`--quick`): default per-step timeout = **600 seconds (10 minutes)**. Recommended for reviewers who want to verify functionality without waiting for heavy model fits.

> Sampling applied in heavy steps (e.g., silhouette search may use a sampled subset; ARIMA autoregression search is limited).

- **Full mode** (use `--force` and a large `--timeout` or `--timeout 0`): runs using full data and full hyperparameter searches. Expect heavy steps (clustering on ~378k rows, ARIMA backtesting) to take significantly longer — potentially tens of minutes to hours depending on machine.

### 🧪 Run Specific Steps Only

```powershell
# EDA
python -m scripts.run_eda --data_path "data.xlsx" --outdir "outputs"

# Shutdown detection
python -m scripts.run_detect_shutdowns --data_path "data.xlsx" --outdir "outputs"

# Clustering (uses shutdowns to mask idle if provided)
python -m scripts.run_cluster_states --data_path "data.xlsx" --outdir "outputs" --shutdowns_csv "outputs/shutdown_periods.csv"

# Contextual anomalies (needs state_labels.csv)
python -m scripts.run_detect_anomalies --data_path "data.xlsx" --outdir "outputs" --clusters_csv "outputs/state_labels.csv"

# 1-hour ahead forecast (12×5-min)
python -m scripts.run_forecast_1h --data_path "data.xlsx" --outdir "outputs" --target_col "Cyclone_Inlet_Gas_Temp"

```

---

## ⚙️ Configuration (edit `src/config.py`)

Key constants (CLI may override some):

* `RANDOM_SEED = 42` — determinism
* `FREQ_MIN = 5` — strict resampling cadence
* `TARGET_COL = "TargetVar"` — set your forecast target (CLI `--target_col`)
* `ROLL_WINDOWS_MIN = [15, 30, 60]` — feature windows
* `SHUTDOWN_THRESHOLDS = {...}` — rule params for shutdown/idle
* `MIN_DURATION_MIN = 30` — minimum continuous shutdown length
* `IDLE_WINDOW_MIN = 20` — smoothing window for idle
* `KMIN = 2, KMAX = 8` — K search range for KMeans
* `IFOREST_N_ESTIMATORS = 200` — IsolationForest size
* `FORECAST_HORIZON_MIN = 60` — horizon = 12×5-min
* `SEASONAL_PERIOD_5MIN = 288` — 24h seasonality for 5-min grid

---

## 🧿 How `analysis.py` Behaves (Transparent Execution)

- It checks for required artifacts in `outputs/`. If files are present, that step is skipped (to preserve reproducibility and save time).
- If a required file is missing, `analysis.py` will run the corresponding script and stream the child process logs to `logs/<step>.log` and to the terminal.
- Each step has a per-step timeout (controlled by `--timeout` when using `analysis.py`). If a step exceeds the timeout, the process is killed and the pipeline reports which outputs are missing.

> This design lets an user either run `--quick` for a fast verification or run full without time limits for exact reproduction.

---

## 🧐 Where to Look for Results

- `outputs/eda_summary.csv` — EDA summary
- `outputs/shutdown_periods.csv` — detected shutdowns (start,end,duration_min,reason)
- `outputs/clusters_summary.csv` & `outputs/state_labels.csv` — cluster labels and summary
- `outputs/anomalous_periods.csv` — anomaly intervals with method & score
- `outputs/forecasts.csv` & `outputs/backtest_metrics.csv` — forecasting predictions and metrics
- `plots/one_year_with_shutdowns.png` and `plots/anomaly_*.png` — curated visuals

---

## 🧠 Methods (short & practical)

**EDA**
Robust timestamp parse → enforce 5-min grid → summarize missingness, basic stats, correlations → time-sliced views (last week / rolling year) for drift & regimes.

**Shutdowns (rule-based)**
Domain thresholds on key sensors + **minimum duration** to avoid chatter; export consolidated intervals.

**Operating States (clustering)**
Rolling features over 15/30/60 min (mean/std/diff), standardize, KMeans with **`K∈[2..8]`**; pick K by **silhouette**; name clusters in comments for interpretability (e.g., *High Load*, *Degraded*, *Idle*).

**Contextual Anomalies**
Per-state **IsolationForest** (so “weird for *this* state”), plus **rolling-MAD** as a simple statistical back-stop; de-duplicate overlaps; export intervals + scores.

**1-Hour Forecast**
Compare **Naïve**, **Seasonal-Naïve** (daily), **ARIMA** (small auto grid), and **GradientBoosting** on lag features. Rolling backtest → `MAE/RMSE` table; store per-timestamp predictions.

---

## 📦 Artifacts (written to `outputs/`)

| Step                  | Files                                                                                            |
| --------------------- | ------------------------------------------------------------------------------------------------ |
| **EDA**               | `eda_summary.csv`, `correlations.png`, `week_view.png`, `year_view.png`                          |
| **Shutdowns**         | `shutdown_periods.csv` *(start,end,duration_min,reason)*, `shutdown_plot.png`                    |
| **States (Clusters)** | `clusters_summary.csv`, `state_labels.csv`, `cluster_scatter.png`, `elbow.png`, `silhouette.png` |
| **Anomalies**         | `anomalous_periods.csv` *(start,end,duration_min,state,method,score)*, `anomalies_plot.png`      |
| **Forecast (1h)**     | `forecasts.csv` *(timestamp,y_true,y_pred,model)*, `backtest_metrics.csv`, `forecast_plot.png`   |

---

## ⚡ Quick-Eval Speed Patches (optional, assignment-safe)

Heavy parts are **silhouette** (O(n²)) and ARIMA backtests. These patches keep the methodology intact while cutting wall-clock time.

* **Clustering** — sample silhouette + reduce K-search:

```powershell
# Windows PowerShell (edit in-place)
(Get-Content -Raw .\src\cluster_states.py) `
  -replace 'silhouette_score\(X, labels\)', 'silhouette_score(X, labels, sample_size=min(20000, len(X)), random_state=CFG.RANDOM_SEED)' `
  | Set-Content .\src\cluster_states.py

(Get-Content -Raw .\src\config.py) `
  -replace 'KMAX:\s*int\s*=\s*8', 'KMAX: int = 5' `
  | Set-Content .\src\config.py
```

* **Forecasting** — stabilize/limit ARIMA work:

```powershell
(Get-Content -Raw .\src\forecast_1h.py) `
  -replace 'def walk_forward_backtest\(y, outdir\):',
           "def walk_forward_backtest(y, outdir):`r`n    y = y.asfreq('5min')" `
  | Set-Content .\src\forecast_1h.py

(Get-Content -Raw .\src\forecast_1h.py) `
  -replace 'fit = model\.fit\(method_kwargs=\{"warn_convergence": False\}\)',
           "fit = model.fit(method_kwargs={""warn_convergence"": False, ""maxiter"": 50})" `
  | Set-Content .\src\forecast_1h.py
```

> These do **not** change the 1-hour horizon, strict indexing, or model set—only compute time.

---

## 🧰 Troubleshooting

* **Pandas/Scipy build errors:** use **Python 3.11** (the venv commands above do this).
* **PowerShell activation blocked:** run the `Set-ExecutionPolicy` line once per shell.
* **`ModuleNotFoundError: src`:** always use **module mode** (`python -m scripts.run_*`).
* **Clustering takes too long:** apply the clustering patch above (sampled silhouette).
* **Backtest slow:** apply ARIMA patch; watch progress in `logs/run_forecast_1h.log`.

---

## 🔁 Reproducibility

- Global seed in `config.py`
- No notebooks; all steps are CLIs with pinned dependencies
- Logs show start/end times, shapes, thresholds, K selection, and backtest folds

---

## 📊 Plots & Annotated Year View

Here, `analysis.py` creates a curated `plots/` folder containing:

- `one_year_with_shutdowns.png` — one full-year overview with shutdown intervals highlighted.
- `anomaly_1.png` .. `anomaly_6.png` — up to 6 annotated anomaly zoom-ins with short root‑cause note placeholders.
- `forecast_plot.png` — forecast vs true for a validation window.

> These images are also copied into `outputs/` (PNG)

---


## 📑 Forecast Notes & Interpretation (RMSE / MAE)

- The forecasting step performs a walk‑forward backtest and compares multiple models (naive, seasonal naive, ARIMA, and a tree model like `GradientBoostingRegressor`).
- The backtest writes `outputs/backtest_metrics.csv` with columns: `model, fold, mae, rmse` and `outputs/forecasts.csv` containing `timestamp,y_true,y_pred,model`.

### 🔎 How to Interpret Metrics

- **MAE (Mean Absolute Error):** average absolute error in the same units as the target. Lower = better. Good for robust error magnitude.
- **RMSE (Root Mean Squared Error):** penalizes larger errors more strongly. Useful to identify models that occasionally make large misses.

A short summary (example you might add to README after running):

```
Forecast Comparison (Example)
-------------------------=-------=------
          Model          |  MAE  | RMSE
-------------------------|-------|------
  Naive                  | 0.84  | 1.12
  Seasonal_Naive         | 0.76  | 1.03
  GradientBoosting       | 0.62  | 0.90  <- winner in this dataset
  ARIMA (auto)           | 0.68  | 0.97
-------------------------=-------=------
```

---

## 🕵🏻 Insights & Recommendations

1. **Frequent short shutdowns dominate downtime.** Many shutdown intervals are short (e.g., < 30 minutes) — investigate whether these are control trips or measurement artifacts and target top 20% culprits for operational fixes.
2. **State clusters separate production vs idle/maintenance modes clearly.** Use cluster labels to build state‑aware dashboards; contextual anomaly detectors should only run in relevant states (avoid flagging expected behavior during maintenance).
3. **Anomalies often precede shutdowns by short windows.** Use anomaly scores as an early‑warning signal to reduce unplanned downtime—test a simple alert: anomaly score > threshold sustained for 15–20 minutes.
4. **Forecasting shows seasonal (daily) structure at 5‑min granularity.** Tree‑based models with lag features gave the best backtest performance (lower RMSE) on our data; deploy these in a short horizon ensemble with a seasonal naive fallback for reliability.
5. **Data hygiene:** enforce strict 5‑min indexing at ingestion and surface gaps > 15 minutes — these gaps materially affect rolling features and ARIMA fitting.

---

## 📜 License & Data

This repository contains **sample documents** and **derived artifacts** strictly for evaluation and learning. If you fork or reuse, ensure you have rights to redistribute your own documents and respect any proprietary content.

---