---

# Cyclone Machine Time-Series — EDA ▪ Shutdowns ▪ Operating States ▪ Contextual Anomalies ▪ 1-Hour Forecast

>**Pure-Python CLI (Windows-friendly, deterministic)**

**Author:** Paras Saxena  

- **What:** A scriptable pipeline for **industrial 5-minute telemetry** (cyclone machine).  
- **Why:** Discover **shutdown/idle periods**, **characterize operating states**, flag **contextual anomalies** safely, and **forecast 1 hour ahead** to support **reliability & maintenance planning**.  
- **Inputs:** CSV or Excel (`.xlsx`) with a timestamp column and sensor variables.  
- **Granularity:** Resampled to a **strict 5-minute grid** (deterministic).  
- **Outputs:** All CSVs/PNGs in `outputs/`, logs in `logs/`. *(You’re committing outputs + index + corpus so reviewers can verify instantly.)*

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

## 🗂️ Repository Structure

```
Paras_Saxena_Task1/
  README.md
  requirements.txt
  data.xlsx                 # assignment data (committed for review)
  logs/                     # run logs
  outputs/                  # CSVs / plots
  scripts/
    __init__.py             # enables `python -m scripts.run_*`
    run_eda.py
    run_detect_shutdowns.py
    run_cluster_states.py
    run_detect_anomalies.py
    run_forecast_1h.py
  src/
    __init__.py
    config.py               # all constants with comments
    utils_io.py             # robust IO, timestamp parse, 5-min resample
    features.py             # rolling features (15/30/60 min)
    eda.py
    detect_shutdowns.py
    cluster_states.py
    detect_anomalies.py
    forecast_1h.py
```

> Tip: keeping `__init__.py` in `scripts/` lets you run commands in **module mode** (`python -m scripts.run_*`), which avoids PYTHONPATH issues.

---

## ✅ Dev Preflight Checklist [Setup-Summary] *(copy-paste)*

> Run from project root `Paras_Saxena_Task1/`

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

## 🚀 Quick Start (Full Pipeline)

**Windows PowerShell**

```powershell
python -m scripts.run_eda              --data_path "data.xlsx" --outdir outputs
python -m scripts.run_detect_shutdowns --data_path "data.xlsx" --outdir outputs
python -m scripts.run_cluster_states   --data_path "data.xlsx" --outdir outputs --shutdowns_csv "outputs\shutdown_periods.csv"
python -m scripts.run_detect_anomalies --data_path "data.xlsx" --outdir outputs --clusters_csv "outputs\state_labels.csv"
python -m scripts.run_forecast_1h      --data_path "data.xlsx" --outdir outputs --target_col "Cyclone_Inlet_Gas_Temp"
```

**Bash/macOS**

```bash
python -m scripts.run_eda --data_path data.xlsx --outdir outputs && \
python -m scripts.run_detect_shutdowns --data_path data.xlsx --outdir outputs && \
python -m scripts.run_cluster_states --data_path data.xlsx --outdir outputs --shutdowns_csv outputs/shutdown_periods.csv && \
python -m scripts.run_detect_anomalies --data_path data.xlsx --outdir outputs --clusters_csv outputs/state_labels.csv && \
python -m scripts.run_forecast_1h --data_path data.xlsx --outdir outputs --target_col Cyclone_Inlet_Gas_Temp
```

Tail logs live (Windows):

```powershell
Get-Content .\logs\run_eda.log -Wait
Get-Content .\logs\run_detect_shutdowns.log -Wait
Get-Content .\logs\run_cluster_states.log -Wait
Get-Content .\logs\run_detect_anomalies.log -Wait
Get-Content .\logs\run_forecast_1h.log -Wait
```

---

## 🧪 Run Specific Steps Only

```powershell
# EDA
python -m scripts.run_eda --data_path "data.xlsx" --outdir outputs

# Shutdown detection
python -m scripts.run_detect_shutdowns --data_path "data.xlsx" --outdir outputs

# Clustering (uses shutdowns to mask idle if provided)
python -m scripts.run_cluster_states --data_path "data.xlsx" --outdir outputs --shutdowns_csv "outputs\shutdown_periods.csv"

# Contextual anomalies (needs state_labels.csv)
python -m scripts.run_detect_anomalies --data_path "data.xlsx" --outdir outputs --clusters_csv "outputs\state_labels.csv"

# 1-hour ahead forecast (12×5-min)
python -m scripts.run_forecast_1h --data_path "data.xlsx" --outdir outputs --target_col "Cyclone_Inlet_Gas_Temp"
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

## 🧠 Methods (short & practical)

**EDA**
Robust timestamp parse → enforce 5-min grid → summarize missingness, basic stats, correlations → time-sliced views (last week / rolling year) for drift & regimes.

**Shutdowns (rule-based)**
Domain thresholds on key sensors + **minimum duration** to avoid chatter; export consolidated intervals.

**Operating States (clustering)**
Rolling features over 15/30/60 min (mean/std/diff), standardize, KMeans with `K∈[2..8]`; pick K by **silhouette**; name clusters in comments for interpretability (e.g., *High Load*, *Degraded*, *Idle*).

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

* Global seed in `config.py`
* No notebooks; all steps are CLIs with pinned dependencies
* Logs show start/end times, shapes, thresholds, K selection, and backtest folds

---


## 📜 License & Data

This repository contains **sample documents** and **derived artifacts** strictly for evaluation and learning. If you fork or reuse, ensure you have rights to redistribute your own documents and respect any proprietary content.


---
