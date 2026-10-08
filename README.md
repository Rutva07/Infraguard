# InfraGuard — Predictive Failure Detection

**Python · XGBoost · scikit-learn · AWS S3 · Amazon Athena · AWS Glue · SQL**

A complete, runnable reference implementation of an infrastructure telemetry pipeline that generates or ingests CPU/memory/power/temperature measurements, computes **causal time-series features**, identifies anomalous behavior, predicts future equipment failures, and evaluates a classifier on a **chronologically held-out test period**. Optional AWS commands store data in S3 as Parquet, register a Glue table, and query it with Athena.

> **Data integrity:** Synthetic telemetry is *simulated*, not recorded from real data centers. No 0.91 AUROC (or any other metric) is claimed before running the benchmark. Realistic production effectiveness requires evaluation on genuine labeled failure logs. AWS usage has not been verified without real credentials.

No Docker. No AWS credentials needed for the local pipeline.

## 1. Quick start (local, no AWS)

Requires Python 3.10+ and a virtual environment.

```bash
cd InfraGuard
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
```

**Small end-to-end demo**, designed for a laptop:

```bash
python -m infraguard generate --devices 18 --steps 550 --output data/raw/demo.csv.gz
python -m infraguard train --input data/raw/demo.csv.gz --output-dir artifacts/demo --estimators 100
python -m infraguard evaluate --input data/raw/demo.csv.gz --model-dir artifacts/demo
python -m infraguard predict --input data/raw/demo.csv.gz --model-dir artifacts/demo --output data/processed/demo_predictions.csv.gz
```

Or `bash scripts/run_demo.sh` (macOS/Linux). If using the console entry point installed by pip, `infraguard` can replace `python -m infraguard`.

**Full 1,040,000-row experiment:**

```bash
python -m infraguard generate --devices 200 --steps 5200 --seed 42 \
  --output data/raw/synthetic_telemetry.csv.gz
python -m infraguard train --input data/raw/synthetic_telemetry.csv.gz \
  --output-dir artifacts/full --estimators 180 --purge-steps 30
python -m infraguard evaluate --input data/raw/synthetic_telemetry.csv.gz \
  --model-dir artifacts/full
```

The full run requires substantially more memory and CPU time than the demo because all raw records and rolling features are processed in memory. Start with the demo if resources are limited. **This repository ships a generated full dataset and its trained artifacts** in the downloadable archive, plus measured results in `reports/full_synthetic_experiment.md`; they are excluded from Git by `.gitignore` to avoid committing large files by default.

## 2. Dataset & target

Each record represents a sampled server reading, including:

| Column | Meaning |
| --- | --- |
| `timestamp` | Time of reading, UTC |
| `device_id` | Server identifier (not given to the classifier) |
| `cpu_pct` / `memory_pct` | Utilization percentages |
| `temperature_c` / `power_w` | Thermal and power measurements |
| `fan_rpm` / `disk_io_mb_s` | Additional equipment readings |
| `failure_event` | Observed failure **at** this timestamp: audit information only |
| `failure_next_horizon` | Binary target: failure within the **next 30 samples**, excluding current time |

`failure_next_horizon` is the supervised target, not an input feature. Simulated devices have isolated periods of leading degradation, faults, noise, and benign spikes. The simulator intentionally yields examples of both true and false warning patterns; it is not a calibrated model of equipment-failure prevalence.

For 1-minute sampling, the default target predicts a failure within the **next 30 minutes**. If changing the simulator's `--horizon-steps`, also adjust `--purge-steps` to be at least that large for training. For real telemetry, you must derive targets from separately recorded future failure-event logs using exactly this prospective convention.

## 3. Feature engineering

`src/infraguard/features.py` computes trailing-only features for each device:

- Raw CPU, memory, temperature, power, fan, and disk telemetry.
- Rolling CPU/memory/temperature/power means over 5, 15, and 30 historical/current samples.
- One-step deltas and deviation from the trailing 15-sample mean.
- CPU/temperature interaction, cyclic hour-of-day encoding, and a simple **heuristic, unsupervised** anomaly score.

All rolling windows include only the current and previous records of the **same device**. The target and current failure-event flag are never used as predictive inputs. For online scoring, supply historical readings (ideally 30 per device) with the current batch. Model input excludes timestamps and device IDs, except for the cyclical hour representation computed from timestamps.

The anomaly score is a relative-deviation heuristic (not an independently calibrated anomaly detection model); predictive failure detection is supervised with XGBoost.

## 4. Training & evaluation methodology

1. Order observations by device and timestamp and compute causal trailing features.
2. Split on the **global chronological timeline**, default **65% train / 17% validation / remaining test**, with a 30-timestamp purge around each boundary. The purge prevents future-looking failure labels from crossing into adjacent periods.
3. Fit missing-value imputation on **training data only**.
4. Train XGBoost with imbalance-aware positive weights and (for sufficiently long training) early stopping on validation average precision.
5. Train a logistic regression baseline with standardized, imputed features.
6. Choose **each threshold only on validation F1**, then evaluate once on the held-out test period.
7. Save both model bundles and `metrics.json` (AUROC, average precision / PR-AUC, precision, recall, F1, confusion matrix, class counts, thresholds and split dates).

The test period contains future timestamps for the *same devices*, so the benchmark measures **temporal generalization**, not generalization to never-before-seen devices. A device-group holdout would be an additional experiment. The current demo uses one simulator seed; robust empirical results should include multiple seeds and a genuine dataset.

### Model output

- `artifacts/<run>/model.joblib`: trained XGBoost + training-fit imputer + features + decision threshold.
- `artifacts/<run>/baseline.joblib`: same structure for logistic regression.
- `artifacts/<run>/metrics.json`: validation/test metrics and reproducibility details.
- `data/processed/*_predictions.csv.gz`: one row per input measurement, with `failure_probability`, `risk_alert`, `decision_threshold` and `anomaly_score`.

Only load `joblib` artifacts you created or trust: they rely on Python pickle and must not be loaded from untrusted sources.

### CLI reference

```text
python -m infraguard --help
python -m infraguard generate --help
python -m infraguard train --help
python -m infraguard predict --help
python -m infraguard evaluate --help
python -m infraguard local-sql --help
python -m infraguard aws --help
```

## 5. Local SQL (without AWS)

Install DuckDB only if you need local SQL exploration:

```bash
python -m pip install -e '.[local-sql]'
python -m infraguard local-sql \
  --input data/raw/demo.csv.gz \
  --sql sql/local_feature_summary.sql \
  --output data/processed/feature_summary.csv
```

This runs `sql/local_feature_summary.sql` against a local DuckDB relation named `telemetry`. It does not pretend to run Athena. This local SQL summary is for exploration, while the training pipeline does feature engineering in Python.

## 6. AWS integration (optional, **real credentials required**)

The code is ready to call AWS APIs, but no account, bucket, query execution or service availability is implied. S3, Athena and Glue can incur charges. Configure an AWS budget or billing alert and avoid long-running large scans.

### 6.1 Install AWS requirements

```bash
python -m pip install -e '.[aws]'
```

AWS integration dependencies: `boto3` (session, STS, S3), `awswrangler` (Parquet/Glue/Athena), `pyarrow` (Parquet), `s3fs` (filesystem support). Core training itself does not require any of them.

### 6.2 Make an account and choose an authentication method

- Create or use an **AWS account** and select a region (such as `us-east-1`).
- Set up AWS IAM Identity Center / AWS CLI SSO (recommended for human accounts) or assume a suitably scoped IAM role. Avoid long-lived access keys where possible.
- Install AWS CLI v2, then run `aws configure sso --profile infraguard-dev`, sign in, and `aws sso login --profile infraguard-dev`.
- Have a uniquely named S3 bucket you control in that AWS region. You can create it via AWS Console if you have permissions.
- Ensure your credentials/role have access to S3 bucket objects, Athena query execution and Glue Data Catalog tables (see `config/iam-policy.example.json`). Ensure Athena's query-results S3 prefix can be written and read.
- Copy `.env.example` to `.env` and **replace the placeholder values**. `.env` is gitignored. Never add secrets to git.

```bash
cp .env.example .env
# edit .env: AWS_PROFILE=infraguard-dev; choose your actual bucket and results path
python -m infraguard aws check
```

This shows your account number and current IAM identity if credentials work. Python automatically loads `.env`; `boto3` uses the standard AWS profile/credential chain. AWS CLI SSO sessions may require periodic sign-in. Make sure `.env`'s profile is an actual configured profile, not the shipped placeholder.

### 6.3 Upload telemetry or publish it as queryable Parquet

**Raw backup only** (no queryable Glue table created):

```bash
python -m infraguard aws upload \
  --input data/raw/demo.csv.gz --key raw/demo.csv.gz
```

**Publish as an Athena table** (automatically creates the Glue database if permitted):

```bash
python -m infraguard aws publish \
  --input data/raw/demo.csv.gz
```

This writes Parquet to `s3://YOUR_BUCKET/telemetry/telemetry/` by default and creates the `infraguard.telemetry` Glue table. **Caution:** `publish` uses `mode='overwrite'`; repeating it replaces the table's dataset, so use a development table/bucket only. Schema and Glue permissions must be correct.

### 6.4 Run a real Athena SQL query

```bash
# A query returning raw, chronologically ordered training rows
python -m infraguard aws query \
  --sql sql/athena_training_rows.sql \
  --output data/raw/from_athena.csv.gz

# Then train locally on the downloaded data
python -m infraguard train \
  --input data/raw/from_athena.csv.gz --output-dir artifacts/from_athena

# Alternative SQL: trailing-window features for exploration
python -m infraguard aws query \
  --sql sql/athena_feature_preview.sql \
  --output data/processed/athena_feature_preview.csv.gz
```

The Athena query includes the same labels if those labels were published from the synthetic generator. A live production table might omit labels and therefore cannot support supervised model training until future failure logs are joined. The SQL examples use the default `infraguard.telemetry` name; update them if changing database/table settings. Athena scans and its result-output bucket can cost money; avoid publishing personal/secret telemetry without proper governance.

### Placeholders left for you

| Placeholder | Where | What to replace with |
| --- | --- | --- |
| `AWS_PROFILE` | `.env` | Existing AWS SSO/CLI profile name |
| `AWS_REGION` | `.env` | AWS region containing resources |
| `INFRAGUARD_S3_BUCKET` | `.env` | Your own unique S3 bucket |
| `INFRAGUARD_ATHENA_RESULTS_S3` | `.env` | Writable `s3://.../athena-results/` prefix |
| `INFRAGUARD_ATHENA_DATABASE/TABLE` | `.env` / SQL | Glue/Athena names |
| IAM resource ARNs | `config/iam-policy.example.json` | Bucket name, region and account ID |

No passwords, access keys, tokens, accounts, cloud resources, or AWS query results are included.

## 7. Project layout

```text
InfraGuard/
├── README.md
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── requirements-aws.txt
├── requirements-dev.txt
├── src/infraguard/
│   ├── __init__.py
│   ├── __main__.py
│   ├── aws.py            # AWS STS, S3, Parquet/Glue, Athena
│   ├── cli.py            # local + AWS CLI
│   ├── features.py       # causal rolling features, anomaly score
│   ├── io.py             # compressed CSV IO
│   ├── model.py          # training, metrics, persistence, inference
│   ├── simulate.py       # seeded labeled telemetry simulator
│   └── split.py          # chronological/purged splits
├── sql/
│   ├── local_feature_summary.sql
│   ├── athena_training_rows.sql
│   └── athena_feature_preview.sql
├── config/iam-policy.example.json
├── scripts/run_demo.sh
├── tests/
│   ├── test_simulate.py
│   ├── test_features.py
│   ├── test_split.py
│   └── test_model.py
├── .github/workflows/tests.yml
├── reports/full_synthetic_experiment.md
├── data/raw/            # generated/ingested data (gitignored)
├── data/processed/      # model predictions, SQL results (gitignored)
└── artifacts/           # trained models, metrics (gitignored)
```

## 8. Tests

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

Tests cover deterministic data generation, ground-truth future labels, causal/no-leak features, invalid data inputs, chronological boundary gaps, training/test metrics, and trained-model prediction/evaluation roundtrips. CI tests run via GitHub Actions with no Docker or AWS credentials.

## 9. Real-data integration checklist

To move beyond a synthetic demonstration, obtain timestamped equipment telemetry and verified outage logs, normalize units/time zones, create per-device future labels, keep identity/label columns out of features, measure event-level lead time and false alarm rates, handle telemetry gaps/device resets, evaluate multiple seeds and future data windows, and review precision–recall tradeoffs under the actual failure base rate. Re-evaluate against an unmodified held-out test set before publishing results or resume metrics.

## 10. Resume reporting

A defensible description after personally executing and documenting the experiments might be:

> Implemented an infrastructure failure-prediction pipeline using Python, XGBoost, S3 and Athena integration; engineered causal temporal CPU, memory, power and temperature features, with performance assessed using chronological holdouts and AUROC/average precision.

Replace this with **your actual test metrics** and clearly state if evaluation used synthetic data. Do not claim AWS or 1M+ records were executed until you have run the relevant commands and confirmed output.

## License

MIT; see `LICENSE`.
