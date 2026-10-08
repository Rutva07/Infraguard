# InfraGuard — Predictive Failure Detection

**Python · XGBoost · scikit-learn · AWS S3 · Amazon Athena · AWS Glue · SQL**

InfraGuard is a Python-based predictive maintenance project combining time-series feature engineering, SQL-based telemetry analysis, and XGBoost classification. The project includes modules for model training and evaluation, as well as AWS S3 storage and Amazon Athena query integration.

## Codebase

```text
InfraGuard/
├── src/infraguard/
│   ├── cli.py           # Command-line interface for project operations
│   ├── io.py            # Dataset loading and saving
│   ├── features.py      # Temporal feature engineering and anomaly scores
│   ├── split.py         # Time-ordered train/validation/test splits
│   ├── model.py         # XGBoost training, inference, and evaluation
│   ├── aws.py           # S3, Glue Data Catalog, and Athena integration
│   ├── sqlutil.py       # SQL utility functions
│   └── __main__.py      # Package entry point
├── sql/
│   ├── athena_training_rows.sql     # Query for training records
│   ├── athena_feature_preview.sql   # SQL feature summaries
│   └── local_feature_summary.sql    # Local SQL analysis
├── config/
│   └── iam-policy.example.json      # Example AWS permissions
├── tests/                           # Automated unit and integration tests
├── pyproject.toml                   # Package configuration and dependencies
└── .env.example                     # AWS configuration placeholders
```

The pipeline processes time-ordered device measurements, derives historical rolling statistics and changes, and trains an XGBoost classifier to estimate failure risk. The AWS module contains code for storing data in S3, publishing Glue tables, and querying data with Athena. The included SQL and feature modules describe the codebase structure.

## Dataset — Backblaze Drive Stats

**Source:** [Backblaze Hard Drive Test Data](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data)

The reported real-data experiment used a smaller subset of the Backblaze Drive Stats dataset. Backblaze publishes dated drive-level records containing drive identifiers, models, SMART health attributes, and failure indicators. These measurements support hard-drive failure prediction using indicators such as temperature, reallocated sectors, pending sectors, and other available SMART signals.

## Experimental Results

**Model:** XGBoost  
**Dataset:** Backblaze Drive Stats (subset)  
**Task:** Hard-drive failure prediction

| Metric | Reported result |
|---|---:|
| AUROC | **0.91** |
| Recall | **75%** |
| False-positive rate | **2.5%** |
| F1 score | **≈ 0.55** |
