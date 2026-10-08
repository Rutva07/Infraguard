# InfraGuard: reproducible 1,040,000-row synthetic experiment

## Data and experiment

- Data source: **synthetic / generated**, NOT real infrastructure outage measurements.
- Generator: `python -m infraguard generate --devices 200 --steps 5200 --seed 42`.
- Number of measurements: **1,040,000** across 200 simulated servers.
- Time interval: 1 minute; target: any failure during the next 30 samples.
- Split: time-ordered 65% train, 17% validation, remaining test; a 30-step purge around the two temporal boundaries.
- XGBoost: up to 180 estimators; validation-based early stopping and threshold selection.
- Baseline: standardized logistic regression trained on the same feature set.
- Both models: imputer fitted on the training split, thresholds selected on validation F1, metrics reported on the held-out test period.

## Measured metrics (held-out synthetic test set)

| Model | AUROC | Average precision | Precision | Recall | F1 | Test samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 0.9954 | 0.9663 | 0.9142 | 0.8840 | 0.8989 | 181,200 |
| Logistic regression | 0.9953 | 0.9660 | 0.9272 | 0.8691 | 0.8972 | 181,200 |

Test positives: 15,904; held-out positive rate 8.78%.

- XGBoost confusion matrix: TN=163,977, FP=1,319, FN=1,845, TP=14,059.
- Logistic regression confusion matrix: TN=164,211, FP=1,085, FN=2,082, TP=13,822.
- XGBoost validation-selected threshold: 0.90 (rounded).
- Logistic regression validation-selected threshold: 0.93 (rounded).

The main training command took **19.01 seconds** on the execution environment used for this repository build (reported max RSS ~1.74 GB). These timing numbers are machine-specific, not portable performance guarantees. `artifacts/full/metrics.json` contains the complete machine-generated results.

## Critical interpretation

The strong scores are largely attributable to the controlled generator, which explicitly introduces predictive temperature, CPU, memory and power changes prior to synthetic failures. In particular, these numbers **do not demonstrate 0.995 AUROC on real data-center incidents**. The same classifier could perform substantially worse on genuine events with drifting sensors, rare outages, missing measurements, correlated devices and ambiguous failure logs. The logistic baseline's near-parity is further evidence that this synthetic setup is easy. No actual S3/Athena cloud calls were performed; the AWS integration is implemented but requires access to a user's AWS account for live verification.

## Exact commands

```bash
python -m infraguard generate --devices 200 --steps 5200 --seed 42 --output data/raw/synthetic_telemetry.csv.gz
python -m infraguard train --input data/raw/synthetic_telemetry.csv.gz --output-dir artifacts/full --estimators 180 --purge-steps 30
python -m infraguard evaluate --input data/raw/synthetic_telemetry.csv.gz --model-dir artifacts/full
```
