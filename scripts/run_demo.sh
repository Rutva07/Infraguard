#!/usr/bin/env bash
set -euo pipefail
# Runs a small end-to-end demonstration on a laptop, without AWS or Docker.
python -m infraguard generate --devices 18 --steps 550 --output data/raw/demo.csv.gz
python -m infraguard train --input data/raw/demo.csv.gz --output-dir artifacts/demo --estimators 100
python -m infraguard evaluate --input data/raw/demo.csv.gz --model-dir artifacts/demo
python -m infraguard predict --input data/raw/demo.csv.gz --model-dir artifacts/demo --output data/processed/demo_predictions.csv.gz
