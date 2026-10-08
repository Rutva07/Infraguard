"""Command-line entry point. Run `python -m infraguard --help`."""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from infraguard.io import read_frame, write_frame
from infraguard.model import evaluate, predict, train
from infraguard.simulate import SimulationConfig, generate
from infraguard.sqlutil import validate_select
from infraguard.split import SplitConfig


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="infraguard", description="InfraGuard failure prediction")
    commands = parser.add_subparsers(dest="command", required=True)
    generate_cmd = commands.add_parser("generate", help="Generate synthetic labeled telemetry")
    generate_cmd.add_argument("--devices", type=int, default=200)
    generate_cmd.add_argument("--steps", type=int, default=5200)
    generate_cmd.add_argument("--interval-minutes", type=int, default=1)
    generate_cmd.add_argument("--horizon-steps", type=int, default=30)
    generate_cmd.add_argument("--seed", type=int, default=42)
    generate_cmd.add_argument("--output", default="data/raw/synthetic_telemetry.csv.gz")

    train_cmd = commands.add_parser("train", help="Train XGBoost and baseline")
    train_cmd.add_argument("--input", default="data/raw/synthetic_telemetry.csv.gz")
    train_cmd.add_argument("--output-dir", default="artifacts")
    train_cmd.add_argument("--seed", type=int, default=42)
    train_cmd.add_argument("--estimators", type=int, default=180)
    train_cmd.add_argument("--purge-steps", type=int, default=30)
    train_cmd.add_argument("--train-fraction", type=float, default=0.65)
    train_cmd.add_argument("--validation-fraction", type=float, default=0.17)

    predict_cmd = commands.add_parser("predict", help="Score telemetry using a trained model")
    predict_cmd.add_argument("--input", required=True)
    predict_cmd.add_argument("--model-dir", default="artifacts")
    predict_cmd.add_argument("--output", default="data/processed/predictions.csv.gz")
    predict_cmd.add_argument("--baseline", action="store_true")

    eval_cmd = commands.add_parser("evaluate", help="Re-evaluate the held-out test window")
    eval_cmd.add_argument("--input", required=True)
    eval_cmd.add_argument("--model-dir", default="artifacts")
    eval_cmd.add_argument("--baseline", action="store_true")

    sql_cmd = commands.add_parser("local-sql", help="Run read-only SQL against local CSV via DuckDB")
    sql_cmd.add_argument("--input", default="data/raw/synthetic_telemetry.csv.gz")
    sql_cmd.add_argument("--sql", default="sql/local_feature_summary.sql")
    sql_cmd.add_argument("--output", default="data/processed/feature_summary.csv")

    cloud = commands.add_parser("aws", help="Optional AWS S3, Glue and Athena operations")
    cloud_commands = cloud.add_subparsers(dest="aws_action", required=True)
    cloud_commands.add_parser("check", help="Verify AWS credentials with STS")
    upload = cloud_commands.add_parser("upload", help="Upload file to S3")
    upload.add_argument("--input", required=True)
    upload.add_argument("--bucket", default=None)
    upload.add_argument("--key", required=True)
    pub = cloud_commands.add_parser("publish", help="Upload Parquet and create Athena Glue table (OVERWRITES)")
    pub.add_argument("--input", required=True)
    pub.add_argument("--bucket", default=None)
    pub.add_argument("--prefix", default=None)
    pub.add_argument("--database", default=None)
    pub.add_argument("--table", default=None)
    query = cloud_commands.add_parser("query", help="Run read-only Athena query to CSV")
    query.add_argument("--sql", default="sql/athena_training_rows.sql")
    query.add_argument("--database", default=None)
    query.add_argument("--output", default="data/raw/from_athena.csv.gz")
    query.add_argument("--results-s3", default=None)
    query.add_argument("--workgroup", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = make_parser().parse_args(argv)
    if args.command == "generate":
        cfg = SimulationConfig(
            devices=args.devices, steps=args.steps, interval_minutes=args.interval_minutes,
            horizon_steps=args.horizon_steps, seed=args.seed,
        )
        path = write_frame(generate(cfg), args.output)
        print(f"Generated {cfg.devices * cfg.steps:,} synthetic records: {path}")
    elif args.command == "train":
        report = train(
            read_frame(args.input), args.output_dir,
            split_config=SplitConfig(args.train_fraction, args.validation_fraction, args.purge_steps),
            seed=args.seed, estimators=args.estimators,
        )
        print(json.dumps({"xgboost_test": report["xgboost"]["test"],
                          "baseline_test": report["logistic_regression"]["test"]}, indent=2))
        print(f"Saved models and full metrics in {args.output_dir}")
    elif args.command == "predict":
        output = write_frame(predict(read_frame(args.input), args.model_dir, baseline=args.baseline), args.output)
        print(f"Saved risk scores and alerts: {output}")
    elif args.command == "evaluate":
        print(json.dumps(evaluate(read_frame(args.input), args.model_dir, baseline=args.baseline), indent=2))
    elif args.command == "local-sql":
        try:
            import duckdb
        except ImportError as error:
            raise SystemExit("Install optional local SQL engine: pip install -e '.[local-sql]'") from error
        sql = Path(args.sql).read_text()
        validate_select(sql)
        connection = duckdb.connect(":memory:")
        connection.register("telemetry", read_frame(args.input))
        rows = connection.execute(sql).fetchdf()
        print(f"SQL returned {len(rows)} rows; saved to {write_frame(rows, args.output)}")
    elif args.command == "aws":
        from infraguard import aws

        if args.aws_action == "check":
            print(json.dumps(aws.check(), indent=2))
        elif args.aws_action == "upload":
            bucket = args.bucket or os.getenv("INFRAGUARD_S3_BUCKET", "")
            print(aws.upload(args.input, bucket, args.key))
        elif args.aws_action == "publish":
            print(aws.publish(
                args.input, args.bucket or os.getenv("INFRAGUARD_S3_BUCKET", ""),
                args.prefix or os.getenv("INFRAGUARD_S3_PREFIX", "telemetry"),
                args.database or os.getenv("INFRAGUARD_ATHENA_DATABASE", "infraguard"),
                args.table or os.getenv("INFRAGUARD_ATHENA_TABLE", "telemetry"),
            ))
        elif args.aws_action == "query":
            print(json.dumps(aws.query(
                args.sql, args.database or os.getenv("INFRAGUARD_ATHENA_DATABASE", "infraguard"),
                args.output, results_s3=args.results_s3,
                workgroup=args.workgroup or os.getenv("INFRAGUARD_ATHENA_WORKGROUP", "primary"),
            ), indent=2))
    return 0
