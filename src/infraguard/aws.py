"""Optional real AWS integration. Never invent credentials or cloud resources."""

import os
from pathlib import Path

import pandas as pd

from infraguard.io import read_frame, write_frame
from infraguard.sqlutil import validate_select


def _boto3():
    try:
        import boto3
    except ImportError as error:
        raise RuntimeError("Install AWS packages: pip install -e '.[aws]'") from error
    return boto3


def _wrangler():
    try:
        import awswrangler as wr
    except ImportError as error:
        raise RuntimeError("Install AWS packages: pip install -e '.[aws]'") from error
    return wr


def session():
    """Use AWS SSO profile or standard boto3 credential chain (never hard-code secrets)."""
    boto3 = _boto3()
    profile = os.getenv("AWS_PROFILE")
    if profile and profile.startswith("REPLACE_"):
        raise ValueError("Replace AWS_PROFILE in .env with your actual configured profile")
    return boto3.Session(profile_name=profile or None, region_name=os.getenv("AWS_REGION", "us-east-1"))


def check() -> dict:
    identity = session().client("sts").get_caller_identity()
    return {"account": identity["Account"], "arn": identity["Arn"]}


def upload(input_path: str, bucket: str, key: str) -> str:
    if not bucket or bucket.startswith("REPLACE_"):
        raise ValueError("Configure a real S3 bucket first")
    session().client("s3").upload_file(str(Path(input_path)), bucket, key)
    return f"s3://{bucket}/{key}"


def publish(input_path: str, bucket: str, prefix: str, database: str, table: str) -> str:
    """Convert local data to Parquet in S3 and register an Athena/Glue table.

    WARNING: mode='overwrite' replaces the existing Glue dataset/table data.
    """
    if not bucket or bucket.startswith("REPLACE_"):
        raise ValueError("Configure a real S3 bucket first")
    wr = _wrangler()
    aws_session = session()
    data = read_frame(input_path)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    wr.catalog.create_database(name=database, exist_ok=True, boto3_session=aws_session)
    location = f"s3://{bucket}/{prefix.strip('/')}/{table}/"
    wr.s3.to_parquet(
        df=data,
        path=location,
        dataset=True,
        database=database,
        table=table,
        mode="overwrite",
        boto3_session=aws_session,
        index=False,
        compression="snappy",
    )
    return location


def query(query_file: str, database: str, output_path: str, *, results_s3: str | None = None, workgroup: str = "primary") -> dict:
    """Execute an Athena SELECT, returning CSV for local exploration/training."""
    sql = Path(query_file).read_text()
    validate_select(sql)
    output_location = results_s3 or os.getenv("INFRAGUARD_ATHENA_RESULTS_S3")
    if not output_location or "REPLACE_" in output_location:
        raise ValueError("Set INFRAGUARD_ATHENA_RESULTS_S3 to a real s3://bucket/prefix/")
    wr = _wrangler()
    frame = wr.athena.read_sql_query(
        sql=sql,
        database=database,
        boto3_session=session(),
        s3_output=output_location,
        workgroup=workgroup,
        ctas_approach=False,
    )
    saved = write_frame(frame, output_path)
    return {"rows": len(frame), "output": str(saved)}
