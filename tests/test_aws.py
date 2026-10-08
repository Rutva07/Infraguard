"""AWS contract tests that do not contact AWS or require credentials."""

from types import SimpleNamespace

import pandas as pd
import pytest

from infraguard import aws


def test_check_calls_sts_without_real_account(monkeypatch):
    class FakeClient:
        def get_caller_identity(self):
            return {"Account": "000000000000", "Arn": "arn:aws:iam::000000000000:role/test"}

    monkeypatch.setattr(aws, "session", lambda: SimpleNamespace(client=lambda name: FakeClient()))
    assert aws.check()["account"] == "000000000000"


def test_upload_validates_bucket_and_calls_s3(monkeypatch, tmp_path):
    src = tmp_path / "raw.csv"
    src.write_text("sample\n1\n")
    calls = []
    fake = SimpleNamespace(upload_file=lambda *args: calls.append(args))
    monkeypatch.setattr(aws, "session", lambda: SimpleNamespace(client=lambda name: fake))
    with pytest.raises(ValueError, match="real S3 bucket"):
        aws.upload(str(src), "REPLACE_WITH_UNIQUE_BUCKET_NAME", "raw/raw.csv")
    assert aws.upload(str(src), "example-bucket", "raw/raw.csv") == "s3://example-bucket/raw/raw.csv"
    assert calls == [(str(src), "example-bucket", "raw/raw.csv")]


def test_query_validates_read_only_sql(monkeypatch, tmp_path):
    sql = tmp_path / "delete.sql"
    sql.write_text("DELETE FROM telemetry")
    with pytest.raises(ValueError, match="read-only"):
        aws.query(str(sql), "infraguard", str(tmp_path / "out.csv"))


def test_query_with_mocked_athena(monkeypatch, tmp_path):
    input_sql = tmp_path / "select.sql"
    input_sql.write_text("SELECT device_id FROM infraguard.telemetry")
    seen = {}

    def fake_query(**kwargs):
        seen.update(kwargs)
        return pd.DataFrame({"device_id": ["server-0001"]})

    monkeypatch.setattr(aws, "_wrangler", lambda: SimpleNamespace(athena=SimpleNamespace(read_sql_query=fake_query)))
    monkeypatch.setattr(aws, "session", lambda: "fake-session")
    info = aws.query(str(input_sql), "infraguard", str(tmp_path / "result.csv"), results_s3="s3://bucket/results/")
    assert info["rows"] == 1
    assert seen["database"] == "infraguard"
    assert seen["boto3_session"] == "fake-session"
    assert pd.read_csv(tmp_path / "result.csv").iloc[0, 0] == "server-0001"
