import pytest

from infraguard.sqlutil import validate_select


def test_queries_with_leading_comments_pass():
    for path in ["sql/local_feature_summary.sql", "sql/athena_training_rows.sql", "sql/athena_feature_preview.sql"]:
        assert validate_select(open(path).read())


def test_multiple_and_nonselect_statements_rejected():
    with pytest.raises(ValueError):
        validate_select("-- hello\nDELETE FROM telemetry")
    with pytest.raises(ValueError, match="exactly one"):
        validate_select("SELECT 1; DROP TABLE telemetry")
