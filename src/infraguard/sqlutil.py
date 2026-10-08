"""Small SQL validation helper for shipped query files (not a SQL security boundary)."""

import re


def validate_select(sql: str) -> str:
    """Accept SELECT/WITH with leading comments; disallow multiple statements.

    For AWS, still use only trusted SQL files, IAM policies and an appropriate
    read-only workgroup. This check is not a complete SQL parser.
    """
    without_blocks = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    without_comments = re.sub(r"(?m)^\s*--[^\n]*", "", without_blocks)
    cleaned = without_comments.strip()
    if not re.match(r"^(SELECT|WITH)\b", cleaned, flags=re.IGNORECASE):
        raise ValueError("Query file must contain SELECT or WITH (read-only) SQL")
    if ";" in cleaned.rstrip(";").strip():
        raise ValueError("Query file must contain exactly one SQL statement")
    return sql
