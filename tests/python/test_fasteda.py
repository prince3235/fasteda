"""
zedda Python tests — runs in CI after wheel build
"""
import sys
import os
import pytest

# ── Import zedda ──────────────────────────────────────────────────────
import zedda as zd
from zedda import fasteda_core as core


# ── Fixtures ─────────────────────────────────────────────────────
@pytest.fixture
def sample_csv(tmp_path):
    """Create a small test CSV."""
    f = tmp_path / "test.csv"
    f.write_text(
        "name,age,salary,city\n"
        "Arjun,25,50000.0,Mumbai\n"
        "Priya,30,75000.0,Delhi\n"
        "Rahul,22,,Bangalore\n"
        "Sneha,28,62000.0,Mumbai\n"
        "Karan,35,90000.0,Pune\n"
    )
    return str(f)


# ── Tests ─────────────────────────────────────────────────────────
def test_version():
    assert zd.__version__ == "0.1.5"


def test_scan_returns_profile(sample_csv):
    p = zd.scan(sample_csv)
    assert p.num_rows == 5
    assert p.num_cols == 4


def test_null_detection(sample_csv):
    p = zd.scan(sample_csv)
    # salary has 1 null out of 5 rows
    salary = next(c for c in p.columns if c.name == "salary")
    assert salary.null_count == 1
    assert abs(salary.null_pct - 20.0) < 0.1


def test_numeric_stats(sample_csv):
    p = zd.scan(sample_csv)
    age = next(c for c in p.columns if c.name == "age")
    assert age.type_str == "int"
    assert abs(age.mean - 28.0) < 0.1
    assert age.val_min == 22.0
    assert age.val_max == 35.0


def test_string_column(sample_csv):
    p = zd.scan(sample_csv)
    city = next(c for c in p.columns if c.name == "city")
    assert city.type_str == "str"
    assert city.null_count == 0


def test_profile_runs(sample_csv, capsys):
    # Should not raise
    p = zd.profile(sample_csv)
    assert p is not None
    assert p.num_rows == 5


def test_overall_null_pct(sample_csv):
    p = zd.scan(sample_csv)
    # 1 null out of 20 cells = 5%
    assert abs(p.overall_null_pct - 5.0) < 0.1


def test_scan_time_reasonable(sample_csv):
    p = zd.scan(sample_csv)
    # Should scan 5 rows in under 1 second
    assert p.scan_time_ms < 1000


def test_column_count(sample_csv):
    p = zd.scan(sample_csv)
    assert p.num_numeric == 2   # age, salary
    assert p.num_string  == 2   # name, city