"""
Phase 3 Integration Test Suite — ZEDDA
=======================================
Tests every feature we shipped in Phase 3.
Run after build: python tests/test_phase3.py
"""
import sys
import os
import tempfile
import csv

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import zedda as zd
from zedda import ZeddaError

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
INFO = "\033[94mℹ\033[0m"

_tests_run    = 0
_tests_passed = 0


def test(name: str, condition: bool, detail: str = ""):
    global _tests_run, _tests_passed
    _tests_run += 1
    if condition:
        _tests_passed += 1
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}  {detail}")


# ─────────────────────────────────────────────────────────────────
#  Helper: create a small in-memory CSV file
# ─────────────────────────────────────────────────────────────────
def make_csv(rows=500) -> str:
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "value", "category", "flag"])
        for i in range(rows):
            category = "A" if i % 3 == 0 else ("B" if i % 3 == 1 else "C")
            flag     = "true" if i % 2 == 0 else "false"
            value    = round(i * 1.5, 2) if i % 10 != 0 else ""  # 10% nulls
            w.writerow([i, value, category, flag])
    return path


def make_parquet(rows=1000) -> str:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return None

    ids   = list(range(rows))
    vals  = [float(i) * 2.5 if i % 5 != 0 else None for i in range(rows)]
    cats  = ["X" if i % 2 == 0 else "Y" for i in range(rows)]

    table = pa.table({
        "id":       pa.array(ids, type=pa.int64()),
        "value":    pa.array(vals, type=pa.float64()),
        "category": pa.array(cats, type=pa.string()),
    })

    fd, path = tempfile.mkstemp(suffix=".parquet")
    os.close(fd)
    pq.write_table(table, path, row_group_size=100)  # 10 row groups
    return path


# ─────────────────────────────────────────────────────────────────
#  Test Group 1: Error Handling (Con 5)
# ─────────────────────────────────────────────────────────────────
print("\n--- Group 1: ZeddaError & File Validation -------------------")

try:
    zd.scan("nonexistent_file_xyz.csv")
    test("Missing file raises ZeddaError", False, "no exception raised")
except ZeddaError as e:
    test("Missing file raises ZeddaError", True)
    test("Error message mentions 'File not found'", "File not found" in str(e))
except Exception as e:
    test("Missing file raises ZeddaError", False, str(e))

try:
    # Create a real file with unsupported extension
    fd, json_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        zd.scan(json_path)
        test("Unsupported extension raises ZeddaError", False)
    except ZeddaError as e:
        test("Unsupported extension raises ZeddaError", True)
        test("Error mentions 'Unsupported format'", "Unsupported format" in str(e), f"got: {e}")
    finally:
        os.unlink(json_path)
except Exception as e:
    test("Unsupported extension raises ZeddaError", False, str(e))


# ─────────────────────────────────────────────────────────────────
#  Test Group 2: CSV Profiling — basic correctness
# ─────────────────────────────────────────────────────────────────
print("\n── Group 2: CSV Profiling (Basic) ─────────────────────────")

csv_path = make_csv(rows=500)
try:
    p = zd.scan(csv_path)
    test("scan() returns DatasetProfile",     hasattr(p, "num_rows"))
    test("Correct column count (4 cols)",     p.num_cols == 4)
    test("Correct row count (500 rows)",      p.num_rows == 500)
    test("scan_time_ms > 0",                  p.scan_time_ms > 0)
    test("is_sampled is False for small file",not p.is_sampled)

    id_col  = next((c for c in p.columns if c.name == "id"),    None)
    val_col = next((c for c in p.columns if c.name == "value"), None)
    cat_col = next((c for c in p.columns if c.name == "category"), None)

    test("id column found",               id_col  is not None)
    test("value column found",            val_col is not None)
    test("category column found",         cat_col is not None)

    if id_col:
        test("id type is int",            id_col.type_str == "int")
        test("id has zero nulls",         id_col.null_count == 0)
        test("id mean ≈ 249.5",           abs(id_col.mean - 249.5) < 1.0)

    if val_col:
        test("value has ~10% nulls",      4.0 < val_col.null_pct < 15.0)
        test("val_min >= 0",              val_col.val_min >= 0)
        test("val_max > val_min",         val_col.val_max > val_col.val_min)
        test("value stddev > 0",          val_col.stddev > 0)

    if cat_col:
        test("category type is str",      cat_col.type_str in ("str", "string", "unknown"))
        test("category unique count ~ 3", 2 <= cat_col.unique_approx <= 5)

finally:
    os.unlink(csv_path)


# ─────────────────────────────────────────────────────────────────
#  Test Group 3: CSV Sampling
# ─────────────────────────────────────────────────────────────────
print("\n── Group 3: CSV Sampling ───────────────────────────────────")

csv_path2 = make_csv(rows=2000)
try:
    p_sampled = zd.scan(csv_path2, sample_size=200)
    test("Sampled scan returns DatasetProfile",  hasattr(p_sampled, "num_rows"))
    test("is_sampled is True when sample_size set", p_sampled.is_sampled)
    test("Sampled rows <= sample_size * threads",   p_sampled.num_rows <= 200 * 8 + 1)
    test("num_cols still correct",                  p_sampled.num_cols == 4)
finally:
    os.unlink(csv_path2)


# ─────────────────────────────────────────────────────────────────
#  Test Group 4: Parquet Profiling + Footer Cheat Code
# ─────────────────────────────────────────────────────────────────
print("\n── Group 4: Parquet / Arrow Profiling ──────────────────────")

parquet_path = make_parquet(rows=1000)
if parquet_path is None:
    print(f"  {INFO}  pyarrow not installed — skipping Parquet tests")
else:
    try:
        p = zd.scan(parquet_path)
        test("Parquet scan returns DatasetProfile",  hasattr(p, "num_rows"))
        test("Correct row count (1000 rows)",        p.num_rows == 1000)
        test("Correct column count (3 cols)",        p.num_cols == 3)
        test("scan_time_ms > 0",                     p.scan_time_ms > 0)

        id_col  = next((c for c in p.columns if c.name == "id"),    None)
        val_col = next((c for c in p.columns if c.name == "value"), None)
        cat_col = next((c for c in p.columns if c.name == "category"), None)

        test("id column found",   id_col  is not None)
        test("val column found",  val_col is not None)
        test("cat column found",  cat_col is not None)

        if val_col:
            # 20% of rows have None value
            test("val null ~20%",        14.0 < val_col.null_pct < 26.0)
            # Footer cheat code gives us exact values
            test("val null_count exact (200)", val_col.null_count == 200)
            # i=0 is null (0 % 5 == 0), so actual min is val at i=1 → 2.5
            test("val_min exact (2.5)",        abs(val_col.val_min - 2.5) < 0.01)
            test("val_max exact (2497.5)",     abs(val_col.val_max - 2497.5) < 1.0)

        # Test is_sampled — 10 row groups <= 6? No → should be sampled with 10 rgs
        # But our Parquet has 10 row groups → 10 > 6 → should be sampled
        # Wait, _scan_arrow is_sampled depends on whether we passed is_sampled=True
        # For a small file (<500MB), we don't auto-sample
        test("Small parquet not sampled",  not p.is_sampled)

    finally:
        os.unlink(parquet_path)

    # Now test explicit sampling for Parquet
    parquet_path2 = make_parquet(rows=5000)
    if parquet_path2:
        try:
            p_s = zd.scan(parquet_path2, sample_size=500)
            test("Explicit Parquet sample is_sampled=True", p_s.is_sampled)
            # Footer cheat code still gives exact total_rows
            test("Parquet total_rows exact from footer",    p_s.num_rows == 5000)
        finally:
            os.unlink(parquet_path2)


# ─────────────────────────────────────────────────────────────────
#  Test Group 5: profile() terminal output (smoke test)
# ─────────────────────────────────────────────────────────────────
print("\n── Group 5: profile() Smoke Test ───────────────────────────")

csv_path3 = make_csv(rows=100)
try:
    try:
        result = zd.profile(csv_path3)
        test("profile() returns DatasetProfile", hasattr(result, "num_rows"))
        test("profile() result has columns",     len(result.columns) > 0)
    except Exception as e:
        test("profile() runs without crash", False, str(e))
finally:
    os.unlink(csv_path3)


# ─────────────────────────────────────────────────────────────────
#  Test Group 6: compare()
# ─────────────────────────────────────────────────────────────────
print("\n── Group 6: compare() ──────────────────────────────────────")

csv_a = make_csv(rows=300)
csv_b = make_csv(rows=400)
try:
    try:
        zd.compare(csv_a, csv_b)
        test("compare() runs without crash", True)
    except Exception as e:
        test("compare() runs without crash", False, str(e))
finally:
    os.unlink(csv_a)
    os.unlink(csv_b)


# ─────────────────────────────────────────────────────────────────
#  Summary
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
status = "\033[92mPASSED\033[0m" if _tests_passed == _tests_run else "\033[91mFAILED\033[0m"
print(f"  Result: {_tests_passed}/{_tests_run} tests {status}")
print(f"{'='*55}\n")

if _tests_passed < _tests_run:
    sys.exit(1)
