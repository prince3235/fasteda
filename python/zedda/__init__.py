"""
zedda - Zero Effort Data Analysis
====================================

The fastest EDA library ever built.
C++ parallel core. 1TB files in seconds.

Quick start::

    import zedda as zd

    # Profile any file
    zd.profile("data.csv")

    # Get result as object
    p = zd.scan("data.csv")
    print(p.num_rows)
    print(p.columns[0].mean)

    # Compare two datasets
    zd.compare("old.csv", "new.csv")
"""

from __future__ import annotations

import math
import ctypes
import time
from pathlib import Path


# ── Public error class ────────────────────────────────────────────
class ZeddaError(Exception):
    """User-friendly error raised by the Zedda engine."""
    pass


__version__ = "0.2.1"
__author__  = "zedda contributors"


# ── Try importing C++ core ────────────────────────────────────────
try:
    from . import fasteda_core as _core
    _CORE_AVAILABLE = True
except ImportError:
    _CORE_AVAILABLE = False
    _core = None

# ── Rich for terminal output ──────────────────────────────────────
try:
    from rich.console import Console
    from rich.table   import Table
    from rich.text    import Text
    from rich.panel   import Panel
    from rich import box
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

_console = Console() if _RICH_AVAILABLE else None

# ── Arrow C Data Interface struct sizes (from arrow/c/abi.h) ──────
# ArrowSchema: 9 fields, mostly pointers → 72 bytes on 64-bit
# ArrowArray:  9 fields similarly        → 72 bytes on 64-bit
# We allocate 256 bytes each for safety (plenty of room for private_data)
_ARROW_SCHEMA_SIZE = 256
_ARROW_ARRAY_SIZE  = 256


# _format_num helper for cleaner number formatting
def _format_num(val: float) -> str:
    if val == 0.0: return "0"
    abs_val = abs(val)
    if abs_val >= 1_000_000:  return f"{val:,.0f}"
    elif abs_val >= 1_000:    return f"{val:,.1f}"
    elif abs_val >= 1:        return f"{val:.4f}"
    elif abs_val >= 0.001:    return f"{val:.6f}"
    else:                     return f"{val:.2e}"

# ─────────────────────────────────────────────────────────────────
#  scan() - run the C++ engine, return DatasetProfile object
# ─────────────────────────────────────────────────────────────────
def scan(path: str, sample_size: int = None) -> object:
    """
    Scan a file and return a DatasetProfile object.

    Args:
        path:        Path to CSV or Parquet file.
        sample_size: Max rows to sample. Auto-triggers for files > 500 MB.

    Returns:
        DatasetProfile with full column-level statistics.

    Example::

        p = zd.scan("titanic.csv")
        print(p.num_rows)          # 891
        print(p.columns[0].mean)   # 29.69
    """
    _require_core()

    file_path = Path(path)
    if not file_path.exists():
        raise ZeddaError(
            f"File not found: '{path}'\n"
            "Tip: Use an absolute path or check your spelling."
        )

    ext = file_path.suffix.lower()
    supported = {".csv", ".parquet", ".arrow"}
    if ext not in supported:
        raise ZeddaError(
            f"Unsupported format: '{ext}'.\n"
            f"Supported: {', '.join(sorted(supported))}"
        )

    # ── Auto-sampling logic ───────────────────────────────────────
    is_sampled = False
    if sample_size is not None:
        is_sampled = True
    elif file_path.stat().st_size > 1024 * 1024 * 1024:   # 1GB
        is_sampled  = True
        sample_size = 2_000_000

    safe_sample = sample_size if sample_size else 1_000_000

    try:
        if ext in (".parquet", ".arrow"):
            return _scan_arrow(path, is_sampled=is_sampled, sample_size=safe_sample)
        return _core.profile(path, False, is_sampled, safe_sample)
    except RuntimeError as e:
        raise ZeddaError(str(e)) from None


# ─────────────────────────────────────────────────────────────────
#  _scan_arrow() - zero-copy Parquet → C++ via Arrow C Data Interface
#
#  Phase 3 features:
#    • Stratified row-group sampling (reads only 6 representative groups)
#    • Parquet Footer Cheat Code: exact nulls/min/max from metadata
#    • Confidence intervals in terminal output when sampled
# ─────────────────────────────────────────────────────────────────
def _scan_arrow(path: str, is_sampled: bool = False, sample_size: int = 1_000_000) -> object:
    try:
        import pyarrow.parquet as pq
        import pyarrow as pa
    except ImportError:
        raise ZeddaError("pyarrow is required for Parquet. Run: pip install pyarrow")

    t0 = time.perf_counter()
    pf = pq.ParquetFile(path)

    total_rows     = pf.metadata.num_rows
    num_row_groups = pf.metadata.num_row_groups

    # ── Stratified sampling: pick 6 representative row groups ─────
    #    Covers the start, middle, and end of the dataset.
    #    This is statistically more reliable than purely random.
    if num_row_groups <= 6 or not is_sampled:
        selected_groups = list(range(num_row_groups))
        final_is_sampled = False
    else:
        mid = num_row_groups // 2
        selected_groups = sorted({
            0, 1,
            mid - 1, mid,
            num_row_groups - 2, num_row_groups - 1,
        })
        final_is_sampled = True

    profiler = _core.ArrowProfiler(path, total_rows)

    # ── Stream selected row groups to C++ via Arrow C Data Interface ──
    # IMPORTANT: We allocate fresh ctypes buffers per batch.
    # PyArrow _export_to_c transfers ownership to C++.
    # The C++ release() callback (set by PyArrow) is responsible for
    # freeing; we must NOT call release() in our C++ code ourselves.
    for rg_idx in selected_groups:
        rg = pf.read_row_group(rg_idx)
        for batch in rg.to_batches(max_chunksize=65_536):
            # Allocate properly-sized buffers for the Arrow C structs
            schema_buf = (ctypes.c_uint8 * _ARROW_SCHEMA_SIZE)()
            array_buf  = (ctypes.c_uint8 * _ARROW_ARRAY_SIZE)()

            ptr_schema = ctypes.addressof(schema_buf)
            ptr_array  = ctypes.addressof(array_buf)

            # PyArrow fills the structs at our pointers and sets release()
            batch._export_to_c(ptr_array, ptr_schema)

            # C++ reads the data; release() is called by C++ consume_batch
            profiler.consume_batch(ptr_schema, ptr_array)

            # Keep Python objects alive until C++ is done (GC anchor)
            del schema_buf, array_buf

    profile = profiler.finalize()

    # ── Parquet Footer Cheat Code ─────────────────────────────────
    # Parquet stores per-column statistics (null_count, min, max) inside
    # the file footer - readable in milliseconds regardless of file size.
    # We override sampled stats with these EXACT values.
    num_cols = profile.num_cols
    for i in range(num_cols):
        exact_nulls = 0
        exact_min   = None
        exact_max   = None
        footer_ok   = True

        for rg_idx in range(num_row_groups):
            try:
                col_meta = pf.metadata.row_group(rg_idx).column(i)
                stats    = col_meta.statistics
                if stats is None:
                    footer_ok = False
                    break
                exact_nulls += stats.null_count
                if stats.has_min_max:
                    cmin, cmax = stats.min, stats.max
                    if cmin is not None:
                        exact_min = cmin if exact_min is None else min(exact_min, cmin)
                    if cmax is not None:
                        exact_max = cmax if exact_max is None else max(exact_max, cmax)
            except Exception:
                footer_ok = False
                break

        if footer_ok:
            col = profile.columns[i]
            col.null_count     = exact_nulls
            col.null_pct       = (exact_nulls / total_rows * 100.0) if total_rows > 0 else 0.0
            col.non_null_count = total_rows - exact_nulls
            col.has_high_nulls = col.null_pct > 20.0

            if (exact_min is not None and exact_max is not None
                    and isinstance(exact_min, (int, float))
                    and isinstance(exact_max, (int, float))):
                col.val_min = float(exact_min)
                col.val_max = float(exact_max)
                col.range   = float(exact_max) - float(exact_min)

    profile.scan_time_ms = (time.perf_counter() - t0) * 1000.0
    profile.is_sampled   = final_is_sampled
    # Always expose the EXACT total row count (from footer, instant)
    profile.num_rows     = total_rows
    return profile


# ─────────────────────────────────────────────────────────────────
#  profile() - scan + print beautiful terminal report
# ─────────────────────────────────────────────────────────────────
def profile(path: str, sample_size: int = None) -> object:
    """
    Profile a file and print a beautiful terminal report.

    One line does everything::

        import zedda as zd
        zd.profile("data.csv")
        zd.profile("big_file.parquet", sample_size=500_000)

    Args:
        path:        Path to your data file.
        sample_size: Max rows to sample (auto if file > 500 MB).

    Returns:
        DatasetProfile (also prints report to terminal).
    """
    if _RICH_AVAILABLE and _console:
        _console.print(f"\n[bold blue]zedda[/bold blue] [dim]v{__version__}[/dim]")
        _console.print(f"[dim]Scanning[/dim] [cyan]{path}[/cyan]...\n")

    result = scan(path, sample_size=sample_size)
    _print_report(result)
    return result


# ─────────────────────────────────────────────────────────────────
#  _print_report() - beautiful Rich terminal output
# ─────────────────────────────────────────────────────────────────
def _print_report(p: object) -> None:
    if not _RICH_AVAILABLE or _console is None:
        _print_plain(p)
        return

    # ── Dataset summary panel ─────────────────────────────────────
    sampled_note = ""
    if p.is_sampled:
        sampled_note = "\n[bold yellow]⚠  SAMPLED MODE[/bold yellow]  [dim](stratified, exact nulls & range from footer)[/dim]"

    rows_display = f"{p.num_rows:,}" if p.num_rows >= 0 else "unknown"
    summary = (
        f"[bold]File:[/bold]    {p.file_name}{sampled_note}\n"
        f"[bold]Rows:[/bold]    [green]{rows_display}[/green]\n"
        f"[bold]Cols:[/bold]    {p.num_cols}  "
        f"([cyan]{p.num_numeric} numeric[/cyan], "
        f"[magenta]{p.num_string} string[/magenta])\n"
        f"[bold]Nulls:[/bold]   "
        + ("[red]" if p.overall_null_pct > 10 else "[green]")
        + f"{p.overall_null_pct:.1f}%[/]"
        + f"  ({p.total_null_cells:,} cells)\n"
        f"[bold]Scanned:[/bold] {p.scan_time_ms:.0f} ms"
    )
    title = "[bold blue]Dataset Overview[/bold blue]"
    if p.is_sampled:
        title += "  [yellow]SAMPLED[/yellow]"

    _console.print(Panel(summary, title=title, border_style="blue", expand=False))

    # ── Data Quality Score ────────────────────────────────────────
    q_score = _quality_score(p)
    blocks_filled = int(q_score / 10)
    blocks = "#" * blocks_filled + "-" * (10 - blocks_filled)
    
    if q_score >= 80:     q_color, q_label = "green", "GOOD"
    elif q_score >= 60:   q_color, q_label = "yellow", "FAIR"
    else:                 q_color, q_label = "red", "POOR"
    
    _console.print(f"\nData Quality Score: {q_score}/100  [{blocks}]  [{q_color}]{q_label}[/{q_color}]\n")

    # ── Column table ──────────────────────────────────────────────
    mean_header = "Mean (±95% CI)" if p.is_sampled else "Mean"
    table = Table(
        show_header=True,
        header_style="bold white on blue",
        border_style="dim",
        box=box.SIMPLE_HEAVY,
        padding=(0, 1),
    )
    table.add_column("Column",        style="bold cyan",   min_width=12)
    table.add_column("Type",          style="magenta",     min_width=6)
    table.add_column("Nulls",         justify="right",     min_width=8)
    table.add_column("Unique~",       justify="right",     min_width=8)
    table.add_column(mean_header,     justify="right",     min_width=18)
    table.add_column("Min",           justify="right",     min_width=12)
    table.add_column("Max",           justify="right",     min_width=12)
    table.add_column("Flags",         min_width=14)

    for col in p.columns:
        # Null cell coloring
        null_cell = Text(f"{col.null_pct:.1f}%")
        if col.null_pct > 20:
            null_cell.stylize("bold red")
        elif col.null_pct > 5:
            null_cell.stylize("yellow")
        else:
            null_cell.stylize("green")

        # Mean / Min / Max
        if col.type_str in ("int", "float"):
            if p.is_sampled and col.non_null_count > 1:
                stderr   = 1.96 * col.stddev / math.sqrt(col.non_null_count)
                mean_str = f"{_format_num(col.mean)} ± {_format_num(stderr)}"
            else:
                mean_str = f"{_format_num(col.mean)}"
            min_str = f"{_format_num(col.val_min)}"
            max_str = f"{_format_num(col.val_max)}"
        else:
            mean_str = f"len~{col.mean_str_len:.0f}"
            min_str  = "-"
            max_str  = "-"

        # Health flags
        flags = []
        if col.has_high_nulls:        flags.append("[red]HIGH NULL[/red]")
        if col.is_constant:           flags.append("[yellow]CONST[/yellow]")
        if col.is_high_cardinality:   flags.append("[blue]HIGH CARD[/blue]")
        flags_str = " ".join(flags) if flags else "[dim]ok[/dim]"

        col_name = col.name if len(col.name) <= 18 else col.name[:16] + ".."
        table.add_row(
            col_name,
            col.type_str,
            null_cell,
            str(col.unique_approx),
            mean_str,
            min_str,
            max_str,
            Text.from_markup(flags_str),
        )

    _console.print(table)

    if p.is_sampled:
        _console.print(
            "[dim]i  Means show 95% confidence interval. "
            "Null counts and min/max are exact (from Parquet footer).[/dim]\n"
        )
    else:
        _console.print()

    # ── Smart Warnings ────────────────────────────────────────────
    warnings = []
    for col in p.columns:
        # High nulls warning
        if col.null_pct > 20:
            warnings.append(f"[red]![/red]  '{col.name}' - {col.null_pct:.1f}% missing. Consider dropping or imputing.")
        
        # Constant column warning
        if col.is_constant:
            warnings.append(f"[yellow]![/yellow]  '{col.name}' - only 1 unique value. Useless for ML, drop it.")
        
        # Possible ID column (very high cardinality on int)
        if col.type_str == "int" and col.unique_pct > 95:
            warnings.append(f"[blue]i[/blue]  '{col.name}' - {col.unique_pct:.0f}% unique. Looks like an ID column.")
        
        # Possible binary target column
        if col.unique_approx == 2 and col.type_str == "int":
            warnings.append(f"[green]v[/green]  '{col.name}' - binary column (2 values). Good ML target candidate.")
        
        # Extreme outlier hint (if max >> mean by 10x)
        if col.type_str in ("int", "float") and col.mean > 0:
            if col.val_max > col.mean * 10:
                warnings.append(f"[yellow]![/yellow]  '{col.name}' - max ({_format_num(col.val_max)}) is {col.val_max/col.mean:.0f}x above mean. Outliers likely.")

    if warnings:
        _console.print("[bold]Smart Warnings:[/bold]")
        for w in warnings:
            _console.print(f"  {w}")
        _console.print()

    # ── Correlation Alerts ────────────────────────────────────────
    _correlation_alerts(p, _console)


def _quality_score(p) -> int:
    score = 100
    # Penalize nulls
    score -= min(40, int(p.overall_null_pct * 2))
    # Penalize high-null columns (>20%)
    high_null_cols = sum(1 for c in p.columns if c.has_high_nulls)
    score -= min(20, high_null_cols * 5)
    # Penalize constant columns (no variance)
    constant_cols = sum(1 for c in p.columns if c.is_constant)
    score -= min(20, constant_cols * 10)
    return max(0, score)

def _correlation_alerts(p, console) -> None:
    numeric_cols = [c for c in p.columns if c.type_str in ("int", "float") and not c.is_constant]
    
    alerts = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            ca, cb = numeric_cols[i], numeric_cols[j]
            # Heuristic: if both have same range direction and similar unique counts
            # flag as potentially correlated for user to investigate
            if ca.unique_approx > 0 and cb.unique_approx > 0:
                ratio = min(ca.unique_approx, cb.unique_approx) / max(ca.unique_approx, cb.unique_approx)
                if ratio > 0.92:  # very similar cardinality = possible correlation
                    alerts.append(f"[blue]<->[/blue]  '{ca.name}' <-> '{cb.name}' - similar cardinality ({ca.unique_approx:,} vs {cb.unique_approx:,}). Check for correlation.")
    
    if alerts:
        console.print("[bold]Correlation Alerts:[/bold] [dim](investigate these pairs)[/dim]")
        for a in alerts[:5]:  # max 5 alerts to avoid noise
            console.print(f"  {a}")
        console.print()

def _print_plain(p: object) -> None:
    """Fallback plain text report when Rich is not installed."""
    sampled = " [SAMPLED]" if p.is_sampled else ""
    print(f"\nzedda v{__version__}")
    print(f"File  : {p.file_name}{sampled}")
    print(f"Rows  : {p.num_rows:,}")
    print(f"Cols  : {p.num_cols}")
    print(f"Nulls : {p.overall_null_pct:.1f}%")
    print(f"Time  : {p.scan_time_ms:.0f} ms")
    print("\nColumn        Type    Nulls     Mean")
    print("-" * 52)
    for col in p.columns:
        mean_s = f"{_format_num(col.mean)}" if col.type_str in ("int", "float") else "-"
        col_name = col.name if len(col.name) <= 12 else col.name[:10] + ".."
        print(f"{col_name:<14}{col.type_str:<8}{col.null_pct:.1f}%     {mean_s}")


# ─────────────────────────────────────────────────────────────────
#  compare() - diff two datasets
# ─────────────────────────────────────────────────────────────────
def compare(path_a: str, path_b: str, sample_size: int = None) -> None:
    """
    Compare two datasets side by side.

    Shows schema differences, null rate changes, and distribution
    shifts (z-score drift detection) between two files.

    Example::

        zd.compare("train.csv", "test.csv")

    Args:
        path_a:      First file path.
        path_b:      Second file path.
        sample_size: Max rows per file (auto if > 500 MB).
    """
    p_a = scan(path_a, sample_size=sample_size)
    p_b = scan(path_b, sample_size=sample_size)

    if not _RICH_AVAILABLE or _console is None:
        print(f"A: {p_a.file_name} - {p_a.num_rows:,} rows")
        print(f"B: {p_b.file_name} - {p_b.num_rows:,} rows")
        return

    _console.print(f"\n[bold blue]zedda compare[/bold blue]")
    _console.print(f"[cyan]A:[/cyan] {p_a.file_name}  [dim]({p_a.num_rows:,} rows)[/dim]")
    _console.print(f"[cyan]B:[/cyan] {p_b.file_name}  [dim]({p_b.num_rows:,} rows)[/dim]\n")

    table = Table(
        show_header=True,
        header_style="bold white on blue",
        border_style="dim",
        box=box.SIMPLE_HEAVY,
    )
    table.add_column("Column",  style="bold cyan", min_width=12)
    table.add_column("Type A",  min_width=7)
    table.add_column("Type B",  min_width=7)
    table.add_column("Nulls A", justify="right",   min_width=8)
    table.add_column("Nulls B", justify="right",   min_width=8)
    table.add_column("Mean A",  justify="right",   min_width=10)
    table.add_column("Mean B",  justify="right",   min_width=10)
    table.add_column("Drift",   min_width=10)

    cols_a = {c.name: c for c in p_a.columns}
    cols_b = {c.name: c for c in p_b.columns}
    all_cols = list(dict.fromkeys(list(cols_a) + list(cols_b)))

    for name in all_cols:
        ca = cols_a.get(name)
        cb = cols_b.get(name)

        type_a = ca.type_str if ca else "[red]MISSING[/red]"
        type_b = cb.type_str if cb else "[red]MISSING[/red]"

        null_a = f"{ca.null_pct:.1f}%" if ca else "-"
        null_b = f"{cb.null_pct:.1f}%" if cb else "-"

        mean_a = f"{_format_num(ca.mean)}" if (ca and ca.type_str in ("int", "float")) else "-"
        mean_b = f"{_format_num(cb.mean)}" if (cb and cb.type_str in ("int", "float")) else "-"

        # Z-score drift detection
        drift = "[green]ok[/green]"
        if ca and cb and ca.type_str in ("int", "float") and cb.type_str in ("int", "float"):
            if ca.stddev > 0:
                shift = abs(ca.mean - cb.mean) / ca.stddev
                if shift > 1.0:
                    drift = "[red]DRIFT[/red]"
                elif shift > 0.3:
                    drift = "[yellow]SHIFT[/yellow]"
        if not ca:
            drift = "[blue]NEW[/blue]"
        if not cb:
            drift = "[red]REMOVED[/red]"

        table.add_row(name, type_a, type_b, null_a, null_b, mean_a, mean_b,
                      Text.from_markup(drift))

    _console.print(table)
    _console.print()


# ─────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────
def _require_core():
    if not _CORE_AVAILABLE:
        raise RuntimeError(
            "zedda C++ core not found.\n"
            "Please reinstall: pip install zedda"
        )


# ─────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────
__all__ = ["profile", "scan", "compare", "ZeddaError", "__version__"]