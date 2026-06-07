"""
zedda — Zero effort data analysis(EDA)
============================

The simplest EDA library ever made.
Built on a C++ core for maximum speed.

Quick start::

    import zedda as zd

    # Profile any file
    zd.profile("data.csv")

    # Get the result as object
    p = zd.scan("data.csv")
    print(p.num_rows)
    print(p.columns[0].mean)

    # Compare two datasets
    zd.compare("old.csv", "new.csv")
"""

from __future__ import annotations

__version__ = "0.1.5"
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


# ─────────────────────────────────────────────────────────────────
#  scan() — run the C++ engine, return DatasetProfile object
#
#  Use when you want to access stats programmatically.
#
#  Example::
#
#      p = zd.scan("data.csv")
#      for col in p.columns:
#          print(col.name, col.mean)
# ─────────────────────────────────────────────────────────────────
def scan(path: str) -> object:
    """
    Scan a file and return a DatasetProfile object.

    Args:
        path: Path to CSV, Excel, JSON, or Parquet file.

    Returns:
        DatasetProfile with full column-level statistics.

    Example::

        p = zd.scan("titanic.csv")
        print(p.num_rows)          # 891
        print(p.columns[0].mean)   # 29.69
    """
    if not _CORE_AVAILABLE:
        raise RuntimeError(
            "zedda C++ core not found. "
            "Please reinstall: pip install zedda"
        )
    return _core.profile(path, False)


# ─────────────────────────────────────────────────────────────────
#  profile() — scan + print beautiful terminal report
#
#  This is the MAIN function. One line does everything.
#
#  Example::
#
#      zd.profile("data.csv")
# ─────────────────────────────────────────────────────────────────
def profile(path: str) -> object:
    """
    Profile a file and print a beautiful terminal report.

    The simplest EDA you will ever do::

        import zedda as zd
        zd.profile("data.csv")

    Args:
        path: Path to your data file.

    Returns:
        DatasetProfile (also prints report to terminal).
    """
    if _RICH_AVAILABLE and _console:
        _console.print(f"\n[bold blue]zedda[/bold blue] [dim]v{__version__}[/dim]")
        _console.print(f"[dim]Scanning[/dim] [cyan]{path}[/cyan]...\n")

    result = scan(path)
    _print_report(result)
    return result


# ─────────────────────────────────────────────────────────────────
#  _print_report() — beautiful Rich terminal output
# ─────────────────────────────────────────────────────────────────
def _print_report(p: object) -> None:
    if not _RICH_AVAILABLE or _console is None:
        _print_plain(p)
        return

    # ── Dataset summary panel ────────────────────────────────────
    summary = (
        f"[bold]File:[/bold]    {p.file_name}\n"
        f"[bold]Rows:[/bold]    [green]{p.num_rows:,}[/green]\n"
        f"[bold]Cols:[/bold]    {p.num_cols}  "
        f"([cyan]{p.num_numeric} numeric[/cyan], "
        f"[magenta]{p.num_string} string[/magenta])\n"
        f"[bold]Nulls:[/bold]   "
        + ("[red]" if p.overall_null_pct > 10 else "[green]")
        + f"{p.overall_null_pct:.1f}%[/]"
        + f"  ({p.total_null_cells:,} cells)\n"
        f"[bold]Scanned:[/bold] {p.scan_time_ms:.0f} ms"
    )
    _console.print(Panel(summary, title="[bold blue]Dataset Overview[/bold blue]",
                         border_style="blue"))

    # ── Column table ─────────────────────────────────────────────
    table = Table(
        show_header=True,
        header_style="bold white on blue",
        border_style="dim",
        box=box.SIMPLE_HEAVY,
        padding=(0, 1),
    )

    table.add_column("Column",   style="bold cyan",  min_width=12)
    table.add_column("Type",     style="magenta",    min_width=6)
    table.add_column("Nulls",    justify="right",    min_width=8)
    table.add_column("Unique~",  justify="right",    min_width=8)
    table.add_column("Mean",     justify="right",    min_width=10)
    table.add_column("Min",      justify="right",    min_width=10)
    table.add_column("Max",      justify="right",    min_width=10)
    table.add_column("Flags",    min_width=14)

    for col in p.columns:
        # null cell
        null_str = f"{col.null_pct:.1f}%"
        null_cell = Text(null_str)
        if col.null_pct > 20:
            null_cell.stylize("bold red")
        elif col.null_pct > 5:
            null_cell.stylize("yellow")
        else:
            null_cell.stylize("green")

        # mean / min / max
        if col.type_str in ("int", "float"):
            mean_str = f"{col.mean:,.2f}"
            min_str  = f"{col.val_min:,.2f}"
            max_str  = f"{col.val_max:,.2f}"
        else:
            mean_str = f"len~{col.mean_str_len:.0f}"
            min_str  = "—"
            max_str  = "—"

        # flags
        flags = []
        if col.has_high_nulls:        flags.append("[red]HIGH NULL[/red]")
        if col.is_constant:           flags.append("[yellow]CONSTANT[/yellow]")
        if col.is_high_cardinality:   flags.append("[blue]HIGH CARD[/blue]")
        flags_str = " ".join(flags) if flags else "[dim]ok[/dim]"

        table.add_row(
            col.name,
            col.type_str,
            null_cell,
            str(col.unique_approx),
            mean_str,
            min_str,
            max_str,
            Text.from_markup(flags_str),
        )

    _console.print(table)
    _console.print()


def _print_plain(p: object) -> None:
    """Fallback if Rich not installed."""
    print(f"\nzedda v{__version__}")
    print(f"File  : {p.file_name}")
    print(f"Rows  : {p.num_rows:,}")
    print(f"Cols  : {p.num_cols}")
    print(f"Nulls : {p.overall_null_pct:.1f}%")
    print(f"Time  : {p.scan_time_ms:.0f} ms")
    print("\nColumn        Type    Nulls    Mean")
    print("-" * 50)
    for col in p.columns:
        mean_s = f"{col.mean:.2f}" if col.type_str in ("int","float") else "—"
        print(f"{col.name:<14}{col.type_str:<8}{col.null_pct:.1f}%    {mean_s}")


# ─────────────────────────────────────────────────────────────────
#  compare() — diff two datasets
#
#  Example::
#
#      zd.compare("train.csv", "test.csv")
# ─────────────────────────────────────────────────────────────────
def compare(path_a: str, path_b: str) -> None:
    """
    Compare two datasets side by side.

    Shows schema differences, null rate changes,
    and distribution shifts between files.

    Example::

        zd.compare("train.csv", "test.csv")

    Args:
        path_a: First file path.
        path_b: Second file path.
    """
    p_a = scan(path_a)
    p_b = scan(path_b)

    if not _RICH_AVAILABLE or _console is None:
        print(f"A: {p_a.file_name} — {p_a.num_rows:,} rows")
        print(f"B: {p_b.file_name} — {p_b.num_rows:,} rows")
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
    table.add_column("Nulls A", justify="right", min_width=8)
    table.add_column("Nulls B", justify="right", min_width=8)
    table.add_column("Mean A",  justify="right", min_width=10)
    table.add_column("Mean B",  justify="right", min_width=10)
    table.add_column("Drift",   min_width=10)

    cols_a = {c.name: c for c in p_a.columns}
    cols_b = {c.name: c for c in p_b.columns}
    all_cols = list(dict.fromkeys(list(cols_a.keys()) + list(cols_b.keys())))

    for name in all_cols:
        ca = cols_a.get(name)
        cb = cols_b.get(name)

        type_a = ca.type_str if ca else "[red]MISSING[/red]"
        type_b = cb.type_str if cb else "[red]MISSING[/red]"

        null_a = f"{ca.null_pct:.1f}%" if ca else "—"
        null_b = f"{cb.null_pct:.1f}%" if cb else "—"

        if ca and ca.type_str in ("int", "float"):
            mean_a = f"{ca.mean:,.2f}"
        else:
            mean_a = "—"

        if cb and cb.type_str in ("int", "float"):
            mean_b = f"{cb.mean:,.2f}"
        else:
            mean_b = "—"

        # drift detection
        drift = "[green]ok[/green]"
        if ca and cb and ca.type_str in ("int","float") and cb.type_str in ("int","float"):
            if ca.stddev > 0:
                shift = abs(ca.mean - cb.mean) / ca.stddev
                if shift > 1.0:
                    drift = "[red]DRIFT[/red]"
                elif shift > 0.3:
                    drift = "[yellow]SHIFT[/yellow]"
        if not ca:
            drift = "[blue]NEW COL[/blue]"
        if not cb:
            drift = "[red]REMOVED[/red]"

        table.add_row(name, type_a, type_b, null_a, null_b,
                      mean_a, mean_b, Text.from_markup(drift))

    _console.print(table)
    _console.print()


# ─────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────
__all__ = ["profile", "scan", "compare", "__version__"]