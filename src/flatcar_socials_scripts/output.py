"""CSV output handler for scraped platform statistics."""

import csv
import io
import logging
from dataclasses import asdict
from pathlib import Path

import polars as pl

from .platforms.base import PlatformStats, UserStats
from .timerange import Granularity

logger = logging.getLogger(__name__)


def write_csv(stats: PlatformStats, output_path: Path, *, append: bool = False) -> Path:
    """Write platform statistics to a CSV file.

    By default the file is overwritten. When ``append=True`` and the file
    already exists, a new row is appended without headers.

    Args:
        stats: The scraped platform statistics.
        output_path: Path to the output CSV file.
        append: If True, append to an existing file instead of overwriting.

    Returns:
        The path to the written CSV file.
    """
    flat = stats.as_flat_dict()
    file_exists = output_path.exists()
    df = pl.DataFrame({k: [v] for k, v in flat.items()})

    if append and file_exists:
        logger.debug("Appending to existing CSV file: %s", output_path)
        # Align append rows to the existing header to avoid column drift.
        with open(output_path, encoding="utf-8", newline="") as f:
            header_line = f.readline()

        if not header_line.strip():
            logger.debug("Existing CSV is empty; writing fresh file: %s", output_path)
            df.write_csv(output_path)
            logger.info("Wrote %d fields to %s", len(flat), output_path)
            return output_path

        existing_cols = next(csv.reader([header_line]))
        missing_in_current = [c for c in existing_cols if c not in df.columns]
        extra_in_current = [c for c in df.columns if c not in existing_cols]

        if extra_in_current:
            raise ValueError(
                "Cannot append due to CSV schema mismatch. "
                f"New columns detected: {extra_in_current}. "
                "Rerun without --append to overwrite with the new schema."
            )

        if missing_in_current:
            df = df.with_columns([pl.lit(None).alias(c) for c in missing_in_current])

        df = df.select(existing_cols)

        buf = io.BytesIO()
        df.write_csv(buf, include_header=False)
        with open(output_path, "ab") as f:
            f.write(buf.getvalue())
    else:
        logger.debug("Creating new CSV file: %s", output_path)
        df.write_csv(output_path)

    logger.info("Wrote %d fields to %s", len(flat), output_path)
    return output_path


def write_user_stats_csv(
    user_stats: list[UserStats],
    output_path: Path,
    server_name: str = "",
    messages_df: pl.DataFrame | None = None,
    granularity: Granularity | None = None,
) -> Path:
    """Write per-user statistics to a CSV file.

    If ``messages_df`` and ``granularity`` are provided, per-user time
    bucket columns (``messages_<bucket>``) are appended.

    Args:
        user_stats: List of per-user statistics.
        output_path: Path to the output CSV file.
        server_name: Server name to include in each row.
        messages_df: Optional Polars DataFrame with raw message data.
        granularity: Time bucket granularity for per-user columns.

    Returns:
        The path to the written CSV file.
    """
    if not user_stats:
        logger.warning("No user stats to write")
        return output_path

    # Build per-user time bucket lookup
    bucket_lookup: dict[int | str, dict[str, int]] = {}
    bucket_cols: list[str] = []
    if messages_df is not None and granularity is not None:
        from .analytics import compute_per_user_time_buckets

        pivoted = compute_per_user_time_buckets(messages_df, granularity)
        bucket_cols = [c for c in pivoted.columns if c != "user_id"]
        for row in pivoted.iter_rows(named=True):
            uid = row["user_id"]
            bucket_lookup[uid] = {f"messages_{col}": row[col] for col in bucket_cols}

    rows = []
    for us in user_stats:
        row = asdict(us)
        row["server_name"] = server_name
        # Append time bucket columns
        if bucket_cols:
            user_buckets = bucket_lookup.get(us.user_id, {})
            for col in bucket_cols:
                row[f"messages_{col}"] = user_buckets.get(f"messages_{col}", 0)
        rows.append(row)

    df = pl.DataFrame(rows)
    df.write_csv(output_path)

    logger.info("Wrote %d user records to %s", len(rows), output_path)
    return output_path
