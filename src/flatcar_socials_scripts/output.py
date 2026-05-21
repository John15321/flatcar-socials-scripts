"""CSV output handler for scraped platform statistics."""

import csv
import logging
from dataclasses import asdict
from pathlib import Path

from .platforms.base import PlatformStats, UserStats

logger = logging.getLogger(__name__)


def write_csv(stats: PlatformStats, output_path: Path) -> Path:
    """Write platform statistics to a CSV file.

    If the file exists, appends a new row. Otherwise creates it with headers.

    Args:
        stats: The scraped platform statistics.
        output_path: Path to the output CSV file.

    Returns:
        The path to the written CSV file.
    """
    flat = stats.as_flat_dict()
    file_exists = output_path.exists()

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        if not file_exists:
            logger.debug("Creating new CSV file: %s", output_path)
            writer.writeheader()
        else:
            logger.debug("Appending to existing CSV file: %s", output_path)
        writer.writerow(flat)

    logger.info("Wrote %d fields to %s", len(flat), output_path)
    return output_path


def write_user_stats_csv(
    user_stats: list[UserStats],
    output_path: Path,
    server_name: str = "",
) -> Path:
    """Write per-user statistics to a CSV file.

    Args:
        user_stats: List of per-user statistics.
        output_path: Path to the output CSV file.
        server_name: Server name to include in each row.

    Returns:
        The path to the written CSV file.
    """
    if not user_stats:
        logger.warning("No user stats to write")
        return output_path

    rows = []
    for us in user_stats:
        row = asdict(us)
        row["server_name"] = server_name
        rows.append(row)

    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d user records to %s", len(rows), output_path)
    return output_path
