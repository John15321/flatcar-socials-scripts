"""Analytics engine using Polars for community health metrics.

Takes raw scraper output and produces aggregated statistics:
- Message counts bucketed by configurable granularity
- Role-based message breakdown
- Message distribution summary (percentiles, histograms)
- Join trend stats
- Per-user activity over time
"""

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl

from .timerange import Granularity

logger = logging.getLogger(__name__)


def _bucket_key_expr(column: str, granularity: Granularity) -> pl.Expr:
    """Build a Polars expression for bucket-key formatting."""
    if granularity is Granularity.DAILY:
        return pl.col(column).dt.strftime("%Y-%m-%d")
    elif granularity is Granularity.WEEKLY:
        return pl.col(column).dt.strftime("%G-W%V")
    elif granularity is Granularity.MONTHLY:
        return pl.col(column).dt.strftime("%Y-%m")
    elif granularity is Granularity.YEARLY:
        return pl.col(column).dt.strftime("%Y")
    raise ValueError(f"Unsupported granularity: {granularity}")


def compute_message_buckets(
    messages: pl.DataFrame,
    granularity: Granularity,
) -> dict[str, int]:
    """Bucket message counts by time granularity.

    Args:
        messages: DataFrame with at least a ``timestamp`` (Datetime) column.
        granularity: How to bucket timestamps.

    Returns:
        Dict of ``{bucket_key: count}``.
    """
    if messages.is_empty():
        return {}

    df = messages.with_columns(
        _bucket_key_expr("timestamp", granularity).alias("bucket")
    )
    counts = df.group_by("bucket").len().sort("bucket")
    return dict(
        zip(
            counts["bucket"].to_list(),
            counts["len"].to_list(),
            strict=False,
        )
    )


def compute_role_breakdown(
    messages: pl.DataFrame,
) -> dict[str, dict[str, int | float]]:
    """Compute message counts and percentages by role.

    Each message should have a ``top_role`` column (the user's highest role).

    Returns:
        Dict with keys ``counts`` and ``percentages``, each mapping
        role name to value.
    """
    if messages.is_empty():
        return {"counts": {}, "percentages": {}}

    role_counts = messages.group_by("top_role").len().sort("len", descending=True)
    total = int(role_counts["len"].sum())  # type: ignore[arg-type]
    counts: dict[str, int] = dict(
        zip(
            role_counts["top_role"].to_list(),
            role_counts["len"].to_list(),
            strict=False,
        )
    )
    percentages: dict[str, float] = {
        role: round(count / total * 100, 1) if total > 0 else 0.0
        for role, count in counts.items()
    }
    result: dict[str, dict[str, int | float]] = {
        "counts": counts,  # type: ignore[dict-item]
        "percentages": percentages,
    }
    return result


def compute_message_distribution(
    user_message_counts: pl.DataFrame,
) -> dict[str, int | float]:
    """Compute message distribution stats for human users.

    Args:
        user_message_counts: DataFrame with ``user_id``, ``message_count``,
            ``is_bot`` columns. Should include ALL members (even those with
            0 messages).

    Returns:
        Dict with histogram buckets, mean, median, p90, p99.
    """
    if user_message_counts.is_empty():
        return {
            "users_with_0_messages": 0,
            "users_with_1_to_5_messages": 0,
            "users_with_6_to_20_messages": 0,
            "users_with_21_to_100_messages": 0,
            "users_with_101_plus_messages": 0,
            "median_messages_per_user": 0,
            "mean_messages_per_user": 0.0,
            "p90_messages_per_user": 0,
            "p99_messages_per_user": 0,
        }
    humans = user_message_counts.filter(pl.col("is_bot").not_())
    if humans.is_empty():
        return {
            "users_with_0_messages": 0,
            "users_with_1_to_5_messages": 0,
            "users_with_6_to_20_messages": 0,
            "users_with_21_to_100_messages": 0,
            "users_with_101_plus_messages": 0,
            "median_messages_per_user": 0,
            "mean_messages_per_user": 0.0,
            "p90_messages_per_user": 0,
            "p99_messages_per_user": 0,
        }

    counts = humans["message_count"]
    return {
        "users_with_0_messages": int(counts.filter(counts == 0).len()),
        "users_with_1_to_5_messages": int(
            counts.filter((counts >= 1) & (counts <= 5)).len()
        ),
        "users_with_6_to_20_messages": int(
            counts.filter((counts >= 6) & (counts <= 20)).len()
        ),
        "users_with_21_to_100_messages": int(
            counts.filter((counts >= 21) & (counts <= 100)).len()
        ),
        "users_with_101_plus_messages": int(counts.filter(counts >= 101).len()),
        "median_messages_per_user": float(counts.median() or 0),  # type: ignore[arg-type]
        "mean_messages_per_user": round(
            float(counts.mean() or 0),  # type: ignore[arg-type]
            1,
        ),
        "p90_messages_per_user": int(
            counts.quantile(0.9, "nearest") or 0  # type: ignore[arg-type]
        ),
        "p99_messages_per_user": int(
            counts.quantile(0.99, "nearest") or 0  # type: ignore[arg-type]
        ),
    }


def compute_join_trends(
    members: pl.DataFrame,
    granularity: Granularity,
    start: str,
    end: str,
) -> dict[str, int]:
    """Bucket join dates by time granularity.

    Args:
        members: DataFrame with ``joined_at`` (Datetime) and ``is_bot`` columns.
        granularity: How to bucket join dates.
        start: ISO date string for range start.
        end: ISO date string for range end.

    Returns:
        Dict of ``{bucket_key: join_count}`` for human members within range.
    """
    if members.is_empty():
        return {}

    humans = members.filter(pl.col("is_bot").not_() & pl.col("joined_at").is_not_null())
    if humans.is_empty():
        return {}

    # Parse boundaries with the same timezone as the data column
    dt_dtype = humans["joined_at"].dtype
    tz: str | None = dt_dtype.time_zone if hasattr(dt_dtype, "time_zone") else None
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(
        tzinfo=ZoneInfo(tz) if tz else None
    )
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(
        tzinfo=ZoneInfo(tz) if tz else None
    )

    # Filter to time range
    in_range = humans.filter(
        (pl.col("joined_at") >= start_dt) & (pl.col("joined_at") <= end_dt)
    )
    if in_range.is_empty():
        return {}

    df = in_range.with_columns(
        _bucket_key_expr("joined_at", granularity).alias("bucket")
    )
    counts = df.group_by("bucket").len().sort("bucket")
    return dict(
        zip(
            counts["bucket"].to_list(),
            counts["len"].to_list(),
            strict=False,
        )
    )


def compute_per_user_time_buckets(
    messages: pl.DataFrame,
    granularity: Granularity,
) -> pl.DataFrame:
    """Pivot messages into per-user × per-bucket counts.

    Args:
        messages: DataFrame with ``user_id`` and ``timestamp`` columns.
        granularity: How to bucket timestamps.

    Returns:
        DataFrame with ``user_id`` and one column per time bucket.
    """
    if messages.is_empty():
        return pl.DataFrame({"user_id": []})

    df = messages.with_columns(
        _bucket_key_expr("timestamp", granularity).alias("bucket")
    )
    pivoted = (
        df.group_by(["user_id", "bucket"])
        .len()
        .pivot(on="bucket", index="user_id", values="len")
        .fill_null(0)
    )
    # Sort columns: user_id first, then bucket columns sorted
    bucket_cols = sorted(c for c in pivoted.columns if c != "user_id")
    return pivoted.select(["user_id", *bucket_cols])
