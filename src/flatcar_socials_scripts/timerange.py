"""Time range parsing utilities.

Supports absolute dates (YYYY-MM-DD) and shorthand expressions like:
  last-month, last-year, last-30d, last-6mo, last-2y
"""

import enum
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


class Granularity(enum.Enum):
    """Time bucket granularity for analytics."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"

    def bucket_key(self, dt: datetime) -> str:
        """Return the bucket key string for the given datetime."""
        if self is Granularity.DAILY:
            return dt.strftime("%Y-%m-%d")
        if self is Granularity.WEEKLY:
            iso = dt.isocalendar()
            return f"{iso[0]}-W{iso[1]:02d}"
        if self is Granularity.MONTHLY:
            return dt.strftime("%Y-%m")
        # YEARLY
        return dt.strftime("%Y")


@dataclass
class TimeRange:
    """A time range with start and end bounds."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError(
                f"Start ({self.start.isoformat()}) must be before "
                f"end ({self.end.isoformat()})"
            )


_SHORTHAND_RE = re.compile(
    r"^last[_-]?(\d+)?(d|day|days|w|week|weeks|mo|month|months|y|year|years)$",
    re.IGNORECASE,
)

_NAMED_SHORTHANDS = {
    "last-month": lambda now: (now.replace(day=1) - timedelta(days=1)).replace(day=1),
    "last-year": lambda now: now.replace(year=now.year - 1, month=1, day=1),
}

_UNIT_TO_DAYS = {
    "d": 1,
    "day": 1,
    "days": 1,
    "w": 7,
    "week": 7,
    "weeks": 7,
    "mo": 30,
    "month": 30,
    "months": 30,
    "y": 365,
    "year": 365,
    "years": 365,
}


def parse_time_range(
    from_str: str | None = None,
    to_str: str | None = None,
    shorthand: str | None = None,
) -> TimeRange:
    """Parse a time range from CLI arguments.

    Args:
        from_str: Start date as YYYY-MM-DD.
        to_str: End date as YYYY-MM-DD (defaults to now).
        shorthand: A shorthand like 'last-30d', 'last-6mo', 'last-month'.

    Returns:
        A TimeRange with UTC-aware datetimes.
    """
    now = datetime.now(UTC)

    if shorthand:
        return _parse_shorthand(shorthand, now)

    end = _parse_date(to_str) if to_str else now
    if from_str:
        start = _parse_date(from_str)
    else:
        # Default: last 6 months
        start = now - timedelta(days=180)

    return TimeRange(start=start, end=end)


def _parse_shorthand(shorthand: str, now: datetime) -> TimeRange:
    """Parse a shorthand time expression."""
    key = shorthand.lower().strip()

    # Named shorthands
    if key in _NAMED_SHORTHANDS:
        start = _NAMED_SHORTHANDS[key](now).replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC
        )
        return TimeRange(start=start, end=now)

    # Pattern: last-Nd, last-6mo, last-2y, etc.
    match = _SHORTHAND_RE.match(key)
    if not match:
        raise click_bad_param(
            f"Unknown time shorthand: '{shorthand}'. "
            "Use formats like: last-30d, last-6mo, last-2y, last-month, last-year"
        )

    count_str, unit = match.groups()
    count = int(count_str) if count_str else 1
    days = count * _UNIT_TO_DAYS[unit.lower()]
    start = now - timedelta(days=days)

    return TimeRange(start=start, end=now)


def _parse_date(date_str: str) -> datetime:
    """Parse a YYYY-MM-DD string into a UTC datetime."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        raise click_bad_param(
            f"Invalid date format: '{date_str}'. Use YYYY-MM-DD."
        ) from None


def click_bad_param(message: str) -> Exception:
    """Return a click.BadParameter for consistent error handling."""
    import click

    return click.BadParameter(message)
