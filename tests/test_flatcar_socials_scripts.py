"""Tests for flatcar_socials_scripts module."""

import csv
from datetime import UTC, datetime
from pathlib import Path

import click
import polars as pl
import pytest
from click.testing import CliRunner

from flatcar_socials_scripts.analytics import (
    compute_join_trends,
    compute_message_buckets,
    compute_message_distribution,
    compute_per_user_time_buckets,
    compute_role_breakdown,
)
from flatcar_socials_scripts.cli import cli
from flatcar_socials_scripts.output import write_csv, write_user_stats_csv
from flatcar_socials_scripts.platforms.base import PlatformStats, UserStats
from flatcar_socials_scripts.timerange import Granularity, TimeRange, parse_time_range


class TestPlatformStats:
    """Tests for the PlatformStats dataclass."""

    def test_as_flat_dict(self) -> None:
        stats = PlatformStats(
            platform="discord",
            server_name="TestServer",
            scraped_at=datetime(2026, 1, 1, tzinfo=UTC),
            stats={"total_members": 100, "text_channels": 5},
        )
        flat = stats.as_flat_dict()
        assert flat["platform"] == "discord"
        assert flat["server_name"] == "TestServer"
        assert flat["total_members"] == 100
        assert flat["text_channels"] == 5
        assert "scraped_at" in flat

    def test_defaults(self) -> None:
        stats = PlatformStats(platform="test", server_name="srv")
        assert stats.stats == {}
        assert stats.user_stats == []
        assert isinstance(stats.scraped_at, datetime)


class TestWriteCsv:
    """Tests for the CSV output handler."""

    def test_write_creates_file(self, tmp_path: Path) -> None:
        output = tmp_path / "stats.csv"
        stats = PlatformStats(
            platform="discord",
            server_name="Test",
            stats={"members": 42},
        )
        result = write_csv(stats, output)
        assert result == output
        assert output.exists()
        content = output.read_text()
        assert "platform" in content  # header
        assert "discord" in content
        assert "42" in content

    def test_write_overwrites_by_default(self, tmp_path: Path) -> None:
        output = tmp_path / "stats.csv"
        stats1 = PlatformStats(
            platform="discord", server_name="A", stats={"members": 10}
        )
        stats2 = PlatformStats(
            platform="discord", server_name="B", stats={"members": 20}
        )
        write_csv(stats1, output)
        write_csv(stats2, output)
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 row (overwritten)
        assert "B" in lines[1]

    def test_write_appends(self, tmp_path: Path) -> None:
        output = tmp_path / "stats.csv"
        stats1 = PlatformStats(
            platform="discord", server_name="A", stats={"members": 10}
        )
        stats2 = PlatformStats(
            platform="discord", server_name="B", stats={"members": 20}
        )
        write_csv(stats1, output)
        write_csv(stats2, output, append=True)
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 rows

    def test_write_append_aligns_to_existing_header(self, tmp_path: Path) -> None:
        output = tmp_path / "stats.csv"
        stats1 = PlatformStats(
            platform="discord",
            server_name="A",
            stats={"members": 10, "admins": 2},
        )
        stats2 = PlatformStats(
            platform="discord",
            server_name="B",
            stats={"admins": 3, "members": 20},
        )

        write_csv(stats1, output)
        write_csv(stats2, output, append=True)

        with output.open(newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))
        assert rows[0]["members"] == "10"
        assert rows[0]["admins"] == "2"
        assert rows[1]["members"] == "20"
        assert rows[1]["admins"] == "3"

    def test_write_append_rejects_new_columns(self, tmp_path: Path) -> None:
        output = tmp_path / "stats.csv"
        stats1 = PlatformStats(
            platform="discord", server_name="A", stats={"members": 10}
        )
        stats2 = PlatformStats(
            platform="discord",
            server_name="B",
            stats={"members": 20, "admins": 3},
        )

        write_csv(stats1, output)

        with pytest.raises(ValueError, match="New columns detected"):
            write_csv(stats2, output, append=True)


class TestWriteUserStatsCsv:
    """Tests for the per-user CSV output."""

    def test_write_user_stats(self, tmp_path: Path) -> None:
        output = tmp_path / "users.csv"
        users = [
            UserStats(
                user_id=1,
                username="alice",
                display_name="Alice",
                is_bot=False,
                joined_at="2025-01-01T00:00:00+00:00",
                roles="Maintainer",
                message_count=50,
                first_message_at="2025-01-02T00:00:00+00:00",
                last_message_at="2026-05-01T00:00:00+00:00",
                days_since_join=500,
                days_to_first_message=1,
                channels_active_in=3,
            ),
            UserStats(
                user_id=2,
                username="bob",
                display_name="Bob",
                is_bot=False,
                joined_at="2025-06-01T00:00:00+00:00",
                roles="",
                message_count=5,
                channels_active_in=1,
            ),
        ]
        result = write_user_stats_csv(users, output, server_name="TestServer")
        assert result == output
        content = output.read_text()
        assert "alice" in content
        assert "bob" in content
        assert "server_name" in content
        assert "TestServer" in content
        lines = content.strip().split("\n")
        assert len(lines) == 3  # header + 2 rows

    def test_write_empty_user_stats(self, tmp_path: Path) -> None:
        output = tmp_path / "users.csv"
        result = write_user_stats_csv([], output)
        assert result == output


class TestTimeRange:
    """Tests for time range parsing."""

    def test_default_range(self) -> None:
        tr = parse_time_range()
        assert tr.start < tr.end
        # Default is ~180 days back
        diff = (tr.end - tr.start).days
        assert 179 <= diff <= 181

    def test_from_to_dates(self) -> None:
        tr = parse_time_range(from_str="2025-01-01", to_str="2025-12-31")
        assert tr.start.year == 2025
        assert tr.start.month == 1
        assert tr.end.month == 12

    def test_shorthand_days(self) -> None:
        tr = parse_time_range(shorthand="last-30d")
        diff = (tr.end - tr.start).days
        assert 29 <= diff <= 31

    def test_shorthand_months(self) -> None:
        tr = parse_time_range(shorthand="last-6mo")
        diff = (tr.end - tr.start).days
        assert 179 <= diff <= 181

    def test_shorthand_years(self) -> None:
        tr = parse_time_range(shorthand="last-2y")
        diff = (tr.end - tr.start).days
        assert 729 <= diff <= 731

    def test_shorthand_last_year(self) -> None:
        tr = parse_time_range(shorthand="last-year")
        assert tr.start.month == 1
        assert tr.start.day == 1

    def test_invalid_date(self) -> None:
        with pytest.raises(click.BadParameter):
            parse_time_range(from_str="not-a-date")

    def test_invalid_shorthand(self) -> None:
        with pytest.raises(click.BadParameter):
            parse_time_range(shorthand="garbage")

    def test_invalid_range_order(self) -> None:
        with pytest.raises(ValueError):
            TimeRange(
                start=datetime(2026, 1, 1, tzinfo=UTC),
                end=datetime(2025, 1, 1, tzinfo=UTC),
            )


class TestCli:
    """Tests for the CLI commands."""

    def test_cli_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Flatcar Socials Scripts" in result.output

    def test_discord_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["discord", "--help"])
        assert result.exit_code == 0
        assert "--token" in result.output
        assert "--guild-id" in result.output
        assert "--output" in result.output
        assert "--from" in result.output
        assert "--to" in result.output
        assert "--range" in result.output
        assert "--user-stats" in result.output
        assert "--granularity" in result.output
        assert "--append" in result.output

    def test_matrix_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["matrix", "--help"])
        assert result.exit_code == 0
        assert "--homeserver" in result.output
        assert "--token" in result.output
        assert "--room-id" in result.output
        assert "--output" in result.output
        assert "--from" in result.output
        assert "--to" in result.output
        assert "--range" in result.output
        assert "--user-stats" in result.output
        assert "--granularity" in result.output
        assert "--append" in result.output


class TestGranularity:
    """Tests for the Granularity enum and bucket_key method."""

    def test_daily_bucket_key(self) -> None:
        dt = datetime(2025, 3, 15, 10, 30, tzinfo=UTC)
        assert Granularity.DAILY.bucket_key(dt) == "2025-03-15"

    def test_weekly_bucket_key(self) -> None:
        dt = datetime(2025, 3, 15, 10, 30, tzinfo=UTC)
        result = Granularity.WEEKLY.bucket_key(dt)
        assert result.startswith("2025-W")

    def test_monthly_bucket_key(self) -> None:
        dt = datetime(2025, 3, 15, 10, 30, tzinfo=UTC)
        assert Granularity.MONTHLY.bucket_key(dt) == "2025-03"

    def test_yearly_bucket_key(self) -> None:
        dt = datetime(2025, 3, 15, 10, 30, tzinfo=UTC)
        assert Granularity.YEARLY.bucket_key(dt) == "2025"

    def test_enum_values(self) -> None:
        assert Granularity("daily") == Granularity.DAILY
        assert Granularity("weekly") == Granularity.WEEKLY
        assert Granularity("monthly") == Granularity.MONTHLY
        assert Granularity("yearly") == Granularity.YEARLY


class TestComputeMessageBuckets:
    """Tests for compute_message_buckets."""

    def test_empty_dataframe(self) -> None:
        df = pl.DataFrame(
            {"timestamp": [], "user_id": [], "is_bot": [], "top_role": []}
        )
        result = compute_message_buckets(df, Granularity.MONTHLY)
        assert result == {}

    def test_typed_empty_dataframe(self) -> None:
        """Empty DataFrame with explicit schema shouldn't crash on filter."""
        df = pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("us", "UTC"),
                "user_id": pl.Int64,
                "is_bot": pl.Boolean,
                "top_role": pl.Utf8,
            }
        )
        # This would crash if is_bot is Null dtype
        filtered = df.filter(pl.col("is_bot").not_())
        assert filtered.is_empty()
        assert compute_message_buckets(df, Granularity.MONTHLY) == {}
        assert compute_role_breakdown(filtered) == {
            "counts": {},
            "percentages": {},
        }

    def test_monthly_bucketing(self) -> None:
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2025, 1, 10, tzinfo=UTC),
                    datetime(2025, 1, 20, tzinfo=UTC),
                    datetime(2025, 2, 5, tzinfo=UTC),
                ],
            }
        )
        result = compute_message_buckets(df, Granularity.MONTHLY)
        assert result == {"2025-01": 2, "2025-02": 1}

    def test_daily_bucketing(self) -> None:
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2025, 1, 10, 8, 0, tzinfo=UTC),
                    datetime(2025, 1, 10, 12, 0, tzinfo=UTC),
                    datetime(2025, 1, 11, 9, 0, tzinfo=UTC),
                ],
            }
        )
        result = compute_message_buckets(df, Granularity.DAILY)
        assert result == {"2025-01-10": 2, "2025-01-11": 1}


class TestComputeRoleBreakdown:
    """Tests for compute_role_breakdown."""

    def test_empty_dataframe(self) -> None:
        df = pl.DataFrame({"top_role": [], "user_id": [], "is_bot": []})
        result = compute_role_breakdown(df)
        assert result == {"counts": {}, "percentages": {}}

    def test_role_counts_and_percentages(self) -> None:
        df = pl.DataFrame(
            {
                "top_role": ["Admin", "Admin", "Member", "Member", "Member"],
            }
        )
        result = compute_role_breakdown(df)
        assert result["counts"]["Admin"] == 2
        assert result["counts"]["Member"] == 3
        assert result["percentages"]["Admin"] == 40.0
        assert result["percentages"]["Member"] == 60.0


class TestComputeMessageDistribution:
    """Tests for compute_message_distribution."""

    def test_empty_dataframe(self) -> None:
        df = pl.DataFrame({"user_id": [], "message_count": [], "is_bot": []})
        result = compute_message_distribution(df)
        assert result["users_with_0_messages"] == 0

    def test_distribution_buckets(self) -> None:
        df = pl.DataFrame(
            {
                "user_id": [1, 2, 3, 4, 5],
                "message_count": [0, 3, 15, 50, 200],
                "is_bot": [False, False, False, False, False],
            }
        )
        result = compute_message_distribution(df)
        assert result["users_with_0_messages"] == 1
        assert result["users_with_1_to_5_messages"] == 1
        assert result["users_with_6_to_20_messages"] == 1
        assert result["users_with_21_to_100_messages"] == 1
        assert result["users_with_101_plus_messages"] == 1

    def test_stats_values(self) -> None:
        df = pl.DataFrame(
            {
                "user_id": [1, 2, 3],
                "message_count": [0, 10, 20],
                "is_bot": [False, False, False],
            }
        )
        result = compute_message_distribution(df)
        assert result["median_messages_per_user"] == 10
        assert result["mean_messages_per_user"] == 10.0
        assert isinstance(result["p90_messages_per_user"], int)
        assert isinstance(result["p99_messages_per_user"], int)

    def test_even_median_preserves_fractional_value(self) -> None:
        df = pl.DataFrame(
            {
                "user_id": [1, 2],
                "message_count": [0, 1],
                "is_bot": [False, False],
            }
        )

        result = compute_message_distribution(df)

        assert result["median_messages_per_user"] == 0.5

    def test_excludes_bots(self) -> None:
        df = pl.DataFrame(
            {
                "user_id": [1, 2],
                "message_count": [100, 500],
                "is_bot": [False, True],
            }
        )
        result = compute_message_distribution(df)
        # Only 1 human user with 100 messages
        assert result["users_with_21_to_100_messages"] == 1
        assert result["users_with_101_plus_messages"] == 0


class TestComputeJoinTrends:
    """Tests for compute_join_trends."""

    def test_empty_dataframe(self) -> None:
        df = pl.DataFrame({"joined_at": [], "is_bot": []})
        result = compute_join_trends(
            df, Granularity.MONTHLY, "2025-01-01", "2025-12-31"
        )
        assert result == {}

    def test_monthly_join_trends(self) -> None:
        df = pl.DataFrame(
            {
                "joined_at": [
                    datetime(2025, 1, 5, tzinfo=UTC),
                    datetime(2025, 1, 20, tzinfo=UTC),
                    datetime(2025, 3, 10, tzinfo=UTC),
                ],
                "is_bot": [False, False, False],
            }
        )
        result = compute_join_trends(
            df, Granularity.MONTHLY, "2025-01-01", "2025-12-31"
        )
        assert result["2025-01"] == 2
        assert result["2025-03"] == 1

    def test_excludes_bots(self) -> None:
        df = pl.DataFrame(
            {
                "joined_at": [
                    datetime(2025, 1, 5, tzinfo=UTC),
                    datetime(2025, 1, 10, tzinfo=UTC),
                ],
                "is_bot": [False, True],
            }
        )
        result = compute_join_trends(
            df, Granularity.MONTHLY, "2025-01-01", "2025-12-31"
        )
        assert result == {"2025-01": 1}

    def test_filters_by_date_range(self) -> None:
        df = pl.DataFrame(
            {
                "joined_at": [
                    datetime(2024, 12, 15, tzinfo=UTC),  # before range
                    datetime(2025, 2, 10, tzinfo=UTC),  # in range
                    datetime(2025, 7, 1, tzinfo=UTC),  # after range
                ],
                "is_bot": [False, False, False],
            }
        )
        result = compute_join_trends(
            df, Granularity.MONTHLY, "2025-01-01", "2025-06-30"
        )
        assert result == {"2025-02": 1}


class TestComputePerUserTimeBuckets:
    """Tests for compute_per_user_time_buckets."""

    def test_empty_dataframe(self) -> None:
        df = pl.DataFrame({"user_id": [], "timestamp": []})
        result = compute_per_user_time_buckets(df, Granularity.MONTHLY)
        assert "user_id" in result.columns

    def test_pivots_correctly(self) -> None:
        df = pl.DataFrame(
            {
                "user_id": [1, 1, 2],
                "timestamp": [
                    datetime(2025, 1, 10, tzinfo=UTC),
                    datetime(2025, 2, 5, tzinfo=UTC),
                    datetime(2025, 1, 15, tzinfo=UTC),
                ],
            }
        )
        result = compute_per_user_time_buckets(df, Granularity.MONTHLY)
        assert "2025-01" in result.columns
        assert "2025-02" in result.columns
        # User 1 has 1 message in Jan, 1 in Feb
        user1 = result.filter(pl.col("user_id") == 1)
        assert user1["2025-01"].item() == 1
        assert user1["2025-02"].item() == 1
        # User 2 has 1 in Jan, 0 in Feb
        user2 = result.filter(pl.col("user_id") == 2)
        assert user2["2025-01"].item() == 1
        assert user2["2025-02"].item() == 0


class TestWriteUserStatsCsvWithBuckets:
    """Tests for write_user_stats_csv with per-user time bucket columns."""

    def test_includes_time_bucket_columns(self, tmp_path: Path) -> None:
        output = tmp_path / "users.csv"
        users = [
            UserStats(
                user_id=1,
                username="alice",
                display_name="Alice",
                is_bot=False,
                joined_at="2025-01-01T00:00:00+00:00",
                roles="Admin",
                message_count=2,
            ),
        ]
        messages_df = pl.DataFrame(
            {
                "user_id": [1, 1],
                "timestamp": [
                    datetime(2025, 1, 10, tzinfo=UTC),
                    datetime(2025, 2, 5, tzinfo=UTC),
                ],
                "is_bot": [False, False],
                "top_role": ["Admin", "Admin"],
            }
        )
        result = write_user_stats_csv(
            users,
            output,
            "TestServer",
            messages_df=messages_df,
            granularity=Granularity.MONTHLY,
        )
        assert result == output
        content = output.read_text()
        assert "messages_2025-01" in content
        assert "messages_2025-02" in content

    def test_silent_members_get_zero_buckets(self, tmp_path: Path) -> None:
        """Silent members should appear with 0 for all time bucket columns."""
        output = tmp_path / "users.csv"
        users = [
            UserStats(
                user_id=1,
                username="alice",
                display_name="Alice",
                is_bot=False,
                joined_at="2025-01-01T00:00:00+00:00",
                roles="Admin",
                message_count=2,
            ),
            UserStats(
                user_id=2,
                username="silent_bob",
                display_name="Silent Bob",
                is_bot=False,
                joined_at="2025-01-15T00:00:00+00:00",
                roles="",
                message_count=0,
                channels_active_in=0,
            ),
        ]
        messages_df = pl.DataFrame(
            {
                "user_id": [1, 1],
                "timestamp": [
                    datetime(2025, 1, 10, tzinfo=UTC),
                    datetime(2025, 2, 5, tzinfo=UTC),
                ],
                "is_bot": [False, False],
                "top_role": ["Admin", "Admin"],
            }
        )
        write_user_stats_csv(
            users,
            output,
            "TestServer",
            messages_df=messages_df,
            granularity=Granularity.MONTHLY,
        )
        content = output.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 3  # header + 2 users
        assert "silent_bob" in content
        # Silent Bob's row should end with 0s for the bucket columns
        bob_line = [line for line in lines if "silent_bob" in line][0]
        # The last two values should be 0 (messages_2025-01, messages_2025-02)
        bob_fields = bob_line.split(",")
        assert bob_fields[-1] == "0"
        assert bob_fields[-2] == "0"
