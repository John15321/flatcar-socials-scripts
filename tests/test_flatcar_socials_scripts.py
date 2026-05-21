"""Tests for flatcar_socials_scripts module."""

from datetime import UTC, datetime
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from flatcar_socials_scripts.cli import cli
from flatcar_socials_scripts.output import write_csv, write_user_stats_csv
from flatcar_socials_scripts.platforms.base import PlatformStats, UserStats
from flatcar_socials_scripts.timerange import TimeRange, parse_time_range


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

    def test_write_appends(self, tmp_path: Path) -> None:
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
        assert len(lines) == 3  # header + 2 rows


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
