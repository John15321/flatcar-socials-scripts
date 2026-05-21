"""CLI entry point for flatcar-socials-scripts."""

import asyncio
import logging
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from .output import write_csv, write_user_stats_csv
from .platforms.discord import DiscordScraper
from .timerange import TimeRange, parse_time_range

console = Console()
logger = logging.getLogger("flatcar_socials")


def _setup_logging(verbose: bool) -> None:
    """Configure logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    # Suppress noisy discord.py logs unless in verbose mode
    logging.getLogger("discord").setLevel(
        logging.WARNING if not verbose else logging.DEBUG
    )
    logging.getLogger("discord.http").setLevel(logging.WARNING)


@click.group()
@click.version_option()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose/debug logging.")
def cli(verbose: bool) -> None:
    """Flatcar Socials Scripts — scrape stats from social platforms."""
    _setup_logging(verbose)


@cli.command()
@click.option(
    "--token",
    envvar="DISCORD_BOT_TOKEN",
    required=True,
    help="Discord bot token (or set DISCORD_BOT_TOKEN env var).",
)
@click.option(
    "--guild-id",
    envvar="DISCORD_GUILD_ID",
    required=True,
    type=int,
    help="Discord server (guild) ID (or set DISCORD_GUILD_ID env var).",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    default="discord_stats.csv",
    show_default=True,
    help="Output CSV file path.",
)
@click.option(
    "--from",
    "from_date",
    default=None,
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--to",
    "to_date",
    default=None,
    help="End date (YYYY-MM-DD). Defaults to now.",
)
@click.option(
    "--range",
    "shorthand",
    default=None,
    help="Time shorthand: last-30d, last-6mo, last-2y, last-month, last-year.",
)
@click.option(
    "--user-stats",
    "user_stats_path",
    default=None,
    help="Output per-user breakdown CSV. Omit to skip user stats.",
)
def discord(
    token: str,
    guild_id: int,
    output_path: str,
    from_date: str | None,
    to_date: str | None,
    shorthand: str | None,
    user_stats_path: str | None,
) -> None:
    """Scrape statistics from a Discord server."""
    time_range = parse_time_range(from_date, to_date, shorthand)
    collect_users = user_stats_path is not None
    asyncio.run(
        _scrape_discord(
            token,
            guild_id,
            Path(output_path),
            time_range,
            collect_users,
            Path(user_stats_path) if user_stats_path else None,
        )
    )


async def _scrape_discord(
    token: str,
    guild_id: int,
    output_path: Path,
    time_range: TimeRange,
    collect_users: bool,
    user_stats_path: Path | None,
) -> None:

    logger.info("Starting Discord scrape for guild %s", guild_id)
    scraper = DiscordScraper(
        token=token,
        guild_id=guild_id,
        time_range=time_range,
        collect_user_stats=collect_users,
    )

    with console.status("[bold green]Connecting to Discord..."):
        try:
            stats = await scraper.scrape()
        except Exception:
            logger.exception("Failed to scrape Discord server")
            raise
        finally:
            await scraper.close()

    logger.info(
        "Scrape complete for '%s' — %d stats collected",
        stats.server_name,
        len(stats.stats),
    )

    # Display results in a rich table
    table = Table(title=f"Discord Stats — {stats.server_name}")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green", justify="right")

    for key, value in stats.stats.items():
        label = key.replace("_", " ").title()
        table.add_row(label, str(value))

    console.print(table)

    # Write server stats CSV
    csv_path = write_csv(stats, output_path)
    console.print(f"\n[bold]CSV written to:[/bold] {csv_path}")

    # Write per-user stats CSV if requested
    if user_stats_path and stats.user_stats:
        user_csv = write_user_stats_csv(
            stats.user_stats, user_stats_path, stats.server_name
        )
        console.print(f"[bold]User stats CSV written to:[/bold] {user_csv}")
        console.print(f"  [dim]{len(stats.user_stats)} users tracked[/dim]")


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
