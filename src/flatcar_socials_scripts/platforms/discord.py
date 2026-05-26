"""Discord platform scraper.

Connects to a Discord server via bot token and collects statistics
such as member count, channel count, posts per month, etc.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import discord
import polars as pl

from ..timerange import Granularity, TimeRange
from .base import PlatformScraper, PlatformStats, UserStats

logger = logging.getLogger(__name__)


@dataclass
class _UserAccumulator:
    """Internal accumulator for per-user message data during scan."""

    user_id: int
    username: str
    display_name: str
    is_bot: bool
    message_count: int = 0
    first_message_at: datetime | None = None
    last_message_at: datetime | None = None
    channels: set[str] = field(default_factory=set)

    def record_message(self, channel_name: str, timestamp: datetime) -> None:
        self.message_count += 1
        self.channels.add(channel_name)
        if self.first_message_at is None or timestamp < self.first_message_at:
            self.first_message_at = timestamp
        if self.last_message_at is None or timestamp > self.last_message_at:
            self.last_message_at = timestamp


class DiscordScraper(PlatformScraper):
    """Scrape statistics from a Discord server (guild)."""

    def __init__(
        self,
        token: str,
        guild_id: int,
        time_range: TimeRange | None = None,
        collect_user_stats: bool = False,
        granularity: Granularity = Granularity.MONTHLY,
    ) -> None:
        self.token = token
        self.guild_id = guild_id
        self.time_range = time_range or TimeRange(
            start=datetime.now(UTC) - timedelta(days=180),
            end=datetime.now(UTC),
        )
        self.collect_user_stats = collect_user_stats
        self.granularity = granularity
        intents = discord.Intents.default()
        intents.members = True
        intents.message_content = True
        self._client = discord.Client(intents=intents)
        self._ready = asyncio.Event()

        @self._client.event
        async def on_ready() -> None:
            self._ready.set()

    @property
    def platform_name(self) -> str:
        return "discord"

    async def _ensure_connected(self) -> None:
        """Start the client if not already running and wait until ready."""
        if not self._client.is_ready():
            logger.debug("Connecting to Discord gateway...")
            asyncio.ensure_future(self._client.start(self.token))
            await self._ready.wait()
            logger.debug("Discord client ready")

    async def _get_guild(self) -> discord.Guild:
        guild = self._client.get_guild(self.guild_id)
        if guild is None:
            logger.error("Guild %s not found", self.guild_id)
            raise ValueError(
                f"Guild {self.guild_id} not found. "
                "Make sure the bot is a member of the server."
            )
        logger.debug("Found guild: %s", guild.name)
        return guild

    @staticmethod
    def _top_role(member: discord.Member) -> str:
        """Return the name of a member's highest non-@everyone role."""
        roles = [r for r in member.roles if r.name != "@everyone"]
        if not roles:
            return "Member"
        return max(roles, key=lambda r: r.position).name

    async def _scan_messages(
        self,
        guild: discord.Guild,
    ) -> tuple[list[dict[str, object]], dict[int, _UserAccumulator]]:
        """Scan all text channels in the time range.

        Returns raw message records (for Polars) and per-user accumulators.
        """
        records: list[dict[str, object]] = []
        users: dict[int, _UserAccumulator] = {}
        members_by_id = {m.id: m for m in guild.members}

        for channel in guild.text_channels:
            try:
                logger.debug("Scanning messages in #%s", channel.name)
                async for message in channel.history(
                    after=self.time_range.start,
                    before=self.time_range.end,
                    limit=None,
                    oldest_first=True,
                ):
                    member = members_by_id.get(message.author.id)
                    top_role = self._top_role(member) if member else "Member"

                    records.append(
                        {
                            "timestamp": message.created_at,
                            "user_id": message.author.id,
                            "is_bot": message.author.bot,
                            "top_role": top_role,
                        }
                    )

                    if self.collect_user_stats and not message.author.bot:
                        uid = message.author.id
                        if uid not in users:
                            users[uid] = _UserAccumulator(
                                user_id=uid,
                                username=str(message.author),
                                display_name=message.author.display_name,
                                is_bot=message.author.bot,
                            )
                        users[uid].record_message(channel.name, message.created_at)
            except discord.Forbidden:
                logger.warning("No permission to read #%s — skipping", channel.name)
                continue

        logger.debug("Scanned %d messages in time range", len(records))
        if self.collect_user_stats:
            logger.debug("Tracked %d unique users", len(users))
        return records, users

    def _build_user_stats(
        self,
        users: dict[int, _UserAccumulator],
        guild: discord.Guild,
    ) -> list[UserStats]:
        """Convert accumulators to UserStats with member metadata.

        Includes ALL human guild members, even those with 0 messages.
        """
        now = datetime.now(UTC)
        result: list[UserStats] = []

        for member in guild.members:
            if member.bot:
                continue

            acc = users.get(member.id)
            joined_at = member.joined_at

            days_since_join = (now - joined_at).days if joined_at else 0
            days_to_first = None
            if joined_at and acc and acc.first_message_at:
                days_to_first = (acc.first_message_at - joined_at).days

            roles = ", ".join(r.name for r in member.roles if r.name != "@everyone")

            result.append(
                UserStats(
                    user_id=member.id,
                    username=str(member),
                    display_name=member.display_name,
                    is_bot=False,
                    joined_at=joined_at.isoformat() if joined_at else "",
                    roles=roles,
                    message_count=acc.message_count if acc else 0,
                    first_message_at=(
                        acc.first_message_at.isoformat()
                        if acc and acc.first_message_at
                        else ""
                    ),
                    last_message_at=(
                        acc.last_message_at.isoformat()
                        if acc and acc.last_message_at
                        else ""
                    ),
                    days_since_join=days_since_join,
                    days_to_first_message=days_to_first,
                    channels_active_in=len(acc.channels) if acc else 0,
                )
            )

        # Sort by message count descending (active users first)
        result.sort(key=lambda u: u.message_count, reverse=True)
        return result

    async def scrape(self) -> PlatformStats:
        """Scrape Discord server statistics."""
        from ..analytics import (
            compute_join_trends,
            compute_message_buckets,
            compute_message_distribution,
            compute_role_breakdown,
        )

        await self._ensure_connected()
        guild = await self._get_guild()
        logger.info("Scraping stats for guild '%s' (%d)", guild.name, guild.id)
        logger.info(
            "Time range: %s to %s",
            self.time_range.start.strftime("%Y-%m-%d"),
            self.time_range.end.strftime("%Y-%m-%d"),
        )

        # Basic counts
        total_members = guild.member_count or 0
        text_channels = len(guild.text_channels)
        voice_channels = len(guild.voice_channels)
        categories = len(guild.categories)
        roles = len(guild.roles)
        emojis = len(guild.emojis)
        online_members = sum(
            1 for m in guild.members if m.status != discord.Status.offline and not m.bot
        )
        bot_count = sum(1 for m in guild.members if m.bot)
        human_members = total_members - bot_count

        # Scan messages in time range (+ per-user if requested)
        message_records, user_accumulators = await self._scan_messages(guild)

        # Build messages DataFrame for analytics
        messages_df = (
            pl.DataFrame(message_records)
            if message_records
            else pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime("us", "UTC"),
                    "user_id": pl.Int64,
                    "is_bot": pl.Boolean,
                    "top_role": pl.Utf8,
                }
            )
        )

        # Count active unique users from the scan
        active_users_in_range = len(user_accumulators) if self.collect_user_stats else 0

        stats: dict[str, str | int | float] = {
            "time_range_start": self.time_range.start.strftime("%Y-%m-%d"),
            "time_range_end": self.time_range.end.strftime("%Y-%m-%d"),
            "granularity": self.granularity.value,
            "total_members": total_members,
            "human_members": human_members,
            "bot_count": bot_count,
            "online_members": online_members,
            "text_channels": text_channels,
            "voice_channels": voice_channels,
            "categories": categories,
            "roles": roles,
            "emojis": emojis,
        }

        # --- Message buckets (configurable granularity) ---
        msg_buckets = compute_message_buckets(messages_df, self.granularity)
        for bucket, count in sorted(msg_buckets.items()):
            stats[f"messages_{bucket}"] = count

        stats["total_messages"] = sum(msg_buckets.values())

        if self.collect_user_stats:
            stats["active_users_in_range"] = active_users_in_range

        # --- Role-based message breakdown ---
        # Filter to human messages only for role breakdown
        human_msgs = messages_df.filter(pl.col("is_bot").not_())
        role_data = compute_role_breakdown(human_msgs)
        for role, count in role_data["counts"].items():  # type: ignore[assignment]
            stats[f"messages_by_role_{role}"] = count
        for role, pct in role_data["percentages"].items():
            stats[f"messages_pct_by_role_{role}"] = pct

        # --- Message distribution summary ---
        if self.collect_user_stats:
            # Build a DataFrame of ALL human members with message counts
            member_rows = []
            for member in guild.members:
                acc = user_accumulators.get(member.id)
                member_rows.append(
                    {
                        "user_id": member.id,
                        "message_count": acc.message_count if acc else 0,
                        "is_bot": member.bot,
                    }
                )
            members_df = pl.DataFrame(member_rows)
            dist = compute_message_distribution(members_df)
            stats.update(dist)

            # --- Join trend stats ---
            join_rows = []
            for member in guild.members:
                join_rows.append(
                    {
                        "joined_at": member.joined_at,
                        "is_bot": member.bot,
                    }
                )
            joins_df = pl.DataFrame(join_rows)
            join_trends = compute_join_trends(
                joins_df,
                self.granularity,
                self.time_range.start.strftime("%Y-%m-%d"),
                self.time_range.end.strftime("%Y-%m-%d"),
            )
            for bucket, count in sorted(join_trends.items()):
                stats[f"joins_{bucket}"] = count
            stats["total_joins_in_range"] = sum(join_trends.values())

        # Build per-user breakdown
        user_stats: list[UserStats] = []
        if self.collect_user_stats:
            user_stats = self._build_user_stats(user_accumulators, guild)
            logger.info("Collected stats for %d users", len(user_stats))

        return PlatformStats(
            platform=self.platform_name,
            server_name=guild.name,
            stats=stats,
            user_stats=user_stats,
            messages_df=messages_df if self.collect_user_stats else None,
        )

    async def close(self) -> None:
        """Disconnect the Discord client."""
        logger.debug("Closing Discord client")
        await self._client.close()
