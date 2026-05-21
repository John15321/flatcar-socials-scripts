"""Discord platform scraper.

Connects to a Discord server via bot token and collects statistics
such as member count, channel count, posts per month, etc.
"""

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import discord

from ..timerange import TimeRange
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
    ) -> None:
        self.token = token
        self.guild_id = guild_id
        self.time_range = time_range or TimeRange(
            start=datetime.now(UTC) - timedelta(days=180),
            end=datetime.now(UTC),
        )
        self.collect_user_stats = collect_user_stats
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

    async def _scan_messages(
        self,
        guild: discord.Guild,
    ) -> tuple[Counter[str], dict[int, _UserAccumulator]]:
        """Scan all text channels in the time range.

        Returns monthly message counts and per-user accumulators.
        """
        monthly: Counter[str] = Counter()
        users: dict[int, _UserAccumulator] = {}

        for channel in guild.text_channels:
            try:
                logger.debug("Scanning messages in #%s", channel.name)
                async for message in channel.history(
                    after=self.time_range.start,
                    before=self.time_range.end,
                    limit=None,
                    oldest_first=True,
                ):
                    key = message.created_at.strftime("%Y-%m")
                    monthly[key] += 1

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

        logger.debug("Message counts by month: %s", dict(monthly))
        if self.collect_user_stats:
            logger.debug("Tracked %d unique users", len(users))
        return monthly, users

    def _build_user_stats(
        self,
        users: dict[int, _UserAccumulator],
        guild: discord.Guild,
    ) -> list[UserStats]:
        """Convert accumulators to UserStats with member metadata."""
        now = datetime.now(UTC)
        result: list[UserStats] = []
        members_by_id = {m.id: m for m in guild.members}

        for acc in sorted(users.values(), key=lambda u: u.message_count, reverse=True):
            member = members_by_id.get(acc.user_id)
            joined_at = member.joined_at if member else None

            days_since_join = (now - joined_at).days if joined_at else 0
            days_to_first = None
            if joined_at and acc.first_message_at:
                days_to_first = (acc.first_message_at - joined_at).days

            roles = ""
            if member:
                roles = ", ".join(r.name for r in member.roles if r.name != "@everyone")

            result.append(
                UserStats(
                    user_id=acc.user_id,
                    username=acc.username,
                    display_name=acc.display_name,
                    is_bot=acc.is_bot,
                    joined_at=joined_at.isoformat() if joined_at else "",
                    roles=roles,
                    message_count=acc.message_count,
                    first_message_at=(
                        acc.first_message_at.isoformat() if acc.first_message_at else ""
                    ),
                    last_message_at=(
                        acc.last_message_at.isoformat() if acc.last_message_at else ""
                    ),
                    days_since_join=days_since_join,
                    days_to_first_message=days_to_first,
                    channels_active_in=len(acc.channels),
                )
            )

        return result

    async def scrape(self) -> PlatformStats:
        """Scrape Discord server statistics."""
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
        messages_per_month, user_accumulators = await self._scan_messages(guild)

        # Count active unique users from the scan
        active_users_in_range = len(user_accumulators) if self.collect_user_stats else 0

        stats: dict[str, str | int | float] = {
            "time_range_start": self.time_range.start.strftime("%Y-%m-%d"),
            "time_range_end": self.time_range.end.strftime("%Y-%m-%d"),
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

        # Add per-month message counts
        for month, count in sorted(messages_per_month.items()):
            stats[f"messages_{month}"] = count

        total_messages = sum(messages_per_month.values())
        stats["total_messages"] = total_messages

        if self.collect_user_stats:
            stats["active_users_in_range"] = active_users_in_range

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
        )

    async def close(self) -> None:
        """Disconnect the Discord client."""
        logger.debug("Closing Discord client")
        await self._client.close()
