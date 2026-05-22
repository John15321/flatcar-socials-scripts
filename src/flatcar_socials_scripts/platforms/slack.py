"""Slack platform scraper.

Connects to a Slack workspace via bot or user token and collects
statistics for a given channel, such as member count, messages per
month, etc.
"""

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from ..timerange import TimeRange
from .base import PlatformScraper, PlatformStats, UserStats

logger = logging.getLogger(__name__)


@dataclass
class _SlackUserAccumulator:
    """Internal accumulator for per-user message data during scan."""

    user_id: str
    display_name: str
    message_count: int = 0
    first_message_at: datetime | None = None
    last_message_at: datetime | None = None

    def record_message(self, timestamp: datetime) -> None:
        self.message_count += 1
        if self.first_message_at is None or timestamp < self.first_message_at:
            self.first_message_at = timestamp
        if self.last_message_at is None or timestamp > self.last_message_at:
            self.last_message_at = timestamp


class SlackScraper(PlatformScraper):
    """Scrape statistics from a Slack channel."""

    def __init__(
        self,
        token: str,
        channel_id: str,
        time_range: TimeRange | None = None,
        collect_user_stats: bool = False,
    ) -> None:
        self.token = token
        self.channel_id = channel_id
        self.time_range = time_range or TimeRange(
            start=datetime.now(UTC) - timedelta(days=180),
            end=datetime.now(UTC),
        )
        self.collect_user_stats = collect_user_stats
        self._client = WebClient(token=token)
        self._user_cache: dict[str, dict[str, str]] = {}

    @property
    def platform_name(self) -> str:
        return "slack"

    def _get_channel_info(self) -> dict[str, str]:
        """Get channel name, topic, purpose."""
        try:
            resp = self._client.conversations_info(channel=self.channel_id)
            channel = resp["channel"]
            return {
                "channel_name": channel.get("name", ""),
                "channel_topic": channel.get("topic", {}).get("value", ""),
                "channel_purpose": channel.get("purpose", {}).get("value", ""),
                "is_private": str(channel.get("is_private", False)),
            }
        except SlackApiError as e:
            logger.error("Failed to get channel info: %s", e.response["error"])
            raise ValueError(
                f"Failed to get channel info: {e.response['error']}"
            ) from e

    def _get_members(self) -> list[str]:
        """Get all member user IDs in the channel (paginated)."""
        members: list[str] = []
        cursor = None
        while True:
            try:
                resp = self._client.conversations_members(
                    channel=self.channel_id,
                    cursor=cursor,
                    limit=200,
                )
                members.extend(resp["members"])
                meta: dict[str, Any] = resp.get("response_metadata", {})
                cursor = meta.get("next_cursor", "")
                if not cursor:
                    break
            except SlackApiError as e:
                logger.error("Failed to get members: %s", e.response["error"])
                raise ValueError(
                    f"Failed to get channel members: {e.response['error']}"
                ) from e
        return members

    def _get_user_info(self, user_id: str) -> dict[str, str]:
        """Get user info, with caching."""
        if user_id in self._user_cache:
            return self._user_cache[user_id]
        try:
            resp = self._client.users_info(user=user_id)
            user = resp["user"]
            info = {
                "display_name": (
                    user.get("profile", {}).get("display_name")
                    or user.get("profile", {}).get("real_name")
                    or user.get("name", user_id)
                ),
                "real_name": user.get("profile", {}).get("real_name", ""),
                "is_bot": str(user.get("is_bot", False)),
                "is_admin": str(user.get("is_admin", False)),
                "is_owner": str(user.get("is_owner", False)),
            }
            self._user_cache[user_id] = info
            return info
        except SlackApiError:
            logger.debug("Could not fetch user info for %s", user_id)
            fallback = {
                "display_name": user_id,
                "real_name": "",
                "is_bot": "False",
                "is_admin": "False",
                "is_owner": "False",
            }
            self._user_cache[user_id] = fallback
            return fallback

    def _scan_messages(
        self,
    ) -> tuple[Counter[str], dict[str, _SlackUserAccumulator]]:
        """Scan messages in the time range.

        Returns monthly message counts and per-user accumulators.
        """
        monthly: Counter[str] = Counter()
        users: dict[str, _SlackUserAccumulator] = {}

        oldest = str(self.time_range.start.timestamp())
        latest = str(self.time_range.end.timestamp())

        cursor = None
        total_scanned = 0

        while True:
            try:
                resp = self._client.conversations_history(
                    channel=self.channel_id,
                    oldest=oldest,
                    latest=latest,
                    cursor=cursor,
                    limit=200,
                )
            except SlackApiError as e:
                logger.warning("Error fetching messages: %s", e.response["error"])
                break

            messages: list[dict[str, Any]] = resp.get("messages", [])  # type: ignore[assignment]
            for message in messages:
                # Skip subtypes like join/leave/bot messages
                if message.get("subtype"):
                    continue

                total_scanned += 1
                ts = float(message["ts"])
                dt = datetime.fromtimestamp(ts, tz=UTC)
                key = dt.strftime("%Y-%m")
                monthly[key] += 1

                if self.collect_user_stats:
                    sender = message.get("user", "")
                    if not sender:
                        continue
                    if sender not in users:
                        user_info = self._get_user_info(sender)
                        users[sender] = _SlackUserAccumulator(
                            user_id=sender,
                            display_name=user_info["display_name"],
                        )
                    users[sender].record_message(dt)

            if not resp.get("has_more", False):
                break
            resp_meta: dict[str, Any] = resp.get("response_metadata", {})  # type: ignore[assignment]
            cursor = resp_meta.get("next_cursor", "")
            if not cursor:
                break

        logger.debug("Scanned %d messages in time range", total_scanned)
        if self.collect_user_stats:
            logger.debug("Tracked %d unique users", len(users))
        return monthly, users

    def _build_user_stats(
        self,
        users: dict[str, _SlackUserAccumulator],
        members: list[str],
    ) -> list[UserStats]:
        """Convert accumulators to UserStats.

        Includes ALL channel members, not just those who sent messages.
        """
        # Ensure every member has an accumulator (0 messages if silent)
        for user_id in members:
            if user_id not in users:
                user_info = self._get_user_info(user_id)
                users[user_id] = _SlackUserAccumulator(
                    user_id=user_id,
                    display_name=user_info["display_name"],
                )

        result: list[UserStats] = []
        for acc in sorted(users.values(), key=lambda u: u.message_count, reverse=True):
            user_info = self._get_user_info(acc.user_id)
            is_bot = user_info.get("is_bot", "False") == "True"
            roles: list[str] = []
            if user_info.get("is_owner") == "True":
                roles.append("owner")
            if user_info.get("is_admin") == "True":
                roles.append("admin")

            result.append(
                UserStats(
                    user_id=acc.user_id,
                    username=acc.user_id,
                    display_name=acc.display_name,
                    is_bot=is_bot,
                    joined_at="",
                    roles=",".join(roles) if roles else "member",
                    message_count=acc.message_count,
                    first_message_at=(
                        acc.first_message_at.isoformat() if acc.first_message_at else ""
                    ),
                    last_message_at=(
                        acc.last_message_at.isoformat() if acc.last_message_at else ""
                    ),
                    days_since_join=0,
                    days_to_first_message=None,
                    channels_active_in=1,
                )
            )

        return result

    async def scrape(self) -> PlatformStats:
        """Scrape Slack channel statistics."""
        logger.info("Scraping stats for channel %s", self.channel_id)
        logger.info(
            "Time range: %s to %s",
            self.time_range.start.strftime("%Y-%m-%d"),
            self.time_range.end.strftime("%Y-%m-%d"),
        )

        # Get channel info
        channel_info = self._get_channel_info()
        channel_name = channel_info.get("channel_name", self.channel_id)

        # Get members
        members = self._get_members()
        total_members = len(members)

        # Count bots, admins, owners
        bot_count = 0
        admin_count = 0
        owner_count = 0
        if self.collect_user_stats:
            for uid in members:
                info = self._get_user_info(uid)
                if info.get("is_bot") == "True":
                    bot_count += 1
                if info.get("is_admin") == "True":
                    admin_count += 1
                if info.get("is_owner") == "True":
                    owner_count += 1

        # Scan messages
        messages_per_month, user_accumulators = self._scan_messages()

        stats: dict[str, str | int | float] = {
            "time_range_start": self.time_range.start.strftime("%Y-%m-%d"),
            "time_range_end": self.time_range.end.strftime("%Y-%m-%d"),
            "total_members": total_members,
            "channel_name": channel_name,
            "channel_topic": channel_info.get("channel_topic", ""),
            "channel_purpose": channel_info.get("channel_purpose", ""),
            "is_private": channel_info.get("is_private", ""),
        }

        if self.collect_user_stats:
            stats["bot_count"] = bot_count
            stats["admin_count"] = admin_count
            stats["owner_count"] = owner_count

        for month, count in sorted(messages_per_month.items()):
            stats[f"messages_{month}"] = count

        total_messages = sum(messages_per_month.values())
        stats["total_messages"] = total_messages

        if self.collect_user_stats:
            stats["active_users_in_range"] = len(user_accumulators)

        # Build per-user breakdown
        user_stats: list[UserStats] = []
        if self.collect_user_stats:
            user_stats = self._build_user_stats(user_accumulators, members)
            logger.info("Collected stats for %d users", len(user_stats))

        return PlatformStats(
            platform=self.platform_name,
            server_name=channel_name,
            stats=stats,
            user_stats=user_stats,
        )

    async def close(self) -> None:
        """No persistent connection to close for Slack SDK."""
        logger.debug("Slack scraper cleanup complete")
