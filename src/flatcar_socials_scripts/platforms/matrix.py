"""Matrix platform scraper.

Connects to a Matrix homeserver via access token and collects statistics
for a given room, such as member count, messages per time bucket, etc.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import polars as pl
from nio import (  # type: ignore[import-untyped]
    AsyncClient,
    JoinedMembersError,
    MessageDirection,
    RoomGetStateEventError,
    RoomMessagesError,
    RoomResolveAliasError,
    SyncError,
)

from ..timerange import Granularity, TimeRange
from .base import PlatformScraper, PlatformStats, UserStats

logger = logging.getLogger(__name__)


@dataclass
class _MatrixUserAccumulator:
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


class MatrixScraper(PlatformScraper):
    """Scrape statistics from a Matrix room."""

    def __init__(
        self,
        homeserver: str,
        token: str,
        room_id: str,
        time_range: TimeRange | None = None,
        collect_user_stats: bool = False,
        granularity: Granularity = Granularity.MONTHLY,
    ) -> None:
        self.homeserver = homeserver
        self.token = token
        self.room_id = room_id
        self.time_range = time_range or TimeRange(
            start=datetime.now(UTC) - timedelta(days=180),
            end=datetime.now(UTC),
        )
        self.collect_user_stats = collect_user_stats
        self.granularity = granularity
        self._client = AsyncClient(homeserver)
        self._resolved_room_id: str = ""
        self._sync_token: str = ""

    @property
    def platform_name(self) -> str:
        return "matrix"

    async def _ensure_connected(self) -> None:
        """Set up the client with access token and do initial sync."""
        self._client.access_token = self.token

        # Resolve room alias to room ID if needed
        if self.room_id.startswith("#"):
            logger.debug("Resolving room alias %s", self.room_id)
            resp = await self._client.room_resolve_alias(self.room_id)
            if isinstance(resp, RoomResolveAliasError):
                raise ValueError(
                    f"Could not resolve room alias {self.room_id}: {resp.message}"
                )
            self._resolved_room_id = resp.room_id
            logger.debug("Resolved to room ID: %s", self._resolved_room_id)
        else:
            self._resolved_room_id = self.room_id

        # Minimal sync to get pagination tokens
        logger.debug("Performing initial sync...")
        sync_filter = {
            "room": {
                "timeline": {"limit": 1},
                "state": {"types": []},
            },
            "presence": {"types": []},
            "account_data": {"types": []},
        }
        resp = await self._client.sync(timeout=30000, sync_filter=sync_filter)
        if isinstance(resp, SyncError):
            raise ValueError(f"Sync failed: {resp.message}")
        self._sync_token = resp.next_batch
        logger.debug("Sync complete, got pagination token")

    async def _get_members(self) -> list[tuple[str, str]]:
        """Get joined members. Returns list of (user_id, display_name)."""
        resp = await self._client.joined_members(self._resolved_room_id)
        if isinstance(resp, JoinedMembersError):
            logger.error("Failed to get members: %s", resp.message)
            raise ValueError(f"Failed to get room members: {resp.message}")
        return [(m.user_id, m.display_name or m.user_id) for m in resp.members]

    async def _get_room_info(self) -> dict[str, str]:
        """Get room name, topic, canonical alias from state."""
        info: dict[str, str] = {}

        # Room name
        resp = await self._client.room_get_state_event(
            self._resolved_room_id, "m.room.name"
        )
        if not isinstance(resp, RoomGetStateEventError):
            info["room_name"] = resp.content.get("name", "")

        # Room topic
        resp = await self._client.room_get_state_event(
            self._resolved_room_id, "m.room.topic"
        )
        if not isinstance(resp, RoomGetStateEventError):
            info["room_topic"] = resp.content.get("topic", "")

        # Canonical alias
        resp = await self._client.room_get_state_event(
            self._resolved_room_id, "m.room.canonical_alias"
        )
        if not isinstance(resp, RoomGetStateEventError):
            info["canonical_alias"] = resp.content.get("alias", "")

        return info

    async def _get_power_levels(self) -> dict[str, int]:
        """Get power levels for users in the room."""
        resp = await self._client.room_get_state_event(
            self._resolved_room_id, "m.room.power_levels"
        )
        if isinstance(resp, RoomGetStateEventError):
            return {}
        users: dict[str, int] = resp.content.get("users", {})
        return users

    async def _scan_messages(
        self,
    ) -> tuple[list[dict[str, object]], dict[str, _MatrixUserAccumulator]]:
        """Scan messages in the time range.

        Returns raw message records (for Polars) and per-user accumulators.
        """
        records: list[dict[str, object]] = []
        users: dict[str, _MatrixUserAccumulator] = {}

        time_start_ms = int(self.time_range.start.timestamp() * 1000)
        time_end_ms = int(self.time_range.end.timestamp() * 1000)

        start_token = self._sync_token
        total_scanned = 0

        while True:
            resp = await self._client.room_messages(
                self._resolved_room_id,
                start=start_token,
                direction=MessageDirection.back,
                limit=100,
                message_filter={"types": ["m.room.message"]},
            )

            if isinstance(resp, RoomMessagesError):
                logger.warning("Error fetching messages: %s", resp.message)
                break

            if not resp.chunk:
                break

            reached_start = False
            for event in resp.chunk:
                ts = event.server_timestamp
                if ts < time_start_ms:
                    reached_start = True
                    break
                if ts > time_end_ms:
                    continue

                total_scanned += 1
                dt = datetime.fromtimestamp(ts / 1000, tz=UTC)
                sender = event.sender

                records.append(
                    {
                        "timestamp": dt,
                        "user_id": sender,
                        "is_bot": False,
                    }
                )

                if self.collect_user_stats:
                    if sender not in users:
                        users[sender] = _MatrixUserAccumulator(
                            user_id=sender,
                            display_name=sender,
                        )
                    users[sender].record_message(dt)

            if reached_start:
                break

            if resp.end is None:
                break
            start_token = resp.end

        logger.debug("Scanned %d messages in time range", total_scanned)
        if self.collect_user_stats:
            logger.debug("Tracked %d unique users", len(users))
        return records, users

    def _build_user_stats(
        self,
        users: dict[str, _MatrixUserAccumulator],
        members: list[tuple[str, str]],
        power_levels: dict[str, int],
    ) -> list[UserStats]:
        """Convert accumulators to UserStats.

        Includes ALL joined members, not just those who sent messages.
        """
        members_by_id = dict(members)

        # Ensure every joined member has an accumulator (0 messages if silent)
        for user_id, display_name in members:
            if user_id not in users:
                users[user_id] = _MatrixUserAccumulator(
                    user_id=user_id,
                    display_name=display_name,
                )

        result: list[UserStats] = []
        for acc in sorted(users.values(), key=lambda u: u.message_count, reverse=True):
            display_name = members_by_id.get(acc.user_id, acc.user_id)
            power_level = power_levels.get(acc.user_id, 0)

            result.append(
                UserStats(
                    user_id=acc.user_id,
                    username=acc.user_id,
                    display_name=display_name,
                    is_bot=False,
                    joined_at="",
                    roles=f"power_level:{power_level}",
                    message_count=acc.message_count,
                    first_message_at=(
                        acc.first_message_at.isoformat() if acc.first_message_at else ""
                    ),
                    last_message_at=(
                        acc.last_message_at.isoformat() if acc.last_message_at else ""
                    ),
                    days_since_join=0,
                    days_to_first_message=None,
                    channels_active_in=1 if acc.message_count > 0 else 0,
                )
            )

        return result

    async def scrape(self) -> PlatformStats:
        """Scrape Matrix room statistics."""
        from ..analytics import compute_message_buckets, compute_message_distribution

        await self._ensure_connected()

        room_id = self._resolved_room_id
        logger.info("Scraping stats for room %s", room_id)
        logger.info(
            "Time range: %s to %s",
            self.time_range.start.strftime("%Y-%m-%d"),
            self.time_range.end.strftime("%Y-%m-%d"),
        )

        # Get room info
        room_info = await self._get_room_info()
        room_name = room_info.get("room_name", room_id)

        # Get members
        members = await self._get_members()
        total_members = len(members)

        # Get power levels
        power_levels = await self._get_power_levels()
        admins = sum(1 for lvl in power_levels.values() if lvl >= 100)
        moderators = sum(1 for lvl in power_levels.values() if 50 <= lvl < 100)

        # Scan messages
        message_records, user_accumulators = await self._scan_messages()

        # Build Polars DataFrame from raw records
        messages_df = (
            pl.DataFrame(message_records)
            if message_records
            else pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime("us", "UTC"),
                    "user_id": pl.Utf8,
                    "is_bot": pl.Boolean,
                }
            )
        )

        active_users_in_range = len(user_accumulators) if self.collect_user_stats else 0

        stats: dict[str, str | int | float] = {
            "time_range_start": self.time_range.start.strftime("%Y-%m-%d"),
            "time_range_end": self.time_range.end.strftime("%Y-%m-%d"),
            "granularity": self.granularity.value,
            "total_members": total_members,
            "admins": admins,
            "moderators": moderators,
            "room_alias": room_info.get("canonical_alias", ""),
            "room_topic": room_info.get("room_topic", ""),
        }

        # Message buckets (configurable granularity)
        msg_buckets = compute_message_buckets(messages_df, self.granularity)
        for bucket, count in sorted(msg_buckets.items()):
            stats[f"messages_{bucket}"] = count

        total_messages = sum(msg_buckets.values())
        stats["total_messages"] = total_messages

        if self.collect_user_stats:
            stats["active_users_in_range"] = active_users_in_range

            # Message distribution summary
            member_rows = [
                {
                    "user_id": uid,
                    "message_count": (
                        user_accumulators[uid].message_count
                        if uid in user_accumulators
                        else 0
                    ),
                    "is_bot": False,
                }
                for uid, _ in members
            ]
            members_dist_df = pl.DataFrame(member_rows)
            dist = compute_message_distribution(members_dist_df)
            stats.update(dist)

        # Build per-user breakdown
        user_stats: list[UserStats] = []
        if self.collect_user_stats:
            user_stats = self._build_user_stats(
                user_accumulators, members, power_levels
            )
            logger.info("Collected stats for %d users", len(user_stats))

        return PlatformStats(
            platform=self.platform_name,
            server_name=room_name,
            stats=stats,
            user_stats=user_stats,
            messages_df=messages_df if self.collect_user_stats else None,
        )

    async def close(self) -> None:
        """Close the Matrix client."""
        logger.debug("Closing Matrix client")
        await self._client.close()
