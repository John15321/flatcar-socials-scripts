"""Base class for all platform scrapers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class UserStats:
    """Per-user statistics."""

    user_id: int | str
    username: str
    display_name: str
    is_bot: bool
    joined_at: str  # ISO format
    roles: str  # comma-separated
    message_count: int = 0
    first_message_at: str = ""
    last_message_at: str = ""
    days_since_join: int = 0
    days_to_first_message: int | None = None
    channels_active_in: int = 0


@dataclass
class PlatformStats:
    """Container for scraped platform statistics."""

    platform: str
    server_name: str
    scraped_at: datetime = field(default_factory=datetime.now)
    stats: dict[str, str | int | float] = field(default_factory=dict)
    user_stats: list[UserStats] = field(default_factory=list)

    def as_flat_dict(self) -> dict[str, str | int | float]:
        """Return all stats as a flat dict suitable for CSV output."""
        return {
            "platform": self.platform,
            "server_name": self.server_name,
            "scraped_at": self.scraped_at.isoformat(),
            **self.stats,
        }


class PlatformScraper(ABC):
    """Abstract base class for platform scrapers.

    Each platform module (Discord, Mastodon, etc.) should subclass this
    and implement the `scrape` method.
    """

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the name of the platform (e.g. 'discord')."""

    @abstractmethod
    async def scrape(self) -> PlatformStats:
        """Scrape statistics and return a PlatformStats object."""

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources (close connections, logout, etc.)."""
