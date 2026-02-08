"""Configuration loader for the DocIntel Discord bot."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    discord_token: str
    api_base_url: str = "http://localhost:8000"
    request_timeout: float = 15.0

    @staticmethod
    def load() -> "Settings":
        token = os.getenv("DISCORD_BOT_TOKEN")
        if not token:
            raise RuntimeError("DISCORD_BOT_TOKEN is not set")
        api_url = os.getenv("DOCINTEL_API_BASE", "http://localhost:8000").rstrip("/")
        timeout = float(os.getenv("DOCINTEL_REQUEST_TIMEOUT", "15"))
        return Settings(
            discord_token=token,
            api_base_url=api_url,
            request_timeout=timeout,
        )
