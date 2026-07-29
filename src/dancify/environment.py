"""Shared process-environment and .env loading."""

from __future__ import annotations

from pathlib import Path

from dotenv import find_dotenv, load_dotenv


def load_environment(path: str | Path | None = None) -> bool:
    """Load the nearest .env without replacing existing process variables."""

    dotenv_path = str(path) if path is not None else find_dotenv(usecwd=True)
    if not dotenv_path:
        return False
    return load_dotenv(dotenv_path=dotenv_path, override=False)
