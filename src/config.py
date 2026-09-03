"""Typed application settings, loaded from a .env file.

Configuration is the app's identity: the chat model, reasoning effort,
embedding model, OpenAI API key, and storage paths all come from a `.env`
file (see `.env.example`). Values found in the `.env` file take precedence
over inherited environment variables; the process environment fills any
gaps the file leaves unset.
"""

from __future__ import annotations

import os
from collections import ChainMap
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values, find_dotenv

DEFAULT_CHAT_MODEL = "gpt-5.6-luna"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_DATA_DIR = "data"


class ConfigError(Exception):
    """Raised when required configuration is missing or unset."""


@dataclass(frozen=True)
class Settings:
    """Typed application settings read from configuration."""

    openai_api_key: str
    chat_model: str
    reasoning_effort: str
    embedding_model: str
    data_dir: Path


def _env_value(env: Mapping[str, str], key: str, default: str) -> str:
    """Read ``key`` from ``env``; empty values fall back to ``default``."""
    return (env.get(key) or "").strip() or default


def load_settings(env_file: Path | None = None) -> Settings:
    """Load settings from ``env_file``, or from a ``.env`` discovered near the working directory.

    Values in the file win over inherited environment variables; the process
    environment fills any gaps the file leaves unset. Empty values are treated
    as unset.

    Raises:
        ConfigError: If required configuration (e.g. ``OPENAI_API_KEY``) is missing.
    """
    if env_file is None:
        env_file = Path(find_dotenv(usecwd=True))
    file_values: dict[str, str] = {}
    if env_file and env_file.is_file():
        file_values = {
            key: value for key, value in dotenv_values(env_file).items() if value is not None
        }

    env: Mapping[str, str] = ChainMap(file_values, os.environ)

    openai_api_key = (env.get("OPENAI_API_KEY") or "").strip()
    if not openai_api_key:
        raise ConfigError(
            "Missing required configuration: OPENAI_API_KEY is not set.\n"
            "Create a .env file in the project root (copy .env.example) containing:\n"
            "    OPENAI_API_KEY=sk-...\n"
            "Get a key at https://platform.openai.com/api-keys."
        )

    return Settings(
        openai_api_key=openai_api_key,
        chat_model=_env_value(env, "DOC_QA_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        reasoning_effort=_env_value(env, "DOC_QA_REASONING_EFFORT", DEFAULT_REASONING_EFFORT),
        embedding_model=_env_value(env, "DOC_QA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        data_dir=Path(_env_value(env, "DOC_QA_DATA_DIR", DEFAULT_DATA_DIR)),
    )
